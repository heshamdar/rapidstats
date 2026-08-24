"""The polars execution engine must be selectable, and must not change results.

Every `.collect()` in the library used the in-memory engine implicitly, with no way for
a caller to choose. Polars state that the streaming engine "will soon become the default
engine", and it is the engine that bounds memory and parallelises by morsel -- so which
one runs needs to be a decision, not an accident.

Measured on this codebase, streaming is *not* the faster choice for these query shapes:
they are dominated by sort, `cum_sum` and `join_asof`, which are order-dependent and
resist morsel parallelism. See `benchmarks/bench_engine.py`. The default therefore stays
in-memory, but the choice is now explicit and overridable, and these tests pin the
contract that the two engines agree.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 2_000
SEED = 208
ENGINES = ["in-memory", "streaming"]


@pytest.fixture(scope="module")
def data() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    protected = rand.choice([True, False], N)

    return {
        "y_true": rand.rand(N) < 0.3 + 0.4 * y_score,
        "y_score": y_score,
        "protected": protected,
        "control": ~protected,
    }


def test_default_engine_is_in_memory():
    assert rs.Config.get_engine() == "in-memory"


def test_set_engine_round_trips():
    try:
        rs.Config.set_engine("streaming")
        assert rs.Config.get_engine() == "streaming"
    finally:
        rs.Config.set_engine("in-memory")

    assert rs.Config.get_engine() == "in-memory"


def test_rejects_unknown_engine():
    with pytest.raises(ValueError, match="engine"):
        rs.Config.set_engine("quantum")

    assert rs.Config.get_engine() == "in-memory", "a rejected value must not be applied"


def test_engine_context_manager_restores_previous():
    with rs.Config.engine("streaming"):
        assert rs.Config.get_engine() == "streaming"

    assert rs.Config.get_engine() == "in-memory"


def test_engine_context_manager_restores_on_error():
    with pytest.raises(RuntimeError):
        with rs.Config.engine("streaming"):
            raise RuntimeError("boom")

    assert rs.Config.get_engine() == "in-memory"


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame, what: str):
    assert a.columns == b.columns, f"{what}: columns differ"
    assert a.height == b.height, f"{what}: height differs"

    for col in a.columns:
        if a[col].dtype.is_numeric():
            left = a[col].cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            right = b[col].cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            left = np.where(a[col].is_null().to_numpy(), np.nan, left)
            right = np.where(b[col].is_null().to_numpy(), np.nan, right)
            np.testing.assert_allclose(
                left,
                right,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{what}: column {col!r} differs between engines",
            )
        else:
            assert a[col].to_list() == b[col].to_list(), f"{what}: {col!r} differs"


@pytest.mark.parametrize(
    "call,sort_by",
    [
        pytest.param(
            lambda d: rs.metrics.confusion_matrix_at_thresholds(
                d["y_true"], d["y_score"], strategy="cum_sum"
            ),
            ["threshold", "metric"],
            id="confusion_matrix_at_thresholds",
        ),
        pytest.param(
            lambda d: rs.metrics.adverse_impact_ratio_at_thresholds(
                d["y_score"], d["protected"], d["control"], strategy="cum_sum"
            ),
            ["threshold"],
            id="adverse_impact_ratio_at_thresholds",
        ),
        pytest.param(
            lambda d: rs.metrics.predicted_positive_ratio_at_thresholds(
                d["y_score"], strategy="cum_sum"
            ),
            ["threshold"],
            id="predicted_positive_ratio_at_thresholds",
        ),
    ],
)
def test_engines_agree_on_threshold_curves(data, call, sort_by):
    """Switching engine is a performance decision; it must never change a number."""
    with rs.Config.engine("in-memory"):
        in_memory = call(data).sort(sort_by)

    with rs.Config.engine("streaming"):
        streaming = call(data).sort(sort_by)

    _frames_equal(in_memory, streaming, "threshold curve")


def test_engines_agree_on_average_precision(data):
    with rs.Config.engine("in-memory"):
        in_memory = rs.metrics.average_precision(data["y_true"], data["y_score"])

    with rs.Config.engine("streaming"):
        streaming = rs.metrics.average_precision(data["y_true"], data["y_score"])

    assert in_memory == pytest.approx(streaming, rel=0, abs=1e-12)


def test_engines_agree_on_psi(data):
    reference, current = data["y_score"][:1000], data["y_score"][1000:]

    with rs.Config.engine("in-memory"):
        in_memory = rs.drift.psi(reference, current)

    with rs.Config.engine("streaming"):
        streaming = rs.drift.psi(reference, current)

    assert in_memory == pytest.approx(streaming, rel=0, abs=1e-12)


def test_engines_agree_on_correlation_matrix():
    rand = np.random.RandomState(SEED)
    frame = pl.DataFrame({f"f{i}": rand.rand(500) for i in range(6)})

    with rs.Config.engine("in-memory"):
        in_memory = rs.correlation_matrix(frame)

    with rs.Config.engine("streaming"):
        streaming = rs.correlation_matrix(frame)

    _frames_equal(in_memory, streaming, "correlation_matrix")


def test_bootstrap_agrees_across_engines(data):
    """The bootstrap's Python paths collect too, so they must be engine-clean."""
    kwargs = dict(iterations=32, seed=SEED, thresholds=[0.25, 0.5, 0.75])

    with rs.Config.engine("in-memory"):
        in_memory = (
            rs.Bootstrap(iterations=32, seed=SEED)
            .confusion_matrix_at_thresholds(
                data["y_true"], data["y_score"], thresholds=kwargs["thresholds"]
            )
            .sort("threshold", "metric")
        )

    with rs.Config.engine("streaming"):
        streaming = (
            rs.Bootstrap(iterations=32, seed=SEED)
            .confusion_matrix_at_thresholds(
                data["y_true"], data["y_score"], thresholds=kwargs["thresholds"]
            )
            .sort("threshold", "metric")
        )

    _frames_equal(in_memory, streaming, "bootstrap confusion_matrix_at_thresholds")


def test_configured_engine_actually_reaches_polars(data, monkeypatch):
    """Guard against the setting being stored but never used.

    Asserting on results alone cannot catch that: the engines agree by design, so a
    config that is silently ignored would pass every other test in this file. This one
    intercepts `LazyFrame.collect` and checks what was actually requested.
    """
    import sys

    seen: list[object] = []
    original = pl.LazyFrame.collect

    def recording_collect(self, *args, **kwargs):
        # Only judge collects issued by our helper. Polars calls `LazyFrame.collect`
        # internally too -- `DataFrame.pivot` and `Series.hist` do on some versions --
        # and those legitimately carry no engine of ours. Bare `.collect()` calls that
        # bypass the helper are caught statically by
        # `test_no_bare_collect_calls_remain`.
        if sys._getframe(1).f_code.co_name == "_collect":
            seen.append(kwargs.get("engine"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", recording_collect)

    for engine in ENGINES:
        seen.clear()
        with rs.Config.engine(engine):
            rs.metrics.confusion_matrix_at_thresholds(
                data["y_true"], data["y_score"], strategy="cum_sum"
            )

        assert seen, "no collect() happened at all"
        assert set(seen) == {engine}, (
            f"configured engine {engine!r} but polars was asked for {set(seen)}"
        )


def test_no_bare_collect_calls_remain():
    """Every collect must route through `_collect`, or the engine is bypassed."""
    import pathlib

    import rapidstats

    package_dir = pathlib.Path(rapidstats.__file__).parent
    offenders = []
    for path in package_dir.rglob("*.py"):
        if path.name == "_utils.py":  # defines the helper
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if ".collect()" in line:
                offenders.append(f"{path.name}:{i}: {line.strip()}")

    assert not offenders, (
        "these bypass the configured engine; use `.pipe(_collect)`:\n  "
        + "\n  ".join(offenders)
    )
