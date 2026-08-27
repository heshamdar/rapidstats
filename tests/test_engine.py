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


# ---------------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------------
#
# `Config` held a module-level global, so `Config.engine(...)` was process-wide: in a
# library whose own default execution model is `_run_concurrent(..., executor="threads")`,
# one thread's `with` block was visible to every other, and two threads entering it with
# different engines restored each other's values on exit.
#
# The fix is a `ContextVar` -- but a bare swap trades one bug for another, because
# `ThreadPoolExecutor` does not propagate the caller's context to its workers. Both
# properties are asserted, because getting either alone is wrong.


def test_engine_context_does_not_leak_across_threads():
    """One thread's `with` block must not be visible to another."""
    import threading

    entered = threading.Event()
    observed: list[str] = []
    release = threading.Event()

    def setter():
        with rs.Config.engine("streaming"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=setter)
    thread.start()
    try:
        assert entered.wait(timeout=5), "setter thread never entered the block"
        observed.append(rs.Config.get_engine())
    finally:
        release.set()
        thread.join(timeout=5)

    assert observed == ["in-memory"], (
        f"the main thread saw {observed[0]!r} while another thread was inside "
        f"`Config.engine('streaming')`; the setting is not thread-scoped"
    )
    assert rs.Config.get_engine() == "in-memory"


def test_library_fan_out_sees_a_caller_set_engine():
    """A caller-set engine must still reach the library's own worker threads.

    `Bootstrap.run` executes `stat_func` on a `ThreadPoolExecutor`, and that function may
    call any metric in the library -- so it has to observe the engine the caller chose.
    `ThreadPoolExecutor` does not propagate context on its own; `_run_concurrent` copies
    it. Without that copy this test fails while the isolation test above still passes.
    """
    import threading

    frame = pl.DataFrame({"y": np.arange(64, dtype=float)})
    seen: set[str] = set()
    lock = threading.Lock()

    def stat_func(df: pl.DataFrame) -> float:
        with lock:
            seen.add(rs.Config.get_engine())

        return float(df["y"].mean())

    with rs.Config.engine("streaming"):
        rs.Bootstrap(iterations=8, seed=SEED, n_jobs=4).run(frame, stat_func)

    assert seen == {"streaming"}, (
        f"worker threads saw {sorted(seen)} while the caller was inside "
        f"`Config.engine('streaming')`; the context is not reaching the fan-out"
    )


def test_collect_does_not_forward_an_engine_to_narwhals():
    """`selection.py` pipes narwhals frames through `_collect`.

    `narwhals.stable.v1.LazyFrame.collect` was `collect(self)` until ~1.30, so forwarding
    `engine=` raised `TypeError` on the floor this project declared. An engine is
    meaningless for a non-polars backend regardless.
    """
    import narwhals.stable.v1 as nw

    from rapidstats._utils import _collect

    lf = nw.from_native(pl.LazyFrame({"a": [1, 2, 3]}))

    with rs.Config.engine("streaming"):
        result = _collect(lf)

    assert result.shape == (3, 1)


def test_collect_still_forwards_an_engine_to_polars():
    """The guard above must not disable engine selection for polars frames."""
    from rapidstats._utils import _collect

    seen: list[str] = []

    class _Spy(pl.LazyFrame):
        def collect(self, **kwargs):
            seen.append(kwargs.get("engine"))

            return pl.DataFrame({"a": [1]})

    with rs.Config.engine("streaming"):
        _collect(_Spy())

    assert seen == ["streaming"]
