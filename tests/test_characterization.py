"""Characterization tests: pin today's output so refactors cannot change it silently.

These are NOT correctness tests -- `test_metrics.py` already checks the numbers against
sklearn/scipy. These lock the *current* behaviour of code that later tranches rewrite
for speed (the threshold mapping, the bootstrap resampling strategy, the lazy plans),
so that any behavioural drift shows up as a diff rather than as a surprise.

Fixtures live in `tests/data/characterization/*.parquet` and are committed. Regenerate
deliberately, never casually:

    pytest tests/test_characterization.py --update-characterization

If a fixture legitimately changes (see the documented breaking changes in the plan),
regenerate it in the *same commit* as the change that caused it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs
from tests.paths import DAT_PATH

# These are deterministic computations, so the net is held to near-machine precision.
# `polars.testing.assert_frame_equal` defaults to rtol=1e-5, which is loose enough to
# hide a real refactor bug; it also renamed `rtol` -> `rel_tol` in 1.32.3, which would
# force version-sniffing in a file that must run across the whole supported range.
RTOL = 1e-12
ATOL = 1e-12

CHAR_PATH = DAT_PATH / "characterization"

SEED = 208
# Deliberately small: the pre-optimisation `cum_sum` bootstrap is super-linear, so a
# large frame here would dominate the whole suite's runtime.
N = 200
ITERATIONS = 25

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = ["standard", "percentile", "basic", "BCa"]

CASES: dict[str, callable] = {}


def case(name: str):
    def decorator(fn):
        if name in CASES:
            raise ValueError(f"duplicate characterization case {name!r}")
        CASES[name] = fn
        return fn

    return decorator


def _data() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    y_true = rand.choice([True, False], N)
    protected = rand.choice([True, False], N)
    sample_weight = rand.rand(N)

    return {
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_score > 0.5,
        "sample_weight": sample_weight,
        "protected": protected,
        "control": ~protected,
        "y_true_reg": rand.rand(N),
    }


DATA = _data()


def _as_frame(value) -> pl.DataFrame:
    """Normalise a scalar, tuple, dataclass or frame into a comparable DataFrame."""
    if isinstance(value, pl.DataFrame):
        return value
    if hasattr(value, "to_polars"):
        return value.to_polars()
    if isinstance(value, tuple):
        return pl.DataFrame(
            {"bound": ["lower", "point", "upper"], "value": [float(x) for x in value]}
        )

    return pl.DataFrame({"value": [float(value)]})


def _canonicalise(df: pl.DataFrame) -> pl.DataFrame:
    """Sort rows and columns so incidental ordering never fails a comparison.

    Row order out of `group_by` is not guaranteed, and `unique()` on the threshold
    column is unordered, so ordering is explicitly not part of what we pin.
    """
    df = df.select(sorted(df.columns))

    return df.sort(by=df.columns)


def _assert_frame_close(actual: pl.DataFrame, expected: pl.DataFrame, name: str):
    """Tight, version-portable frame comparison.

    Nulls and NaNs are treated as interchangeable missing markers: several metrics
    convert non-finite results to null via `_fill_infinite`/`fill_nan`, and which one
    you land on is an implementation detail we do not want to pin.
    """
    assert actual.columns == expected.columns, (
        f"{name}: columns differ\n  actual  : {actual.columns}\n"
        f"  expected: {expected.columns}"
    )
    assert actual.height == expected.height, (
        f"{name}: height differs ({actual.height} vs {expected.height})"
    )

    for col in expected.columns:
        a, e = actual[col], expected[col]

        if e.dtype.is_numeric():
            a_np = a.cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            e_np = e.cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            # Nulls surface as NaN once cast through Float64 numpy.
            a_np = np.where(a.is_null().to_numpy(), np.nan, a_np)
            e_np = np.where(e.is_null().to_numpy(), np.nan, e_np)

            np.testing.assert_allclose(
                a_np,
                e_np,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
                err_msg=f"{name}: column {col!r} drifted",
            )
        else:
            assert a.to_list() == e.to_list(), f"{name}: column {col!r} differs"


# --------------------------------------------------------------------------------
# Scalar metrics
# --------------------------------------------------------------------------------


@case("metrics_confusion_matrix")
def _():
    return rs.metrics.confusion_matrix(DATA["y_true"], DATA["y_pred"])


@case("metrics_confusion_matrix_weighted")
def _():
    return rs.metrics.confusion_matrix(
        DATA["y_true"], DATA["y_pred"], sample_weight=DATA["sample_weight"]
    )


@case("metrics_roc_auc")
def _():
    return rs.metrics.roc_auc(DATA["y_true"], DATA["y_score"])


@case("metrics_roc_auc_weighted")
def _():
    return rs.metrics.roc_auc(
        DATA["y_true"], DATA["y_score"], sample_weight=DATA["sample_weight"]
    )


@case("metrics_average_precision")
def _():
    return rs.metrics.average_precision(DATA["y_true"], DATA["y_score"])


@case("metrics_max_ks")
def _():
    return rs.metrics.max_ks(DATA["y_true"], DATA["y_score"])


@case("metrics_brier_loss")
def _():
    return rs.metrics.brier_loss(DATA["y_true"], DATA["y_score"])


@case("metrics_r2")
def _():
    return rs.metrics.r2(DATA["y_true_reg"], DATA["y_score"])


@case("metrics_mean_squared_error")
def _():
    return rs.metrics.mean_squared_error(DATA["y_true_reg"], DATA["y_score"])


@case("metrics_adverse_impact_ratio")
def _():
    return rs.metrics.adverse_impact_ratio(
        DATA["y_pred"], DATA["protected"], DATA["control"]
    )


# --------------------------------------------------------------------------------
# Threshold curves -- the code paths that Tranche 2a rewrites
# --------------------------------------------------------------------------------


@case("confusion_matrix_at_thresholds_cum_sum")
def _():
    return rs.metrics.confusion_matrix_at_thresholds(
        DATA["y_true"], DATA["y_score"], strategy="cum_sum"
    )


@case("confusion_matrix_at_thresholds_explicit")
def _():
    return rs.metrics.confusion_matrix_at_thresholds(
        DATA["y_true"], DATA["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
    )


@case("confusion_matrix_at_thresholds_weighted")
def _():
    return rs.metrics.confusion_matrix_at_thresholds(
        DATA["y_true"],
        DATA["y_score"],
        thresholds=THRESHOLDS,
        strategy="cum_sum",
        sample_weight=DATA["sample_weight"],
    )


@case("adverse_impact_ratio_at_thresholds_cum_sum")
def _():
    return rs.metrics.adverse_impact_ratio_at_thresholds(
        DATA["y_score"], DATA["protected"], DATA["control"], strategy="cum_sum"
    )


@case("adverse_impact_ratio_at_thresholds_explicit")
def _():
    return rs.metrics.adverse_impact_ratio_at_thresholds(
        DATA["y_score"],
        DATA["protected"],
        DATA["control"],
        thresholds=THRESHOLDS,
        strategy="cum_sum",
    )


@case("predicted_positive_ratio_at_thresholds")
def _():
    return rs.metrics.predicted_positive_ratio_at_thresholds(
        DATA["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
    )


# --------------------------------------------------------------------------------
# Bootstrap intervals -- the code paths that Tranche 2c/2d rewrite
# --------------------------------------------------------------------------------


def _bootstrap(method: str, sampling_method: str = "multinomial") -> rs.Bootstrap:
    return rs.Bootstrap(
        iterations=ITERATIONS,
        method=method,
        sampling_method=sampling_method,
        seed=SEED,
    )


for _method in METHODS:

    @case(f"bootstrap_roc_auc_{_method}")
    def _(method=_method):
        return _bootstrap(method).roc_auc(DATA["y_true"], DATA["y_score"])

    @case(f"bootstrap_mean_{_method}")
    def _(method=_method):
        return _bootstrap(method).mean(DATA["y_score"])

    @case(f"bootstrap_confusion_matrix_{_method}")
    def _(method=_method):
        return _bootstrap(method).confusion_matrix(DATA["y_true"], DATA["y_pred"])


for _sampling in ["poisson", "multinomial"]:

    @case(f"bootstrap_roc_auc_percentile_{_sampling}")
    def _(sampling=_sampling):
        return _bootstrap("percentile", sampling).roc_auc(
            DATA["y_true"], DATA["y_score"]
        )


# `Bootstrap.roc_auc` is handled entirely in Rust, so the cases above never exercise the
# Python resampling helpers. These do: `run` and the `cum_sum` paths route through
# `_poisson_sample` / `_multinomial_sample`, which Tranche 2c replaces with a
# weight-based equivalent.
for _sampling in ["poisson", "multinomial"]:

    @case(f"bootstrap_run_mean_{_sampling}")
    def _(sampling=_sampling):
        frame = pl.DataFrame({"y": DATA["y_score"]})

        return _bootstrap("percentile", sampling).run(
            frame, lambda df: float(df["y"].mean())
        )

    @case(f"bootstrap_cm_at_thresholds_percentile_{_sampling}")
    def _(sampling=_sampling):
        return _bootstrap("percentile", sampling).confusion_matrix_at_thresholds(
            DATA["y_true"], DATA["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
        )

    @case(f"bootstrap_air_at_thresholds_percentile_{_sampling}")
    def _(sampling=_sampling):
        return _bootstrap("percentile", sampling).adverse_impact_ratio_at_thresholds(
            DATA["y_score"],
            DATA["protected"],
            DATA["control"],
            thresholds=THRESHOLDS,
            strategy="cum_sum",
        )


# All four methods, including BCa: the `cum_sum` paths used to raise
# NotImplementedError for it, on an upstream polars bug that is now fixed.
for _method in METHODS:

    @case(f"bootstrap_cm_at_thresholds_{_method}")
    def _(method=_method):
        return _bootstrap(method).confusion_matrix_at_thresholds(
            DATA["y_true"], DATA["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
        )

    @case(f"bootstrap_air_at_thresholds_{_method}")
    def _(method=_method):
        return _bootstrap(method).adverse_impact_ratio_at_thresholds(
            DATA["y_score"],
            DATA["protected"],
            DATA["control"],
            thresholds=THRESHOLDS,
            strategy="cum_sum",
        )


@pytest.mark.parametrize("name", sorted(CASES))
def test_characterization(name: str, request):
    result = _canonicalise(_as_frame(CASES[name]()))
    path = CHAR_PATH / f"{name}.parquet"

    if request.config.getoption("--update-characterization"):
        CHAR_PATH.mkdir(parents=True, exist_ok=True)
        result.write_parquet(path)
        pytest.skip(f"regenerated {path.name}")

    if not path.exists():
        pytest.fail(
            f"missing characterization fixture {path}. Generate it with:\n"
            f"    pytest tests/test_characterization.py --update-characterization"
        )

    _assert_frame_close(result, pl.read_parquet(path), name)


def test_every_case_has_a_fixture():
    """Guard against a case being added but its fixture never committed."""
    missing = sorted(n for n in CASES if not (CHAR_PATH / f"{n}.parquet").exists())

    assert not missing, (
        f"characterization fixtures missing for {missing}; run "
        f"`pytest tests/test_characterization.py --update-characterization`"
    )
