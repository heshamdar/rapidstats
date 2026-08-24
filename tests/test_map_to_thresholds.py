"""Equivalence tests for `_map_to_thresholds`.

`_map_to_thresholds` answers: for each requested threshold `t`, which row of an
already-computed curve applies? The answer is the row with the *smallest* curve
threshold that is still `>= t`.

The original implementation expressed that as a cross join followed by a filter and a
per-target `min`, which is O(n*m) and dominates every bootstrapped threshold metric
(measured at n=m=8000: ~6.0s of a ~6.8s call). The same question is exactly what
`join_asof(strategy="forward")` answers in O(n log n).

`reference_map_to_thresholds` below is the original implementation, verbatim. It is the
oracle: the fast path must agree with it on every input, including the awkward ones --
targets outside the curve range, exact ties, duplicated targets, and empty inputs.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rapidstats.metrics import _map_to_thresholds


def reference_map_to_thresholds(pf, thresholds):
    """The original cross-join implementation, kept as the oracle."""
    if thresholds is None:
        return pf.lazy()

    lf = pf.lazy()
    target = pl.LazyFrame({"target_threshold": thresholds})

    mapping = (
        target.join(lf.select("threshold"), how="cross")
        .filter(pl.col("threshold").ge(pl.col("target_threshold")))
        .group_by("target_threshold")
        .agg(pl.col("threshold").min())
    )

    mapping = target.join(
        mapping,
        on="target_threshold",
        how="left",
        validate="1:1",
    )

    return (
        mapping.join(lf, on="threshold", how="left", validate="m:m")
        .rename({"threshold": "_threshold_actual"})
        .rename({"target_threshold": "threshold"})
    )


def _curve(thresholds, descending: bool = True) -> pl.LazyFrame:
    """A curve frame shaped like the confusion-matrix output feeding this function."""
    values = np.asarray(thresholds, dtype=float)

    return pl.LazyFrame(
        {
            "threshold": values,
            "tp": np.arange(len(values), dtype=float),
            "fp": np.arange(len(values), dtype=float) * 2.0,
        }
    ).sort("threshold", descending=descending)


def _compare(curve: pl.LazyFrame, targets, case: str):
    expected = reference_map_to_thresholds(curve, targets)
    actual = _map_to_thresholds(curve, targets)

    if targets is None:
        # Passthrough: both must return the curve untouched.
        assert actual.collect().equals(expected.collect()), case
        return

    sort_cols = ["threshold", "_threshold_actual"]
    expected_df = expected.collect().sort(sort_cols, nulls_last=True)
    actual_df = actual.collect().sort(sort_cols, nulls_last=True)

    assert actual_df.height == expected_df.height, (
        f"{case}: row count {actual_df.height} != {expected_df.height}"
    )

    for col in ["threshold", "_threshold_actual", "tp", "fp"]:
        assert actual_df[col].to_list() == expected_df[col].to_list(), (
            f"{case}: column {col!r} differs\n"
            f"  actual  : {actual_df[col].to_list()[:12]}\n"
            f"  expected: {expected_df[col].to_list()[:12]}"
        )


@pytest.mark.parametrize("n,m", [(10, 5), (200, 50), (1000, 300), (500, 500)])
def test_matches_reference_on_random_inputs(n, m):
    rand = np.random.RandomState(n * 7 + m)
    curve = _curve(np.unique(rand.rand(n)))

    _compare(curve, list(rand.rand(m)), f"random n={n} m={m}")


def test_targets_outside_the_curve_range():
    """Targets above every curve threshold have no match and must yield nulls."""
    curve = _curve([0.2, 0.4, 0.6, 0.8])

    _compare(curve, [0.05, 0.5, 0.95, 1.5, -1.0], "outside range")


def test_exact_ties_with_curve_thresholds():
    """A target equal to a curve threshold maps to that threshold, not the next one."""
    curve = _curve([0.2, 0.4, 0.6, 0.8])

    _compare(curve, [0.2, 0.4, 0.6, 0.8], "exact ties")


def test_duplicate_targets():
    """Duplicate targets are legal and each maps independently.

    Stated directly rather than compared against `reference_map_to_thresholds`: the old
    implementation could not do this at all. Its `validate="1:1"` raised
    `ComputeError: join keys did not fulfill 1:1 validation`, which reached users via
    `Bootstrap.adverse_impact_ratio_at_thresholds` on any score column with ties (see
    tests/test_bootstrap.py::test_air_at_thresholds_with_tied_scores).
    """
    curve = _curve([0.1, 0.3, 0.5, 0.7, 0.9])

    result = _map_to_thresholds(curve, [0.35, 0.35, 0.35, 0.62, 0.62]).collect()

    assert result.height == 5, "one row per target, duplicates included"
    # 0.35 -> 0.5 (smallest curve threshold >= 0.35); 0.62 -> 0.7.
    assert result["threshold"].to_list() == [0.35, 0.35, 0.35, 0.62, 0.62]
    assert result["_threshold_actual"].to_list() == [0.5, 0.5, 0.5, 0.7, 0.7]
    # tp is the positional index in the ascending curve: 0.5 is index 2, 0.7 is index 3.
    assert result["tp"].to_list() == [2.0, 2.0, 2.0, 3.0, 3.0]


def test_reference_rejects_duplicate_targets():
    """Pin the old failure mode, so the regression above cannot quietly come back."""
    curve = _curve([0.1, 0.3, 0.5, 0.7, 0.9])

    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        reference_map_to_thresholds(curve, [0.35, 0.35]).collect()


def test_single_row_curve():
    _compare(_curve([0.5]), [0.1, 0.5, 0.9], "single-row curve")


def test_empty_targets():
    _compare(_curve([0.2, 0.4, 0.6]), [], "empty targets")


def test_none_targets_is_passthrough():
    _compare(_curve([0.2, 0.4, 0.6]), None, "None targets")


def test_ascending_curve_input():
    """The AIR path feeds an ascending curve; sortedness must not be assumed."""
    curve = _curve([0.2, 0.4, 0.6, 0.8], descending=False)

    _compare(curve, [0.1, 0.45, 0.75, 0.99], "ascending curve")


def test_integer_thresholds():
    """Score columns are not always floats -- test_metrics covers integer scores."""
    curve = _curve([300.0, 500.0, 700.0, 850.0])

    _compare(curve, [400.0, 500.0, 900.0], "integer-like thresholds")


@pytest.mark.perf
def test_scales_to_large_inputs():
    """Guard against the O(n*m) cross join being reintroduced.

    Budget picked to discriminate cleanly rather than to be tight: at n=m=8000 the
    cross join measured ~640ms on polars 1.44 (and ~6s on 1.43 -- 1.44 improved it
    substantially, but it is still quadratic), while `join_asof` measures ~3ms. 100ms
    sits ~6x below the former and ~30x above the latter, so it stays meaningful on a
    slower machine without becoming flaky.
    """
    import time

    n = 8_000
    rand = np.random.RandomState(0)
    scores = np.unique(rand.rand(n))
    curve = _curve(scores).collect().lazy()

    start = time.perf_counter()
    _map_to_thresholds(curve, list(scores)).collect()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, (
        f"_map_to_thresholds took {elapsed * 1000:.0f}ms for n=m={n}; the O(n*m) cross "
        f"join has likely been reintroduced"
    )
