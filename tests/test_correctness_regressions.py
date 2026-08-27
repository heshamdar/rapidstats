"""Regression tests for bugs found by review.

Each test here reproduces a defect that shipped in v0.4.1, and each failed before the
corresponding fix. Grouped in one file so the set is easy to run and easy to audit
against the review; the fixes themselves live across `_corr.py`, `metrics.py`,
`_bootstrap.py` and `src/bootstrap.rs`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 500
SEED = 208


# ---------------------------------------------------------------------------------
# 1. Quantile definition mismatch between the Rust and polars interval paths
# ---------------------------------------------------------------------------------


def test_percentile_interval_uses_linear_interpolation():
    """The scalar and cum_sum bootstraps must define a quantile the same way.

    Rust's `percentile` interpolates linearly, matching numpy and scipy. The polars
    path called `Series.quantile(alpha)`, whose default interpolation is "nearest", so
    `Bootstrap.roc_auc` and `Bootstrap.confusion_matrix_at_thresholds` reported bounds
    computed under different definitions. The gap widens as iterations fall.
    """
    from rapidstats._bootstrap import _percentile_interval_polars
    from rapidstats._rustystats import _percentile_interval

    rand = np.random.RandomState(SEED)
    values = rand.rand(1_000)
    alpha = 0.025

    rust = _percentile_interval(0.5, list(values), alpha)
    grouped = pl.LazyFrame({"group": ["a"] * len(values), "value": values}).group_by(
        "group"
    )
    polars_result = _percentile_interval_polars(grouped, alpha).collect()

    assert polars_result["lower"].item() == pytest.approx(rust[0], rel=0, abs=1e-12)
    assert polars_result["upper"].item() == pytest.approx(rust[2], rel=0, abs=1e-12)

    # And both must match numpy, which is what the docstring's percentile means.
    assert rust[0] == pytest.approx(
        np.percentile(values, alpha * 100), rel=0, abs=1e-12
    )


# ---------------------------------------------------------------------------------
# 2 & 3. correlation_matrix batching
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("colliding", ["c1", "c2", "correlation"])
def test_batched_correlation_matrix_survives_colliding_column_names(
    tmp_path, colliding
):
    """Internal sentinel names must not collide with the user's columns.

    The batched path built a long frame with hardcoded `c1`/`c2` columns and then
    pivoted on them. Any input column actually named `c2` collided with the pivot index
    and raised `DuplicateError: column with name 'c2' has more than one occurrence`.
    The unbatched path was unaffected, and the test fixture happened to have no such
    column, so this went unnoticed.
    """
    rand = np.random.RandomState(SEED)
    frame = pl.DataFrame({nm: rand.rand(200) for nm in [colliding, "x", "y", "z"]})

    result = rs.correlation_matrix(
        frame,
        batch_options=rs.CorrelationBatchOptions(cache_dir=tmp_path, quiet=True),
    )

    assert colliding in result.columns
    assert result.height == 3


def test_batch_size_fraction_actually_batches(tmp_path):
    """`batch_size` as a fraction is a share of the work, not a multiple of it.

    The computation was `int(len(combinations) / batch_size)`, so `0.1` produced a batch
    ten times *larger* than the total and everything landed in one file -- defeating the
    entire purpose of batching, which exists to bound memory on wide frames.
    """
    rand = np.random.RandomState(SEED)
    # 10 columns -> 45 pairs; 10% per batch should be ~10 batches.
    frame = pl.DataFrame({f"f{i}": rand.rand(200) for i in range(10)})

    rs.correlation_matrix(
        frame,
        format="long",
        batch_options=rs.CorrelationBatchOptions(
            batch_size=0.1, cache_dir=tmp_path, quiet=True
        ),
    )

    batches = sorted(tmp_path.glob("*.parquet"))
    assert len(batches) >= 8, (
        f"batch_size=0.1 over 45 pairs produced {len(batches)} batch(es); expected ~10"
    )


def test_batched_matches_unbatched(tmp_path):
    """Batching is an execution strategy; it must not change the numbers."""
    rand = np.random.RandomState(SEED)
    frame = pl.DataFrame({f"f{i}": rand.rand(300) for i in range(6)})

    unbatched = rs.correlation_matrix(frame, format="long").sort("c1", "c2")
    batched = rs.correlation_matrix(
        frame,
        format="long",
        batch_options=rs.CorrelationBatchOptions(
            batch_size=0.25, cache_dir=tmp_path, quiet=True
        ),
    ).sort("c1", "c2")

    assert batched["c1"].to_list() == unbatched["c1"].to_list()
    np.testing.assert_allclose(
        batched["correlation"].to_numpy().astype(float),
        unbatched["correlation"].to_numpy().astype(float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------------
# 4. sample_weight alignment when nulls are dropped
# ---------------------------------------------------------------------------------


def test_confusion_matrix_at_thresholds_loop_handles_nulls_with_weights():
    """Nulls must be dropped jointly across all three inputs.

    The loop strategy dropped nulls from `y_true`/`y_score` but then passed the original
    full-length `sample_weight` through, so any null raised
    `ShapeError: height of column 'sample_weight' (5) does not match ... (4)`.
    """
    y_true = [True, False, True, None, False]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.5]
    sample_weight = [1.0, 2.0, 1.5, 3.0, 0.5]

    result = rs.metrics.confusion_matrix_at_thresholds(
        y_true,
        y_score,
        thresholds=[0.75],
        sample_weight=sample_weight,
        strategy="loop",
    )

    assert result.height > 0

    # The dropped row's weight must be dropped with it, not shifted onto its neighbour.
    expected = rs.metrics.confusion_matrix_at_thresholds(
        [True, False, True, False],
        [0.9, 0.8, 0.7, 0.5],
        thresholds=[0.75],
        sample_weight=[1.0, 2.0, 1.5, 0.5],
        strategy="loop",
    )

    np.testing.assert_allclose(
        result.sort("metric")["value"].to_numpy().astype(float),
        expected.sort("metric")["value"].to_numpy().astype(float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_confusion_matrix_handles_nulls_with_weights():
    """Same alignment requirement for the scalar entry point."""
    result = rs.metrics.confusion_matrix(
        [True, False, None, True],
        [True, False, True, True],
        sample_weight=[1.0, 2.0, 5.0, 1.0],
    )

    expected = rs.metrics.confusion_matrix(
        [True, False, True], [True, False, True], sample_weight=[1.0, 2.0, 1.0]
    )

    assert result.tp == expected.tp
    assert result.tn == expected.tn


# ---------------------------------------------------------------------------------
# 5. predicted_positive_ratio_at_thresholds passed the wrong array
# ---------------------------------------------------------------------------------


def test_ppr_auto_strategy_decides_on_thresholds_not_scores():
    """`strategy="auto"` picks by how many *thresholds* were asked for.

    The call passed `y_score` where `thresholds` was intended, so the decision was made
    on the wrong array: a handful of requested thresholds against a large score column
    chose `cum_sum` when `loop` was intended.
    """
    from rapidstats.metrics import _set_loop_strategy

    assert _set_loop_strategy([0.1, 0.5], "auto") == "loop"
    assert _set_loop_strategy(list(np.linspace(0, 1, 50)), "auto") == "cum_sum"

    # Comparing results cannot detect this: both strategies compute the same numbers,
    # so a wrong dispatch is invisible in the output. Assert on what was passed.
    import rapidstats.metrics as metrics_module

    seen: list = []
    original = metrics_module._set_loop_strategy

    def recording(thresholds, strategy):
        seen.append(thresholds)
        return original(thresholds, strategy)

    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    thresholds = [0.25, 0.5, 0.75]

    metrics_module._set_loop_strategy = recording
    try:
        rs.metrics.predicted_positive_ratio_at_thresholds(
            y_score, thresholds=thresholds, strategy="auto"
        )
    finally:
        metrics_module._set_loop_strategy = original

    assert seen, "_set_loop_strategy was never called"
    passed = seen[0]
    assert passed is not None and len(passed) == len(thresholds), (
        f"strategy was decided on an array of length "
        f"{None if passed is None else len(passed)}; expected the {len(thresholds)} "
        f"requested thresholds, not the {len(y_score)} scores"
    )


# ---------------------------------------------------------------------------------
# 6. standard interval centring
# ---------------------------------------------------------------------------------


def test_standard_interval_is_centred_on_the_point_estimate():
    r"""The documented interval is \hat{\theta} +/- z * \hat{\sigma}.

    The implementation centred on the *bootstrap mean* instead, while reporting
    `original_stat` as the point. On a skewed bootstrap distribution the reported point
    could therefore sit off-centre, or outside its own interval.
    """
    from rapidstats._rustystats import _standard_interval

    rand = np.random.RandomState(SEED)
    # Deliberately skewed, so the bootstrap mean and the point estimate differ.
    bootstrap_stats = list(rand.exponential(1.0, 2_000))
    original_stat = 0.25
    alpha = 0.025

    lower, point, upper = _standard_interval(original_stat, bootstrap_stats, alpha)

    assert point == original_stat
    midpoint = (lower + upper) / 2
    assert midpoint == pytest.approx(original_stat, rel=0, abs=1e-9), (
        f"interval centred on {midpoint} rather than the point estimate {original_stat}"
    )
    assert lower <= point <= upper
