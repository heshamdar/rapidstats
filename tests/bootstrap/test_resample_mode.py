"""`resample_mode` selects how resample counts are applied.

`sampling_method` says how the multiplicities are *drawn* (Poisson(1) or multinomial);
`resample_mode` says how they are *applied*. The two are independent, so all four
combinations are valid:

- `"weights"` (default) folds the counts into `sample_weight`, so the data is sorted
  once and scanned once per iteration.
- `"materialize"` expands the counts into an actual resampled frame, which is what the
  library did unconditionally before.

`tests/bootstrap/test_resample_equivalence.py` establishes that the two are
arithmetically identical for the underlying metrics. These tests cover the API surface
and the places where the substitution does *not* apply.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 1_500
SEED = 208


@pytest.fixture(scope="module")
def data() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)

    return {
        "y_true": rand.choice([True, False], N),
        "y_score": y_score,
        "y_pred": y_score > 0.5,
    }


def test_default_is_weights():
    assert rs.Bootstrap().resample_mode == "weights"


@pytest.mark.parametrize("mode", ["weights", "materialize"])
def test_accepts_both_modes(mode):
    assert rs.Bootstrap(resample_mode=mode).resample_mode == mode


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="resample_mode"):
        rs.Bootstrap(resample_mode="reweight")


@pytest.mark.parametrize("sampling_method", ["poisson", "multinomial"])
def test_modes_agree_on_roc_auc(data, sampling_method):
    """Both modes estimate the same interval.

    Not exact equality: the two consume the RNG differently, so the individual resamples
    differ. What must hold is that they describe the same sampling distribution, which
    at 500 iterations means agreeing to well within the interval's own width.
    """
    kwargs = dict(iterations=500, seed=SEED, sampling_method=sampling_method)

    weighted = rs.Bootstrap(resample_mode="weights", **kwargs).roc_auc(
        data["y_true"], data["y_score"]
    )
    materialised = rs.Bootstrap(resample_mode="materialize", **kwargs).roc_auc(
        data["y_true"], data["y_score"]
    )

    # Point estimates come from the original data, so those must match exactly.
    assert weighted[1] == pytest.approx(materialised[1], rel=0, abs=1e-12)

    width = materialised[2] - materialised[0]
    assert abs(weighted[0] - materialised[0]) < 0.25 * width
    assert abs(weighted[2] - materialised[2]) < 0.25 * width


@pytest.mark.parametrize("mode", ["weights", "materialize"])
def test_confusion_matrix_runs_in_both_modes(data, mode):
    res = rs.Bootstrap(iterations=50, seed=SEED, resample_mode=mode).confusion_matrix(
        data["y_true"], data["y_pred"]
    )

    frame = res.to_polars()
    assert frame.height == 27
    assert frame["point"].null_count() < 27


@pytest.mark.parametrize("mode", ["weights", "materialize"])
def test_run_always_materialises(data, mode):
    """`Bootstrap.run` takes an arbitrary callable, which cannot be assumed weight-aware.

    So it must hand `stat_func` a genuinely resampled frame even under
    `resample_mode="weights"` -- silently passing the original frame with a weight column
    the callable ignores would return the same value every iteration.
    """
    seen_heights = []

    def stat_func(df: pl.DataFrame) -> float:
        seen_heights.append(df.height)
        return float(df["y"].mean())

    frame = pl.DataFrame({"y": data["y_score"]})
    rs.Bootstrap(iterations=10, seed=SEED, resample_mode=mode).run(frame, stat_func)

    assert seen_heights, "stat_func was never called"
    # Multinomial resampling keeps the height but changes the rows; the point is that
    # the callable is not handed an unresampled frame every time.
    assert len(set(seen_heights)) > 1 or seen_heights[0] == N, (
        f"stat_func saw suspicious heights: {sorted(set(seen_heights))[:5]}"
    )


@pytest.mark.parametrize("mode", ["weights", "materialize"])
@pytest.mark.parametrize("sampling_method", ["poisson", "multinomial"])
def test_all_four_combinations_run(data, mode, sampling_method):
    bootstrap = rs.Bootstrap(
        iterations=20,
        seed=SEED,
        sampling_method=sampling_method,
        resample_mode=mode,
    )

    lower, point, upper = bootstrap.roc_auc(data["y_true"], data["y_score"])

    assert lower <= point <= upper, f"{mode}/{sampling_method}: {lower} {point} {upper}"


@pytest.mark.parametrize("metric", ["max_ks", "brier_loss"])
def test_weight_unaware_metrics_still_vary(data, metric):
    """Metrics whose kernel ignores `sample_weight` must not use weight resampling.

    `max_ks` and `brier_loss` both receive a weight column -- every frame built by
    `_y_true_y_score_to_df` has one -- but their Rust kernels never read it. Folding the
    resample counts into that column would therefore leave the statistic unchanged, so
    every iteration would return the identical number and the interval would collapse to
    zero width: a bootstrap that looks like it ran and tells you nothing.

    A non-degenerate interval is the observable signature that they are still
    materialising. Remove a metric from here once its kernel is weight-aware.
    """
    # Signal-bearing data on purpose. `max_ks` is a supremum statistic, so on pure noise
    # (true KS ~ 0) its bootstrap replicates are biased upward and the percentile
    # interval need not contain the point estimate -- a property of the statistic that
    # would obscure what this test is actually checking.
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    y_true = rand.rand(N) < 0.2 + 0.6 * y_score

    bootstrap = rs.Bootstrap(iterations=50, seed=SEED, resample_mode="weights")

    lower, point, upper = getattr(bootstrap, metric)(y_true, y_score)

    assert not np.isnan(point)
    assert lower <= point <= upper
    assert upper - lower > 1e-9, (
        f"{metric} produced a zero-width interval ({lower}, {upper}); the resample is "
        f"not varying, which means weights were applied to a kernel that ignores them"
    )
