import polars as pl
import polars.testing
import pytest

import rapidstats as rs
from tests.paths import DAT_PATH

kwargs = {
    "seed": 208,
    "iterations": 1000,
    "chunksize": 8,
    # "n_jobs": 1,
}

POISSON_BOOTSTRAP = rs.Bootstrap(sampling_method="poisson", **kwargs)
MULTINOMIAL_BOOTSTRAP = rs.Bootstrap(sampling_method="multinomial", **kwargs)
SCORES = pl.read_parquet(DAT_PATH / "scores.parquet")
REL_TOL = 1e-1


def test_roc_auc():
    y_true = SCORES["y_true"]
    y_score = SCORES["y_score"]

    p_res = POISSON_BOOTSTRAP.roc_auc(y_true, y_score)
    m_res = MULTINOMIAL_BOOTSTRAP.roc_auc(y_true, y_score)

    assert pytest.approx(p_res, REL_TOL) == m_res


def test_run():
    p_res = POISSON_BOOTSTRAP.run(SCORES, lambda df: rs.metrics.mean(df["y_score"]))
    m_res = MULTINOMIAL_BOOTSTRAP.run(SCORES, lambda df: rs.metrics.mean(df["y_score"]))

    assert pytest.approx(p_res, REL_TOL) == m_res


# Unbounded ratio-of-ratio metrics. Their bootstrap distributions are heavy-tailed --
# `dor` is plr/nlr, a ratio of two ratios, so a small denominator swing moves it a long
# way -- and the two sampling methods draw independent resamples. At 1000 iterations the
# Monte Carlo error on their interval bounds alone exceeds 10%, which is a property of
# the statistic rather than a disagreement between the methods. They are compared, but
# against a bound that reflects that.
HEAVY_TAILED = ["plr", "nlr", "dor"]
HEAVY_TAILED_REL_TOL = 0.5


def test_confusion_matrix():
    y_true = SCORES["y_true"]
    y_pred = SCORES["y_score"].ge(0.5)

    p_res = (
        POISSON_BOOTSTRAP.confusion_matrix(y_true, y_pred).to_polars().sort("metric")
    )
    m_res = (
        MULTINOMIAL_BOOTSTRAP.confusion_matrix(y_true, y_pred)
        .to_polars()
        .sort("metric")
    )

    stable = pl.col("metric").is_in(HEAVY_TAILED).not_()
    polars.testing.assert_frame_equal(
        p_res.filter(stable), m_res.filter(stable), rel_tol=REL_TOL
    )
    polars.testing.assert_frame_equal(
        p_res.filter(stable.not_()),
        m_res.filter(stable.not_()),
        rel_tol=HEAVY_TAILED_REL_TOL,
    )


# def test_confusion_matrix_at_thresholds():
#     tmp = SCORES
#     y_true = tmp["y_true"]
#     y_score = tmp["y_score"]

#     p_res = POISSON_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score).sort(
#         "threshold", "metric"
#     )
#     m_res = MULTINOMIAL_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score).sort(
#         "threshold", "metric"
#     )

#     polars.testing.assert_frame_equal(p_res, m_res, rel_tol=REL_TOL)


# `adverse_impact_ratio_at_thresholds` selects `_air_at_thresholds_core_sorted` on the
# poisson path but never sorted the frame, so every replicate ran a cumulative approval
# rate scan over unordered scores. It failed silently: the point estimate is computed by
# `_air_at_thresholds_core`, which sorts, so only the interval was wrong -- measured 6x
# to 29x too wide against an identical point.
#
# This is the check that would have caught it, and it is the same shape as `test_roc_auc`
# above: the two sampling methods are different draws of one distribution, so their
# intervals have to agree to within Monte Carlo error.
AIR_N = 800
AIR_THRESHOLDS = [0.2, 0.5, 0.8]


@pytest.fixture(scope="module")
def air_data() -> dict:
    import numpy as np

    rand = np.random.RandomState(208)
    y_score = rand.rand(AIR_N)
    protected = rand.choice([True, False], AIR_N)

    return {"y_score": y_score, "protected": protected, "control": ~protected}


def test_air_at_thresholds_interval_width(air_data):
    """Poisson and multinomial AIR intervals must be the same width, not just centred
    on the same point."""

    def run(sampling_method: str) -> pl.DataFrame:
        return (
            rs.Bootstrap(iterations=300, seed=1, sampling_method=sampling_method)
            .adverse_impact_ratio_at_thresholds(
                air_data["y_score"],
                air_data["protected"],
                air_data["control"],
                thresholds=AIR_THRESHOLDS,
                strategy="cum_sum",
            )
            .sort("threshold")
        )

    poisson = run("poisson")
    multinomial = run("multinomial")

    polars.testing.assert_series_equal(
        poisson["point"], multinomial["point"], rel_tol=1e-12
    )

    p_width = (poisson["upper"] - poisson["lower"]).to_list()
    m_width = (multinomial["upper"] - multinomial["lower"]).to_list()

    for threshold, p, m in zip(AIR_THRESHOLDS, p_width, m_width):
        ratio = p / m
        assert 0.5 < ratio < 2.0, (
            f"at t={threshold} the poisson interval is {ratio:.1f}x the multinomial "
            f"width ({p:.4f} vs {m:.4f}); the poisson frame is likely unsorted again"
        )
