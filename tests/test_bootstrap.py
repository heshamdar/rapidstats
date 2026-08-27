import numpy as np
import polars as pl
import pytest
import scipy.stats

import rapidstats

np.random.seed(208)

N = 1_000
CONFIDENCE_LEVEL = 0.95
ALPHA = (1 - CONFIDENCE_LEVEL) / 2
BOOTSTRAP_STATS = np.random.uniform(size=N)
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
POINT = 0.5


def _alpha(confidence_level: float) -> float:
    return (1 - confidence_level) / 2


def reference_standard_interval(point, bootstrap_stats, confidence_level):
    r"""The first-order normal approximation: \hat{\theta} +/- z * \hat{\sigma}.

    Centred on the observed statistic, which is what the class docstring specifies and
    what is reported as the point estimate. The implementation used to centre on the
    bootstrap *mean* instead, so on a skewed bootstrap distribution the reported point
    sat off-centre in its own interval -- measured at 0.014 off for a bootstrapped
    ROC-AUC. This reference moved with that fix.
    """
    alpha = _alpha(confidence_level)

    stderr = np.std(bootstrap_stats, ddof=1)
    x = scipy.stats.norm.ppf(1 - alpha) * stderr

    return (point - x, point + x)


def test_standard_interval():
    rs = rapidstats._bootstrap._standard_interval(POINT, BOOTSTRAP_STATS, ALPHA)
    ref = reference_standard_interval(POINT, BOOTSTRAP_STATS, CONFIDENCE_LEVEL)

    assert pytest.approx((rs[0], rs[2])) == ref


def test_standard_interval_is_symmetric_about_the_point():
    lower, point, upper = rapidstats._bootstrap._standard_interval(
        POINT, BOOTSTRAP_STATS, ALPHA
    )

    assert point == POINT
    assert (lower + upper) / 2 == pytest.approx(POINT)


def reference_percentile_interval(bootstrap_stats, confidence_level):
    alpha = (1 - confidence_level) / 2
    interval = alpha, 1 - alpha

    def percentile_fun(a, q):
        return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(bootstrap_stats, interval[0] * 100)
    ci_u = percentile_fun(bootstrap_stats, interval[1] * 100)

    return (ci_l, ci_u)


def test_percentile_interval():
    n = 1_000
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    bootstrap_stats = np.random.uniform(size=n)

    rs = rapidstats._bootstrap._percentile_interval(POINT, bootstrap_stats, alpha)
    rs = (rs[0], rs[2])
    ref = reference_percentile_interval(bootstrap_stats, confidence_level)

    for a, b in zip(rs, ref):
        assert pytest.approx(a) == b


def reference_basic_interval(data, bootstrap_stats, confidence_level):
    ci_l, ci_u = reference_percentile_interval(bootstrap_stats, confidence_level)
    theta_hat = np.mean(data)
    ci_l, ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

    return ci_l, ci_u


def test_basic_interval():
    n = 1_000
    confidence_level = 0.95
    alpha = _alpha(confidence_level)
    bootstrap_stats = np.random.uniform(size=n)
    original_stat = np.mean(np.random.uniform(size=n))

    rs = rapidstats._bootstrap._basic_interval(original_stat, bootstrap_stats, alpha)
    rs = (rs[0], rs[2])
    ref = reference_basic_interval(original_stat, bootstrap_stats, confidence_level)

    for a, b in zip(rs, ref):
        assert pytest.approx(a) == b


def reference_bca_interval(data, bootstrap_stats, confidence_level):
    alpha = (1 - confidence_level) / 2

    interval = scipy.stats._resampling._bca_interval(
        (data,),
        statistic=np.mean,
        axis=-1,
        alpha=alpha,
        theta_hat_b=bootstrap_stats,
        batch=None,
    )[:2]

    percentile_fun = scipy.stats._resampling._percentile_along_axis

    ci_l = percentile_fun(bootstrap_stats, interval[0] * 100)
    ci_u = percentile_fun(bootstrap_stats, interval[1] * 100)

    return (ci_l, ci_u)


def test_bca_interval():
    n = 1_000
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    bootstrap_stats = np.random.uniform(size=n) * 100
    data = np.random.randint(1, 100, size=n)

    def _mean(df):
        return df["x"].mean()

    rs = rapidstats._bootstrap._bca_interval(
        original_stat=np.mean(data),
        bootstrap_stats=bootstrap_stats.tolist(),
        jacknife_stats=rapidstats._bootstrap._jacknife(
            pl.DataFrame({"x": data.tolist()}), _mean
        ),
        alpha=alpha,
    )
    rs = (rs[0], rs[2])

    ref = reference_bca_interval(data, bootstrap_stats, confidence_level)

    for a, b in zip(rs, ref):
        assert pytest.approx(a, rel=1e-5) == b


@pytest.mark.parametrize(
    "bs",
    [
        rapidstats.Bootstrap(method=method, iterations=5)
        for method in ["standard", "percentile", "basic", "BCa"]
    ],
)
def test_loop_cum_sum_shape(bs: rapidstats.Bootstrap):
    y_true_score = np.random.rand(100)
    y_true = y_true_score >= 0.5
    y_score = np.random.rand(100)
    protected = [True] * 50 + [False] * 50
    control = [False] * 50 + [True] * 50

    if bs.method != "BCa":
        ref = bs.confusion_matrix_at_thresholds(y_true, y_score, strategy="loop")
        res = bs.confusion_matrix_at_thresholds(y_true, y_score, strategy="cum_sum")

        assert ref.shape == res.shape

        ref = bs.confusion_matrix_at_thresholds(
            y_true, y_score, thresholds=THRESHOLDS, strategy="loop"
        )

        res = bs.confusion_matrix_at_thresholds(
            y_true, y_score, thresholds=THRESHOLDS, strategy="cum_sum"
        )

        assert ref.shape == res.shape

        ref = bs.adverse_impact_ratio_at_thresholds(
            y_score, protected, control, strategy="loop"
        )
        res = bs.adverse_impact_ratio_at_thresholds(
            y_score, protected, control, strategy="cum_sum"
        )

        assert ref.shape == res.shape

        ref = bs.adverse_impact_ratio_at_thresholds(
            y_score, protected, control, thresholds=THRESHOLDS, strategy="loop"
        )
        res = bs.adverse_impact_ratio_at_thresholds(
            y_score, protected, control, thresholds=THRESHOLDS, strategy="cum_sum"
        )

        assert ref.shape == res.shape


@pytest.mark.parametrize(
    "method",
    ["standard", "percentile", "basic", "BCa"],
)
@pytest.mark.parametrize("sampling_method", ["poisson", "multinomial"])
def test_bootstrap_succesfully_runs(method, sampling_method):
    y_true_score = np.random.rand(100)
    y_true = y_true_score >= 0.5
    y_score = np.random.rand(100)
    y_pred = y_score >= 0.5
    protected = [True] * 50 + [False] * 50
    control = [False] * 50 + [True] * 50

    bs = rapidstats.Bootstrap(
        method=method, sampling_method=sampling_method, iterations=5
    )

    bs.roc_auc(y_true, y_score)
    bs.average_precision(y_true, y_score)
    bs.brier_loss(y_true, y_score)
    bs.mean(y_score)
    bs.confusion_matrix(y_true, y_pred)

    if bs.method != "BCa":
        bs.confusion_matrix_at_thresholds(y_true, y_score)
        bs.confusion_matrix_at_thresholds(y_true, y_score, thresholds=THRESHOLDS)
        bs.adverse_impact_ratio_at_thresholds(y_score, protected, control)
        bs.adverse_impact_ratio_at_thresholds(
            y_score, protected, control, thresholds=THRESHOLDS
        )

    bs.adverse_impact_ratio(y_pred, y_true, y_score <= 0.5)
    bs.max_ks(y_true, y_score)
    bs.mean_squared_error(y_true_score, y_score)
    bs.root_mean_squared_error(y_true_score, y_score)
    bs.r2(y_true_score, y_score)


@pytest.mark.parametrize("strategy", ["loop", "cum_sum"])
def test_air_at_thresholds_with_tied_scores(strategy):
    """Tied scores must not crash the bootstrap.

    `Bootstrap.adverse_impact_ratio_at_thresholds` passed the raw `y_score` column as
    the threshold list when `thresholds` was None, where the non-bootstrap version
    deduplicates first. Duplicated targets then tripped a `validate="1:1"` join inside
    `_map_to_thresholds`, so any score column with repeats -- rounded scores, integer
    scores, credit-style bands -- raised `ComputeError: join keys did not fulfill 1:1
    validation`.
    """
    rand = np.random.RandomState(0)
    n = 200
    # Rounding to 2dp guarantees ties: ~89 distinct values across 200 rows.
    y_score = np.round(rand.rand(n), 2)
    protected = rand.choice([True, False], n)

    assert len(np.unique(y_score)) < n, "fixture must contain tied scores"

    bs = rapidstats.Bootstrap(iterations=3, seed=208)
    res = bs.adverse_impact_ratio_at_thresholds(
        y_score, protected, ~protected, strategy=strategy
    )

    assert res.height == len(np.unique(y_score))
    assert res["threshold"].n_unique() == res.height


def test_cm_at_thresholds_with_duplicate_user_thresholds():
    """A caller passing duplicate thresholds should not crash either."""
    rand = np.random.RandomState(1)
    n = 200
    y_true = rand.choice([True, False], n)
    y_score = rand.rand(n)

    res = rapidstats.metrics.confusion_matrix_at_thresholds(
        y_true, y_score, thresholds=[0.25, 0.5, 0.5, 0.75], strategy="cum_sum"
    )

    assert res.height > 0
