"""BCa intervals for the `cum_sum` threshold strategies.

`Bootstrap.confusion_matrix_at_thresholds` and
`Bootstrap.adverse_impact_ratio_at_thresholds` raised `NotImplementedError` for
`method="BCa"`, citing pola-rs/polars#20951 -- a dynamic `quantile` inside
`group_by().agg()`. The implementation was left in place below the raise, as dead code.

That polars issue is fixed: on 1.44,

    df.lazy().group_by("g").agg(pl.col("value").quantile(pl.col("p").first()))

returns correctly. So the path is re-enabled, and the dead code it guarded is now
exercised.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 300
SEED = 208
THRESHOLDS = [0.2, 0.4, 0.6, 0.8]


@pytest.fixture(scope="module")
def data() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    protected = rand.choice([True, False], N)

    return {
        "y_true": rand.rand(N) < 0.25 + 0.5 * y_score,
        "y_score": y_score,
        "protected": protected,
        "control": ~protected,
    }


def test_polars_dynamic_quantile_in_agg_works():
    """The upstream blocker itself, pinned.

    If this ever regresses, the failure should point at polars rather than at the BCa
    implementation built on top of it.
    """
    rand = np.random.RandomState(SEED)
    frame = pl.DataFrame(
        {
            "g": ["a"] * 50 + ["b"] * 50,
            "value": rand.rand(100),
            "p": [0.1] * 50 + [0.9] * 50,
        }
    )

    result = (
        frame.lazy()
        .group_by("g")
        .agg(pl.col("value").quantile(pl.col("p").first()).alias("q"))
        .sort("g")
        .collect()
    )

    assert result.height == 2
    assert result["q"].null_count() == 0


# `sampling_method` and `resample_mode` are orthogonal to `method` and `strategy`, and
# every combination is meant to be valid. Testing one axis at a time is what let the
# poisson BCa crash ship: `df` is a LazyFrame on that path, and the jackknife indexes
# rows. `test_resample_mode.py` already gives `roc_auc` this grid.
MODES = pytest.mark.parametrize(
    ("sampling_method", "resample_mode"),
    [
        ("multinomial", "weights"),
        ("multinomial", "materialize"),
        ("poisson", "weights"),
        ("poisson", "materialize"),
    ],
)


@MODES
def test_bca_confusion_matrix_at_thresholds(data, sampling_method, resample_mode):
    result = rs.Bootstrap(
        iterations=100,
        seed=SEED,
        method="BCa",
        sampling_method=sampling_method,
        resample_mode=resample_mode,
    ).confusion_matrix_at_thresholds(
        data["y_true"], data["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
    )

    assert result.height > 0
    assert set(result.columns) == {"threshold", "metric", "lower", "point", "upper"}
    assert sorted(result["threshold"].unique().to_list()) == THRESHOLDS

    ordered = result.drop_nulls(["lower", "point", "upper"])
    assert ordered.height > 0, "every interval came back null"
    assert (ordered["lower"] <= ordered["point"]).all()
    assert (ordered["point"] <= ordered["upper"]).all()


@MODES
def test_bca_adverse_impact_ratio_at_thresholds(data, sampling_method, resample_mode):
    result = rs.Bootstrap(
        iterations=100,
        seed=SEED,
        method="BCa",
        sampling_method=sampling_method,
        resample_mode=resample_mode,
    ).adverse_impact_ratio_at_thresholds(
        data["y_score"],
        data["protected"],
        data["control"],
        thresholds=THRESHOLDS,
        strategy="cum_sum",
    )

    assert result.height > 0
    assert set(result.columns) == {"threshold", "lower", "point", "upper"}

    ordered = result.drop_nulls(["lower", "point", "upper"])
    assert ordered.height > 0, "every interval came back null"
    assert (ordered["lower"] <= ordered["point"]).all()
    assert (ordered["point"] <= ordered["upper"]).all()


def test_bca_agrees_with_the_loop_strategy(data):
    """`cum_sum` is an optimisation of `loop`; BCa must agree between them.

    Not exact: the two consume the RNG differently, so they are different draws of the
    same distribution. Compared on a bounded, well-behaved metric at a tolerance that
    reflects the iteration count.
    """
    kwargs = dict(iterations=200, seed=SEED, method="BCa")

    loop = (
        rs.Bootstrap(**kwargs)
        .confusion_matrix_at_thresholds(
            data["y_true"], data["y_score"], thresholds=[0.5], strategy="loop"
        )
        .filter(pl.col("metric") == "tpr")
    )
    cum_sum = (
        rs.Bootstrap(**kwargs)
        .confusion_matrix_at_thresholds(
            data["y_true"], data["y_score"], thresholds=[0.5], strategy="cum_sum"
        )
        .filter(pl.col("metric") == "tpr")
    )

    assert loop["point"].item() == pytest.approx(cum_sum["point"].item(), abs=1e-12)

    width = loop["upper"].item() - loop["lower"].item()
    assert abs(loop["lower"].item() - cum_sum["lower"].item()) < 0.3 * width
    assert abs(loop["upper"].item() - cum_sum["upper"].item()) < 0.3 * width


def test_bca_no_longer_raises_not_implemented(data):
    """The guard is gone -- catching this explicitly so its return is obvious."""
    for call in (
        lambda bs: bs.confusion_matrix_at_thresholds(
            data["y_true"], data["y_score"], thresholds=THRESHOLDS, strategy="cum_sum"
        ),
        lambda bs: bs.adverse_impact_ratio_at_thresholds(
            data["y_score"],
            data["protected"],
            data["control"],
            thresholds=THRESHOLDS,
            strategy="cum_sum",
        ),
    ):
        try:
            call(rs.Bootstrap(iterations=20, seed=SEED, method="BCa"))
        except NotImplementedError as exc:  # pragma: no cover - the regression itself
            pytest.fail(f"BCa still raises NotImplementedError: {exc}")


@pytest.mark.perf
def test_bca_jackknife_scales_near_linearly():
    """The jackknife must not be O(n^2).

    The confusion matrix is a weighted bincount, so a leave-one-out replicate is the
    totals minus one row -- O(1) each once the totals are known. The generic jackknife
    re-filtered and re-scanned the whole frame per row instead, which made BCa grow ~4.5x
    per doubling of n: 21ms at n=2000, 306ms at n=8000. The closed form measured 3.3ms
    and 6.0ms respectively, a 51x improvement at n=8000.

    Asserted as a growth ratio rather than absolute times, so it stays meaningful on a
    slower machine.
    """
    import statistics
    import time

    import numpy as np

    def timed(n: int, repeats: int = 3) -> float:
        rand = np.random.RandomState(0)
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score

        def call():
            return rs.Bootstrap(iterations=25, seed=1, method="BCa").confusion_matrix(
                y_true, y_score > 0.5
            )

        call()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            timings.append(time.perf_counter() - start)

        return statistics.median(timings)

    small = timed(4_000)
    large = timed(16_000)

    # 4x the rows. Quadratic would be ~16x; the closed form measured ~2.2x.
    growth = large / small
    assert growth < 6.0, (
        f"quadrupling n multiplied BCa cost by {growth:.1f}x ({small * 1000:.1f}ms -> "
        f"{large * 1000:.1f}ms); the O(n^2) jackknife has likely come back"
    )
