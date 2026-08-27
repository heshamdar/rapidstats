"""Metrics can take a frame plus column names instead of materialised arrays.

Every entry point used to coerce its inputs into an eager `pl.DataFrame` before doing any
work (`_utils.py::_y_true_y_score_to_df` and siblings), so the whole dataset had to fit in
memory whatever the metric actually needed.

Passing `data=` instead lets the coercion build a `LazyFrame`, which keeps polars'
optimiser in play. The concrete win is projection pushdown: against a `scan_parquet`
source only the columns a metric names are read from disk, which matters for the wide
feature tables this library is aimed at. It applies even to the Rust-backed metrics,
which must still collect before calling their kernel.

Four of the pure-polars entry points can go further and defer collection entirely, via
`lazy=True`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 1_000
SEED = 208


@pytest.fixture(scope="module")
def arrays() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)
    protected = rand.choice([True, False], N)

    return {
        "y_true": rand.rand(N) < 0.25 + 0.5 * y_score,
        "y_score": y_score,
        "y_pred": y_score > 0.5,
        "sample_weight": rand.rand(N) + 0.5,
        "protected": protected,
        "control": ~protected,
        "y_true_reg": rand.rand(N),
    }


@pytest.fixture(scope="module")
def frame(arrays) -> pl.DataFrame:
    return pl.DataFrame(arrays)


@pytest.fixture(scope="module")
def wide_parquet(tmp_path_factory, arrays) -> str:
    """A frame with many irrelevant columns, so pushdown has something to prune."""
    rand = np.random.RandomState(SEED + 1)
    padding = {f"filler_{i}": rand.rand(N) for i in range(40)}
    path = tmp_path_factory.mktemp("lazy") / "wide.parquet"
    pl.DataFrame({**arrays, **padding}).write_parquet(path)

    return str(path)


# ---------------------------------------------------------------------------------
# Equivalence: `data=` must match the ArrayLike path exactly
# ---------------------------------------------------------------------------------


def _close(a, b, what: str):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"{what}: lazy input disagreed with the array input",
    )


SCALAR_CASES = {
    "roc_auc": (
        lambda m, a: m.roc_auc(a["y_true"], a["y_score"]),
        lambda m, d: m.roc_auc("y_true", "y_score", data=d),
    ),
    "roc_auc_weighted": (
        lambda m, a: m.roc_auc(a["y_true"], a["y_score"], a["sample_weight"]),
        lambda m, d: m.roc_auc("y_true", "y_score", "sample_weight", data=d),
    ),
    "max_ks": (
        lambda m, a: m.max_ks(a["y_true"], a["y_score"]),
        lambda m, d: m.max_ks("y_true", "y_score", data=d),
    ),
    "brier_loss": (
        lambda m, a: m.brier_loss(a["y_true"], a["y_score"]),
        lambda m, d: m.brier_loss("y_true", "y_score", data=d),
    ),
    "average_precision": (
        lambda m, a: m.average_precision(a["y_true"], a["y_score"]),
        lambda m, d: m.average_precision("y_true", "y_score", data=d),
    ),
    "mean": (
        lambda m, a: m.mean(a["y_score"]),
        lambda m, d: m.mean("y_score", data=d),
    ),
    "r2": (
        lambda m, a: m.r2(a["y_true_reg"], a["y_score"]),
        lambda m, d: m.r2("y_true_reg", "y_score", data=d),
    ),
    "mean_squared_error": (
        lambda m, a: m.mean_squared_error(a["y_true_reg"], a["y_score"]),
        lambda m, d: m.mean_squared_error("y_true_reg", "y_score", data=d),
    ),
    "root_mean_squared_error": (
        lambda m, a: m.root_mean_squared_error(a["y_true_reg"], a["y_score"]),
        lambda m, d: m.root_mean_squared_error("y_true_reg", "y_score", data=d),
    ),
    "adverse_impact_ratio": (
        lambda m, a: m.adverse_impact_ratio(a["y_pred"], a["protected"], a["control"]),
        lambda m, d: m.adverse_impact_ratio("y_pred", "protected", "control", data=d),
    ),
}


@pytest.mark.parametrize("name", sorted(SCALAR_CASES))
@pytest.mark.parametrize("lazy", [False, True], ids=["eager_frame", "lazy_frame"])
def test_scalar_metrics_accept_data(arrays, frame, name, lazy):
    from_arrays, from_data = SCALAR_CASES[name]
    data = frame.lazy() if lazy else frame

    _close(from_data(rs.metrics, data), from_arrays(rs.metrics, arrays), name)


def test_confusion_matrix_accepts_data(arrays, frame):
    from_arrays = rs.metrics.confusion_matrix(arrays["y_true"], arrays["y_pred"])
    from_data = rs.metrics.confusion_matrix("y_true", "y_pred", data=frame.lazy())

    for metric, expected in from_arrays.__dict__.items():
        _close([getattr(from_data, metric)], [expected], f"confusion_matrix.{metric}")


CURVE_CASES = {
    "confusion_matrix_at_thresholds": (
        lambda m, a: m.confusion_matrix_at_thresholds(a["y_true"], a["y_score"]),
        lambda m, d, **kw: m.confusion_matrix_at_thresholds(
            "y_true", "y_score", data=d, **kw
        ),
        ["threshold", "metric"],
    ),
    "predicted_positive_ratio_at_thresholds": (
        lambda m, a: m.predicted_positive_ratio_at_thresholds(a["y_score"]),
        lambda m, d, **kw: m.predicted_positive_ratio_at_thresholds(
            "y_score", data=d, **kw
        ),
        ["threshold"],
    ),
    "adverse_impact_ratio_at_thresholds": (
        lambda m, a: m.adverse_impact_ratio_at_thresholds(
            a["y_score"], a["protected"], a["control"]
        ),
        lambda m, d, **kw: m.adverse_impact_ratio_at_thresholds(
            "y_score", "protected", "control", data=d, **kw
        ),
        ["threshold"],
    ),
    "capture_rate_at_quantiles": (
        lambda m, a: m.capture_rate_at_quantiles(a["y_true"], a["y_score"]),
        lambda m, d, **kw: m.capture_rate_at_quantiles(
            "y_true", "y_score", data=d, **kw
        ),
        ["quantile"],
    ),
}


@pytest.mark.parametrize("name", sorted(CURVE_CASES))
def test_curve_metrics_accept_data(arrays, frame, name):
    from_arrays, from_data, sort_by = CURVE_CASES[name]

    expected = from_arrays(rs.metrics, arrays).sort(sort_by)
    actual = from_data(rs.metrics, frame.lazy()).sort(sort_by)

    assert actual.columns == expected.columns
    for col in expected.columns:
        if expected[col].dtype.is_numeric():
            _close(
                actual[col].fill_null(np.nan).to_numpy(),
                expected[col].fill_null(np.nan).to_numpy(),
                f"{name}.{col}",
            )
        else:
            assert actual[col].to_list() == expected[col].to_list()


# ---------------------------------------------------------------------------------
# lazy=True defers collection
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURVE_CASES))
def test_lazy_true_returns_a_lazyframe_with_the_same_contents(arrays, frame, name):
    _, from_data, sort_by = CURVE_CASES[name]

    eager = from_data(rs.metrics, frame.lazy()).sort(sort_by)
    lazy = from_data(rs.metrics, frame.lazy(), lazy=True)

    assert isinstance(lazy, pl.LazyFrame), f"{name} did not defer collection"
    assert lazy.collect().sort(sort_by).equals(eager)


def test_lazy_output_composes_into_a_larger_plan(frame):
    """The point of lazy output: keep building, collect once."""
    curve = rs.metrics.confusion_matrix_at_thresholds(
        "y_true", "y_score", data=frame.lazy(), lazy=True
    )

    result = (
        curve.filter(pl.col("metric") == "tpr")
        .filter(pl.col("threshold") > 0.5)
        .select(pl.col("value").mean().alias("mean_tpr"))
        .collect()
    )

    assert result.height == 1


def test_lazy_true_rejected_for_kernel_backed_metrics(frame):
    """`roc_auc` returns a float from Rust; there is nothing to defer."""
    with pytest.raises(TypeError):
        rs.metrics.roc_auc("y_true", "y_score", data=frame.lazy(), lazy=True)


# ---------------------------------------------------------------------------------
# Projection pushdown -- the mechanism, not a timing
# ---------------------------------------------------------------------------------


def test_only_named_columns_are_read_from_a_scan(wide_parquet):
    """A scan source must not read the 40 irrelevant columns.

    Asserted on the query plan rather than the clock: pushdown is a structural property,
    and a timing assertion here would be noise-bound.
    """
    from rapidstats._utils import _y_true_y_score_to_lf

    scan = pl.scan_parquet(wide_parquet)
    assert len(scan.collect_schema().names()) > 40

    plan = _y_true_y_score_to_lf(scan, "y_true", "y_score").explain()

    assert "PROJECT 2/" in plan or 'PROJECT["y_true", "y_score"]' in plan, (
        f"expected only y_true and y_score to be projected; plan was:\n{plan}"
    )


def test_scan_source_matches_in_memory_result(wide_parquet, arrays):
    """End to end from disk, never materialising the full table."""
    scan = pl.scan_parquet(wide_parquet)

    _close(
        [rs.metrics.roc_auc("y_true", "y_score", data=scan)],
        [rs.metrics.roc_auc(arrays["y_true"], arrays["y_score"])],
        "roc_auc from scan_parquet",
    )

    from_scan = rs.metrics.confusion_matrix_at_thresholds(
        "y_true", "y_score", data=scan, thresholds=[0.25, 0.5, 0.75]
    ).sort("threshold", "metric")
    from_arrays = rs.metrics.confusion_matrix_at_thresholds(
        arrays["y_true"], arrays["y_score"], thresholds=[0.25, 0.5, 0.75]
    ).sort("threshold", "metric")

    _close(
        from_scan["value"].fill_null(np.nan).to_numpy(),
        from_arrays["value"].fill_null(np.nan).to_numpy(),
        "confusion_matrix_at_thresholds from scan_parquet",
    )


# ---------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------


def test_bootstrap_accepts_data(arrays, frame):
    kwargs = dict(iterations=50, seed=SEED)

    from_arrays = rs.Bootstrap(**kwargs).roc_auc(arrays["y_true"], arrays["y_score"])
    from_data = rs.Bootstrap(**kwargs).roc_auc("y_true", "y_score", data=frame.lazy())

    _close(from_data, from_arrays, "Bootstrap.roc_auc")


def test_bootstrap_curve_accepts_data(arrays, frame):
    kwargs = dict(iterations=20, seed=SEED)
    thresholds = [0.3, 0.6]

    from_arrays = (
        rs.Bootstrap(**kwargs)
        .confusion_matrix_at_thresholds(
            arrays["y_true"], arrays["y_score"], thresholds=thresholds
        )
        .sort("threshold", "metric")
    )
    from_data = (
        rs.Bootstrap(**kwargs)
        .confusion_matrix_at_thresholds(
            "y_true", "y_score", data=frame.lazy(), thresholds=thresholds
        )
        .sort("threshold", "metric")
    )

    _close(
        from_data["point"].fill_null(np.nan).to_numpy(),
        from_arrays["point"].fill_null(np.nan).to_numpy(),
        "Bootstrap.confusion_matrix_at_thresholds",
    )


# ---------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------


def test_missing_column_is_reported_clearly(frame):
    with pytest.raises(Exception, match="nope"):
        rs.metrics.roc_auc("nope", "y_score", data=frame.lazy())


def test_data_requires_string_column_names(arrays, frame):
    """Passing arrays *and* `data` is a mistake worth catching."""
    with pytest.raises(TypeError, match="column name"):
        rs.metrics.roc_auc(arrays["y_true"], "y_score", data=frame.lazy())


@pytest.mark.perf
def test_streaming_is_not_slower_on_a_scan_source(wide_parquet):
    """The payoff for lazy inputs, and the one shape where streaming is the right engine.

    Every in-memory shape in this library measures *worse* under streaming -- sorts,
    `cum_sum` and as-of joins resist morsel parallelism. From a scan there is a real
    stream to process and columns to prune, and the ordering inverts: benchmarked at 2M
    rows over 2 of 42 columns, 2638ms in-memory against 488ms streaming, a 5.4x win.

    This fixture is far smaller than that, so fixed costs dominate and the bar is only
    that streaming is not markedly worse. `benchmarks/bench_engine.py` carries the real
    numbers.
    """
    import time

    def best_of(engine: str, repeats: int = 3) -> float:
        timings = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            with rs.Config.engine(engine):
                rs.metrics.confusion_matrix_at_thresholds(
                    "y_true",
                    "y_score",
                    data=pl.scan_parquet(wide_parquet),
                    strategy="cum_sum",
                )
            timings.append(time.perf_counter() - start)

        return min(timings[1:])  # drop the warm-up

    in_memory = best_of("in-memory")
    streaming = best_of("streaming")

    assert streaming < in_memory * 3, (
        f"streaming ({streaming * 1000:.0f}ms) was far slower than in-memory "
        f"({in_memory * 1000:.0f}ms) on a scan source, which is the shape it should win"
    )


# ---------------------------------------------------------------------------------
# `data=` under `strategy="loop"`
# ---------------------------------------------------------------------------------
#
# Five call sites derived their threshold set from `set(thresholds or y_score)`. With
# `data=`, `y_score` is a column *name*, so that is a set of single characters, each then
# compared against a float column. The existing `data=` curve tests all pass an explicit
# `thresholds` list, so `thresholds or y_score` never evaluated `y_score` and the defect
# passed CI. These omit it.
#
# `strategy="loop"` has to be explicit: `_set_loop_strategy` routes
# `strategy="auto", thresholds=None` to `cum_sum`, which reads the frame already.

LOOP_CASES = [
    pytest.param(
        lambda a: rs.metrics.predicted_positive_ratio_at_thresholds(
            a["y_score"], strategy="loop"
        ),
        lambda f: rs.metrics.predicted_positive_ratio_at_thresholds(
            "y_score", strategy="loop", data=f
        ),
        id="predicted_positive_ratio_at_thresholds",
    ),
    pytest.param(
        lambda a: rs.metrics.adverse_impact_ratio_at_thresholds(
            a["y_score"], a["protected"], a["control"], strategy="loop"
        ),
        lambda f: rs.metrics.adverse_impact_ratio_at_thresholds(
            "y_score", "protected", "control", strategy="loop", data=f
        ),
        id="adverse_impact_ratio_at_thresholds",
    ),
    pytest.param(
        lambda a: rs.metrics.confusion_matrix_at_thresholds(
            a["y_true"], a["y_score"], strategy="loop"
        ),
        lambda f: rs.metrics.confusion_matrix_at_thresholds(
            "y_true", "y_score", strategy="loop", data=f
        ),
        id="confusion_matrix_at_thresholds",
    ),
    pytest.param(
        lambda a: rs.Bootstrap(
            iterations=3, seed=SEED, quiet=True
        ).confusion_matrix_at_thresholds(a["y_true"], a["y_score"], strategy="loop"),
        lambda f: rs.Bootstrap(
            iterations=3, seed=SEED, quiet=True
        ).confusion_matrix_at_thresholds("y_true", "y_score", strategy="loop", data=f),
        id="Bootstrap.confusion_matrix_at_thresholds",
    ),
]


@pytest.mark.parametrize(("from_arrays", "from_data"), LOOP_CASES)
def test_loop_strategy_accepts_data_without_thresholds(
    arrays, frame, from_arrays, from_data
):
    """The threshold set must come from the frame, not from the `y_score` argument."""
    expected = from_arrays(arrays).sort(pl.all())
    actual = from_data(frame.lazy()).sort(pl.all())

    assert actual.height == expected.height
    assert set(actual["threshold"].to_list()) == set(expected["threshold"].to_list())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda f, s: rs.metrics.confusion_matrix_at_thresholds(
                "y_true", "y_score", strategy=s, data=f
            ),
            id="confusion_matrix_at_thresholds",
        ),
        pytest.param(
            lambda f, s: rs.metrics.predicted_positive_ratio_at_thresholds(
                "y_score", strategy=s, data=f
            ),
            id="predicted_positive_ratio_at_thresholds",
        ),
    ],
)
def test_loop_and_cum_sum_resolve_the_same_thresholds(frame, call):
    """The two strategies compute the same curve, so they must evaluate the same set.

    `cum_sum` always read the frame; `loop` read the raw argument, which also meant it
    carried duplicates and nulls the frame had already dropped.
    """
    loop = call(frame.lazy(), "loop")
    cum_sum = call(frame.lazy(), "cum_sum")

    assert set(loop["threshold"].to_list()) == set(cum_sum["threshold"].to_list())


def test_quiet_silences_the_loop_strategy_bar(arrays, capsys):
    """The two `strategy="loop"` bars in `_bootstrap.py` never received the flag."""
    rs.Bootstrap(
        iterations=3, seed=SEED, n_jobs=1, quiet=True
    ).confusion_matrix_at_thresholds(
        arrays["y_true"], arrays["y_score"], thresholds=[0.3, 0.6], strategy="loop"
    )
    rs.Bootstrap(
        iterations=3, seed=SEED, n_jobs=1, quiet=True
    ).adverse_impact_ratio_at_thresholds(
        arrays["y_score"],
        arrays["protected"],
        arrays["control"],
        thresholds=[0.3, 0.6],
        strategy="loop",
    )

    assert capsys.readouterr().err == "", "progress output leaked despite quiet=True"
