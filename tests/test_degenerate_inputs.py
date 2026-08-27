"""Absent is null; undefined is NaN.

Two different things get conflated when a metric has nothing useful to return, and they
deserve two different answers:

- **Nothing to compute.** Empty input, all-null input, an input that filters away to
  nothing. There is no number here, and the polars way to say that is `null` -- surfaced
  to Python as `None`. Every frame-returning path in this library already says it that
  way; the scalar metrics used to say `NaN` instead, which is the same word polars uses
  for a real (if undefined) float.
- **Undefined arithmetic.** There *is* data, and the formula divides by zero: `tpr` when
  no row is positive, `roc_auc` when every label is the same class, `r2` when `y_true`
  has no variance. That is what NaN means, and it stays NaN.

So the rule these tests pin is: a metric returns `None` when its (cleaned) input is
empty, and otherwise returns the arithmetic result, NaN included.

The bootstrap depends on the first half: a resample can legitimately come out empty or
single-class, and those iterations are skipped rather than taking the whole run down.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import rapidstats as rs

# Every metric that takes two aligned score-like arrays, and how to call it.
BINARY_METRICS = {
    "roc_auc": lambda a, b: rs.metrics.roc_auc(a, b),
    "brier_loss": lambda a, b: rs.metrics.brier_loss(a, b),
    "max_ks": lambda a, b: rs.metrics.max_ks(a, b),
    "average_precision": lambda a, b: rs.metrics.average_precision(a, b),
    "mean_squared_error": lambda a, b: rs.metrics.mean_squared_error(a, b),
    "root_mean_squared_error": lambda a, b: rs.metrics.root_mean_squared_error(a, b),
    "r2": lambda a, b: rs.metrics.r2(a, b),
}


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


@pytest.mark.parametrize("name", sorted(BINARY_METRICS))
def test_empty_input_returns_none(name):
    result = BINARY_METRICS[name]([], [])

    assert result is None, f"{name} on empty input returned {result!r}, expected None"


@pytest.mark.parametrize("name", sorted(BINARY_METRICS))
def test_all_null_input_returns_none(name):
    """Nulls are dropped first, so this reduces to the empty case."""
    result = BINARY_METRICS[name]([None, None, None], [None, None, None])

    assert result is None, (
        f"{name} on all-null input returned {result!r}, expected None"
    )


def test_empty_mean_returns_none():
    assert rs.metrics.mean([]) is None


def test_empty_adverse_impact_ratio_returns_none():
    assert rs.metrics.adverse_impact_ratio([], [], []) is None


def test_empty_confusion_matrix_counts_zero_and_rates_nan():
    """A weighted bincount of nothing really is zero; the rates really are 0/0."""
    result = rs.metrics.confusion_matrix([], [])

    assert result.tp == 0.0
    assert result.tn == 0.0
    assert _is_nan(result.tpr)
    assert _is_nan(result.precision)


def test_single_class_roc_auc_is_nan_not_none():
    """There is data -- there are just no discordant pairs to rank, so 0/0."""
    result = rs.metrics.roc_auc([True, True, True], [0.1, 0.5, 0.9])

    assert _is_nan(result), f"expected NaN, got {result!r}"


def test_single_class_max_ks_is_nan_not_none():
    result = rs.metrics.max_ks([True, True, True], [0.1, 0.5, 0.9])

    assert _is_nan(result), f"expected NaN, got {result!r}"


def test_zero_variance_r2_is_nan_not_none():
    """`y_true` constant and predicted exactly: 1 - 0/0."""
    result = rs.metrics.r2([1.0, 1.0], [1.0, 1.0])

    assert _is_nan(result), f"expected NaN, got {result!r}"


def test_single_row_metrics_do_not_crash():
    """One row leaves zero variance, which is a different degenerate path."""
    for name, call in BINARY_METRICS.items():
        result = call([1.0], [0.5])
        assert isinstance(result, float), f"{name} returned {result!r}"


def test_empty_bootstrap_returns_null_interval():
    lower, point, upper = rs.Bootstrap(iterations=5, seed=1).roc_auc([], [])

    assert (lower, point, upper) == (None, None, None)


def test_bootstrap_survives_degenerate_resamples():
    """The reason absent beats raising: one unlucky resample must not kill the run.

    A tiny single-class-heavy input makes many resamples degenerate. The bootstrap should
    come back with a number from the resamples that did work.
    """
    y_true = [True, True, True, False]
    y_score = [0.9, 0.8, 0.7, 0.1]

    lower, point, upper = rs.Bootstrap(iterations=50, seed=1).roc_auc(y_true, y_score)

    assert isinstance(point, float)


@pytest.mark.parametrize(
    "name,call",
    [
        (
            "mean_squared_error",
            lambda: rs.Bootstrap(iterations=5, seed=1).mean_squared_error([], []),
        ),
        ("r2", lambda: rs.Bootstrap(iterations=5, seed=1).r2([], [])),
        ("mean", lambda: rs.Bootstrap(iterations=5, seed=1).mean([])),
    ],
)
def test_empty_bootstrap_of_regression_metrics(name, call):
    assert call() == (None, None, None), name


def test_empty_bootstrap_confusion_matrix_separates_the_two():
    """Both answers in one tuple, which is the clearest statement of the rule.

    The point estimate is the confusion matrix of no rows: its counts are zero and its
    `tpr` is 0/0, undefined, NaN -- exactly what the scalar metric returns. The bounds are
    a different question: every replicate was undefined too, so there was nothing to take
    a percentile of, and that is absent rather than undefined.
    """
    cm = rs.Bootstrap(iterations=5, seed=1).confusion_matrix([], [])
    lower, point, upper = cm.tpr

    assert (lower, upper) == (None, None)
    assert _is_nan(point)

    assert cm.tp == (0.0, 0.0, 0.0)


def test_run_skips_stat_func_iterations_that_return_none():
    """`Bootstrap.run` takes an arbitrary `stat_func`, so it sees `None` too.

    A user statistic reports "nothing to compute here" the same way the built-in metrics
    do. Those iterations are skipped; the rest still make an interval.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    calls = []

    def stat_func(resample: pl.DataFrame):
        calls.append(1)

        return None if len(calls) % 2 == 0 else resample["x"].mean()

    lower, point, upper = rs.Bootstrap(iterations=20, seed=1, n_jobs=1).run(
        df, stat_func
    )

    assert lower is not None
    assert upper is not None
    assert lower <= point <= upper


def test_run_with_no_usable_iterations_returns_null_bounds():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    lower, point, upper = rs.Bootstrap(iterations=5, seed=1, n_jobs=1).run(
        df, lambda _: None
    )

    assert (lower, point, upper) == (None, None, None)
