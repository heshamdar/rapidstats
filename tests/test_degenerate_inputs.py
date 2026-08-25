"""Degenerate inputs must return NaN, not abort the interpreter.

Most metrics already return NaN when there is nothing to compute -- `roc_auc`,
`brier_loss`, `mean`, `max_ks` and `confusion_matrix` all do. Four did not:
`mean_squared_error`, `root_mean_squared_error`, `r2` and `adverse_impact_ratio` called
`.unwrap()` on an empty aggregate and raised `pyo3_runtime.PanicException`, a
`BaseException` that ordinary `except Exception` handling does not catch.

NaN rather than a raised error is the right answer, and not only for consistency: a
bootstrap resample can legitimately come out degenerate -- every row one class, or a
Poisson draw that keeps almost nothing -- and `drop_nans` already discards those
iterations. A metric that raised instead would take the whole bootstrap down because one
of a thousand resamples was unlucky.
"""

from __future__ import annotations

import math

import numpy as np
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


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@pytest.mark.parametrize("name", sorted(BINARY_METRICS))
def test_empty_input_returns_nan(name):
    result = BINARY_METRICS[name]([], [])

    assert _is_missing(result), f"{name} on empty input returned {result!r}"


@pytest.mark.parametrize("name", sorted(BINARY_METRICS))
def test_all_null_input_returns_nan(name):
    """Nulls are dropped first, so this reduces to the empty case."""
    result = BINARY_METRICS[name]([None, None, None], [None, None, None])

    assert _is_missing(result), f"{name} on all-null input returned {result!r}"


def test_empty_confusion_matrix_is_all_nan_or_zero():
    result = rs.metrics.confusion_matrix([], [])

    assert result.tp == 0.0
    assert _is_missing(result.tpr)


def test_empty_adverse_impact_ratio_returns_nan():
    result = rs.metrics.adverse_impact_ratio([], [], [])

    assert _is_missing(result)


def test_single_row_metrics_do_not_crash():
    """One row leaves zero variance, which is a different degenerate path."""
    for name, call in BINARY_METRICS.items():
        result = call([1.0], [0.5])
        assert isinstance(result, float), f"{name} returned {result!r}"


def test_empty_bootstrap_returns_nan_interval():
    lower, point, upper = rs.Bootstrap(iterations=5, seed=1).roc_auc([], [])

    assert all(_is_missing(v) for v in (lower, point, upper))


def test_bootstrap_survives_degenerate_resamples():
    """The reason NaN beats raising: one unlucky resample must not kill the run.

    A tiny single-class input makes most resamples degenerate. The bootstrap should come
    back with NaN where it could not compute, not propagate an exception.
    """
    y_true = np.array([True, True, True, False])
    y_score = np.array([0.9, 0.8, 0.7, 0.1])

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
    lower, point, upper = call()

    assert all(_is_missing(v) for v in (lower, point, upper)), f"{name}"
