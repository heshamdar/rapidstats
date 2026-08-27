"""A bootstrap resample is exactly a reweighting.

Drawing row `i` of the data `c[i]` times and computing a metric is arithmetically
identical to computing the same metric once with `sample_weight` multiplied by `c`.
Every metric here is already weight-aware, so the library can replace per-iteration
resampling -- which materialises a new frame and re-sorts it every time -- with a single
sorted pass carrying the counts as weights.

These tests are the contract that makes that substitution legal. They compare against
`tests/helpers.py::materialised_resample`, which does the expansion explicitly with
`np.repeat` and is deliberately obvious rather than fast.

The tolerance is 0: this is an algebraic identity, not an approximation. Weighted and
materialised sums visit the same values in the same order, so they agree bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rapidstats as rs
from rapidstats.metrics import DefaultConfusionMatrixMetrics
from tests.helpers import materialised_resample, multinomial_counts, poisson_counts

N = 2_000
SEEDS = [0, 1, 7, 208]


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rand = np.random.RandomState(208)

    return pl.DataFrame(
        {
            "y_true": rand.choice([True, False], N),
            "y_score": rand.rand(N),
            "protected": rand.choice([True, False], N),
        }
    ).with_columns(pl.col("protected").not_().alias("control"))


def _counts(kind: str, seed: int) -> np.ndarray:
    return poisson_counts(N, seed) if kind == "poisson" else multinomial_counts(N, seed)


@pytest.mark.parametrize("kind", ["poisson", "multinomial"])
@pytest.mark.parametrize("seed", SEEDS)
def test_roc_auc_weighted_equals_materialised(data, kind, seed):
    counts = _counts(kind, seed)
    expanded = materialised_resample(data, counts)

    weighted = rs.metrics.roc_auc(
        data["y_true"], data["y_score"], sample_weight=counts.astype(float)
    )
    materialised = rs.metrics.roc_auc(expanded["y_true"], expanded["y_score"])

    assert weighted == materialised, (
        f"{kind} seed={seed}: weighted {weighted!r} != materialised {materialised!r}"
    )


@pytest.mark.parametrize("kind", ["poisson", "multinomial"])
@pytest.mark.parametrize("seed", SEEDS)
def test_confusion_matrix_weighted_equals_materialised(data, kind, seed):
    counts = _counts(kind, seed)
    expanded = materialised_resample(data, counts)
    y_pred = data["y_score"] > 0.5

    weighted = rs.metrics.confusion_matrix(
        data["y_true"], y_pred, sample_weight=counts.astype(float)
    )
    materialised = rs.metrics.confusion_matrix(
        expanded["y_true"], expanded["y_score"] > 0.5
    )

    for metric in DefaultConfusionMatrixMetrics:
        a = getattr(weighted, metric)
        b = getattr(materialised, metric)

        if a is None or (isinstance(a, float) and np.isnan(a)):
            assert b is None or np.isnan(b), f"{metric}: {a!r} vs {b!r}"
        else:
            assert a == pytest.approx(b, rel=0, abs=1e-12), (
                f"{kind} seed={seed} metric={metric}: {a!r} != {b!r}"
            )


@pytest.mark.parametrize("kind", ["poisson", "multinomial"])
@pytest.mark.parametrize("seed", SEEDS)
def test_adverse_impact_ratio_weighted_equals_materialised(data, kind, seed):
    counts = _counts(kind, seed)
    expanded = materialised_resample(data, counts)
    y_pred = data["y_score"] > 0.5

    weighted = rs.metrics.adverse_impact_ratio(
        y_pred,
        data["protected"],
        data["control"],
        sample_weight=counts.astype(float),
    )
    materialised = rs.metrics.adverse_impact_ratio(
        expanded["y_score"] > 0.5, expanded["protected"], expanded["control"]
    )

    assert weighted == pytest.approx(materialised, rel=0, abs=1e-12)


@pytest.mark.parametrize("kind", ["poisson", "multinomial"])
def test_confusion_matrix_at_thresholds_weighted_equals_materialised(data, kind):
    """The curve, not just the scalar -- this is the path the bootstrap rewrites."""
    counts = _counts(kind, 3)
    expanded = materialised_resample(data, counts)
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]

    weighted = rs.metrics.confusion_matrix_at_thresholds(
        data["y_true"],
        data["y_score"],
        thresholds=thresholds,
        sample_weight=counts.astype(float),
        strategy="cum_sum",
    ).sort("threshold", "metric")

    materialised = rs.metrics.confusion_matrix_at_thresholds(
        expanded["y_true"],
        expanded["y_score"],
        thresholds=thresholds,
        strategy="cum_sum",
    ).sort("threshold", "metric")

    assert weighted["metric"].to_list() == materialised["metric"].to_list()

    a = weighted["value"].to_numpy().astype(float)
    b = materialised["value"].to_numpy().astype(float)
    a = np.where(weighted["value"].is_null().to_numpy(), np.nan, a)
    b = np.where(materialised["value"].is_null().to_numpy(), np.nan, b)

    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def test_zero_weight_equals_dropping_the_row(data):
    """The degenerate case the identity depends on: count 0 means 'not present'."""
    counts = np.ones(N)
    counts[:50] = 0.0

    weighted = rs.metrics.roc_auc(data["y_true"], data["y_score"], sample_weight=counts)
    dropped = rs.metrics.roc_auc(data["y_true"][50:], data["y_score"][50:])

    assert weighted == dropped
