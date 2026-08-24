"""Shared pytest configuration, fixtures, and opt-in markers.

Fixtures here mirror the module-level constants that `tests/test_metrics.py` already
uses, so new tests can share one seeded dataset instead of re-deriving their own.
"""

import numpy as np
import pytest

SEED = 208
N_ROWS = 1_000


def pytest_addoption(parser):
    parser.addoption(
        "--runperf",
        action="store_true",
        default=False,
        help="run performance-guard tests (deselected by default)",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (deselected by default)",
    )
    parser.addoption(
        "--update-characterization",
        action="store_true",
        default=False,
        help=(
            "regenerate the committed characterization fixtures under "
            "tests/data/characterization instead of asserting against them"
        ),
    )


def pytest_collection_modifyitems(config, items):
    skips = []

    if not config.getoption("--runperf"):
        skips.append(("perf", pytest.mark.skip(reason="need --runperf to run")))

    if not config.getoption("--runslow"):
        skips.append(("slow", pytest.mark.skip(reason="need --runslow to run")))

    for keyword, marker in skips:
        for item in items:
            if keyword in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def binary_data() -> dict:
    """A single seeded binary-classification dataset shared across tests.

    Deliberately built with a legacy `RandomState` so the values are stable regardless
    of NumPy's default-generator changes.
    """
    rs = np.random.RandomState(SEED)

    y_score = rs.rand(N_ROWS)
    y_true = rs.choice([True, False], N_ROWS)
    protected = rs.choice([True, False], N_ROWS)

    sample_weight = rs.rand(N_ROWS)
    # Keep a block of unit weights so weighted and unweighted paths overlap.
    sample_weight[:100] = 1.0

    return {
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_score > 0.5,
        "sample_weight": sample_weight,
        "protected": protected,
        "control": ~protected,
    }


@pytest.fixture(scope="session")
def regression_data() -> dict:
    rs = np.random.RandomState(SEED + 1)

    return {"y_true": rs.rand(N_ROWS), "y_score": rs.rand(N_ROWS)}
