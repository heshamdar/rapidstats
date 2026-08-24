"""Guards on the polars version contract.

The fork was previously unusable on current polars: `rapidstats` imported
`ArrayLike` from `polars.series.series`, a private path that moved, so
`import rapidstats` raised ImportError outright. These tests pin the supported
range so that regression cannot recur silently.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import subprocess
import sys

import polars as pl
import pytest

MIN_POLARS = (1, 33)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)

    return tuple(parts)


def test_package_imports():
    """The whole package must import, including every submodule `__init__` pulls in."""
    module = importlib.import_module("rapidstats")

    for submodule in (
        "bin",
        "drift",
        "metrics",
        "polars",
        "preprocessing",
        "selection",
        "viz",
    ):
        assert hasattr(module, submodule), f"rapidstats.{submodule} did not import"


def test_imports_in_a_clean_interpreter():
    """Catch import-time breakage that a warm test session would mask."""
    result = subprocess.run(
        [sys.executable, "-c", "import rapidstats"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"`import rapidstats` failed:\n{result.stderr}"


def test_installed_polars_satisfies_declared_floor():
    installed = _version_tuple(pl.__version__)

    assert installed >= MIN_POLARS, (
        f"polars {pl.__version__} is below the declared floor "
        f"{'.'.join(str(p) for p in MIN_POLARS)}"
    )


def test_no_private_polars_paths():
    """`polars.series.series.ArrayLike` moved; nothing may reach into it again."""
    import rapidstats

    package_dir = __import__("pathlib").Path(rapidstats.__file__).parent
    offenders = [
        path.name
        for path in package_dir.rglob("*.py")
        if "from polars.series.series import" in path.read_text()
    ]

    assert not offenders, (
        f"these modules import from the private path polars.series.series: {offenders}"
    )


def test_declared_dependency_bounds_polars():
    """An unbounded polars requirement can pair a wheel with a polars that broke it."""
    requires = importlib.metadata.requires("rapidstats") or []
    # Requirement strings have no space between name and specifier
    # (e.g. "polars>=1.0.0,!=1.26.0"), so isolate the leading name.
    polars_reqs = [
        r for r in requires if re.match(r"^\s*polars\b(?!-)", r) and "extra ==" not in r
    ]

    assert polars_reqs, "rapidstats does not declare a polars dependency"
    assert any("<" in r for r in polars_reqs), (
        f"polars requirement has no upper bound: {polars_reqs}. The compiled plugin "
        f"embeds a polars-rs ABI, so an unbounded range can install an incompatible "
        f"polars against an old wheel."
    )


def test_no_deprecation_warnings_from_rapidstats_code():
    """No polars API we call may be deprecated.

    Asserting on the warning's originating filename rather than configuring a pytest
    `filterwarnings` module regex: the latter depends on polars choosing the right
    `stacklevel`, which would make the guard silently stop working.
    """
    import pathlib
    import warnings

    import numpy as np

    import rapidstats

    package_dir = pathlib.Path(rapidstats.__file__).parent
    rand = np.random.RandomState(0)
    y_score = rand.rand(300)
    y_true = rand.choice([True, False], 300)
    protected = rand.choice([True, False], 300)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        rapidstats.metrics.confusion_matrix_at_thresholds(y_true, y_score)
        rapidstats.metrics.adverse_impact_ratio_at_thresholds(
            y_score, protected, ~protected
        )
        rapidstats.metrics.predicted_positive_ratio_at_thresholds(y_score)
        rapidstats.metrics.average_precision(y_true, y_score)
        rapidstats.drift.psi(y_score[:150], y_score[150:])

        # Cover both sampling methods *and* both dispatch routes. `roc_auc` is handled
        # entirely in Rust, so it never reaches the Python resampling helpers; `run`
        # and the `cum_sum` paths do. Exercising only the former hides deprecations in
        # `_poisson_sample`.
        for sampling_method in ("poisson", "multinomial"):
            bootstrap = rapidstats.Bootstrap(
                iterations=5, seed=1, sampling_method=sampling_method
            )
            bootstrap.roc_auc(y_true, y_score)
            bootstrap.confusion_matrix_at_thresholds(y_true, y_score)
            bootstrap.adverse_impact_ratio_at_thresholds(y_score, protected, ~protected)
            bootstrap.run(
                pl.DataFrame({"y": y_score}), lambda df: float(df["y"].mean())
            )

    offenders = sorted(
        {
            f"{pathlib.Path(w.filename).name}:{w.lineno} {w.category.__name__}: "
            f"{str(w.message).splitlines()[0]}"
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and package_dir in pathlib.Path(w.filename).parents
        }
    )

    assert not offenders, (
        "deprecated polars APIs used by rapidstats:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "expression",
    ["auc", "is_pareto"],
)
def test_plugin_expressions_load(expression):
    """The compiled plugin must still register against the installed polars."""
    import rapidstats.polars as rps

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 6.0, 7.0]})
    result = df.select(getattr(rps, expression)("x", "y"))

    assert result.height >= 1
