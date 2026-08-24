"""A bootstrap must not take the interpreter down at high iteration counts.

v0.4.1 segfaulted on `Bootstrap(sampling_method="poisson").roc_auc(...)` at the
*default* 1000 iterations. The cause was a rayon worker stack overflow: each bootstrap
task calls into polars, polars parallelises internally on its own rayon pool, and that
nested work runs on the calling worker's stack -- which rayon defaults to 2 MiB. Deep
enough nesting overflows it and the process dies with SIGSEGV, no Python traceback.

Diagnosis that pinned it: `n_jobs=1` (no rayon) never crashed, and setting
`RUST_MIN_STACK=16777216` made the crash disappear. The fix runs bootstrap work on a
dedicated pool with an explicit stack size instead of the global pool.

Every test here runs in a subprocess. A segfault cannot be caught -- it would take
pytest with it -- so the assertion is on the child's exit status.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

SAMPLING_METHODS = ["poisson", "multinomial"]
RESAMPLE_MODES = ["weights", "materialize"]


def _run(body: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_survived(result: subprocess.CompletedProcess, what: str):
    if result.returncode == -11 or result.returncode == 139:
        pytest.fail(
            f"{what} segfaulted (exit {result.returncode}). This is the rayon worker "
            f"stack overflow -- check the pool stack size in src/bootstrap.rs."
        )

    assert result.returncode == 0, (
        f"{what} exited {result.returncode}\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout, f"{what} did not complete: {result.stdout}"


@pytest.mark.parametrize("sampling_method", SAMPLING_METHODS)
@pytest.mark.parametrize("resample_mode", RESAMPLE_MODES)
def test_default_iterations_do_not_crash(sampling_method, resample_mode):
    """The regression proper: 1000 iterations is the default and used to segfault."""
    result = _run(f"""
        import numpy as np, rapidstats as rs
        rand = np.random.RandomState(0)
        n = 3_000
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score
        bootstrap = rs.Bootstrap(
            iterations=1_000,
            seed=11,
            sampling_method="{sampling_method}",
            resample_mode="{resample_mode}",
        )
        lower, point, upper = bootstrap.roc_auc(y_true, y_score)
        assert lower <= point <= upper
        print("ok")
    """)

    _assert_survived(result, f"roc_auc {sampling_method}/{resample_mode} @1000")


@pytest.mark.parametrize("sampling_method", SAMPLING_METHODS)
def test_many_iterations_do_not_crash(sampling_method):
    """Well past the default, since the old failure threshold moved with the workload."""
    result = _run(f"""
        import numpy as np, rapidstats as rs
        rand = np.random.RandomState(0)
        n = 2_000
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score
        bootstrap = rs.Bootstrap(
            iterations=10_000, seed=11, sampling_method="{sampling_method}"
        )
        lower, point, upper = bootstrap.roc_auc(y_true, y_score)
        assert lower <= point <= upper
        print("ok")
    """)

    _assert_survived(result, f"roc_auc {sampling_method} @10000")


def test_confusion_matrix_many_iterations_do_not_crash():
    result = _run("""
        import numpy as np, rapidstats as rs
        rand = np.random.RandomState(0)
        n = 2_000
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score
        res = rs.Bootstrap(iterations=10_000, seed=11).confusion_matrix(
            y_true, y_score > 0.5
        )
        assert res.to_polars().height == 27
        print("ok")
    """)

    _assert_survived(result, "confusion_matrix @10000")


def test_jackknife_does_not_crash():
    """BCa runs one jackknife task per row, so it nests sooner than the bootstrap."""
    result = _run("""
        import numpy as np, rapidstats as rs
        rand = np.random.RandomState(0)
        n = 4_000
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score
        lower, point, upper = rs.Bootstrap(
            iterations=200, seed=11, method="BCa"
        ).roc_auc(y_true, y_score)
        assert lower <= point <= upper
        print("ok")
    """)

    _assert_survived(result, "BCa roc_auc (jackknife)")
