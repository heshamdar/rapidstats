"""The Rust extension must release the GIL while computing.

Every `#[pyfunction]` in `src/lib.rs` used to hold the GIL for its whole body, so
`_run_concurrent(..., executor="threads")` -- the default across the Python layer --
was pure overhead for Rust-backed work: threads queued on the interpreter lock instead
of running.

The GIL is observed directly rather than inferred from a speedup. A background thread
spins on a Python-level counter; if the extension holds the GIL, that thread cannot
advance at all, because a Rust call has no bytecode boundaries for the interpreter to
preempt. Comparing its progress during a call against its progress during an equally
long `time.sleep` gives a ratio that is ~0 when the GIL is held and ~1 when it is not.
Measured against the pre-fix build, a long bootstrap scored 0.003.

Two design notes, both learned by measuring:

- The workload must be long and predominantly Rust. Short calls are dominated by
  thread-startup and by the surrounding Python-level polars work (which already
  releases the GIL), which pushes the ratio up regardless and destroys the signal.
- Wall-clock speedup is a poor proxy for this property. `metrics.roc_auc` sorts, and
  polars' sort is already rayon-parallel, so a single call saturates every core and
  fanning out cannot help however the GIL behaves. Releasing it is still correct: it
  lets unrelated Python threads run.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

pytestmark = pytest.mark.perf

# A held GIL scores ~0.003. Anything above this is unambiguous, and the bar is low
# enough to survive a loaded or throttled machine.
MIN_GIL_RATIO = 0.25


def _spin_progress(work) -> tuple[int, float]:
    """Run `work` while a background thread increments a Python counter."""
    counter = 0
    stop = threading.Event()

    def spin():
        nonlocal counter
        while not stop.is_set():
            counter += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        start = time.perf_counter()
        work()
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        thread.join()

    return counter, elapsed


def _gil_ratio(work, min_seconds: float = 0.2) -> tuple[float, float]:
    """Counter progress during `work` relative to progress during an equal sleep."""
    work()  # warm up so first-call costs land in neither arm

    during_work, elapsed = _spin_progress(work)

    if elapsed < min_seconds:
        pytest.skip(
            f"workload only ran for {elapsed * 1000:.0f}ms; too short to separate the "
            f"GIL signal from setup overhead"
        )

    during_sleep, _ = _spin_progress(lambda: time.sleep(elapsed))

    if during_sleep == 0:
        pytest.skip("background thread made no progress even during sleep")

    return during_work / during_sleep, elapsed


@pytest.fixture(scope="module")
def data() -> dict:
    rand = np.random.RandomState(0)
    n = 200_000

    return {
        "y_true": pl.Series(rand.rand(n) > 0.5),
        "y_score": pl.Series(rand.rand(n)),
        "y_pred": pl.Series(rand.rand(n) > 0.5),
    }


def test_bootstrap_releases_the_gil(data):
    """The headline case: a bootstrap is seconds of pure Rust."""
    bootstrap = rs.Bootstrap(iterations=100, seed=1)

    ratio, elapsed = _gil_ratio(
        lambda: bootstrap.roc_auc(data["y_true"], data["y_score"])
    )

    assert ratio > MIN_GIL_RATIO, (
        f"another Python thread made only {ratio:.1%} of its unblocked progress during "
        f"a {elapsed * 1000:.0f}ms bootstrap. The extension is holding the GIL -- check "
        f"py.allow_threads in src/lib.rs."
    )


def test_bootstrap_confusion_matrix_releases_the_gil(data):
    # More iterations than the roc_auc case: the bincount kernel is cheap, so fewer
    # would finish before the measurement window opens and the test would skip.
    bootstrap = rs.Bootstrap(iterations=600, seed=1)

    ratio, elapsed = _gil_ratio(
        lambda: bootstrap.confusion_matrix(data["y_true"], data["y_pred"])
    )

    assert ratio > MIN_GIL_RATIO, (
        f"another Python thread made only {ratio:.1%} of its unblocked progress during "
        f"a {elapsed * 1000:.0f}ms bootstrapped confusion matrix"
    )


def test_threaded_fanout_of_a_single_threaded_kernel():
    """End-to-end payoff on a kernel that is not already rayon-parallel internally.

    `confusion_matrix`'s Rust body is a serial bincount, so concurrent calls genuinely
    overlap once the GIL is released. The bar is modest because much of the surrounding
    work is memory-bandwidth-bound and does not scale with cores; the pre-fix build
    measured 0.90x here, i.e. slower than sequential.
    """
    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs >= 4 cores to observe fan-out")

    rand = np.random.RandomState(0)
    n = 4_000_000
    y_true = pl.Series(rand.rand(n) > 0.5)
    y_pred = pl.Series(rand.rand(n) > 0.5)

    def call():
        return rs.metrics.confusion_matrix(y_true, y_pred)

    call()

    start = time.perf_counter()
    for _ in range(4):
        call()
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: call(), range(4)))
    threaded = time.perf_counter() - start

    speedup = sequential / threaded

    assert speedup > 1.2, (
        f"4 concurrent confusion_matrix calls achieved only {speedup:.2f}x "
        f"({sequential * 1000:.0f}ms sequential vs {threaded * 1000:.0f}ms threaded)"
    )
