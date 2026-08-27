"""`n_jobs` and `quiet` must actually reach the Python-side execution paths.

`Bootstrap` stores `n_jobs` and `chunksize` and forwards them to the Rust kernels, but
the Python paths -- `run`, the `cum_sum` bootstraps, and the `strategy="loop"` metrics --
call `_run_concurrent` without passing them on. `_run_concurrent` only serialises when it
sees `max_workers == 1` in its kwargs, so `n_jobs=1` did not serialise anything on that
side, and `quiet` did not suppress the progress bars.

That matters beyond tidiness: `n_jobs=1` is how you get a deterministic, debuggable run,
and it is the documented escape hatch when a `stat_func` is not thread-safe.
"""

from __future__ import annotations

import threading

import numpy as np
import polars as pl
import pytest

import rapidstats as rs

N = 400
SEED = 208


@pytest.fixture(scope="module")
def data() -> dict:
    rand = np.random.RandomState(SEED)
    y_score = rand.rand(N)

    return {
        "y_true": rand.rand(N) < 0.3 + 0.4 * y_score,
        "y_score": y_score,
        "protected": rand.choice([True, False], N),
    }


def _thread_ids_used(bootstrap: rs.Bootstrap, frame: pl.DataFrame) -> set[int]:
    seen: set[int] = set()
    lock = threading.Lock()

    def stat_func(df: pl.DataFrame) -> float:
        with lock:
            seen.add(threading.get_ident())
        return float(df["y"].mean())

    bootstrap.run(frame, stat_func)

    return seen


def test_run_is_sequential_when_n_jobs_is_one(data):
    frame = pl.DataFrame({"y": data["y_score"]})
    bootstrap = rs.Bootstrap(iterations=32, seed=SEED, n_jobs=1)

    seen = _thread_ids_used(bootstrap, frame)

    assert seen == {threading.get_ident()}, (
        f"n_jobs=1 ran on {len(seen)} threads ({seen}); it must run inline on the "
        f"calling thread"
    )


def test_run_uses_threads_when_n_jobs_is_not_one(data):
    """The complement, so the test above cannot pass by everything being serial."""
    frame = pl.DataFrame({"y": data["y_score"]})
    bootstrap = rs.Bootstrap(iterations=64, seed=SEED, n_jobs=4)

    seen = _thread_ids_used(bootstrap, frame)

    assert len(seen) > 1, f"n_jobs=4 ran on a single thread ({seen})"


@pytest.mark.parametrize(
    ("call", "method"),
    [
        pytest.param(
            lambda bs, d: bs.confusion_matrix_at_thresholds(
                d["y_true"], d["y_score"], strategy="cum_sum"
            ),
            "percentile",
            id="confusion_matrix_at_thresholds",
        ),
        pytest.param(
            lambda bs, d: bs.adverse_impact_ratio_at_thresholds(
                d["y_score"], d["protected"], ~d["protected"], strategy="cum_sum"
            ),
            "percentile",
            id="adverse_impact_ratio_at_thresholds",
        ),
        pytest.param(
            lambda bs, d: bs.average_precision(d["y_true"], d["y_score"]),
            "percentile",
            id="average_precision",
        ),
        # The BCa branches reach `_jacknife`, a second `_run_concurrent` call site inside
        # the same method. The confusion-matrix one passed no executor kwargs at all, so
        # `n_jobs` stopped at the resampling loop and the jackknife still fanned out.
        # Explicit thresholds keep the jackknife frame small.
        pytest.param(
            lambda bs, d: bs.confusion_matrix_at_thresholds(
                d["y_true"], d["y_score"], thresholds=[0.4, 0.6], strategy="cum_sum"
            ),
            "BCa",
            id="confusion_matrix_at_thresholds-BCa",
        ),
        pytest.param(
            lambda bs, d: bs.adverse_impact_ratio_at_thresholds(
                d["y_score"],
                d["protected"],
                ~d["protected"],
                thresholds=[0.4, 0.6],
                strategy="cum_sum",
            ),
            "BCa",
            id="adverse_impact_ratio_at_thresholds-BCa",
        ),
    ],
)
@pytest.mark.parametrize("sampling_method", ["multinomial", "poisson"])
def test_cum_sum_paths_accept_n_jobs(data, call, method, sampling_method):
    """These route through `_run_concurrent` too, and must honour the setting."""
    result = call(
        rs.Bootstrap(
            iterations=8,
            seed=SEED,
            n_jobs=1,
            method=method,
            sampling_method=sampling_method,
        ),
        data,
    )

    assert result is not None


def test_n_jobs_one_matches_parallel_result(data):
    """Serialising must not change the answer, only how it is computed."""
    kwargs = dict(iterations=64, seed=SEED)

    sequential = rs.Bootstrap(n_jobs=1, **kwargs).roc_auc(
        data["y_true"], data["y_score"]
    )
    parallel = rs.Bootstrap(n_jobs=4, **kwargs).roc_auc(data["y_true"], data["y_score"])

    assert sequential == pytest.approx(parallel, rel=0, abs=1e-12)


def test_quiet_suppresses_progress_bars(data, capsys):
    """`quiet=True` must silence tqdm on the Python paths."""
    frame = pl.DataFrame({"y": data["y_score"]})

    rs.Bootstrap(iterations=8, seed=SEED, n_jobs=1, quiet=True).run(
        frame, lambda df: float(df["y"].mean())
    )

    assert capsys.readouterr().err == "", "progress output leaked despite quiet=True"


@pytest.mark.parametrize("sampling_method", ["multinomial", "poisson"])
def test_quiet_suppresses_the_bca_jackknife_bar(data, capsys, sampling_method):
    """The cum_sum BCa jackknife is a `_run_concurrent` call of its own.

    `_jacknife` defaults `quiet` to True, so the confusion-matrix site passing no
    executor kwargs was silent by accident rather than by request -- and `n_jobs` was
    dropped on the same line. Asserting on both together keeps them from drifting apart.
    """
    rs.Bootstrap(
        iterations=8,
        seed=SEED,
        n_jobs=1,
        quiet=True,
        method="BCa",
        sampling_method=sampling_method,
    ).confusion_matrix_at_thresholds(
        data["y_true"], data["y_score"], thresholds=[0.4, 0.6], strategy="cum_sum"
    )

    assert capsys.readouterr().err == "", "progress output leaked despite quiet=True"
