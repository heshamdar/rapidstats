"""Which polars engine should rapidstats default to?

Run directly (not under `cli-pybench`, which targets per-function timings):

    uv run python benchmarks/bench_engine.py

Polars state that the streaming engine "will soon become the default engine", and it is
the engine that bounds memory and parallelises by morsel. That makes it the obvious
default in general -- but this library's hot paths are unusual, so the choice is settled
by measurement here rather than by that general expectation.

What it reports:

- **Library shapes** -- the queries rapidstats actually runs. These are dominated by
  sort, `cum_sum` and `join_asof`: order-dependent operations that resist morsel
  parallelism.
- **Generic shapes** -- `group_by` and filter scans, which streaming is built for, as a
  control showing the harness is not simply biased against streaming.

Numbers are medians of repeated runs. Single-shot timings on a small box swing by up to
2x, which is wider than most of the differences being judged; an early single-shot pass
at this comparison produced a conclusion that repeated measurement reversed.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import polars as pl

import rapidstats as rs

ENGINES = ["in-memory", "streaming"]
REPEATS = 5
SEED = 208


def _bench(fn, repeats: int = REPEATS) -> float:
    """Median milliseconds, after a warm-up run."""
    fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)

    return statistics.median(timings)


def _row(label: str, per_engine: dict[str, float]):
    best = min(per_engine, key=per_engine.get)
    cells = "".join(f"{per_engine[e]:>12.1f}ms" for e in ENGINES)
    ratio = per_engine["in-memory"] / per_engine["streaming"]
    print(f"  {label:<44}{cells}   {ratio:>5.2f}x   {best}")


def _header(title: str):
    print(f"\n{title}")
    cols = "".join(f"{e:>14}" for e in ENGINES)
    print(f"  {'shape':<44}{cols}   {'in/str':>6}   winner")
    print(f"  {'-' * 44}{'-' * 28}   {'-' * 6}   {'-' * 10}")


def bench_library_shapes():
    _header("Library query shapes (what rapidstats actually runs)")

    rand = np.random.RandomState(SEED)

    for n in (10_000, 100_000):
        y_score = rand.rand(n)
        y_true = rand.rand(n) < 0.3 + 0.4 * y_score
        protected = rand.choice([True, False], n)

        cases = {
            f"confusion_matrix_at_thresholds (n={n:,})": lambda: (
                rs.metrics.confusion_matrix_at_thresholds(
                    y_true, y_score, strategy="cum_sum"
                )
            ),
            f"adverse_impact_ratio_at_thresholds (n={n:,})": lambda: (
                rs.metrics.adverse_impact_ratio_at_thresholds(
                    y_score, protected, ~protected, strategy="cum_sum"
                )
            ),
            f"average_precision (n={n:,})": lambda: rs.metrics.average_precision(
                y_true, y_score
            ),
        }

        for label, call in cases.items():
            per_engine = {}
            for engine in ENGINES:
                with rs.Config.engine(engine):
                    per_engine[engine] = _bench(call)
            _row(label, per_engine)


def bench_bootstrap():
    _header("Bootstrap (Python-side cum_sum paths)")

    rand = np.random.RandomState(SEED)
    n = 5_000
    y_score = rand.rand(n)
    y_true = rand.rand(n) < 0.3 + 0.4 * y_score
    thresholds = list(np.linspace(0.05, 0.95, 20))

    def call():
        return rs.Bootstrap(iterations=50, seed=SEED).confusion_matrix_at_thresholds(
            y_true, y_score, thresholds=thresholds
        )

    per_engine = {}
    for engine in ENGINES:
        with rs.Config.engine(engine):
            per_engine[engine] = _bench(call, repeats=3)
    _row(f"confusion_matrix_at_thresholds (n={n:,}, 50 iters)", per_engine)


def bench_generic_shapes():
    """Control: shapes the streaming engine is designed for."""
    _header("Generic shapes (control -- streaming's home ground)")

    rand = np.random.RandomState(SEED)
    n = 5_000_000
    frame = pl.LazyFrame(
        {
            "g": rand.randint(0, 1000, n),
            "x": rand.rand(n),
            "y": rand.rand(n),
        }
    )

    cases = {
        f"group_by aggregate (n={n:,})": frame.group_by("g").agg(
            pl.col("x").mean(), pl.col("y").sum(), pl.len()
        ),
        f"filter + elementwise + group_by (n={n:,})": (
            frame.filter(pl.col("x") > 0.1)
            .select((pl.col("x") * pl.col("y")).alias("z"), "g")
            .group_by("g")
            .agg(pl.col("z").sum())
        ),
    }

    for label, query in cases.items():
        per_engine = {
            engine: _bench(lambda q=query, e=engine: q.collect(engine=e), repeats=3)
            for engine in ENGINES
        }
        _row(label, per_engine)


def main():
    print(f"polars {pl.__version__} | rapidstats engine benchmark")
    print(f"medians of {REPEATS} runs; 'in/str' > 1 means streaming is faster")

    bench_library_shapes()
    bench_bootstrap()
    bench_generic_shapes()

    print(
        "\nIf the library shapes stop favouring in-memory, revisit the default in "
        "python/rapidstats/_config.py."
    )


if __name__ == "__main__":
    main()
