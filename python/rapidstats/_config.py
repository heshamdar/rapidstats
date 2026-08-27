"""Library-wide execution settings."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator, Literal

Engine = Literal["in-memory", "streaming"]

_VALID_ENGINES: tuple[Engine, ...] = ("in-memory", "streaming")

# In-memory rather than streaming, on measurement rather than habit -- but the right
# choice depends on where the data comes from, so this is a default, not a verdict.
#
# From an in-memory frame, the library's hot paths are sorts, `cum_sum` scans and as-of
# joins: order-dependent work that resists morsel parallelism. Benchmarked on polars 1.44,
# streaming measured 0.18-0.94x on those shapes.
#
# From a `scan_parquet` source (see `data=` in `metrics.py`) the picture inverts, because
# there is a real scan to stream and columns to prune:
#
#     confusion_matrix_at_thresholds, 2M rows, 2 of 42 columns
#         in-memory 2638.5ms    streaming 487.9ms    5.4x
#
# So prefer streaming when reading from disk:
#
#     with rs.Config.engine("streaming"):
#         rs.metrics.confusion_matrix_at_thresholds("y", "p", data=pl.scan_parquet(...))
#
# `benchmarks/bench_engine.py` reports the whole grid; revisit this default when polars
# flips its own.
_DEFAULT_ENGINE: Engine = "in-memory"

# A `ContextVar` rather than a module global, so `Config.engine(...)` scopes to the
# caller instead of the process. The library's own default execution model is
# `_run_concurrent(..., executor="threads")`, so a process-wide setting meant one thread's
# `with` block was visible to every other, and two threads entering it with different
# engines restored each other's values on exit.
#
# `_run_concurrent` copies the caller's context into its workers, so a caller-set engine
# still reaches the library's fan-out. Processes cannot inherit a context; a spawned
# worker re-imports this module and sees `_DEFAULT_ENGINE`, which is what it did before
# this change too.
_engine: contextvars.ContextVar[Engine] = contextvars.ContextVar(
    "rapidstats_engine", default=_DEFAULT_ENGINE
)


class Config:
    """Library-wide settings.

    Examples
    --------
    ``` py
    import rapidstats as rs

    rs.Config.set_engine("streaming")           # for the rest of the session
    with rs.Config.engine("streaming"):         # or just for a block
        ...
    ```

    Added in version 0.5.0
    ----------------------
    """

    @staticmethod
    def get_engine() -> Engine:
        """The polars engine used to execute queries.

        Returns
        -------
        Engine
            Either `"in-memory"` or `"streaming"`
        """
        return _engine.get()

    @staticmethod
    def set_engine(engine: Engine) -> None:
        """Set the polars engine used to execute queries.

        The engine changes how a query runs, never what it returns.

        Parameters
        ----------
        engine : Engine
            Either `"in-memory"` (the default) or `"streaming"`

        Raises
        ------
        ValueError
            If `engine` is not a supported value
        """
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"Invalid engine `{engine}`, only "
                f"{' and '.join(repr(e) for e in _VALID_ENGINES)} are supported"
            )

        _engine.set(engine)

    @staticmethod
    @contextlib.contextmanager
    def engine(engine: Engine) -> Iterator[None]:
        """Use `engine` for the duration of the block, then restore the previous value.

        Parameters
        ----------
        engine : Engine
            Either `"in-memory"` or `"streaming"`
        """
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"Invalid engine `{engine}`, only "
                f"{' and '.join(repr(e) for e in _VALID_ENGINES)} are supported"
            )

        # A token rather than saving and restoring the value, so nesting unwinds
        # correctly even if an inner block sets the engine without a context manager.
        token = _engine.set(engine)
        try:
            yield
        finally:
            _engine.reset(token)
