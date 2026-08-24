"""Library-wide execution settings."""

from __future__ import annotations

import contextlib
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

_engine: Engine = _DEFAULT_ENGINE


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
        return _engine

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
        global _engine

        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"Invalid engine `{engine}`, only "
                f"{' and '.join(repr(e) for e in _VALID_ENGINES)} are supported"
            )

        _engine = engine

    @staticmethod
    @contextlib.contextmanager
    def engine(engine: Engine) -> Iterator[None]:
        """Use `engine` for the duration of the block, then restore the previous value.

        Parameters
        ----------
        engine : Engine
            Either `"in-memory"` or `"streaming"`
        """
        previous = Config.get_engine()
        Config.set_engine(engine)
        try:
            yield
        finally:
            Config.set_engine(previous)
