"""Library-wide execution settings."""

from __future__ import annotations

import contextlib
from typing import Iterator, Literal

Engine = Literal["in-memory", "streaming"]

_VALID_ENGINES: tuple[Engine, ...] = ("in-memory", "streaming")

# In-memory rather than streaming, on measurement rather than habit. This library's hot
# paths are sorts, `cum_sum` scans and as-of joins -- order-dependent operations that
# resist morsel parallelism -- and the bootstrap runs many small queries, where the
# streaming engine's per-query setup dominates. Benchmarked on polars 1.44, streaming
# measured 0.18-0.94x on these shapes while winning 3.5-4.4x on generic group_by and
# filter scans. See `benchmarks/bench_engine.py`; revisit when polars flips its own
# default, and prefer streaming for lazy `scan_*` sources where the data need never be
# materialised at all.
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
