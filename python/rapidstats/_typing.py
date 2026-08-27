from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar, Union

import polars as pl

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)

# Anything the polars Series constructor accepts: lists, numpy arrays, pandas Series,
# Arrow arrays, polars Series.
#
# Defined here rather than imported from polars on purpose. Upstream has kept this alias
# at `polars.series.series.ArrayLike` (<= 1.40) and `polars._typing.ArrayLike` (>= 1.44),
# and neither path is public API -- importing either one pins the library to a narrow
# polars range for the sake of a type annotation that is never evaluated at runtime.
ArrayLike = Union[Iterable[Any], "pl.Series"]
