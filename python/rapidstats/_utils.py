import concurrent.futures
import multiprocessing
from typing import Literal, Optional, Union

import polars as pl
from tqdm.auto import tqdm

from ._typing import ArrayLike

PolarsFrame = Union[pl.LazyFrame, pl.DataFrame]


def _regression_to_df(
    y_true: ArrayLike, y_score: ArrayLike, *, data=None
) -> pl.DataFrame:
    if data is not None:
        return _collect(_regression_to_lf(data, y_true, y_score))

    return (
        pl.DataFrame({"y_true": y_true, "y_score": y_score})
        .with_columns(pl.col("y_true", "y_score").cast(pl.Float64))
        .drop_nulls()
    )


def _y_true_y_score_to_df(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> pl.DataFrame:
    """`y_true` as boolean, `y_score` and `sample_weight` as float64, nulls dropped.

    With `data`, the arguments are column names and the frame is built lazily before
    collecting -- so a scan source only reads the columns named here.
    """
    if data is not None:
        return _collect(_y_true_y_score_to_lf(data, y_true, y_score, sample_weight))

    return (
        pl.DataFrame(
            {
                "y_true": y_true,
                "y_score": y_score,
                "sample_weight": 1.0 if sample_weight is None else sample_weight,
            }
        )
        .with_columns(
            pl.col("y_true").cast(pl.Boolean),
            pl.col("y_score", "sample_weight").cast(pl.Float64),
        )
        .drop_nulls()
    )


def _y_true_y_pred_to_df(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> pl.DataFrame:
    if data is not None:
        return _collect(_y_true_y_pred_to_lf(data, y_true, y_pred, sample_weight))

    return (
        pl.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "sample_weight": (1.0 if sample_weight is None else sample_weight),
            }
        )
        .select(
            pl.col("y_true", "y_pred").cast(pl.Boolean),
            pl.col("sample_weight").cast(pl.Float64),
        )
        .drop_nulls()
    )


def _fill_infinite(
    pf: PolarsFrame, value: Union[pl.Expr, int, float, None] = None
) -> PolarsFrame:
    return pf.with_columns(
        pl.when(pl.selectors.float().is_infinite())
        .then(value)
        .otherwise(pl.selectors.float())
        .name.keep()
    )


def _expr_fill_infinite(
    expr: pl.Expr, value: Union[pl.Expr, int, float, None] = None
) -> pl.Expr:
    return pl.when(expr.is_infinite()).then(value).otherwise(expr)


def _run_concurrent(
    fn,
    iterable,
    executor: Union[
        Literal["threads", "processes"],
        concurrent.futures.ThreadPoolExecutor,
        concurrent.futures.ProcessPoolExecutor,
    ] = "threads",
    preserve_order: bool = False,
    quiet: bool = False,
    **executor_kwargs,
) -> list:
    if executor_kwargs.get("max_workers") == 1:
        return [fn(i) for i in tqdm(iterable, disable=quiet)]

    if executor == "threads":
        executor = concurrent.futures.ThreadPoolExecutor(**executor_kwargs)
    elif executor == "processes":
        if "context" not in executor_kwargs:
            executor_kwargs["context"] = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)

    if preserve_order:
        with executor as pool:
            res = pool.map(fn, iterable)

        return list(res)

    with executor as pool:
        futures = [pool.submit(fn, i) for i in iterable]
        res = []
        for future in concurrent.futures.as_completed(futures):
            res.append(future.result())

    return res


def _collect(lf: pl.LazyFrame, **kwargs):
    """Collect through the configured engine.

    Every `.collect()` in the library goes through here so the engine is one decision in
    one place rather than 24 implicit ones. See `rapidstats.Config.set_engine`.
    """
    from ._config import Config

    return lf.collect(engine=Config.get_engine(), **kwargs)


def _resolve_data(data) -> pl.LazyFrame:
    """Normalise anything frame-like into a `LazyFrame`.

    Mirrors the conversion `_corr.py::_prepare_inputs` already performs, so a pandas or
    pyarrow table works here for the same reason it works there.
    """
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()

    import narwhals.stable.v1 as nw

    return nw.from_native(data).to_polars().lazy()


def _resolve_thresholds(thresholds, df: pl.DataFrame, column: str) -> set:
    """The thresholds to evaluate, read from the frame rather than the argument.

    The caller's `y_score` may be a column name when `data=` is given, so the raw argument
    cannot be iterated -- `set("score")` is five single characters, each then compared
    against a float column.

    Reading the already-materialised frame is strictly better without `data=` too: it
    deduplicates, and it drops nulls along with the rest of the frame instead of carrying
    them into the loop. It also stops the `loop` and `cum_sum` strategies deriving their
    threshold sets from different sources -- `cum_sum` already reads the frame.
    """
    if thresholds is not None:
        return set(thresholds)

    return set(df[column].unique().to_list())


def _column(name, argument: str) -> pl.Expr:
    """A column reference, insisting the caller passed a name rather than an array.

    Mixing the two -- an array for one argument and `data=` for another -- cannot be
    honoured, and silently ignoring one of them would be worse than refusing.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"`{argument}` must be a column name when `data` is given, got "
            f"{type(name).__name__}"
        )

    return pl.col(name)


def _y_true_y_score_to_lf(
    data,
    y_true,
    y_score,
    sample_weight=None,
    *,
    y_true_dtype: pl.DataType = pl.Boolean,
) -> pl.LazyFrame:
    """Lazy sibling of `_y_true_y_score_to_df`, selecting from `data` by name.

    Same canonical column names, casts and null handling, so everything downstream is
    unchanged. Selecting only the named columns is what lets polars prune the rest: on a
    `scan_parquet` source the untouched columns are never read from disk.
    """
    weight = (
        pl.lit(1.0)
        if sample_weight is None
        else _column(sample_weight, "sample_weight").cast(pl.Float64)
    )

    return (
        _resolve_data(data)
        .select(
            _column(y_true, "y_true").cast(y_true_dtype).alias("y_true"),
            _column(y_score, "y_score").cast(pl.Float64).alias("y_score"),
            weight.alias("sample_weight"),
        )
        .drop_nulls()
    )


def _y_true_y_pred_to_lf(data, y_true, y_pred, sample_weight=None) -> pl.LazyFrame:
    """Lazy sibling of `_y_true_y_pred_to_df`."""
    weight = (
        pl.lit(1.0)
        if sample_weight is None
        else _column(sample_weight, "sample_weight").cast(pl.Float64)
    )

    return (
        _resolve_data(data)
        .select(
            _column(y_true, "y_true").cast(pl.Boolean).alias("y_true"),
            _column(y_pred, "y_pred").cast(pl.Boolean).alias("y_pred"),
            weight.alias("sample_weight"),
        )
        .drop_nulls()
    )


def _regression_to_lf(data, y_true, y_score) -> pl.LazyFrame:
    """Lazy sibling of `_regression_to_df`."""
    return (
        _resolve_data(data)
        .select(
            _column(y_true, "y_true").cast(pl.Float64).alias("y_true"),
            _column(y_score, "y_score").cast(pl.Float64).alias("y_score"),
        )
        .drop_nulls()
    )
