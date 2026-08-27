import concurrent.futures
import contextvars
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


def _in_context(context: contextvars.Context, fn, i):
    """Run `fn(i)` inside `context`. Module level so it stays picklable."""
    return context.run(fn, i)


def _context_per_task(fn, items: list) -> list:
    """One `(context, fn, item)` triple per task, contexts copied on the calling thread.

    `ThreadPoolExecutor` does not propagate the caller's context to its workers, so a
    `ContextVar` set by the caller -- `rapidstats.Config.engine(...)` -- would not be
    visible to work submitted there. `Bootstrap.run` cares: its user-supplied `stat_func`
    runs on those threads and may call any metric in the library.

    A copy per task rather than one shared copy, because a `Context` cannot be entered by
    two threads at once -- sharing one raises `RuntimeError: cannot enter context ... is
    already entered` as soon as the pool has more than one worker. And copied here rather
    than inside the worker, which would snapshot the worker's context instead of the
    caller's.
    """
    return [(contextvars.copy_context(), fn, item) for item in items]


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

    # Every task as a `(callable, *args)` pair, so the two dispatch branches below stay
    # identical whether or not a context is being carried.
    if executor == "threads":
        call, args = _in_context, _context_per_task(fn, list(iterable))
    else:
        # No context carried for processes, or for a caller-supplied executor instance:
        # a context cannot cross a process boundary, and a spawned worker re-imports the
        # package and sees the defaults -- as it did before contextvars.
        call, args = fn, [(item,) for item in iterable]

    if not args:
        return []

    if executor == "threads":
        executor = concurrent.futures.ThreadPoolExecutor(**executor_kwargs)
    elif executor == "processes":
        if "context" not in executor_kwargs:
            executor_kwargs["context"] = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)

    if preserve_order:
        with executor as pool:
            res = pool.map(call, *zip(*args))

        return list(res)

    with executor as pool:
        futures = [pool.submit(call, *a) for a in args]
        res = []
        for future in concurrent.futures.as_completed(futures):
            res.append(future.result())

    return res


def _collect(lf, **kwargs):
    """Collect through the configured engine.

    Every `.collect()` in the library goes through here so the engine is one decision in
    one place rather than 24 implicit ones. See `rapidstats.Config.set_engine`.

    Only polars takes an engine. `selection.py` pipes *narwhals* frames through here, and
    `narwhals.stable.v1.LazyFrame.collect` was `collect(self)` until ~1.30 -- so on the
    floor this project declares, forwarding `engine=` raises `TypeError`. An engine is
    meaningless for a non-polars backend anyway, so it is not forwarded there.
    """
    from ._config import Config

    if not isinstance(lf, pl.LazyFrame):
        return lf.collect(**kwargs)

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
