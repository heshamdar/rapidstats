import dataclasses
import typing
from collections.abc import Iterable
from typing import Literal, Optional, Union, cast

import polars as pl

from ._rustystats import (
    _adverse_impact_ratio,
    _brier_loss,
    _confusion_matrix,
    _max_ks,
    _mean,
    _mean_squared_error,
    _r2,
    _roc_auc,
    _root_mean_squared_error,
)
from ._typing import ArrayLike, PolarsFrameT
from ._utils import (
    _collect,
    _column,
    _resolve_data,
    _resolve_thresholds,
    _y_true_y_score_to_lf,
    _fill_infinite,
    _regression_to_df,
    _run_concurrent,
    _y_true_y_pred_to_df,
    _y_true_y_score_to_df,
)

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]
ConfusionMatrixMetric = Literal[
    "tn",
    "fp",
    "fn",
    "tp",
    "tpr",
    "fpr",
    "fnr",
    "tnr",
    "prevalence",
    "prevalence_threshold",
    "informedness",
    "precision",
    "false_omission_rate",
    "plr",
    "nlr",
    "acc",
    "balanced_accuracy",
    "fbeta",
    "folkes_mallows_index",
    "mcc",
    "threat_score",
    "markedness",
    "fdr",
    "npv",
    "dor",
    "ppr",
    "pnr",
]

DefaultConfusionMatrixMetrics: tuple[ConfusionMatrixMetric] = typing.get_args(
    ConfusionMatrixMetric
)
LoopStrategy = Literal["auto", "loop", "cum_sum"]


@dataclasses.dataclass
class ConfusionMatrix:
    r"""Result object returned by `rapidstats.metrics.confusion_matrix`

    Attributes
    ----------
    tn : float
        ↑Count of True Negatives; y_true == False and y_pred == False
    fp : float
        ↓Count of False Positives; y_true == False and y_pred == True
    fn : float
        ↓Count of False Negatives; y_true == True and y_pred == False
    tp : float
        ↑Count of True Positives; y_true == True, y_pred == True
    tpr : float
        ↑True Positive Rate, Recall, Sensitivity; Probability that an actual positive
        will be predicted positive; \( \frac{TP}{TP + FN} \)
    fpr : float
        ↓False Positive Rate, Type I Error; Probability that an actual negative will
        be predicted positive; \( \frac{FP}{FP + TN} \)
    fnr : float
        ↓False Negative Rate, Type II Error; Probability an actual positive will be
        predicted negative; \( \frac{FN}{TP + FN} \)
    tnr : float
        ↑True Negative Rate, Specificity; Probability an actual negative will be
        predicted negative; \( \frac{TN}{FP + TN} \)
    prevalence : float
        Prevalence; Proportion of positive classes; \( \frac{TP + FN}{TN + FP + FN + TP} \)
    prevalence_threshold : float
        Prevalence Threshold; \( \frac{\sqrt{TPR \times FPR} - FPR}{TPR - FPR} \)
    informedness : float
        ↑Informedness, Youden's J; \( TPR + TNR - 1 \)
    precision : float
        ↑Precision, Positive Predicted Value (PPV); Probability a predicted positive was
        actually correct; \( \frac{TP}{TP + FP} \)
    false_omission_rate : float
        ↓False Omission Rate (FOR); Proportion of predicted negatives that were wrong
        \( \frac{FN}{FN + TN} \)
    plr : float
        ↑Positive Likelihood Ratio, LR+; \( \frac{TPR}{FPR} \)
    nlr : float
        Negative Likelihood Ratio, LR-; \( \frac{FNR}{TNR} \)
    acc : float
        ↑Accuracy (ACC); Probability of a correct prediction; \( \frac{TP + TN}{TN + FP + FN + TP} \)
    balanced_accuracy : float
        ↑Balanced Accuracy (BA); \( \frac{TP + TN}{2} \)
    fbeta : float
        ↑\( F_{\beta} \); Harmonic mean of Precision and Recall; \( \frac{(1 + \beta)^2 \times PPV \times TPR}{(\beta^2 \times PPV) + TPR} \)
    folkes_mallows_index : float
        ↑Folkes Mallows Index (FM); \( \sqrt{PPV \times TPR} \)
    mcc : float
        ↑Matthew Correlation Coefficient (MCC), Yule Phi Coefficient; \( \sqrt{TPR \times TNR \times PPV \times NPV} - \sqrt{FNR \times FPR \times FOR \times FDR} \)
    threat_score : float
        ↑Threat Score (TS), Critical Success Index (CSI), Jaccard Index; \( \frac{TP}{TP + FN + FP} \)
    markedness : float
        Markedness (MP), deltaP; \( PPV + NPV - 1 \)
    fdr : float
        ↓False Discovery Rate, Proportion of predicted positives that are wrong; \( \frac{FP}{TP + FP} \)
    ↑npv : float
        Negative Predictive Value; Proportion of predicted negatives that are correct; \( \frac{TN}{FN + TN} \)
    dor : float
        Diagnostic Odds Ratio; \( \frac{LR+}{LR-} \)
    ppr : float
        Predicted Positive Ratio; Proportion that are predicted positive; \( \frac{TP + FP}{TN + FP + FN + TP} \)
    pnr : float
        Predicted Negative Ratio; Proportion that are predicted negative; \( \frac{TN + FN}{TN + FP + FN + TP} \)
    """

    tn: float
    fp: float
    fn: float
    tp: float
    tpr: float
    fpr: float
    fnr: float
    tnr: float
    prevalence: float
    prevalence_threshold: float
    informedness: float
    precision: float
    false_omission_rate: float
    plr: float
    nlr: float
    acc: float
    balanced_accuracy: float
    fbeta: float
    folkes_mallows_index: float
    mcc: float
    threat_score: float
    markedness: float
    fdr: float
    npv: float
    dor: float
    ppr: float
    pnr: float

    def to_polars(self) -> pl.DataFrame:
        """Convert the dataclass to a long Polars DataFrame with columns `metric` and
        `value`.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns `metric` and `value`
        """
        dct = self.__dict__

        return pl.DataFrame({"metric": dct.keys(), "value": dct.values()})


def _air_frame(y_pred, protected, control, sample_weight=None, *, data=None):
    """Eager AIR input frame, from arrays or from named columns of `data`."""
    if data is not None:
        weight = (
            pl.lit(1.0)
            if sample_weight is None
            else _column(sample_weight, "sample_weight").cast(pl.Float64)
        )

        return _collect(
            _resolve_data(data).select(
                _column(y_pred, "y_pred").alias("y_pred"),
                _column(protected, "protected").alias("protected"),
                _column(control, "control").alias("control"),
                weight.alias("sample_weight"),
            )
        )

    return pl.DataFrame(
        {
            "y_pred": y_pred,
            "protected": protected,
            "control": control,
            "sample_weight": 1.0 if sample_weight is None else sample_weight,
        }
    ).with_columns(pl.col("sample_weight").cast(pl.Float64))


def _air_score_frame(y_score, protected, control, sample_weight=None, *, data=None):
    """Lazy AIR-at-thresholds input, from arrays or from named columns of `data`."""
    if data is not None:
        weight = (
            pl.lit(1.0)
            if sample_weight is None
            else _column(sample_weight, "sample_weight").cast(pl.Float64)
        )

        return _resolve_data(data).select(
            _column(y_score, "y_score").cast(pl.Float64).alias("y_score"),
            _column(protected, "protected").cast(pl.Boolean).alias("protected"),
            _column(control, "control").cast(pl.Boolean).alias("control"),
            weight.alias("sample_weight"),
        )

    return pl.LazyFrame(
        {
            "y_score": y_score,
            "protected": protected,
            "control": control,
            "sample_weight": 1.0 if sample_weight is None else sample_weight,
        }
    ).select(
        pl.col("y_score").cast(pl.Float64),
        pl.col("protected", "control").cast(pl.Boolean),
        pl.col("sample_weight").cast(pl.Float64),
    )


def _score_frame(y_score, sample_weight=None, *, data=None) -> pl.LazyFrame:
    """Lazy `y_score`/`sample_weight` input, from arrays or named columns of `data`."""
    if data is not None:
        weight = (
            pl.lit(1.0)
            if sample_weight is None
            else _column(sample_weight, "sample_weight").cast(pl.Float64)
        )

        return (
            _resolve_data(data)
            .select(
                _column(y_score, "y_score").cast(pl.Float64).alias("y_score"),
                weight.alias("sample_weight"),
            )
            .drop_nulls()
        )

    return pl.LazyFrame(
        {
            "y_score": y_score,
            "sample_weight": 1.0 if sample_weight is None else sample_weight,
        }
    ).drop_nulls()


def _cm_curve_frame(y_true, y_score, sample_weight=None, *, data=None) -> pl.LazyFrame:
    """Lazy `y_true`/`threshold`/`sample_weight` input for the confusion-matrix curve."""
    if data is not None:
        return _y_true_y_score_to_lf(data, y_true, y_score, sample_weight).rename(
            {"y_score": "threshold"}
        )

    return (
        pl.LazyFrame(
            {
                "y_true": y_true,
                "threshold": y_score,
                "sample_weight": 1.0 if sample_weight is None else sample_weight,
            }
        )
        # Cast all three, not just `y_true`. An empty input infers Null columns, and the
        # weighted cum_sum downstream then fails with "`multiply` operation not supported
        # for dtype `bool`" rather than returning NaN.
        .select(
            pl.col("y_true").cast(pl.Boolean),
            pl.col("threshold").cast(pl.Float64),
            pl.col("sample_weight").cast(pl.Float64),
        )
        .drop_nulls()
    )


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    beta: float = 1.0,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> ConfusionMatrix:
    r"""Computes confusion matrix metrics (TP, FP, TN, FN, TPR, Fbeta, etc.).

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_pred : ArrayLike
        Predicted target
    beta : float, optional
        \( \beta \) to use in \( F_\beta \), by default 1
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0

    Returns
    -------
    ConfusionMatrix
        Dataclass of confusion matrix metrics

    Added in version 0.1.0
    ----------------------
    """
    df = _y_true_y_pred_to_df(y_true, y_pred, sample_weight, data=data).with_columns(
        pl.col("y_true").cast(pl.UInt8)
    )

    return ConfusionMatrix(*_confusion_matrix(df, beta))


def roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> Optional[float]:
    """Computes Area Under the Receiver Operating Characteristic Curve.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0

    Returns
    -------
    Optional[float]
        ROC-AUC, or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    df = _y_true_y_score_to_df(y_true, y_score, sample_weight, data=data).with_columns(
        pl.col("y_true").cast(pl.Float64)
    )

    return _roc_auc(df)


def max_ks(y_true: ArrayLike, y_score: ArrayLike, *, data=None) -> Optional[float]:
    """Performs the two-sample Kolmogorov-Smirnov test on the predicted scores of the
    ground truth positive and ground truth negative classes. The KS test measures the
    highest distance between two CDFs, so the Max-KS metric measures how well the model
    separates two classes. In pseucode:

    ``` py
    df = Frame(y_true, y_score)
    class0 = df.filter(~y_true)["y_score"]
    class1 = df.filter(y_true)["y_score"]

    ks(class0, class1)
    ```

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    Optional[float]
        Max-KS, or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    df = _y_true_y_score_to_df(y_true, y_score, data=data)

    return _max_ks(df)


def brier_loss(y_true: ArrayLike, y_score: ArrayLike, *, data=None) -> Optional[float]:
    r"""Computes the Brier loss (smaller is better). The Brier loss measures the mean
    squared difference between the predicted scores and the ground truth target.
    Calculated as:

    \[ \frac{1}{N} \sum_{i=1}^N (yt_i - ys_i)^2 \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    Optional[float]
        Brier loss, or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    df = _y_true_y_score_to_df(y_true, y_score, data=data)

    return _brier_loss(df)


def mean(y: ArrayLike, *, data=None) -> Optional[float]:
    """Computes the mean of the input array.

    Parameters
    ----------
    y : ArrayLike
        A 1D-array of numbers

    Returns
    -------
    Optional[float]
        Mean, or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    if data is not None:
        frame = _collect(_resolve_data(data).select(_column(y, "y").alias("y")))
    else:
        frame = pl.DataFrame({"y": y})

    return _mean(frame)


def _weighted_mean(x: pl.Series, sample_weight: pl.Series):
    return (x * sample_weight).sum() / (sample_weight.sum())


def predicted_positive_ratio_at_thresholds(
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    thresholds: Optional[list[float]] = None,
    strategy: LoopStrategy = "auto",
    *,
    data=None,
    lazy: bool = False,
) -> pl.DataFrame:
    """Computes the Predicted Positive Ratio (PPR) at each threshold, where the PPR is
    the ratio of predicted positive to the total, and a positive is defined as
    `y_score` >= threshold.

    Parameters
    ----------
    y_score : ArrayLike
        Predicted scores
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0
    thresholds : Optional[list[float]], optional
        The thresholds to compute `y_pred` at, i.e. y_score >= t. If None,
        uses every score present in `y_score`, by default None
    strategy : LoopStrategy, optional
        Computation method, by default "auto"

    Returns
    -------
    pl.DataFrame
        A DataFrame of `threshold` and `ppr`

    Added in version 0.1.0
    ----------------------
    """
    lf = _score_frame(y_score, sample_weight, data=data)

    strategy = _set_loop_strategy(thresholds, strategy)

    if lazy and strategy == "loop":
        raise ValueError("`lazy=True` requires `strategy='cum_sum'`")

    if strategy == "loop":
        df = lf.pipe(_collect)

        def _ppr(t: float) -> dict:
            return {
                "threshold": t,
                "ppr": _weighted_mean(df["y_score"].ge(t), df["sample_weight"]),
            }

        return pl.DataFrame(
            _run_concurrent(_ppr, _resolve_thresholds(thresholds, df, "y_score"))
        )
    elif strategy == "cum_sum":

        def _cumulative_ppr(lf: pl.LazyFrame, has_sample_weight: bool):
            if not has_sample_weight:
                return lf.with_row_index(
                    "cumulative_predicted_positive", offset=1
                ).with_columns(
                    pl.col("cumulative_predicted_positive")
                    .truediv(pl.len())
                    .alias("ppr")
                )
            else:
                return lf.with_columns(
                    pl.col("sample_weight")
                    .cum_sum()
                    .alias("cumulative_predicted_positive")
                ).with_columns(
                    pl.col("cumulative_predicted_positive")
                    .truediv(pl.col("sample_weight").sum())
                    .alias("ppr")
                )

        result = (
            lf.sort("y_score", descending=True)
            .pipe(_cumulative_ppr, sample_weight is not None)
            .rename({"y_score": "threshold"})
            .select("threshold", "ppr")
            .pipe(_dedup_ties)
            .pipe(_map_to_thresholds, thresholds)
            .drop("_threshold_actual", strict=False)
        )

        return result if lazy else _collect(result)


def adverse_impact_ratio(
    y_pred: ArrayLike,
    protected: ArrayLike,
    control: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> Optional[float]:
    """Computes the Adverse Impact Ratio (AIR), which is the ratio of negative
    predictions for the protected class and the control class. The ideal ratio is 1.
    For example, in an underwriting context, this means that the model is equally as
    likely to approve protected applicants as it is unprotected applicants, given that
    the model score is probability of bad.

    Parameters
    ----------
    y_pred : ArrayLike
        Predicted negative
    protected : ArrayLike
        An array of booleans identifying the protected class
    control : ArrayLike
        An array of booleans identifying the control class
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0

    Returns
    -------
    Optional[float]
        Adverse Impact Ratio (AIR), or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    return _adverse_impact_ratio(
        _air_frame(y_pred, protected, control, sample_weight, data=data)
        .with_columns(pl.col("y_pred", "protected", "control").cast(pl.Boolean))
        .with_columns(pl.col("y_pred").cast(pl.Float64))
    )


def _air_at_thresholds_core_sorted(
    pf: PolarsFrame,
    thresholds: Optional[list[float]],
    has_sample_weight: bool,
) -> pl.LazyFrame:
    def _appr_rate(pf: PolarsFrame) -> pl.LazyFrame:
        # An approve is score < t
        return (
            pf.lazy()
            .with_row_index("cumulative_approved")
            .with_columns(
                pl.col("cumulative_approved").truediv(pl.len()).alias("appr_rate")
            )
            .rename({"y_score": "threshold"})
            .pipe(_dedup_ties, keep="first")
        )

    def _appr_rate_sample_weight(pf: PolarsFrame) -> pl.LazyFrame:
        # An approve is score < t
        return (
            pf.lazy()
            .with_columns(
                pl.col("sample_weight")
                .shift(1, fill_value=0.0)
                .cum_sum()
                .alias("cumulative_approved")
            )
            .with_columns(
                pl.col("cumulative_approved")
                .truediv(pl.col("sample_weight").sum())
                .alias("appr_rate")
            )
            .rename({"y_score": "threshold"})
            .pipe(_dedup_ties, keep="first")
        )

    _appr_rate_func = _appr_rate_sample_weight if has_sample_weight else _appr_rate

    p = _appr_rate_func(pf.filter(pl.col("protected"))).rename(
        {"appr_rate": "appr_rate_protected"}
    )

    c = _appr_rate_func(pf.filter(pl.col("control"))).rename(
        {"appr_rate": "appr_rate_control"}
    )

    thresholds = (
        thresholds
        if thresholds is not None
        else pf.lazy().select("y_score").unique().pipe(_collect).to_series()
    )

    # The protected/control join below is 1:1, so targets must be distinct. Enforce it
    # here rather than trusting every caller: a duplicated target is meaningless for AIR
    # (it would emit the same ratio twice) and previously raised a bare polars
    # ComputeError about join validation.
    thresholds = pl.Series("threshold", thresholds).unique()

    p = p.pipe(_map_to_thresholds, thresholds).with_columns(
        pl.when(pl.col("_threshold_actual").is_null())
        .then(1)
        .otherwise(pl.col("appr_rate_protected"))
        .alias("appr_rate_protected"),
    )
    c = c.pipe(_map_to_thresholds, thresholds).with_columns(
        pl.when(pl.col("_threshold_actual").is_null())
        .then(1)
        .otherwise(pl.col("appr_rate_control"))
        .alias("appr_rate_control"),
    )

    return (
        p.join(
            c,
            on="threshold",
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("appr_rate_protected")
            .truediv(pl.col("appr_rate_control"))
            .alias("air")
        )
        .select("threshold", "air")
    )


def _air_at_thresholds_core(
    pf: PolarsFrame,
    thresholds: Optional[list[float]],
    has_sample_weight: bool,
) -> pl.LazyFrame:
    pf = pf.sort("y_score", descending=False)

    return _air_at_thresholds_core_sorted(pf, thresholds, has_sample_weight)


def adverse_impact_ratio_at_thresholds(
    y_score: ArrayLike,
    protected: ArrayLike,
    control: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    thresholds: Optional[list[float]] = None,
    strategy: LoopStrategy = "auto",
    *,
    data=None,
    lazy: bool = False,
) -> pl.DataFrame:
    """Computes the Adverse Impact Ratio (AIR) at each threshold of `y_score`. See
    [rapidstats.metrics.adverse_impact_ratio][] for more details. When the `strategy` is
    `cum_sum`, computes


    ``` py
    for t in y_score:
        is_predicted_negative = y_score < t
        adverse_impact_ratio(is_predicted_negative, protected, control)
    ```

    Parameters
    ----------
    y_score : ArrayLike
        Predicted scores
    protected : ArrayLike
        An array of booleans identifying the protected class
    control : ArrayLike
        An array of booleans identifying the control class
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0
    thresholds : Optional[list[float]], optional
        The thresholds to compute `is_predicted_negative` at, i.e. y_score < t. If None,
        uses every score present in `y_score`, by default None
    strategy : LoopStrategy, optional
        Computation method, by default "auto"

    Returns
    -------
    pl.DataFrame
        A DataFrame of `threshold` and `air`

    Added in version 0.1.0
    ----------------------
    """
    has_sample_weight = sample_weight is not None
    lf = _air_score_frame(y_score, protected, control, sample_weight, data=data)

    strategy = _set_loop_strategy(thresholds, strategy)

    if lazy and strategy == "loop":
        raise ValueError("`lazy=True` requires `strategy='cum_sum'`")

    if strategy == "loop":
        df = _collect(lf)

        def _air(t):
            return {
                "threshold": t,
                "air": _adverse_impact_ratio(
                    df.select(
                        pl.col("y_score").lt(t).cast(pl.Float64).alias("y_pred"),
                        "protected",
                        "control",
                        "sample_weight",
                    )
                ),
            }

        airs = _run_concurrent(_air, _resolve_thresholds(thresholds, df, "y_score"))

        res = pl.LazyFrame(airs)
    elif strategy == "cum_sum":
        res = _air_at_thresholds_core(
            lf, thresholds, has_sample_weight=has_sample_weight
        )

    res = res.pipe(_fill_infinite, None).fill_nan(None)

    return res if lazy else _collect(res)


def mean_squared_error(
    y_true: ArrayLike, y_score: ArrayLike, *, data=None
) -> Optional[float]:
    r"""Computes Mean Squared Error (MSE) as

    \[ \frac{1}{N} \sum_{i=1}^{N} (yt_i - ys_i)^2 \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    Optional[float]
        Mean Squared Error (MSE), or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    return _mean_squared_error(_regression_to_df(y_true, y_score, data=data))


def root_mean_squared_error(
    y_true: ArrayLike, y_score: ArrayLike, *, data=None
) -> Optional[float]:
    r"""Computes Root Mean Squared Error (RMSE) as

    \[ \sqrt{\frac{1}{N} \sum_{i=1}^{N} (yt_i - ys_i)^2} \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    Optional[float]
        Root Mean Squared Error (RMSE), or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    return _root_mean_squared_error(_regression_to_df(y_true, y_score, data=data))


def _dedup_ties(
    pf: PolarsFrameT,
    col: str = "threshold",
    keep: Literal["first", "last"] = "last",
) -> PolarsFrameT:
    """Deduplicate tied values in `col`, keeping either the first or last row per group.

    Assumes `pf` is already sorted on `col`. Use `keep="last"` for descending sorts
    (e.g. confusion matrix, PPR) and `keep="first"` for ascending sorts (e.g. AIR).
    """
    if keep == "last":
        filter_expr = pl.col("_tie_group") != pl.col("_tie_group").shift(-1).fill_null(
            -1
        )
    else:
        filter_expr = pl.col("_tie_group") != pl.col("_tie_group").shift(1).fill_null(
            -1
        )

    return cast(
        PolarsFrameT,
        pf.with_columns(pl.col(col).rle_id().alias("_tie_group"))
        .filter(filter_expr)
        .drop("_tie_group"),
    )


def _set_loop_strategy(
    thresholds: Optional[list[float]], strategy: LoopStrategy
) -> Literal["loop", "cum_sum"]:
    if strategy == "auto":
        if thresholds is not None and len(thresholds) < 10:
            return "loop"
        else:
            return "cum_sum"

    if strategy not in ("loop", "cum_sum"):
        raise ValueError(
            f"Invalid strategy {strategy}, please specify one of `auto`, `loop`, or `cum_sum`."
        )

    return strategy


def _base_confusion_matrix_at_thresholds_sorted(pf: PolarsFrameT) -> PolarsFrameT:
    """Compute basic confusion matrix. Assumes that it is sorted.

    Parameters
    ----------
    pf : PolarsFrame
        Needs `y_true`, `threshold`, and `sample_weight`

    Returns
    -------
    PolarsFrame
        A frame of `threshold`, `tn`, `fp`, `fn`, and `tp`
    """
    return (
        pf.with_columns(
            (pl.col("y_true") * pl.col("sample_weight")).cum_sum().alias("tp"),
            (pl.col("y_true").not_() * pl.col("sample_weight")).cum_sum().alias("fp"),
        )
        .with_columns(
            pl.col("tp").tail(1).first().alias("total_positives"),
        )
        .with_columns(
            pl.col("sample_weight")
            .sum()
            .sub(pl.col("total_positives"))
            .alias("total_negatives")
        )
        .with_columns(
            pl.col("total_positives").sub(pl.col("tp")).alias("fn"),
            pl.col("total_negatives").sub(pl.col("fp")).round(12).alias("tn"),
        )
        .with_columns(
            pl.col("tp").add(pl.col("fn")).alias("p"),
            pl.col("fp").add(pl.col("tn")).alias("n"),
        )
        .pipe(_dedup_ties)
        .select("threshold", "tn", "fp", "fn", "tp")
    )


def _base_confusion_matrix_at_thresholds(pf: PolarsFrameT) -> PolarsFrameT:
    return pf.sort("threshold", descending=True).pipe(
        _base_confusion_matrix_at_thresholds_sorted
    )


def _full_confusion_matrix_from_base(
    pf: PolarsFrameT, beta: float = 1.0
) -> PolarsFrameT:
    beta_squared = beta**2

    res = (
        pf.with_columns(
            pl.col("tp").add(pl.col("fn")).alias("p"),
            pl.col("fp").add(pl.col("tn")).alias("n"),
            pl.col("tp").add(pl.col("fp")).alias("pp"),
            pl.col("tn").add(pl.col("fn")).alias("pn"),
        )
        .with_columns(
            pl.col("tp").truediv("p").alias("tpr"),
            pl.col("fp").truediv("n").alias("fpr"),
            pl.col("tp").truediv(pl.col("tp").add(pl.col("fp"))).alias("precision"),
            pl.col("fn")
            .truediv(pl.col("fn").add(pl.col("tn")))
            .alias("false_omission_rate"),
            pl.col("p").truediv(pl.col("p").add(pl.col("n"))).alias("prevalence"),
            pl.col("p").add(pl.col("n")).alias("total"),
        )
        .with_columns(
            pl.lit(1).sub(pl.col("fpr")).alias("tnr"),
            pl.lit(1).sub(pl.col("precision")).alias("fdr"),
            pl.col("tpr")
            .add(pl.lit(1).sub(pl.col("fpr")))
            .sub(1)
            .alias("informedness"),
            pl.col("precision").sub(pl.col("false_omission_rate")).alias("markedness"),
            pl.lit(1 + beta_squared)
            .mul(pl.col("precision"))
            .mul(pl.col("tpr"))
            .truediv(pl.lit(beta_squared).mul(pl.col("precision")).add(pl.col("tpr")))
            .alias("fbeta"),
            (pl.col("precision").mul(pl.col("tpr")))
            .sqrt()
            .alias("folkes_mallows_index"),
            pl.col("tp").add(pl.col("tn")).truediv(pl.col("total")).alias("acc"),
            pl.col("tp")
            .truediv(pl.col("tp").add(pl.col("fn")).add(pl.col("fp")))
            .alias("threat_score"),
            pl.col("pp").truediv(pl.col("total")).alias("ppr"),
            pl.col("pn").truediv(pl.col("total")).alias("pnr"),
        )
        .with_columns(
            pl.lit(1).sub(pl.col("tpr")).alias("fnr"),
            pl.lit(1).sub(pl.col("false_omission_rate")).alias("npv"),
            pl.col("tpr").truediv(pl.col("fpr")).alias("plr"),
        )
        .with_columns(pl.col("fnr").truediv(pl.lit(1).sub(pl.col("fpr"))).alias("nlr"))
        .with_columns(
            pl.col("tpr")
            .mul(pl.col("fpr"))
            .sqrt()
            .sub(pl.col("fpr"))
            .truediv(pl.col("tpr").sub(pl.col("fpr")))
            .alias("prevalence_threshold"),
            pl.col("tpr")
            .mul(pl.lit(1).sub(pl.col("fpr")))
            .mul(pl.col("precision"))
            .mul(pl.col("npv"))
            .sqrt()
            .sub(
                pl.col("fnr")
                .mul(pl.col("fpr"))
                .mul(pl.col("false_omission_rate"))
                .mul(pl.col("fdr"))
                .sqrt()
            )
            .alias("mcc"),
            pl.col("tpr")
            .add(pl.lit(1).sub(pl.col("fpr")))
            .truediv(2)
            .alias("balanced_accuracy"),
            pl.col("plr").truediv(pl.col("nlr")).alias("dor"),
        )
        .drop("p", "n", "pp", "pn", "total")
        .pipe(_fill_infinite, None)
        .fill_nan(None)
    )

    return cast(PolarsFrameT, res)


def _map_to_thresholds(
    pf: PolarsFrame,
    thresholds: Optional[list[float]],
) -> pl.LazyFrame:
    """Map each requested threshold onto the curve row that applies to it.

    For target `t`, that is the row with the smallest curve threshold still `>= t`, or
    nulls when no such row exists. Returns the targets as `threshold` and the matched
    curve threshold as `_threshold_actual`, ascending.

    This is an as-of join. It was previously a cross join plus filter plus per-target
    `min`, which is O(n*m) and dominated every bootstrapped threshold metric -- the
    statistics themselves are near-free by comparison. It also carried a
    `validate="1:1"` that made duplicate targets a hard error.
    """
    if thresholds is None:
        return pf.lazy()

    lf = pf.lazy()
    targets = (
        thresholds
        if isinstance(thresholds, pl.Series)
        else pl.Series("threshold", thresholds)
    )

    # An integer score column paired with float thresholds (or the reverse) compared
    # fine under the old value-wise join. `join_asof` requires one dtype on both sides,
    # so reconcile before joining rather than raising at the user.
    curve_dtype = lf.collect_schema()["threshold"]
    if targets.dtype != curve_dtype:
        targets = targets.cast(pl.Float64)
        lf = lf.with_columns(pl.col("threshold").cast(pl.Float64))

    return (
        pl.LazyFrame({"threshold": targets})
        .sort("threshold")
        .join_asof(
            lf.rename({"threshold": "_threshold_actual"}).sort("_threshold_actual"),
            left_on="threshold",
            right_on="_threshold_actual",
            strategy="forward",
        )
    )


def confusion_matrix_at_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    thresholds: Optional[list[float]] = None,
    metrics: Iterable[ConfusionMatrixMetric] = DefaultConfusionMatrixMetrics,
    strategy: LoopStrategy = "auto",
    beta: float = 1.0,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
    lazy: bool = False,
) -> pl.DataFrame:
    r"""Computes the confusion matrix at each threshold. When the `strategy` is
    "cum_sum", computes

    ``` py
    for t in y_score:
        y_pred = y_score >= t
        confusion_matrix(y_true, y_pred)
    ```

    using fast DataFrame operations.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores
    thresholds : Optional[list[float]], optional
        The thresholds to compute `y_pred` at, i.e. y_score >= t. If None,
        uses every score present in `y_score`, by default None
    metrics : Iterable[ConfusionMatrixMetric], optional
        The metrics to compute, by default DefaultConfusionMatrixMetrics
    strategy : LoopStrategy, optional
        Computation method, by default "auto"
    beta : float, optional
        \( \beta \) to use in \( F_\beta \), by default 1
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame of `threshold`, `metric`, and `value`

    Added in version 0.1.0
    ----------------------
    """
    strategy = _set_loop_strategy(thresholds, strategy)

    if lazy and strategy == "loop":
        raise ValueError(
            "`lazy=True` requires `strategy='cum_sum'`; the loop strategy computes each "
            "threshold through the Rust kernel, which needs materialised data"
        )

    if strategy == "loop":
        # Nulls must be dropped jointly across all three inputs. This built the frame
        # from `y_true`/`y_score` alone and then passed the *original* full-length
        # `sample_weight` into each call, so a single null raised `ShapeError: height of
        # column 'sample_weight' (5) does not match height of column 'y_true' (4)` --
        # and, had the lengths happened to match, would have shifted every weight after
        # the null onto the wrong row.
        df = _y_true_y_score_to_df(y_true, y_score, sample_weight, data=data)
        weights = df["sample_weight"]

        def _cm(t):
            return (
                confusion_matrix(
                    df["y_true"],
                    df["y_score"].ge(t),
                    beta=beta,
                    sample_weight=weights,
                )
                .to_polars()
                .with_columns(pl.lit(t).alias("threshold"))
            )

        cms: list[pl.DataFrame] = _run_concurrent(
            _cm, _resolve_thresholds(thresholds, df, "y_score")
        )

        return pl.concat(cms, how="vertical").fill_nan(None)
    elif strategy == "cum_sum":
        result = (
            _cm_curve_frame(y_true, y_score, sample_weight, data=data)
            .pipe(_base_confusion_matrix_at_thresholds)
            .pipe(_full_confusion_matrix_from_base, beta=beta)
            .select("threshold", *metrics)
            # .unique("threshold")
            .pipe(_map_to_thresholds, thresholds)
            .drop("_threshold_actual", strict=False)
            .unpivot(index="threshold")
            .rename({"variable": "metric"})
        )

        return result if lazy else _collect(result)


def _ap_from_pr_curve(precision: pl.Expr, recall: pl.Expr) -> pl.Expr:
    return (
        recall.extend_constant(0.0, 1)
        .diff(null_behavior="drop")
        .mul(precision)
        .sum()
        .mul(-1)
    )


def average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    data=None,
) -> Optional[float]:
    """Computes Average Precision.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores
    sample_weight: Optional[ArrayLike], optional
        Sample weights, set to 1 if None

        !!! Version
            Added 0.2.0

    Returns
    -------
    Optional[float]
        Average Precision (AP), or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    curve = (
        _cm_curve_frame(y_true, y_score, sample_weight, data=data)
        .pipe(_base_confusion_matrix_at_thresholds)
        .pipe(_full_confusion_matrix_from_base)
        .select("threshold", "precision", "tpr")
        .drop_nulls()
        .sort("threshold")
        .select(
            _ap_from_pr_curve(pl.col("precision"), pl.col("tpr")).alias("ap"),
            pl.len().alias("points"),
        )
        .pipe(_collect)
    )

    # An empty curve sums to -0.0, which reads as a real score of zero. Nothing was
    # computed, so say so with a null rather than a number -- or a NaN, which here would
    # claim the arithmetic was undefined when there was no arithmetic at all.
    if curve["points"].item() == 0:
        return None

    return curve["ap"].item()


def capture_rate_at_quantiles(
    y_true: ArrayLike,
    y_score: ArrayLike,
    quantiles: int = 10,
    drop_intermediate: bool = True,
    raw_quantiles: bool = False,
    *,
    data=None,
    lazy: bool = False,
) -> pl.DataFrame:
    source = (
        _y_true_y_score_to_lf(data, y_true, y_score)
        if data is not None
        else _y_true_y_score_to_df(y_true=y_true, y_score=y_score).lazy()
    )
    lf = (
        source.with_columns(
            pl.col("y_score").qcut(quantiles, allow_duplicates=True).alias("quantile")
        )
        .group_by("quantile")
        .agg(
            pl.col("y_true").sum().alias("n_pos"),
            pl.col("y_true").count().alias("n"),
            pl.col("y_true").mean().alias("capture_rate"),
        )
    )

    if drop_intermediate:
        lf = lf.drop("n_pos", "n")

    if not raw_quantiles:
        lf = lf.sort("quantile").drop("quantile").with_row_index("quantile", offset=1)

    return lf if lazy else _collect(lf)


def r2(y_true: ArrayLike, y_score: ArrayLike, *, data=None) -> Optional[float]:
    r"""Computes R2 as

    \[
        1 - \frac{\sum{(y_i - \hat{y_i})^2}{}}{\sum{(y_{i} - \bar{y})^2}}
    \]

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    Optional[float]
        R2, or `None` if there is nothing to compute it from

    Added in version 0.1.0
    ----------------------
    """
    return _r2(_regression_to_df(y_true, y_score, data=data))
