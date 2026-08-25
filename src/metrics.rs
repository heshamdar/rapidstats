use core::f64;

use crate::bootstrap;
use polars::prelude::*;

pub type ConfusionMatrixArray = [f64; 27];

pub fn base_confusion_matrix(df: DataFrame) -> DataFrame {
    df.lazy()
        .select([
            (lit(2) * col("y_true") + col("y_pred")).alias("y"),
            col("sample_weight"),
        ])
        .collect()
        .unwrap()
}

/// Weighted counts of the four confusion-matrix cells, indexed by `2*y_true + y_pred`.
fn base_counts(base_cm: &DataFrame) -> [f64; 4] {
    let mut s = [0.0; 4];
    for (i, w) in base_cm["y"]
        .cast(&DataType::UInt64)
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .zip(base_cm["sample_weight"].f64().unwrap().into_no_null_iter())
    {
        s[i as usize] += w;
    }

    s
}

/// Leave-one-out confusion matrices, in O(n) rather than O(n^2).
///
/// The confusion matrix is a weighted bincount, so dropping row `i` just removes its
/// weight from its own cell -- the other three are untouched. Computing the totals once
/// and subtracting therefore costs O(1) per replicate, where the generic
/// `bootstrap::run_jacknife` re-filters and re-scans the whole frame for every row.
///
/// This is what makes BCa usable at realistic sizes: the jackknife dominated it, growing
/// 4.5x per doubling of n.
pub fn jacknife_confusion_matrix(base_cm: &DataFrame, beta: f64) -> Vec<ConfusionMatrixArray> {
    let totals = base_counts(base_cm);

    base_cm["y"]
        .cast(&DataType::UInt64)
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .zip(base_cm["sample_weight"].f64().unwrap().into_no_null_iter())
        .map(|(bin, weight)| {
            let mut s = totals;
            s[bin as usize] -= weight;

            confusion_matrix_from_counts(s, beta)
        })
        .collect()
}

pub fn confusion_matrix(base_cm: DataFrame, beta: f64) -> ConfusionMatrixArray {
    confusion_matrix_from_counts(base_counts(&base_cm), beta)
}

fn confusion_matrix_from_counts(s: [f64; 4], beta: f64) -> ConfusionMatrixArray {
    let tn = s[0];
    let fp = s[1];
    let fn_ = s[2];
    let tp = s[3];

    let p = tp + fn_;
    let n = fp + tn;
    let total = p + n;
    let tpr = tp / p;
    let fnr = 1.0 - tpr;
    let fpr = fp / n;
    let tnr = 1.0 - fpr;
    let precision = tp / (tp + fp);
    let false_omission_rate = fn_ / (fn_ + tn);
    let plr = tpr / fpr;
    let nlr = fnr / tnr;
    let npv = 1.0 - false_omission_rate;
    let fdr = 1.0 - precision;
    let prevalence = p / (p + n);
    let informedness = tpr + tnr - 1.0;
    let prevalence_threshold = ((tpr * fpr).sqrt() - fpr) / (tpr - fpr);
    let markedness = precision - false_omission_rate;
    let dor = plr / nlr;
    let balanced_accuracy = (tpr + tnr) / 2.0;
    let fbeta = ((1.0 + beta.powi(2)) * precision * tpr) / ((beta.powi(2) * precision) + tpr);
    let folkes_mallows_index = (precision * tpr).sqrt();
    let mcc = (tpr * tnr * precision * npv).sqrt() - (fnr * fpr * false_omission_rate * fdr).sqrt();
    let acc = (tp + tn) / total;
    let threat_score = tp / (tp + fn_ + fp);
    let ppr = (tp + fp) / total;
    let pnr = (tn + fn_) / total;

    [
        tn,
        fp,
        fn_,
        tp,
        tpr,
        fpr,
        fnr,
        tnr,
        prevalence,
        prevalence_threshold,
        informedness,
        precision,
        false_omission_rate,
        plr,
        nlr,
        acc,
        balanced_accuracy,
        fbeta,
        folkes_mallows_index,
        mcc,
        threat_score,
        markedness,
        fdr,
        npv,
        dor,
        ppr,
        pnr,
    ]
    .map(|x| if x.is_infinite() { f64::NAN } else { x })
}

fn transpose_confusion_matrix_results(results: Vec<[f64; 27]>) -> [Vec<f64>; 27] {
    let mut transposed: [Vec<f64>; 27] = Default::default();
    for arr in results {
        for (i, v) in arr.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }

    transposed
}

pub fn bootstrap_confusion_matrix(
    df: DataFrame,
    beta: f64,
    iterations: u64,
    alpha: f64,
    method: &str,
    seed: Option<u64>,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
    poisson: bool,
    weights: bool,
) -> Vec<bootstrap::ConfidenceInterval> {
    let base_cm = base_confusion_matrix(df);

    let bootstrap_stats = bootstrap::run_bootstrap(
        base_cm.clone(),
        iterations,
        seed,
        |x| confusion_matrix(x, beta),
        n_jobs,
        chunksize,
        poisson,
        weights,
    );
    let bs_transposed = transpose_confusion_matrix_results(bootstrap_stats);

    let original_stat = confusion_matrix(base_cm.clone(), beta);

    // The confusion matrix is a weighted bincount: it always has an answer, even from no
    // rows (every cell zero, every rate 0/0). So its replicates are never absent, only
    // undefined -- wrap them as present and let the interval code drop the NaNs.
    let bs_transposed: Vec<Vec<Option<f64>>> = bs_transposed
        .into_iter()
        .map(|bs| bs.into_iter().map(Some).collect())
        .collect();

    if method == "standard" {
        bs_transposed
            .into_iter()
            .zip(original_stat)
            .map(|(bs, o)| bootstrap::standard_interval(Some(o), bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "percentile" {
        bs_transposed
            .into_iter()
            .zip(original_stat)
            .map(|(bs, o)| bootstrap::percentile_interval(Some(o), bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "basic" {
        original_stat
            .into_iter()
            .zip(bs_transposed)
            .map(|(original_stat, bs)| bootstrap::basic_interval(Some(original_stat), bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "BCa" {
        let jacknife_stats = jacknife_confusion_matrix(&base_cm, beta);
        let js_transposed: Vec<Vec<Option<f64>>> =
            transpose_confusion_matrix_results(jacknife_stats)
                .into_iter()
                .map(|js| js.into_iter().map(Some).collect())
                .collect();

        original_stat
            .into_iter()
            .zip(bs_transposed)
            .zip(js_transposed)
            .map(|((original_stat, bs), js)| {
                bootstrap::bca_interval(Some(original_stat), bs, js, alpha)
            })
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else {
        panic!("Invalid method");
    }
}

/// Absent is `None`; undefined is `NaN`.
///
/// A metric with no rows to read has no answer at all, and that is a null -- polars
/// spells it `None` here and Python sees `None`. A metric with rows whose formula
/// divides by zero (`roc_auc` on a single class, `tpr` with no positives) has a real,
/// undefined answer, and that is `NaN`. Conflating the two hands the caller a float that
/// looks like a computed result when nothing was computed.
///
/// The bootstrap relies on the first half: a resample can come out empty, and those
/// iterations are skipped rather than poisoning the interval.
pub fn roc_auc_sorted(df: DataFrame) -> Option<f64> {
    if df.height() == 0 {
        return None;
    }

    let y_true = df["y_true"].f64().unwrap();
    let y_score = df["y_score"].f64().unwrap();
    let sample_weight = df["sample_weight"].f64().unwrap();

    // Rechunk to single chunk so .cont_slice() succeeds,
    // avoiding per-element chunk lookups in the hot loop
    let y_true = y_true.rechunk();
    let y_score = y_score.rechunk();
    let sample_weight = sample_weight.rechunk();

    let y_true = y_true.cont_slice().unwrap();
    let y_score = y_score.cont_slice().unwrap();
    let sample_weight = sample_weight.cont_slice().unwrap();

    let mut auc = 0.0f64;
    let mut n_false = 0.0f64;
    let mut n_true = 0.0f64;
    let mut i = 0usize;

    while i < y_true.len() {
        let score_i = y_score[i];
        let mut j = i + 1;
        while j < y_true.len() && y_score[j] == score_i {
            j += 1;
        }

        let mut group_pos = 0.0f64;
        let mut group_neg = 0.0f64;
        for k in i..j {
            let w = sample_weight[k];
            if y_true[k] == 1.0 {
                group_pos += w;
            } else {
                group_neg += w;
            }
        }

        auc += group_pos * (n_false + 0.5 * group_neg);
        n_false += group_neg;
        n_true += group_pos;
        i = j;
    }

    // Single-class input leaves one of these at zero: 0/0, genuinely undefined, NaN.
    Some(auc / (n_false * n_true))
}

pub fn roc_auc(df: DataFrame) -> Option<f64> {
    let df = df.sort(["y_score"], Default::default()).unwrap();

    roc_auc_sorted(df)
}

// Max KS code taken largely from https://github.com/abstractqqq/polars_ds_extension/blob/main/src/stats/ks.rs

fn binary_search_right<T: PartialOrd>(arr: &[T], t: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + ((right - left) >> 1);
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid,
                _ => left = mid + 1,
            }
        } else {
            return None;
        }
    }
    Some(left)
}

fn ks_2samp(v1: &[f64], v2: &[f64]) -> f64 {
    // https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L8644-L8875

    // v1 and v2 must be sorted
    let n1: f64 = v1.len() as f64;
    let n2: f64 = v2.len() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return f64::NAN;
    }

    let stats = v1
        .iter()
        .chain(v2.iter())
        .map(|x| {
            (
                (binary_search_right(v1, x).unwrap() as f64) / n1,
                (binary_search_right(v2, x).unwrap() as f64) / n2,
            )
        })
        .fold(f64::MIN, |acc, (x, y)| acc.max((x - y).abs()));

    if stats.is_infinite() {
        f64::NAN
    } else {
        stats
    }
}

pub fn max_ks(df: DataFrame) -> Option<f64> {
    if df.height() == 0 {
        return None;
    }

    let y_score = df["y_score"].f64().unwrap();
    let y_true = df["y_true"].bool().unwrap();

    // A single-class input leaves one sample empty: the statistic is undefined rather
    // than absent, so `ks_2samp` returns NaN and that is what comes back.
    Some(ks_2samp(
        y_score
            .filter(y_true)
            .unwrap()
            .sort(false)
            .cont_slice()
            .unwrap(),
        y_score
            .filter(&!y_true)
            .unwrap()
            .sort(false)
            .cont_slice()
            .unwrap(),
    ))
}

pub fn brier_loss(df: DataFrame) -> Option<f64> {
    df.lazy()
        .with_column((col("y_true") - col("y_score")).pow(2).alias("x"))
        .collect()
        .unwrap()
        .column("x")
        .unwrap()
        .f64()
        .unwrap()
        .mean()
}

pub fn mean(df: DataFrame) -> Option<f64> {
    df["y"].as_series().unwrap().mean()
}

pub fn adverse_impact_ratio(df: DataFrame) -> Option<f64> {
    if df.height() == 0 {
        return None;
    }

    let is_protected = df["protected"].bool().unwrap();
    let is_control = df["control"].bool().unwrap();
    let y_pred = df["y_pred"].f64().unwrap();
    let sample_weight = df["sample_weight"].f64().unwrap();
    let protected = y_pred.filter(is_protected).unwrap();
    let protected_sample_weight = sample_weight.filter(is_protected).unwrap();
    let control = y_pred.filter(is_control).unwrap();
    let control_sample_weight = sample_weight.filter(is_control).unwrap();

    // `sum()` over an empty ChunkedArray is None, and dividing those unwraps was the
    // panic path for a fully-filtered input. There are rows here -- just none in this
    // group -- so the rate is undefined rather than absent.
    let rate = |weighted: Option<f64>, weight_total: Option<f64>| match (
        weighted,
        weight_total,
    ) {
        (Some(weighted), Some(total)) => weighted / total,
        _ => f64::NAN,
    };

    let protected_approval_rate = rate(
        (&protected * &protected_sample_weight).sum(),
        protected_sample_weight.sum(),
    );
    let control_approval_rate = rate(
        (&control * &control_sample_weight).sum(),
        control_sample_weight.sum(),
    );

    let res = protected_approval_rate / control_approval_rate;

    Some(if res.is_infinite() { f64::NAN } else { res })
}

pub fn mean_squared_error(df: DataFrame) -> Option<f64> {
    let y_true = df["y_true"].f64().unwrap();
    let y_score = df["y_score"].f64().unwrap();

    let x = &(y_true - y_score);

    // `mean()` of nothing is None, not a panic and not NaN: there is no error to average.
    (x * x).mean()
}

pub fn root_mean_squared_error(df: DataFrame) -> Option<f64> {
    mean_squared_error(df).map(f64::sqrt)
}

pub fn r2(df: DataFrame) -> Option<f64> {
    let y_true = df["y_true"].f64().unwrap();
    let y_score = df["y_score"].f64().unwrap();

    // No rows, no mean, no R2 -- absent rather than undefined.
    let mean = y_true.mean()?;

    let residual = &(y_true - y_score);
    let squared_residual = residual * residual;
    let error = &(y_true - mean);
    let squared_error = error * error;

    // Zero variance in `y_true` makes `total_ss` zero: 1 - 0/0 is undefined, so NaN.
    match (squared_residual.sum(), squared_error.sum()) {
        (Some(residual_ss), Some(total_ss)) => Some(1.0 - (residual_ss / total_ss)),
        _ => None,
    }
}
