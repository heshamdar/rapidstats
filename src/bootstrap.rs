use crate::distributions;
use polars::prelude::*;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

pub type ConfidenceInterval = (f64, f64, f64);

// The resulting bootstrap vectors are small vectors, usually around 500-10_000 in
// length, so let's just operate on these vectors directly instead of converting into
// ndarray or ChunkedArray
trait VecUtils {
    fn mean(&self) -> f64;
    fn std(&self) -> f64;
    fn drop_nans(&self) -> Vec<f64>;
    fn percentile(&self, q: f64) -> f64;
}

impl VecUtils for Vec<f64> {
    #[allow(clippy::manual_range_contains)]
    fn percentile(&self, q: f64) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }

        if q < 0.0 || q > 100.0 {
            panic!("Percentile must be between 0 and 100");
        }

        let mut sorted_data = self.clone();
        sorted_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        if q == 0.0 {
            return sorted_data[0];
        }
        if q == 100.0 {
            return sorted_data[sorted_data.len() - 1];
        }

        let rank = (q / 100.0) * (sorted_data.len() - 1) as f64;
        let lower_index = rank.floor() as usize;
        let upper_index = rank.ceil() as usize;

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let lower_value = sorted_data[lower_index];
            let upper_value = sorted_data[upper_index];
            let fraction = rank - lower_index as f64;

            lower_value + (upper_value - lower_value) * fraction
        }
    }

    fn drop_nans(&self) -> Vec<f64> {
        // copied is a no-op for f64
        self.iter().copied().filter(|x| !x.is_nan()).collect()
    }

    fn mean(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }

        self.iter().sum::<f64>() / self.len() as f64
    }

    fn std(&self) -> f64 {
        if self.len() < 2 {
            return f64::NAN;
        }
        let mean = self.mean();
        let variance =
            self.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (self.len() - 1) as f64;

        variance.sqrt()
    }
}

// fn repeat<T: Copy>(a: &[T], repeats: &[u64], capacity: usize) -> Vec<T> {
//     let mut res: Vec<T> = Vec::with_capacity(capacity);

//     for value in a.iter().copied() {
//         for _ in repeats {
//             res.push(value);
//         }
//     }

//     res
// }

/// Worker stack size for the bootstrap pool.
///
/// Each bootstrap task calls into polars, which parallelises internally on its own rayon
/// pool. That nesting is executed on the calling worker's stack, and rayon's 2 MiB
/// default is not enough: past a few thousand iterations it overflows and takes the
/// interpreter down with a SIGSEGV. This reproduced on v0.4.1 at the *default* 1000
/// iterations for `Bootstrap(sampling_method="poisson").roc_auc(...)`.
///
/// 16 MiB is what the crash was empirically shown to need; stacks are virtual
/// allocations, so the headroom costs address space rather than resident memory.
const BOOTSTRAP_STACK_SIZE: usize = 16 * 1024 * 1024;

fn create_rayon_pool(n_jobs: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .stack_size(BOOTSTRAP_STACK_SIZE)
        .build()
        .unwrap()
}

/// A dedicated pool for bootstrap work, built once.
///
/// Bootstrapping deliberately does not run on rayon's global pool: that is the pool
/// polars nests onto, and its workers carry the too-small default stack. Running the
/// outer loop on our own pool lets us set the stack size, and keeps a long bootstrap
/// from monopolising the pool polars uses for everything else.
fn bootstrap_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

    POOL.get_or_init(|| {
        let n_jobs = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        create_rayon_pool(n_jobs)
    })
}

fn sample(df: DataFrame, df_height: usize, seed: Option<u64>) -> DataFrame {
    df.sample_n_literal(df_height, true, false, seed).unwrap()
}

/// Apply resample multiplicities as weights instead of materialising the resample.
///
/// Drawing row `i` exactly `c[i]` times and computing a weight-aware metric is
/// arithmetically identical to computing it once with `sample_weight * c`. Using that
/// identity avoids building a new frame per iteration, and -- because row order is
/// untouched -- lets a pre-sorted frame be sorted once for the whole bootstrap rather
/// than once per iteration.
///
/// Frames reaching here always carry a `sample_weight` column: the Python layer adds one
/// (defaulting to 1.0) in `_y_true_y_score_to_df` and friends. Frames without one -- the
/// regression metrics and `mean` -- are handled by the caller, which falls back to
/// materialising.
fn weight_sample(df: DataFrame, df_height: usize, seed: Option<u64>, poisson: bool) -> DataFrame {
    let counts: Vec<f64> = if poisson {
        // Qualified: the `poisson` parameter shadows the imported function.
        distributions::poisson(1.0, df_height, seed)
            .into_iter()
            .map(|c| c as f64)
            .collect()
    } else {
        multinomial_counts(df_height, seed)
    };

    // Deliberately eager ChunkedArray arithmetic rather than `df.lazy()...collect()`.
    // This runs inside a rayon `par_iter`, and re-entering the polars query engine from
    // a rayon worker lets the engine's own pool steal work recursively; deep enough
    // nesting overflows the worker stack and segfaults the interpreter. A plain
    // elementwise multiply avoids the engine entirely -- and is faster besides.
    let counts = Float64Chunked::from_vec("__rapidstats_resample_count__".into(), counts);
    let weighted = df
        .column("sample_weight")
        .expect("checked by `resample_fn`")
        .f64()
        .expect("`sample_weight` is always Float64")
        * &counts;

    let mut out = df;
    out.with_column(weighted.into_series().with_name("sample_weight".into()))
        .expect("replacing a column with one of equal length");

    out
}

/// Multiplicities from sampling `n` rows with replacement, i.e. a Multinomial(n, 1/n)
/// draw, produced by counting `n` uniform draws over the row indices.
fn multinomial_counts(n: usize, seed: Option<u64>) -> Vec<f64> {
    use rand::Rng;
    use rand::SeedableRng;

    let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut counts = vec![0.0f64; n];
    for _ in 0..n {
        counts[rng.gen_range(0..n)] += 1.0;
    }

    counts
}

fn poisson_sample(df: DataFrame, df_height: usize, seed: Option<u64>) -> DataFrame {
    // Was `with_row_index` + `repeat_by` + `explode` through the lazy engine. That
    // re-entered polars' query engine from inside a rayon worker, which could recurse
    // deeply enough to overflow the worker stack and segfault the interpreter -- at the
    // default 1000 iterations, on the default `roc_auc` path. Expanding the counts to
    // gather indices and taking them is eager, engine-free, and faster.
    let indices = distributions::poisson_repeat_indices(1.0, df_height, seed);
    let idx = IdxCa::from_vec("".into(), indices);

    df.take(&idx).expect("indices are all < df.height()")
}

/// Pick the resampling strategy.
///
/// `weights` is only honoured when the frame actually has a `sample_weight` column;
/// otherwise the identity does not apply and we fall back to materialising, so callers
/// never silently get an unresampled statistic.
fn resample_fn(
    df: &DataFrame,
    poisson: bool,
    weights: bool,
) -> Box<dyn Fn(DataFrame, usize, Option<u64>) -> DataFrame + Send + Sync> {
    if weights && df.column("sample_weight").is_ok() {
        Box::new(move |d, h, s| weight_sample(d, h, s, poisson))
    } else if poisson {
        Box::new(poisson_sample)
    } else {
        Box::new(sample)
    }
}

fn bootstrap_core<T: Send + Sync, F>(
    df: DataFrame,
    iterations: u64,
    seed: Option<u64>,
    func: F,
    chunksize: Option<usize>,
    poisson: bool,
    weights: bool,
) -> Vec<T>
where
    F: Fn(DataFrame) -> T + Send + Sync,
{
    let df_height = df.height();

    let seeds: Vec<u64> = (0..iterations).collect();

    let sample_func = resample_fn(&df, poisson, weights);

    let res: Vec<T> = if chunksize.is_none() {
        seeds
            .par_iter()
            .map(|i| {
                func(sample_func(
                    df.clone(),
                    df_height,
                    seed.map(|seed| seed + i),
                ))
            })
            .collect()
    } else {
        let chunksize = chunksize.unwrap();
        seeds
            .par_chunks(chunksize)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|i| {
                        func(sample_func(
                            df.clone(),
                            df_height,
                            seed.map(|seed| seed + i),
                        ))
                    })
                    .collect::<Vec<T>>()
            })
            .collect()
    };

    res
}

pub fn run_bootstrap<T: Send + Sync, F>(
    df: DataFrame,
    iterations: u64,
    seed: Option<u64>,
    func: F,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
    poisson: bool,
    weights: bool,
) -> Vec<T>
where
    F: Fn(DataFrame) -> T + Send + Sync,
{
    let df_height = df.height();

    let bootstrap_stats: Vec<T> = if n_jobs == Some(1) {
        let sample_func = resample_fn(&df, poisson, weights);
        (0..iterations)
            .map(|i| {
                func(sample_func(
                    df.clone(),
                    df_height,
                    seed.map(|seed| seed + i),
                ))
            })
            .collect()
    } else if n_jobs.is_none() {
        bootstrap_pool()
            .install(|| bootstrap_core(df, iterations, seed, func, chunksize, poisson, weights))
    } else {
        create_rayon_pool(n_jobs.unwrap())
            .install(|| bootstrap_core(df, iterations, seed, func, chunksize, poisson, weights))
    };

    bootstrap_stats
}

pub fn run_jacknife<T: Send + Sync, F>(df: DataFrame, func: F) -> Vec<T>
where
    F: Fn(DataFrame) -> T + Send + Sync,
{
    let df_height = df.height();
    let index = ChunkedArray::new("index".into(), 0..df_height as u64);

    // Same reasoning as `run_bootstrap`: this nests polars work inside rayon tasks, and
    // the jackknife runs one task per row, so it hits the deep end sooner than the
    // bootstrap does.
    bootstrap_pool().install(|| {
        (0..df_height)
            .into_par_iter()
            .map(|i| func(df.filter(&index.not_equal(i)).unwrap()))
            .collect()
    })
}

pub fn standard_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let runs = bootstrap_stats.drop_nans();
    let stderr = runs.std();
    let z = distributions::norm_ppf(1.0 - alpha);
    let x = z * stderr;

    // Centred on the point estimate, per the documented interval
    // `theta_hat +/- z * sigma_hat`. This used the bootstrap *mean* as the centre while
    // still reporting `original_stat` as the point, so on a skewed bootstrap
    // distribution the reported point sat off-centre in its own interval -- and could
    // fall outside it.
    (original_stat - x, original_stat, original_stat + x)
}

pub fn percentile_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let runs = bootstrap_stats.drop_nans();

    (
        runs.percentile(alpha * 100.0),
        original_stat,
        runs.percentile((1.0 - alpha) * 100.0),
    )
}

pub fn basic_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let interval = percentile_interval(original_stat, bootstrap_stats, alpha);
    let lower = interval.0;
    let upper = interval.2;

    let x = 2.0 * original_stat;

    (x - upper, original_stat, x - lower)
}

fn percentile_of_score(arr: &[f64], score: f64) -> f64 {
    let a1 = arr.iter().filter(|x| x < &&score).count() as f64;
    let a2 = arr.iter().filter(|x| x <= &&score).count() as f64;

    (a1 + a2) / (2.0 * arr.len() as f64)
}

pub fn bca_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    jacknife_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let bootstrap_stats = bootstrap_stats.drop_nans();
    let jacknife_stats = jacknife_stats.drop_nans();
    let z1 = distributions::norm_ppf(alpha);
    let z2 = -z1;

    let bias_correction_factor =
        distributions::norm_ppf(percentile_of_score(&bootstrap_stats, original_stat));

    let jacknife_mean = jacknife_stats.mean();
    let n = jacknife_stats.len() as f64;
    let n1 = n - 1.0;
    let diff: Vec<f64> = jacknife_stats
        .iter()
        .map(|x| n1 * (jacknife_mean - x))
        .collect();
    let numerator = diff.iter().map(|x| x.powi(3)).sum::<f64>() / n.powi(3);
    let denominator = diff.iter().map(|x| x.powi(2)).sum::<f64>() / n.powi(2);
    let acceleration_factor = numerator / (6.0 * denominator.powf(1.5));

    let lower_p = distributions::norm_cdf(
        bias_correction_factor
            + (bias_correction_factor + z1)
                / (1.0 - acceleration_factor * (bias_correction_factor + z1)),
    );
    let upper_p = distributions::norm_cdf(
        bias_correction_factor
            + (bias_correction_factor + z2)
                / (1.0 - acceleration_factor * (bias_correction_factor + z2)),
    );

    (
        bootstrap_stats.percentile(lower_p * 100.0),
        original_stat,
        bootstrap_stats.percentile(upper_p * 100.0),
    )
}
