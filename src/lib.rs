#![allow(clippy::too_many_arguments)]

use bootstrap::ConfidenceInterval;
use paste::paste;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

mod bootstrap;
mod distributions;
mod general;
mod metrics;
mod string;
mod viz;

use pyo3_polars::PolarsAllocator;
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

macro_rules! generate_functions {
    ($func_name:ident, $metric_func:path) => {
        #[pyfunction]
        fn $func_name(py: Python<'_>, df: PyDataFrame) -> PyResult<Option<f64>> {
            let df: DataFrame = df.into();

            Ok(py.allow_threads(move || $metric_func(df)))
        }

        paste! {
            #[pyfunction]
            #[pyo3(signature = (df, iterations, alpha, method, seed = None, n_jobs = None, chunksize = None, poisson = true, weights = true))]
            fn [<_bootstrap $func_name>] (
                py: Python<'_>,
                df: PyDataFrame,
                iterations: u64,
                alpha: f64,
                method: &str,
                seed: Option<u64>,
                n_jobs: Option<usize>,
                chunksize: Option<usize>,
                poisson: bool,
                weights: bool,
            ) -> PyResult<bootstrap::ConfidenceInterval> {
                let df: DataFrame = df.into();
                // `method` borrows Python-owned memory, which cannot cross into
                // `allow_threads`. Own it first.
                let method = method.to_owned();

                py.allow_threads(move || {
                    let original_stat = $metric_func(df.clone());
                    let bootstrap_stats =
                        bootstrap::run_bootstrap(df.clone(), iterations, seed, $metric_func, n_jobs, chunksize, poisson, weights);
                    if method == "standard" {
                        Ok(bootstrap::standard_interval(original_stat, bootstrap_stats, alpha))
                    }
                    else if method == "percentile" {
                        Ok(bootstrap::percentile_interval(original_stat, bootstrap_stats, alpha))
                    } else if method == "basic" {
                        Ok(bootstrap::basic_interval(original_stat, bootstrap_stats, alpha))
                    } else if method == "BCa" {
                        let jacknife_stats = bootstrap::run_jacknife(df, $metric_func);
                        Ok(bootstrap::bca_interval(
                            original_stat,
                            bootstrap_stats,
                            jacknife_stats,
                            alpha,
                        ))
                    } else {
                        Err(PyValueError::new_err(format!(
                            "Invalid confidence interval method `{}`, only `percentile`, `basic`, and `BCa` are supported",
                            method
                        )))
                    }
                })
            }
        }
    };
}

#[pyfunction]
fn _confusion_matrix(
    py: Python<'_>,
    df: PyDataFrame,
    beta: f64,
) -> PyResult<metrics::ConfusionMatrixArray> {
    let df: DataFrame = df.into();

    Ok(py.allow_threads(move || {
        let base_cm = metrics::base_confusion_matrix(df);

        metrics::confusion_matrix(base_cm, beta)
    }))
}

#[pyfunction]
#[pyo3(signature = (df, beta, iterations, alpha, method, seed = None, n_jobs = None, chunksize = None, poisson = true, weights = true))]
fn _bootstrap_confusion_matrix(
    py: Python<'_>,
    df: PyDataFrame,
    beta: f64,
    iterations: u64,
    alpha: f64,
    method: &str,
    seed: Option<u64>,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
    poisson: bool,
    weights: bool,
) -> PyResult<Vec<bootstrap::ConfidenceInterval>> {
    let df: DataFrame = df.into();
    let method = method.to_owned();

    Ok(py.allow_threads(move || {
        metrics::bootstrap_confusion_matrix(
            df, beta, iterations, alpha, &method, seed, n_jobs, chunksize, poisson, weights,
        )
    }))
}

generate_functions!(_roc_auc, metrics::roc_auc);
generate_functions!(_roc_auc_sorted, metrics::roc_auc_sorted);
generate_functions!(_max_ks, metrics::max_ks);
generate_functions!(_brier_loss, metrics::brier_loss);
generate_functions!(_mean, metrics::mean);
generate_functions!(_adverse_impact_ratio, metrics::adverse_impact_ratio);
generate_functions!(_mean_squared_error, metrics::mean_squared_error);
generate_functions!(_root_mean_squared_error, metrics::root_mean_squared_error);
generate_functions!(_r2, metrics::r2);

#[pyfunction]
fn _standard_interval(
    py: Python<'_>,
    original_stat: Option<f64>,
    bootstrap_stats: Vec<Option<f64>>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(py.allow_threads(move || {
        bootstrap::standard_interval(original_stat, bootstrap_stats, alpha)
    }))
}

#[pyfunction]
fn _percentile_interval(
    py: Python<'_>,
    original_stat: Option<f64>,
    bootstrap_stats: Vec<Option<f64>>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(py.allow_threads(move || {
        bootstrap::percentile_interval(original_stat, bootstrap_stats, alpha)
    }))
}

#[pyfunction]
fn _basic_interval(
    py: Python<'_>,
    original_stat: Option<f64>,
    bootstrap_stats: Vec<Option<f64>>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(py.allow_threads(move || {
        bootstrap::basic_interval(original_stat, bootstrap_stats, alpha)
    }))
}

#[pyfunction]
fn _bca_interval(
    py: Python<'_>,
    original_stat: Option<f64>,
    bootstrap_stats: Vec<Option<f64>>,
    jacknife_stats: Vec<Option<f64>>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(py.allow_threads(move || {
        bootstrap::bca_interval(original_stat, bootstrap_stats, jacknife_stats, alpha)
    }))
}

#[pyfunction]
fn _norm_ppf(q: f64) -> PyResult<f64> {
    Ok(distributions::norm_ppf(q))
}

#[pyfunction]
fn _norm_cdf(x: f64) -> PyResult<f64> {
    Ok(distributions::norm_cdf(x))
}

#[pyfunction]
fn _poisson(py: Python<'_>, lam: f64, size: usize, seed: Option<u64>) -> PyResult<Vec<u64>> {
    Ok(py.allow_threads(move || distributions::poisson(lam, size, seed)))
}

#[pyfunction]
#[pyo3(signature = (lam, size, seed = None))]
fn _poisson_repeat_indices(
    py: Python<'_>,
    lam: f64,
    size: usize,
    seed: Option<u64>,
) -> PyResult<Vec<u32>> {
    Ok(py.allow_threads(move || distributions::poisson_repeat_indices(lam, size, seed)))
}

#[pyfunction]
fn _trapezoidal_auc(py: Python<'_>, df: PyDataFrame) -> PyResult<f64> {
    let df: DataFrame = df.into();

    Ok(py.allow_threads(move || {
        general::trapezoidal_auc(
            df["x"].f64().unwrap().cont_slice().unwrap(),
            df["y"].f64().unwrap().cont_slice().unwrap(),
        )
    }))
}

#[pyfunction]
fn _rectangular_auc(py: Python<'_>, df: PyDataFrame) -> PyResult<f64> {
    let df: DataFrame = df.into();

    Ok(py.allow_threads(move || {
        general::rectangular_auc(
            df["x"].f64().unwrap().cont_slice().unwrap(),
            df["y"].f64().unwrap().cont_slice().unwrap(),
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (df, x, y, min_distance, always_keep, order))]
fn _thin_points_greedy(
    py: Python<'_>,
    df: PyDataFrame,
    x: &str,
    y: &str,
    min_distance: f64,
    always_keep: &str,
    order: Option<&str>,
) -> PyResult<Vec<bool>> {
    let df: DataFrame = df.into();
    // Column names borrow Python-owned memory; own them before releasing the GIL.
    let (x, y, always_keep) = (x.to_owned(), y.to_owned(), always_keep.to_owned());
    let order = order.map(|o| o.to_owned());

    Ok(py.allow_threads(move || {
        viz::thin_points_greedy(
            df,
            &x,
            &y,
            min_distance,
            &always_keep,
            order.as_deref(),
        )
    }))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_roc_auc_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_roc_auc_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(_max_ks, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_max_ks, m)?)?;
    m.add_function(wrap_pyfunction!(_brier_loss, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_brier_loss, m)?)?;
    m.add_function(wrap_pyfunction!(_mean, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_mean, m)?)?;
    m.add_function(wrap_pyfunction!(_adverse_impact_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_adverse_impact_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_root_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_root_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_r2, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_r2, m)?)?;
    m.add_function(wrap_pyfunction!(_standard_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_basic_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_bca_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_norm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(_norm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(_poisson_repeat_indices, m)?)?;
    m.add_function(wrap_pyfunction!(_rectangular_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_trapezoidal_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_thin_points_greedy, m)?)?;

    Ok(())
}
