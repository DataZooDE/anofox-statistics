//! C FFI for regularisation paths and CV curves (#154), quantile-regression
//! inference and the regression-quantile process (#157), coefficient
//! covariances, predictions with standard errors and partial residuals
//! (#158). Every result struct is mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::{
    coefficient_vcov, cv_reg_path_strategy, fit_reg_path, lars_path, nested_cv_reg_path,
    partial_residuals_binomial, partial_residuals_gamma, partial_residuals_logistic,
    partial_residuals_negbinomial, partial_residuals_ols, partial_residuals_poisson,
    partial_residuals_tweedie, partial_residuals_wls, predict_se_glm, predict_se_linear,
    quantile_process, CvSpec, CvStrategy, PartialResidualsResult, PredictInterval, QuantileSeType,
    RegPathOptions, RegPathResult,
};
use std::ffi::CStr;

// =============================================================================
// Result structs
// =============================================================================

/// Options of `anofox_reg_path` / `anofox_cv_reg_path`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RegPathOptionsFFI {
    /// Elastic-net mixing in [0, 1] (0 = ridge, 1 = lasso).
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub standardize: bool,
    /// Penalties (glmnet scale); NULL / 0 for the default grid.
    pub lambdas: *const f64,
    pub lambdas_len: usize,
    /// Size of the default grid.
    pub n_lambda: usize,
    /// NaN for glmnet's default.
    pub lambda_min_ratio: f64,
}

/// A coefficient path: `n_lambda` entries, coefficients row-major per lambda.
#[repr(C)]
#[derive(Debug)]
pub struct RegPathResultFFI {
    pub lambdas: *mut f64,
    pub intercepts: *mut f64,
    pub coefficients: *mut f64,
    pub df: *mut f64,
    pub n_lambda: usize,
    pub n_features: usize,
    pub n_observations: usize,
}

impl Default for RegPathResultFFI {
    fn default() -> Self {
        Self {
            lambdas: std::ptr::null_mut(),
            intercepts: std::ptr::null_mut(),
            coefficients: std::ptr::null_mut(),
            df: std::ptr::null_mut(),
            n_lambda: 0,
            n_features: 0,
            n_observations: 0,
        }
    }
}

/// A k-fold CV curve over `n_lambda` penalties.
#[repr(C)]
#[derive(Debug)]
pub struct CvPathResultFFI {
    pub lambdas: *mut f64,
    pub cv_mean: *mut f64,
    pub cv_se: *mut f64,
    pub df: *mut f64,
    pub n_lambda: usize,
    pub index_min: usize,
    pub index_1se: usize,
    pub n_folds: usize,
    pub n_observations: usize,
}

impl Default for CvPathResultFFI {
    fn default() -> Self {
        Self {
            lambdas: std::ptr::null_mut(),
            cv_mean: std::ptr::null_mut(),
            cv_se: std::ptr::null_mut(),
            df: std::ptr::null_mut(),
            n_lambda: 0,
            index_min: 0,
            index_1se: 0,
            n_folds: 0,
            n_observations: 0,
        }
    }
}

/// CV strategy of `anofox_cv_reg_path` / `anofox_nested_cv_reg_path` (#204).
pub const CV_STRATEGY_KFOLD: i32 = 0;
pub const CV_STRATEGY_FOLD_IDS: i32 = 1;
pub const CV_STRATEGY_GROUP: i32 = 2;
pub const CV_STRATEGY_LOGO: i32 = 3;
pub const CV_STRATEGY_STRATIFIED: i32 = 4;
pub const CV_STRATEGY_TIME_SERIES: i32 = 5;

/// How the CV splits are built (mirrors `CvSpec`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CvSpecFFI {
    pub strategy: i32,
    pub n_folds: usize,
    pub seed: u64,
    pub repeats: usize,
    pub test_size: usize,
    pub gap: usize,
    pub max_train_size: usize,
    pub n_bins: usize,
    pub mae: bool,
    pub inner_folds: usize,
    pub one_se: bool,
}

/// Nested CV: one entry per outer split, plus the mean and its standard error.
#[repr(C)]
#[derive(Debug)]
pub struct NestedCvResultFFI {
    pub lambda: *mut f64,
    pub test_error: *mut f64,
    pub inner_cv_mean: *mut f64,
    pub inner_cv_se: *mut f64,
    pub n_train: *mut f64,
    pub n_test: *mut f64,
    pub n_inner_folds: *mut f64,
    pub n_folds: usize,
    pub mean: f64,
    pub se: f64,
    pub n_observations: usize,
}

impl Default for NestedCvResultFFI {
    fn default() -> Self {
        Self {
            lambda: std::ptr::null_mut(),
            test_error: std::ptr::null_mut(),
            inner_cv_mean: std::ptr::null_mut(),
            inner_cv_se: std::ptr::null_mut(),
            n_train: std::ptr::null_mut(),
            n_test: std::ptr::null_mut(),
            n_inner_folds: std::ptr::null_mut(),
            n_folds: 0,
            mean: f64::NAN,
            se: f64::NAN,
            n_observations: 0,
        }
    }
}

/// Standard-error type of `anofox_quantile_process`.
pub const QUANTILE_SE_NONE: i32 = 0;
pub const QUANTILE_SE_IID: i32 = 1;
pub const QUANTILE_SE_NID: i32 = 2;
pub const QUANTILE_SE_BOOT: i32 = 3;

/// Coefficient tables of `n_tau` quantiles, each `k` entries (intercept
/// first when `has_intercept`), row-major per tau.
#[repr(C)]
#[derive(Debug)]
pub struct QuantileProcessResultFFI {
    pub taus: *mut f64,
    pub estimate: *mut f64,
    pub std_error: *mut f64,
    pub statistic: *mut f64,
    pub p_value: *mut f64,
    pub conf_low: *mut f64,
    pub conf_high: *mut f64,
    pub n_tau: usize,
    pub k: usize,
    pub n_observations: usize,
    pub conf_level: f64,
    pub df_residual: f64,
    pub has_intercept: bool,
}

impl Default for QuantileProcessResultFFI {
    fn default() -> Self {
        Self {
            taus: std::ptr::null_mut(),
            estimate: std::ptr::null_mut(),
            std_error: std::ptr::null_mut(),
            statistic: std::ptr::null_mut(),
            p_value: std::ptr::null_mut(),
            conf_low: std::ptr::null_mut(),
            conf_high: std::ptr::null_mut(),
            n_tau: 0,
            k: 0,
            n_observations: 0,
            conf_level: f64::NAN,
            df_residual: f64::NAN,
            has_intercept: false,
        }
    }
}

/// One prediction with its standard error and interval.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PredictSeResultFFI {
    pub yhat: f64,
    pub se_fit: f64,
    pub yhat_lower: f64,
    pub yhat_upper: f64,
}

/// Interval of `anofox_predict_se`.
pub const PREDICT_INTERVAL_NONE: i32 = 0;
pub const PREDICT_INTERVAL_CONFIDENCE: i32 = 1;
pub const PREDICT_INTERVAL_PREDICTION: i32 = 2;

/// Partial-residual / added-variable data, term-major `n_terms * len`.
#[repr(C)]
#[derive(Debug)]
pub struct PartialResidualsResultFFI {
    pub x: *mut f64,
    pub component: *mut f64,
    pub partial_residual: *mut f64,
    pub x_residual: *mut f64,
    pub y_residual: *mut f64,
    pub n_terms: usize,
    pub len: usize,
    pub has_added_variable: bool,
}

impl Default for PartialResidualsResultFFI {
    fn default() -> Self {
        Self {
            x: std::ptr::null_mut(),
            component: std::ptr::null_mut(),
            partial_residual: std::ptr::null_mut(),
            x_residual: std::ptr::null_mut(),
            y_residual: std::ptr::null_mut(),
            n_terms: 0,
            len: 0,
            has_added_variable: false,
        }
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

unsafe fn set_error(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
}

/// Run `compute`, mapping a panic or a `StatsError` into `out_error`.
unsafe fn run_core<T>(
    label: &str,
    out_error: *mut AnofoxError,
    compute: impl FnOnce() -> Result<T, StatsError>,
) -> Option<T> {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(compute)) {
        Err(_) => {
            set_error(
                out_error,
                ErrorCode::InternalError,
                &format!("Internal panic in {label}"),
            );
            None
        }
        Ok(Err(e)) => {
            set_error(out_error, error_to_code(&e), &e.to_string());
            None
        }
        Ok(Ok(r)) => Some(r),
    }
}

/// Free each non-NULL pointer and reset it to NULL.
unsafe fn free_all(ptrs: &mut [&mut *mut f64]) {
    for p in ptrs.iter_mut() {
        if !p.is_null() {
            libc::free(**p as *mut libc::c_void);
            **p = std::ptr::null_mut();
        }
    }
}

unsafe fn read_x(x: *const DataArray, x_count: usize) -> Vec<Vec<f64>> {
    slice::from_raw_parts(x, x_count)
        .iter()
        .map(|arr| arr.to_vec())
        .collect()
}

unsafe fn path_options(options: &RegPathOptionsFFI) -> RegPathOptions {
    RegPathOptions {
        l1_ratio: options.l1_ratio,
        fit_intercept: options.fit_intercept,
        standardize: options.standardize,
        lambdas: if options.lambdas.is_null() || options.lambdas_len == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(options.lambdas, options.lambdas_len).to_vec()
        },
        n_lambda: options.n_lambda,
        lambda_min_ratio: options.lambda_min_ratio,
    }
}

fn cv_spec(spec: &CvSpecFFI) -> CvSpec {
    CvSpec {
        strategy: match spec.strategy {
            CV_STRATEGY_FOLD_IDS => CvStrategy::FoldIds,
            CV_STRATEGY_GROUP => CvStrategy::Group,
            CV_STRATEGY_LOGO => CvStrategy::Logo,
            CV_STRATEGY_STRATIFIED => CvStrategy::Stratified,
            CV_STRATEGY_TIME_SERIES => CvStrategy::TimeSeries,
            _ => CvStrategy::Kfold,
        },
        n_folds: spec.n_folds,
        seed: spec.seed,
        repeats: spec.repeats,
        test_size: spec.test_size,
        gap: spec.gap,
        max_train_size: spec.max_train_size,
        n_bins: spec.n_bins,
        mae: spec.mae,
        inner_folds: spec.inner_folds,
        one_se: spec.one_se,
    }
}

unsafe fn write_path(
    r: &RegPathResult,
    out: *mut RegPathResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    let df: Vec<f64> = r.df.iter().map(|&d| d as f64).collect();
    let mut res = RegPathResultFFI {
        lambdas: alloc_f64(&r.lambdas),
        intercepts: alloc_f64(&r.intercepts),
        coefficients: alloc_f64(&r.coefficients),
        df: alloc_f64(&df),
        n_lambda: r.lambdas.len(),
        n_features: r.n_features,
        n_observations: r.n_observations,
    };
    if res.n_lambda > 0
        && (res.lambdas.is_null()
            || res.intercepts.is_null()
            || res.df.is_null()
            || (res.coefficients.is_null() && !r.coefficients.is_empty()))
    {
        anofox_free_reg_path_result(&mut res);
        set_error(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate path result",
        );
        return false;
    }
    *out = res;
    true
}

// =============================================================================
// #154: regularisation paths and CV
// =============================================================================

/// Ridge / elastic-net coefficient path (glmnet's Gaussian objective), from
/// `anofox_regression::solvers::path::fit_path` on glmnet's default grid or
/// the given lambdas. Rows with a non-finite value are dropped.
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`;
/// `options.lambdas` to `options.lambdas_len` doubles (or be NULL). `out` is
/// required and is released with `anofox_free_reg_path_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_reg_path(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: RegPathOptionsFFI,
    out: *mut RegPathResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = RegPathResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = read_x(x, x_count);
        let opts = path_options(&options);
        match run_core("regularisation path", out_error, || {
            fit_reg_path(&y_vec, &x_vecs, &opts)
        }) {
            Some(r) => write_path(&r, out, out_error),
            None => false,
        }
    })
}

/// LARS / Lasso-LARS coefficient path (`FittedLars::path`), one entry per knot.
///
/// # Safety
/// As [`anofox_reg_path`].
#[no_mangle]
pub unsafe extern "C" fn anofox_lars_path(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: LarsOptionsFFI,
    out: *mut RegPathResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = RegPathResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = read_x(x, x_count);
        let opts = LarsOptions {
            method_lasso: options.method_lasso,
            fit_intercept: options.fit_intercept,
            alpha: options.alpha,
            n_nonzero_coefs: options.n_nonzero_coefs,
            standardize: options.standardize,
        };
        match run_core("LARS path", out_error, || lars_path(&y_vec, &x_vecs, &opts)) {
            Some(r) => write_path(&r, out, out_error),
            None => false,
        }
    })
}

/// Release the arrays of a `RegPathResultFFI` (NULL-safe, idempotent).
///
/// # Safety
/// `result` must be NULL or come from `anofox_reg_path` / `anofox_lars_path`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_reg_path_result(result: *mut RegPathResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_all(&mut [
            &mut r.lambdas,
            &mut r.intercepts,
            &mut r.coefficients,
            &mut r.df,
        ]);
        r.n_lambda = 0;
    })
}

/// Cross-validation curve of the ridge / elastic-net path
/// (`solvers::cv::cv_path_splits`, cv.glmnet's `type.measure = "mse"` or
/// `"mae"`) on the splits of a CV strategy (#204). `keys` holds one value per
/// row (fold label, group, stratum or time, by strategy) or is NULL.
///
/// # Safety
/// As [`anofox_reg_path`]; `keys` is NULL or has the length of `y`.
/// Release `out` with `anofox_free_cv_path_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_cv_reg_path(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    keys: *const f64,
    spec: CvSpecFFI,
    options: RegPathOptionsFFI,
    out: *mut CvPathResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = CvPathResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = read_x(x, x_count);
        let key_vec = if keys.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(keys, y_vec.len()).to_vec())
        };
        let opts = path_options(&options);
        let spec = cv_spec(&spec);
        let Some(r) = run_core("cross-validation", out_error, || {
            cv_reg_path_strategy(&y_vec, &x_vecs, key_vec.as_deref(), &spec, &opts)
        }) else {
            return false;
        };
        let df: Vec<f64> = r.df.iter().map(|&d| d as f64).collect();
        let mut res = CvPathResultFFI {
            lambdas: alloc_f64(&r.lambdas),
            cv_mean: alloc_f64(&r.cv_mean),
            cv_se: alloc_f64(&r.cv_se),
            df: alloc_f64(&df),
            n_lambda: r.lambdas.len(),
            index_min: r.index_min,
            index_1se: r.index_1se,
            n_folds: r.n_folds,
            n_observations: r.n_observations,
        };
        if res.n_lambda > 0
            && (res.lambdas.is_null()
                || res.cv_mean.is_null()
                || res.cv_se.is_null()
                || res.df.is_null())
        {
            anofox_free_cv_path_result(&mut res);
            set_error(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate CV result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Nested cross-validation of the ridge / elastic-net path
/// (`solvers::cv::nested_cv_path`, #204): outer splits from the strategy,
/// inner folds from each outer training set only.
///
/// # Safety
/// As [`anofox_cv_reg_path`]. Release `out` with `anofox_free_nested_cv_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_nested_cv_reg_path(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    keys: *const f64,
    spec: CvSpecFFI,
    options: RegPathOptionsFFI,
    out: *mut NestedCvResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = NestedCvResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = read_x(x, x_count);
        let key_vec = if keys.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(keys, y_vec.len()).to_vec())
        };
        let opts = path_options(&options);
        let spec = cv_spec(&spec);
        let Some(r) = run_core("nested cross-validation", out_error, || {
            nested_cv_reg_path(&y_vec, &x_vecs, key_vec.as_deref(), &spec, &opts)
        }) else {
            return false;
        };
        let as_f64 = |v: &[usize]| v.iter().map(|&a| a as f64).collect::<Vec<f64>>();
        let mut res = NestedCvResultFFI {
            lambda: alloc_f64(&r.lambda),
            test_error: alloc_f64(&r.test_error),
            inner_cv_mean: alloc_f64(&r.inner_cv_mean),
            inner_cv_se: alloc_f64(&r.inner_cv_se),
            n_train: alloc_f64(&as_f64(&r.n_train)),
            n_test: alloc_f64(&as_f64(&r.n_test)),
            n_inner_folds: alloc_f64(&as_f64(&r.n_inner_folds)),
            n_folds: r.lambda.len(),
            mean: r.mean,
            se: r.se,
            n_observations: r.n_observations,
        };
        if res.n_folds > 0
            && (res.lambda.is_null()
                || res.test_error.is_null()
                || res.inner_cv_mean.is_null()
                || res.inner_cv_se.is_null()
                || res.n_train.is_null()
                || res.n_test.is_null()
                || res.n_inner_folds.is_null())
        {
            anofox_free_nested_cv_result(&mut res);
            set_error(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate nested CV result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Release the arrays of a `NestedCvResultFFI` (NULL-safe, idempotent).
///
/// # Safety
/// `result` must be NULL or come from `anofox_nested_cv_reg_path`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_nested_cv_result(result: *mut NestedCvResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_all(&mut [
            &mut r.lambda,
            &mut r.test_error,
            &mut r.inner_cv_mean,
            &mut r.inner_cv_se,
            &mut r.n_train,
            &mut r.n_test,
            &mut r.n_inner_folds,
        ]);
        r.n_folds = 0;
    })
}

/// Release the arrays of a `CvPathResultFFI` (NULL-safe, idempotent).
///
/// # Safety
/// `result` must be NULL or come from `anofox_cv_reg_path`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_cv_path_result(result: *mut CvPathResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_all(&mut [&mut r.lambdas, &mut r.cv_mean, &mut r.cv_se, &mut r.df]);
        r.n_lambda = 0;
    })
}

// =============================================================================
// #157: quantile-regression inference and the quantile process
// =============================================================================

/// Fit a quantile regression at each of `taus` and compute its coefficient
/// table (`anofox_regression::solvers::quantile_inference::quantile_process`,
/// `quantreg::summary.rq(se = ...)`). `se_type` is one of `QUANTILE_SE_*`;
/// `seed` / `n_boot` apply to the bootstrap. Rows with a non-finite value are
/// dropped.
///
/// # Safety
/// As [`anofox_reg_path`]; `taus` points to `n_taus` doubles. Release `out`
/// with `anofox_free_quantile_process_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_quantile_process(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    taus: *const f64,
    n_taus: usize,
    fit_intercept: bool,
    se_type: i32,
    seed: u64,
    n_boot: usize,
    confidence_level: f64,
    out: *mut QuantileProcessResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 || taus.is_null() || n_taus == 0 {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "x, taus or out is NULL or empty",
            );
            return false;
        }
        *out = QuantileProcessResultFFI::default();
        let se = match se_type {
            QUANTILE_SE_NONE => None,
            QUANTILE_SE_IID => Some(QuantileSeType::Iid),
            QUANTILE_SE_NID => Some(QuantileSeType::Nid),
            QUANTILE_SE_BOOT => Some(QuantileSeType::Boot { seed, n_boot }),
            other => {
                set_error(
                    out_error,
                    ErrorCode::InvalidInput,
                    &format!("unknown se type {other}"),
                );
                return false;
            }
        };
        let y_vec = y.to_vec();
        let x_vecs = read_x(x, x_count);
        let taus = slice::from_raw_parts(taus, n_taus).to_vec();
        let Some((rows, n_obs)) = run_core("quantile inference", out_error, || {
            quantile_process(&y_vec, &x_vecs, &taus, fit_intercept, se, confidence_level)
        }) else {
            return false;
        };
        let k = rows.first().map_or(0, |r| r.estimate.len());
        let flat =
            |f: fn(&anofox_stats_core::models::QuantileProcessRow) -> &Vec<f64>| -> Vec<f64> {
                rows.iter().flat_map(|r| f(r).iter().copied()).collect()
            };
        let tau_vec: Vec<f64> = rows.iter().map(|r| r.tau).collect();
        let mut res = QuantileProcessResultFFI {
            taus: alloc_f64(&tau_vec),
            estimate: alloc_f64(&flat(|r| &r.estimate)),
            std_error: alloc_f64(&flat(|r| &r.std_error)),
            statistic: alloc_f64(&flat(|r| &r.statistic)),
            p_value: alloc_f64(&flat(|r| &r.p_value)),
            conf_low: alloc_f64(&flat(|r| &r.conf_low)),
            conf_high: alloc_f64(&flat(|r| &r.conf_high)),
            n_tau: rows.len(),
            k,
            n_observations: n_obs,
            conf_level: rows.first().map_or(f64::NAN, |r| r.conf_level),
            df_residual: rows.first().map_or(f64::NAN, |r| r.df_residual),
            has_intercept: fit_intercept,
        };
        let any_null = [
            res.taus,
            res.estimate,
            res.std_error,
            res.statistic,
            res.p_value,
            res.conf_low,
            res.conf_high,
        ]
        .iter()
        .any(|p| p.is_null());
        if res.n_tau * res.k > 0 && any_null {
            anofox_free_quantile_process_result(&mut res);
            set_error(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate quantile result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Release the arrays of a `QuantileProcessResultFFI` (NULL-safe, idempotent).
///
/// # Safety
/// `result` must be NULL or come from `anofox_quantile_process`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_quantile_process_result(
    result: *mut QuantileProcessResultFFI,
) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_all(&mut [
            &mut r.taus,
            &mut r.estimate,
            &mut r.std_error,
            &mut r.statistic,
            &mut r.p_value,
            &mut r.conf_low,
            &mut r.conf_high,
        ]);
        r.n_tau = 0;
    })
}

// =============================================================================
// #158: coefficient covariance, predictions with SE, partial residuals
// =============================================================================

/// Coefficient covariance `σ² M` of an OLS / WLS / Ridge fit from its
/// training design (`M` as `anofox_interval_matrix`), row-major `dim x dim`
/// with the intercept first. Release with `anofox_free_interval_matrix`.
///
/// # Safety
/// As `anofox_interval_matrix`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_coefficient_vcov(
    x: *const DataArray,
    x_count: usize,
    coefficients: *const f64,
    coefficients_len: usize,
    fit_intercept: bool,
    weights: *const DataArray,
    ridge_lambda: f64,
    residual_std_error: f64,
    out_matrix: *mut *mut f64,
    out_dim: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out_matrix.is_null() || out_dim.is_null() {
            set_error(
                out_error,
                ErrorCode::InvalidInput,
                "out_matrix or out_dim is NULL",
            );
            return false;
        }
        *out_matrix = std::ptr::null_mut();
        *out_dim = 0;
        if x.is_null() || x_count == 0 || coefficients.is_null() || coefficients_len != x_count {
            set_error(
                out_error,
                ErrorCode::DimensionMismatch,
                "x is empty or coefficients_len != x_count",
            );
            return false;
        }
        let x_vecs = read_x(x, x_count);
        let coef = slice::from_raw_parts(coefficients, coefficients_len).to_vec();
        let w_vec = if weights.is_null() {
            None
        } else {
            Some((*weights).to_vec())
        };
        let Some((m, dim)) = run_core("coefficient covariance", out_error, || {
            coefficient_vcov(
                &x_vecs,
                &coef,
                fit_intercept,
                w_vec.as_deref(),
                ridge_lambda,
                residual_std_error,
            )
        }) else {
            return false;
        };
        let ptr = alloc_f64(&m);
        if ptr.is_null() && !m.is_empty() {
            set_error(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate covariance",
            );
            return false;
        }
        *out_matrix = ptr;
        *out_dim = dim;
        true
    })
}

/// One prediction with its standard error and interval from a model's
/// coefficients and covariance. `link` NULL is a linear model (OLS / WLS /
/// Ridge: t intervals with `df_residual`, prediction intervals with
/// `residual_std_error`); otherwise the GLM link name, with intervals on the
/// link scale (`response_scale` maps them through the inverse link).
/// `interval` is one of `PREDICT_INTERVAL_*`.
///
/// # Safety
/// `coefficients` and `x` point to `coefficients_len` doubles, `vcov` to
/// `dim * dim`; `link` is NULL or NUL-terminated; `out` is required.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_predict_se(
    coefficients: *const f64,
    coefficients_len: usize,
    intercept: f64,
    x: *const f64,
    vcov: *const f64,
    dim: usize,
    link: *const c_char,
    response_scale: bool,
    residual_std_error: f64,
    df_residual: f64,
    interval: i32,
    level: f64,
    out: *mut PredictSeResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null()
            || vcov.is_null()
            || (coefficients_len > 0 && (coefficients.is_null() || x.is_null()))
        {
            set_error(out_error, ErrorCode::InvalidInput, "NULL input");
            return false;
        }
        let coef = slice::from_raw_parts(coefficients, coefficients_len).to_vec();
        let x0 = slice::from_raw_parts(x, coefficients_len).to_vec();
        let v = slice::from_raw_parts(vcov, dim * dim).to_vec();
        let interval = match interval {
            PREDICT_INTERVAL_NONE => PredictInterval::None,
            PREDICT_INTERVAL_CONFIDENCE => PredictInterval::Confidence,
            PREDICT_INTERVAL_PREDICTION => PredictInterval::Prediction,
            other => {
                set_error(
                    out_error,
                    ErrorCode::InvalidInput,
                    &format!("unknown interval {other}"),
                );
                return false;
            }
        };
        let link = if link.is_null() {
            None
        } else {
            Some(CStr::from_ptr(link).to_string_lossy().into_owned())
        };
        let r = run_core("predict", out_error, || match &link {
            None => predict_se_linear(
                &coef,
                intercept,
                &x0,
                &v,
                dim,
                residual_std_error,
                df_residual,
                interval,
                level,
            ),
            Some(l) => predict_se_glm(
                &coef,
                intercept,
                &x0,
                &v,
                dim,
                l,
                response_scale,
                interval,
                level,
            ),
        });
        match r {
            Some(p) => {
                *out = PredictSeResultFFI {
                    yhat: p.fit,
                    se_fit: p.se_fit,
                    yhat_lower: p.lower,
                    yhat_upper: p.upper,
                };
                true
            }
            None => false,
        }
    })
}

unsafe fn write_partial(
    r: &PartialResidualsResult,
    out: *mut PartialResidualsResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    let mut res = PartialResidualsResultFFI {
        x: alloc_f64(&r.x),
        component: alloc_f64(&r.component),
        partial_residual: alloc_f64(&r.partial_residual),
        x_residual: alloc_f64(&r.x_residual),
        y_residual: alloc_f64(&r.y_residual),
        n_terms: r.n_terms,
        len: r.len,
        has_added_variable: r.has_added_variable,
    };
    let any_null = [
        res.x,
        res.component,
        res.partial_residual,
        res.x_residual,
        res.y_residual,
    ]
    .iter()
    .any(|p| p.is_null());
    if r.n_terms * r.len > 0 && any_null {
        anofox_free_partial_residuals_result(&mut res);
        set_error(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate partial residuals",
        );
        return false;
    }
    *out = res;
    true
}

/// Shared body of the partial-residual entry points.
unsafe fn partial_entry(
    label: &str,
    x: *const DataArray,
    x_count: usize,
    out: *mut PartialResidualsResultFFI,
    out_error: *mut AnofoxError,
    compute: impl FnOnce(Vec<Vec<f64>>) -> Result<PartialResidualsResult, StatsError>,
) -> bool {
    if out.is_null() || x.is_null() || x_count == 0 {
        set_error(
            out_error,
            ErrorCode::InvalidInput,
            "x or out is NULL or empty",
        );
        return false;
    }
    *out = PartialResidualsResultFFI::default();
    let x_vecs = read_x(x, x_count);
    match run_core(label, out_error, || compute(x_vecs)) {
        Some(r) => write_partial(&r, out, out_error),
        None => false,
    }
}

/// Partial residuals (CCPR) and added-variable coordinates of an OLS fit on
/// the usable rows, one entry per input row and x column
/// (`anofox_regression::diagnostics::{partial_residuals, added_variable}`).
///
/// # Safety
/// As `anofox_ols_augment`. Release `out` with
/// `anofox_free_partial_residuals_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_ols_partial_residuals(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: OlsOptionsFFI,
    out: *mut PartialResidualsResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let y_vec = y.to_vec();
        let opts = OlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: false,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: None,
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
        };
        partial_entry("OLS partial residuals", x, x_count, out, out_error, |xv| {
            partial_residuals_ols(&y_vec, &xv, &opts)
        })
    })
}

/// As [`anofox_ols_partial_residuals`] for a WLS fit.
///
/// # Safety
/// As `anofox_wls_augment`.
#[no_mangle]
pub unsafe extern "C" fn anofox_wls_partial_residuals(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    weights: DataArray,
    options: WlsOptionsFFI,
    out: *mut PartialResidualsResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let y_vec = y.to_vec();
        let w_vec = weights.to_vec();
        let opts = WlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: false,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: None,
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
        };
        partial_entry("WLS partial residuals", x, x_count, out, out_error, |xv| {
            partial_residuals_wls(&y_vec, &xv, &w_vec, &opts)
        })
    })
}

macro_rules! glm_partial_residuals {
    ($name:ident, $opt_ffi:ty, $convert:ident, $core:ident, $label:expr) => {
        /// GLM partial residuals (working residual + component,
        /// `anofox_regression::diagnostics::partial_residuals_glm`) on the
        /// usable rows; no added-variable coordinates.
        ///
        /// # Safety
        /// As the matching `anofox_*_augment`.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            y: DataArray,
            x: *const DataArray,
            x_count: usize,
            options: $opt_ffi,
            out: *mut PartialResidualsResultFFI,
            out_error: *mut AnofoxError,
        ) -> bool {
            ffi_guard(out_error, false, || {
                let y_vec = y.to_vec();
                let opts = $convert(&options);
                partial_entry($label, x, x_count, out, out_error, |xv| {
                    $core(&y_vec, &xv, &opts)
                })
            })
        }
    };
}

glm_partial_residuals!(
    anofox_poisson_partial_residuals,
    PoissonOptionsFFI,
    poisson_options_from_ffi,
    partial_residuals_poisson,
    "Poisson partial residuals"
);
glm_partial_residuals!(
    anofox_binomial_partial_residuals,
    BinomialOptionsFFI,
    binomial_options_from_ffi,
    partial_residuals_binomial,
    "Binomial partial residuals"
);
glm_partial_residuals!(
    anofox_logistic_partial_residuals,
    LogisticOptionsFFI,
    logistic_options_from_ffi,
    partial_residuals_logistic,
    "logistic partial residuals"
);
glm_partial_residuals!(
    anofox_negbinomial_partial_residuals,
    NegBinomialOptionsFFI,
    negbinomial_options_from_ffi,
    partial_residuals_negbinomial,
    "Negative Binomial partial residuals"
);
glm_partial_residuals!(
    anofox_gamma_partial_residuals,
    GammaOptionsFFI,
    gamma_options_from_ffi,
    partial_residuals_gamma,
    "Gamma partial residuals"
);
glm_partial_residuals!(
    anofox_tweedie_partial_residuals,
    TweedieOptionsFFI,
    tweedie_options_from_ffi,
    partial_residuals_tweedie,
    "Tweedie partial residuals"
);

/// Release the arrays of a `PartialResidualsResultFFI` (NULL-safe, idempotent).
///
/// # Safety
/// `result` must be NULL or come from an `anofox_*_partial_residuals` call.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_partial_residuals_result(
    result: *mut PartialResidualsResultFFI,
) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_all(&mut [
            &mut r.x,
            &mut r.component,
            &mut r.partial_residual,
            &mut r.x_residual,
            &mut r.y_residual,
        ]);
        r.len = 0;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Mirrors the static_asserts in src/include/model_struct.cpp.
    #[test]
    fn layouts_match_c_header() {
        let p = size_of::<*mut f64>();
        let z = size_of::<usize>();
        assert_eq!(size_of::<RegPathResultFFI>(), 4 * p + 3 * z);
        assert_eq!(size_of::<CvPathResultFFI>(), 4 * p + 5 * z);
        assert_eq!(offset_of!(QuantileProcessResultFFI, n_tau), 7 * p);
        assert_eq!(
            offset_of!(QuantileProcessResultFFI, has_intercept),
            7 * p + 3 * z + 16
        );
        assert_eq!(size_of::<PredictSeResultFFI>(), 32);
        assert_eq!(offset_of!(PartialResidualsResultFFI, n_terms), 5 * p);
        assert_eq!(
            offset_of!(PartialResidualsResultFFI, has_added_variable),
            5 * p + 2 * z
        );
    }

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 30;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 3.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 / 2.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + x1[i] - 0.5 * x2[i] + ((i * 3) % 7) as f64 / 4.0)
            .collect();
        (y, x1, x2)
    }

    #[test]
    fn reg_path_round_trip() {
        let (y, x1, x2) = data();
        let x = [arr(&x1), arr(&x2)];
        let opts = RegPathOptionsFFI {
            l1_ratio: 0.5,
            fit_intercept: true,
            standardize: false,
            lambdas: std::ptr::null(),
            lambdas_len: 0,
            n_lambda: 5,
            lambda_min_ratio: f64::NAN,
        };
        let mut out = RegPathResultFFI::default();
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_reg_path(
                arr(&y),
                x.as_ptr(),
                2,
                opts,
                &mut out,
                &mut err
            ));
            assert_eq!((out.n_lambda, out.n_features), (5, 2));
            assert_eq!(*out.df, 0.0);
            anofox_free_reg_path_result(&mut out);
            anofox_free_reg_path_result(&mut out);
            let spec = CvSpecFFI {
                strategy: CV_STRATEGY_KFOLD,
                n_folds: 5,
                seed: 1,
                repeats: 1,
                test_size: 0,
                gap: 0,
                max_train_size: 0,
                n_bins: 5,
                mae: false,
                inner_folds: 3,
                one_se: false,
            };
            let mut cv = CvPathResultFFI::default();
            assert!(anofox_cv_reg_path(
                arr(&y),
                x.as_ptr(),
                2,
                std::ptr::null(),
                spec,
                opts,
                &mut cv,
                &mut err
            ));
            assert_eq!(cv.n_folds, 5);
            anofox_free_cv_path_result(&mut cv);
            let mut nested = NestedCvResultFFI::default();
            assert!(anofox_nested_cv_reg_path(
                arr(&y),
                x.as_ptr(),
                2,
                std::ptr::null(),
                spec,
                opts,
                &mut nested,
                &mut err
            ));
            assert_eq!(nested.n_folds, 5);
            assert!(nested.mean.is_finite());
            anofox_free_nested_cv_result(&mut nested);
            anofox_free_nested_cv_result(&mut nested);
        }
    }

    #[test]
    fn quantile_process_round_trip() {
        let (y, x1, _) = data();
        let x = [arr(&x1)];
        let taus = [0.25, 0.5, 0.75];
        let mut out = QuantileProcessResultFFI::default();
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_quantile_process(
                arr(&y),
                x.as_ptr(),
                1,
                taus.as_ptr(),
                3,
                true,
                QUANTILE_SE_NID,
                0,
                0,
                0.95,
                &mut out,
                &mut err
            ));
            assert_eq!((out.n_tau, out.k), (3, 2));
            assert!(*out.std_error.add(5) > 0.0);
            anofox_free_quantile_process_result(&mut out);
        }
    }
}
