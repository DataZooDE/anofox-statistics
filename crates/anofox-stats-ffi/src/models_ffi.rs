//! C FFI for Bayesian Ridge and ARD regression (#61) and for
//! Passive-Aggressive regression, lmDynamic and LOWESS (#63). Every struct is
//! mirrored in `src/include/anofox_stats_ffi.h`; the layouts are pinned by the
//! tests below and the static_asserts in `src/include/model_struct.cpp`.

use super::*;
use anofox_stats_core::models::{
    fit_bayesian, fit_lm_dynamic, fit_lowess, fit_passive_aggressive, fit_predict_bayesian,
    BayesianOptions, PassiveAggressiveOptions,
};
use anofox_stats_core::{LmDynamicOptions, LowessOptions};

/// Options of `anofox_bayesian_fit` / `anofox_bayesian_fit_predict`
/// (scikit-learn's `BayesianRidge` / `ARDRegression` parameters).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BayesianOptionsFFI {
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
    /// ARD only.
    pub threshold_lambda: f64,
}

impl From<&BayesianOptionsFFI> for BayesianOptions {
    fn from(o: &BayesianOptionsFFI) -> Self {
        Self {
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations as usize,
            tolerance: o.tolerance,
            alpha_1: o.alpha_1,
            alpha_2: o.alpha_2,
            lambda_1: o.lambda_1,
            lambda_2: o.lambda_2,
            threshold_lambda: o.threshold_lambda,
        }
    }
}

/// A fitted Bayesian Ridge / ARD model. `coefficients` and `posterior_sd`
/// have `coefficients_len` entries; `weight_precisions` (ARD) has
/// `weight_precisions_len` (0 and NULL for Bayesian Ridge). Released with
/// `anofox_free_bayesian_result`. The doubles come first so the layout has no
/// padding on 64-bit or wasm32.
#[repr(C)]
#[derive(Debug)]
pub struct BayesianFitResultFFI {
    pub intercept: f64,
    pub r_squared: f64,
    pub noise_precision: f64,
    /// NaN for ARD.
    pub weight_precision: f64,
    /// NaN for ARD.
    pub log_marginal_likelihood: f64,
    pub coefficients: *mut f64,
    pub posterior_sd: *mut f64,
    pub weight_precisions: *mut f64,
    pub coefficients_len: usize,
    pub weight_precisions_len: usize,
    pub n_observations: usize,
    pub n_features: usize,
    pub iterations: usize,
    pub summary: ModelSummaryFFI,
}

impl Default for BayesianFitResultFFI {
    fn default() -> Self {
        Self {
            intercept: f64::NAN,
            r_squared: f64::NAN,
            noise_precision: f64::NAN,
            weight_precision: f64::NAN,
            log_marginal_likelihood: f64::NAN,
            coefficients: std::ptr::null_mut(),
            posterior_sd: std::ptr::null_mut(),
            weight_precisions: std::ptr::null_mut(),
            coefficients_len: 0,
            weight_precisions_len: 0,
            n_observations: 0,
            n_features: 0,
            iterations: 0,
            summary: ModelSummaryFFI::default(),
        }
    }
}

unsafe fn report(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
}

/// Run `compute`, mapping a panic or a `StatsError` into `out_error`.
unsafe fn run_model<T>(
    label: &str,
    out_error: *mut AnofoxError,
    compute: impl FnOnce() -> Result<T, StatsError>,
) -> Option<T> {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(compute)) {
        Err(_) => {
            report(
                out_error,
                ErrorCode::InternalError,
                &format!("Internal panic in {label}"),
            );
            None
        }
        Ok(Err(e)) => {
            report(out_error, error_to_code(&e), &e.to_string());
            None
        }
        Ok(Ok(r)) => Some(r),
    }
}

unsafe fn x_columns(x: *const DataArray, x_count: usize) -> Vec<Vec<f64>> {
    slice::from_raw_parts(x, x_count)
        .iter()
        .map(|arr| arr.to_vec())
        .collect()
}

/// Fit Bayesian Ridge (`ard = false`) or ARD (`ard = true`) regression on the
/// complete rows of (y, x).
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`. `out` is
/// required and is released with `anofox_free_bayesian_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayesian_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: BayesianOptionsFFI,
    ard: bool,
    out: *mut BayesianFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            report(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = BayesianFitResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = x_columns(x, x_count);
        let opts = BayesianOptions::from(&options);
        let label = if ard { "ARD fit" } else { "Bayesian Ridge fit" };
        let Some(r) = run_model(label, out_error, || {
            fit_bayesian(&y_vec, &x_vecs, &opts, ard)
        }) else {
            return false;
        };
        let mut res = BayesianFitResultFFI {
            intercept: r.intercept.unwrap_or(f64::NAN),
            r_squared: r.r_squared,
            noise_precision: r.noise_precision,
            weight_precision: r.weight_precision,
            log_marginal_likelihood: r.log_marginal_likelihood,
            coefficients: alloc_f64(&r.coefficients),
            posterior_sd: alloc_f64(&r.posterior_sd),
            weight_precisions: alloc_f64(&r.weight_precisions),
            coefficients_len: r.coefficients.len(),
            weight_precisions_len: r.weight_precisions.len(),
            n_observations: r.n_observations,
            n_features: r.n_features,
            iterations: r.iterations,
            summary: ModelSummaryFFI::from(&r.summary),
        };
        if res.coefficients.is_null()
            || res.posterior_sd.is_null()
            || (res.weight_precisions.is_null() && !r.weight_precisions.is_empty())
        {
            anofox_free_bayesian_result(&mut res);
            report(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate Bayesian fit result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Fit Bayesian Ridge / ARD on the complete rows of (y, x) and predict the
/// `n_new` rows of `x_new` (row-major, `x_count` columns) with posterior
/// predictive intervals at `level`. Rows of `x_new` with a non-finite value
/// predict NaN.
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`; `x_new` to
/// `n_new * x_count` doubles; `out_yhat`, `out_lower` and `out_upper` to
/// `n_new` doubles each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_bayesian_fit_predict(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    x_new: *const f64,
    n_new: usize,
    options: BayesianOptionsFFI,
    ard: bool,
    level: f64,
    out_yhat: *mut f64,
    out_lower: *mut f64,
    out_upper: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if x.is_null()
            || x_count == 0
            || (n_new > 0
                && (x_new.is_null()
                    || out_yhat.is_null()
                    || out_lower.is_null()
                    || out_upper.is_null()))
        {
            report(
                out_error,
                ErrorCode::InvalidInput,
                "x, x_new or an output is NULL",
            );
            return false;
        }
        let y_vec = y.to_vec();
        let x_vecs = x_columns(x, x_count);
        let rows = if n_new == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(x_new, n_new * x_count).to_vec()
        };
        let opts = BayesianOptions::from(&options);
        let label = if ard {
            "ARD fit-predict"
        } else {
            "Bayesian Ridge fit-predict"
        };
        let Some(p) = run_model(label, out_error, || {
            fit_predict_bayesian(&y_vec, &x_vecs, &rows, &opts, ard, level)
        }) else {
            return false;
        };
        if n_new > 0 {
            std::ptr::copy_nonoverlapping(p.yhat.as_ptr(), out_yhat, n_new);
            std::ptr::copy_nonoverlapping(p.lower.as_ptr(), out_lower, n_new);
            std::ptr::copy_nonoverlapping(p.upper.as_ptr(), out_upper, n_new);
        }
        true
    })
}

/// Release the arrays of a `BayesianFitResultFFI`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_bayesian_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_bayesian_result(result: *mut BayesianFitResultFFI) {
    if result.is_null() {
        return;
    }
    let r = &mut *result;
    for p in [
        &mut r.coefficients,
        &mut r.posterior_sd,
        &mut r.weight_precisions,
    ] {
        if !p.is_null() {
            libc::free(*p as *mut libc::c_void);
            *p = std::ptr::null_mut();
        }
    }
    r.coefficients_len = 0;
    r.weight_precisions_len = 0;
}

// =============================================================================
// Passive-Aggressive regression (#63)
// =============================================================================

/// Options of `anofox_passive_aggressive_fit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PassiveAggressiveOptionsFFI {
    pub fit_intercept: bool,
    /// PA-II (squared epsilon-insensitive loss) instead of PA-I.
    pub squared_loss: bool,
    /// Passes over the rows; 1 = a single online pass.
    pub max_epochs: u32,
    pub c: f64,
    pub epsilon: f64,
    pub tolerance: f64,
}

/// A fitted Passive-Aggressive model; `coefficients` has `n_features`
/// entries. Released with `anofox_free_passive_aggressive_result`.
#[repr(C)]
#[derive(Debug)]
pub struct PassiveAggressiveResultFFI {
    pub intercept: f64,
    pub r_squared: f64,
    pub coefficients: *mut f64,
    pub n_observations: usize,
    pub n_features: usize,
    pub epochs: usize,
    pub summary: ModelSummaryFFI,
}

impl Default for PassiveAggressiveResultFFI {
    fn default() -> Self {
        Self {
            intercept: f64::NAN,
            r_squared: f64::NAN,
            coefficients: std::ptr::null_mut(),
            n_observations: 0,
            n_features: 0,
            epochs: 0,
            summary: ModelSummaryFFI::default(),
        }
    }
}

/// Fit a Passive-Aggressive regressor on the complete rows of (y, x), in
/// input order.
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`. `out` is
/// required and is released with `anofox_free_passive_aggressive_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_passive_aggressive_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: PassiveAggressiveOptionsFFI,
    out: *mut PassiveAggressiveResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            report(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = PassiveAggressiveResultFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = x_columns(x, x_count);
        let opts = PassiveAggressiveOptions {
            fit_intercept: options.fit_intercept,
            c: options.c,
            epsilon: options.epsilon,
            squared_loss: options.squared_loss,
            max_epochs: options.max_epochs as usize,
            tolerance: options.tolerance,
        };
        let Some(r) = run_model("Passive-Aggressive fit", out_error, || {
            fit_passive_aggressive(&y_vec, &x_vecs, &opts)
        }) else {
            return false;
        };
        let coefficients = alloc_f64(&r.coefficients);
        if coefficients.is_null() {
            report(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate Passive-Aggressive coefficients",
            );
            return false;
        }
        *out = PassiveAggressiveResultFFI {
            intercept: r.intercept.unwrap_or(f64::NAN),
            r_squared: r.r_squared,
            coefficients,
            n_observations: r.n_observations,
            n_features: r.n_features,
            epochs: r.epochs,
            summary: ModelSummaryFFI::from(&r.summary),
        };
        true
    })
}

/// Release the coefficients of a `PassiveAggressiveResultFFI`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_passive_aggressive_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_passive_aggressive_result(
    result: *mut PassiveAggressiveResultFFI,
) {
    if result.is_null() || (*result).coefficients.is_null() {
        return;
    }
    libc::free((*result).coefficients as *mut libc::c_void);
    (*result).coefficients = std::ptr::null_mut();
}

// =============================================================================
// lmDynamic (#63)
// =============================================================================

/// A fitted lmDynamic model. `coefficients` (time-averaged) has `n_features`
/// entries; `dynamic_coefficients` is `n_rows x n_coefs_per_row` row-major
/// (`[intercept, coef_1..coef_p]` per input row) and `fitted_values` has
/// `n_rows`; rows dropped for non-finite values are NaN. Released with
/// `anofox_free_lm_dynamic_fit_result`.
#[repr(C)]
#[derive(Debug)]
pub struct LmDynamicFitFFI {
    pub intercept: f64,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residual_std_error: f64,
    pub coefficients: *mut f64,
    pub dynamic_coefficients: *mut f64,
    pub fitted_values: *mut f64,
    pub n_rows: usize,
    pub n_coefs_per_row: usize,
    pub n_observations: usize,
    pub n_features: usize,
    pub n_models: usize,
    pub summary: ModelSummaryFFI,
}

impl Default for LmDynamicFitFFI {
    fn default() -> Self {
        Self {
            intercept: f64::NAN,
            r_squared: f64::NAN,
            adj_r_squared: f64::NAN,
            residual_std_error: f64::NAN,
            coefficients: std::ptr::null_mut(),
            dynamic_coefficients: std::ptr::null_mut(),
            fitted_values: std::ptr::null_mut(),
            n_rows: 0,
            n_coefs_per_row: 0,
            n_observations: 0,
            n_features: 0,
            n_models: 0,
            summary: ModelSummaryFFI::default(),
        }
    }
}

/// Fit lmDynamic (greybox `lmDynamic(bruteforce = TRUE)`) on (y, x) in input
/// order. `options.lowess_span` 0 disables the weight smoothing;
/// `options.max_models` 0 averages every regressor subset.
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`. `out` is
/// required and is released with `anofox_free_lm_dynamic_fit_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_lm_dynamic_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: LmDynamicOptionsFFI,
    out: *mut LmDynamicFitFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() || x.is_null() || x_count == 0 {
            report(
                out_error,
                ErrorCode::InvalidInput,
                "x or out is NULL or empty",
            );
            return false;
        }
        *out = LmDynamicFitFFI::default();
        let y_vec = y.to_vec();
        let x_vecs = x_columns(x, x_count);
        let opts = LmDynamicOptions {
            fit_intercept: options.fit_intercept,
            ic: convert_ic_ffi(options.ic),
            distribution: convert_alm_distribution(options.distribution),
            lowess_span: (options.lowess_span > 0.0).then_some(options.lowess_span),
            max_models: (options.max_models > 0).then_some(options.max_models as usize),
            confidence_level: options.confidence_level,
        };
        let Some(r) = run_model("lmDynamic fit", out_error, || {
            fit_lm_dynamic(&y_vec, &x_vecs, &opts)
        }) else {
            return false;
        };
        let per_row = x_count + 1;
        let flat: Vec<f64> = r.dynamic_coefficients.iter().flatten().copied().collect();
        let mut res = LmDynamicFitFFI {
            intercept: r.intercept.unwrap_or(f64::NAN),
            r_squared: r.r_squared,
            adj_r_squared: r.adj_r_squared,
            residual_std_error: r.rmse,
            coefficients: alloc_f64(&r.coefficients),
            dynamic_coefficients: alloc_f64(&flat),
            fitted_values: alloc_f64(&r.fitted_values),
            n_rows: r.fitted_values.len(),
            n_coefs_per_row: per_row,
            n_observations: r.n_observations,
            n_features: r.n_features,
            n_models: r.n_models,
            summary: ModelSummaryFFI::from(&r.summary),
        };
        if res.coefficients.is_null()
            || res.dynamic_coefficients.is_null()
            || res.fitted_values.is_null()
        {
            anofox_free_lm_dynamic_fit_result(&mut res);
            report(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate lmDynamic result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Release the arrays of an `LmDynamicFitFFI`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_lm_dynamic_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_lm_dynamic_fit_result(result: *mut LmDynamicFitFFI) {
    if result.is_null() {
        return;
    }
    let r = &mut *result;
    for p in [
        &mut r.coefficients,
        &mut r.dynamic_coefficients,
        &mut r.fitted_values,
    ] {
        if !p.is_null() {
            libc::free(*p as *mut libc::c_void);
            *p = std::ptr::null_mut();
        }
    }
}

// =============================================================================
// LOWESS (#63)
// =============================================================================

/// The LOWESS curve: `len` points, x increasing. Released with
/// `anofox_free_lowess_result`.
#[repr(C)]
#[derive(Debug)]
pub struct LowessResultFFI {
    pub x: *mut f64,
    pub fitted: *mut f64,
    pub len: usize,
}

impl Default for LowessResultFFI {
    fn default() -> Self {
        Self {
            x: std::ptr::null_mut(),
            fitted: std::ptr::null_mut(),
            len: 0,
        }
    }
}

/// R's `stats::lowess(x, y, f = frac, iter = iterations, delta)` on the
/// complete rows of (x, y); a NaN `delta` uses R's default, 1% of the range of x.
///
/// # Safety
/// `x` and `y` must be valid DataArrays of equal length. `out` is required
/// and is released with `anofox_free_lowess_result`.
#[no_mangle]
pub unsafe extern "C" fn anofox_lowess(
    x: DataArray,
    y: DataArray,
    frac: f64,
    iterations: u32,
    delta: f64,
    out: *mut LowessResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if out.is_null() {
            report(out_error, ErrorCode::InvalidInput, "out is NULL");
            return false;
        }
        *out = LowessResultFFI::default();
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let opts = LowessOptions {
            frac,
            iterations: iterations as usize,
            delta: (!delta.is_nan()).then_some(delta),
        };
        let Some(r) = run_model("LOWESS", out_error, || fit_lowess(&x_vec, &y_vec, &opts)) else {
            return false;
        };
        let mut res = LowessResultFFI {
            x: alloc_f64(&r.x),
            fitted: alloc_f64(&r.fitted),
            len: r.x.len(),
        };
        if res.x.is_null() || res.fitted.is_null() {
            anofox_free_lowess_result(&mut res);
            report(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate LOWESS result",
            );
            return false;
        }
        *out = res;
        true
    })
}

/// Release the arrays of a `LowessResultFFI`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_lowess`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_lowess_result(result: *mut LowessResultFFI) {
    if result.is_null() {
        return;
    }
    let r = &mut *result;
    for p in [&mut r.x, &mut r.fitted] {
        if !p.is_null() {
            libc::free(*p as *mut libc::c_void);
            *p = std::ptr::null_mut();
        }
    }
    r.len = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Mirrors the static_asserts in src/include/model_struct.cpp.
    #[test]
    fn bayesian_layout() {
        let ptr = size_of::<*mut f64>();
        let sz = size_of::<usize>();
        let d = size_of::<f64>();
        assert_eq!(offset_of!(BayesianFitResultFFI, coefficients), 5 * d);
        assert_eq!(
            offset_of!(BayesianFitResultFFI, coefficients_len),
            5 * d + 3 * ptr
        );
        assert_eq!(
            offset_of!(BayesianFitResultFFI, summary),
            5 * d + 3 * ptr + 5 * sz
        );
        assert_eq!(
            size_of::<BayesianFitResultFFI>(),
            offset_of!(BayesianFitResultFFI, summary) + size_of::<ModelSummaryFFI>()
        );
        assert_eq!(offset_of!(BayesianOptionsFFI, tolerance), d);
        assert_eq!(size_of::<BayesianOptionsFFI>(), 7 * d);
    }

    /// Mirrors the #63 static_asserts in src/include/model_struct.cpp.
    #[test]
    fn online_dynamic_lowess_layout() {
        let ptr = size_of::<*mut f64>();
        let sz = size_of::<usize>();
        let d = size_of::<f64>();
        assert_eq!(offset_of!(PassiveAggressiveOptionsFFI, max_epochs), 4);
        assert_eq!(offset_of!(PassiveAggressiveOptionsFFI, c), d);
        assert_eq!(size_of::<PassiveAggressiveOptionsFFI>(), 4 * d);
        assert_eq!(offset_of!(PassiveAggressiveResultFFI, coefficients), 2 * d);
        assert_eq!(
            offset_of!(PassiveAggressiveResultFFI, summary),
            2 * d + ptr + 3 * sz
        );
        assert_eq!(
            size_of::<PassiveAggressiveResultFFI>(),
            offset_of!(PassiveAggressiveResultFFI, summary) + size_of::<ModelSummaryFFI>()
        );
        assert_eq!(offset_of!(LmDynamicFitFFI, coefficients), 4 * d);
        assert_eq!(
            offset_of!(LmDynamicFitFFI, summary),
            4 * d + 3 * ptr + 5 * sz
        );
        assert_eq!(
            size_of::<LmDynamicFitFFI>(),
            offset_of!(LmDynamicFitFFI, summary) + size_of::<ModelSummaryFFI>()
        );
        assert_eq!(size_of::<LowessResultFFI>(), 2 * ptr + sz);
    }

    #[test]
    fn online_dynamic_lowess_round_trip() {
        let x1: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = (0..40)
            .map(|i| 1.0 + if i < 20 { 0.5 } else { 2.0 } * x1[i] + 0.05 * (i as f64).cos())
            .collect();
        let xs = [arr(&x1)];
        unsafe {
            let mut err = AnofoxError::success();
            let mut pa = PassiveAggressiveResultFFI::default();
            let pa_opts = PassiveAggressiveOptionsFFI {
                fit_intercept: true,
                squared_loss: true,
                max_epochs: 1,
                c: 1.0,
                epsilon: 0.1,
                tolerance: 1e-3,
            };
            assert!(anofox_passive_aggressive_fit(
                arr(&y),
                xs.as_ptr(),
                1,
                pa_opts,
                &mut pa,
                &mut err
            ));
            assert_eq!(pa.n_features, 1);
            assert_eq!(pa.epochs, 1);
            assert!((*pa.coefficients).is_finite());
            anofox_free_passive_aggressive_result(&mut pa);
            assert!(pa.coefficients.is_null());

            let mut dynamic = LmDynamicFitFFI::default();
            let opts = LmDynamicOptionsFFI {
                lowess_span: 0.0,
                ..Default::default()
            };
            assert!(anofox_lm_dynamic_fit(
                arr(&y),
                xs.as_ptr(),
                1,
                opts,
                &mut dynamic,
                &mut err
            ));
            assert_eq!(dynamic.n_rows, 40);
            assert_eq!(dynamic.n_coefs_per_row, 2);
            assert_eq!(dynamic.n_models, 2);
            // The x1 effect grows after the step change.
            let early = *dynamic.dynamic_coefficients.add(2 * 2 + 1);
            let late = *dynamic.dynamic_coefficients.add(2 * 35 + 1);
            assert!(late > early);
            anofox_free_lm_dynamic_fit_result(&mut dynamic);
            assert!(dynamic.fitted_values.is_null());

            let mut curve = LowessResultFFI::default();
            assert!(anofox_lowess(
                arr(&x1),
                arr(&y),
                2.0 / 3.0,
                3,
                f64::NAN,
                &mut curve,
                &mut err
            ));
            assert_eq!(curve.len, 40);
            assert!(*curve.x <= *curve.x.add(1));
            anofox_free_lowess_result(&mut curve);
            assert!(curve.x.is_null());

            // Invalid options are reported, not panicked on.
            assert!(!anofox_lowess(
                arr(&x1),
                arr(&y),
                0.0,
                3,
                f64::NAN,
                &mut curve,
                &mut err
            ));
            assert_eq!(err.code, ErrorCode::InvalidInput);
        }
    }

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    fn options() -> BayesianOptionsFFI {
        BayesianOptionsFFI {
            fit_intercept: true,
            max_iterations: 300,
            tolerance: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 1e4,
        }
    }

    #[test]
    fn bayesian_fit_and_predict_round_trip() {
        let x1: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let x2: Vec<f64> = (0..30).map(|i| (i as f64 * 0.7).cos()).collect();
        let y: Vec<f64> = (0..30)
            .map(|i| 1.0 + 2.0 * x1[i] - x2[i] + 0.01 * (i as f64).sin())
            .collect();
        let xs = [arr(&x1), arr(&x2)];
        unsafe {
            for ard in [false, true] {
                let mut res = BayesianFitResultFFI::default();
                let mut err = AnofoxError::success();
                assert!(anofox_bayesian_fit(
                    arr(&y),
                    xs.as_ptr(),
                    2,
                    options(),
                    ard,
                    &mut res,
                    &mut err
                ));
                assert_eq!(res.coefficients_len, 2);
                assert_eq!(res.weight_precisions_len, if ard { 2 } else { 0 });
                assert!((*res.coefficients - 2.0).abs() < 0.05);
                anofox_free_bayesian_result(&mut res);
                assert!(res.coefficients.is_null());

                let rows = [0.5, 0.5, f64::NAN, 0.0];
                let (mut yhat, mut lo, mut hi) = ([0.0; 2], [0.0; 2], [0.0; 2]);
                assert!(anofox_bayesian_fit_predict(
                    arr(&y),
                    xs.as_ptr(),
                    2,
                    rows.as_ptr(),
                    2,
                    options(),
                    ard,
                    0.95,
                    yhat.as_mut_ptr(),
                    lo.as_mut_ptr(),
                    hi.as_mut_ptr(),
                    &mut err
                ));
                assert!(lo[0] < yhat[0] && yhat[0] < hi[0]);
                assert!(yhat[1].is_nan());
            }
            // NULL output is rejected, not dereferenced.
            let mut err = AnofoxError::success();
            assert!(!anofox_bayesian_fit(
                arr(&y),
                xs.as_ptr(),
                2,
                options(),
                false,
                std::ptr::null_mut(),
                &mut err
            ));
        }
    }
}
