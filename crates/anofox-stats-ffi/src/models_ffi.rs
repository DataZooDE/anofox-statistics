//! C FFI for Bayesian Ridge and ARD regression (#61). Every struct is
//! mirrored in `src/include/anofox_stats_ffi.h`; the layouts are pinned by the
//! tests below and the static_asserts in `src/include/model_struct.cpp`.

use super::*;
use anofox_stats_core::models::{fit_bayesian, fit_predict_bayesian, BayesianOptions};

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
