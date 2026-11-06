//! C FFI for the option pass-throughs of #305 that need an entry point of
//! their own (the options themselves ride on the existing option structs).
//! Mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::{
    fit_gamma_weighted, fit_poisson_weighted, fit_tweedie_weighted, ridge_selected_lambda,
    WeightedGlmFit,
};
use anofox_stats_core::StatsResult;

unsafe fn fail(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
    false
}

/// The λ `anofox_ridge_fit` uses on `(y, x)` with these options: `alpha`, or
/// with `conditioning` upstream's conditioning-aware choice for the rows and
/// columns the fit uses. The fit-predict intervals and the coefficient
/// covariance need it alongside the fit's coefficients.
///
/// # Safety
/// `x` must point to `x_count` valid arrays; `out_lambda` must be valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_ridge_selected_lambda(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: RidgeOptionsFFI,
    out_lambda: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if out_lambda.is_null() || x.is_null() || x_count == 0 {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        }
        let y_vec = y.to_vec();
        let x_vecs: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
            .iter()
            .map(|arr| arr.to_vec())
            .collect();
        let opts = RidgeOptions {
            alpha: options.alpha,
            fit_intercept: options.fit_intercept,
            compute_inference: false,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            lambda_scaling: convert_lambda_scaling_ffi(options.lambda_scaling),
            compute_loo: false,
            conditioning: options.conditioning_pair(),
        };
        match ridge_selected_lambda(&y_vec, &x_vecs, &opts) {
            Ok(lambda) => {
                *out_lambda = lambda;
                true
            }
            Err(e) => fail(out_error, error_to_code(&e), &e.to_string()),
        }
    })
}

// =============================================================================
// Prior weights for Poisson / Tweedie / Gamma (#305)
// =============================================================================

/// Write a weighted GLM fit into the same result structs `anofox_poisson_fit`
/// fills.
unsafe fn write_weighted_glm(
    fit: StatsResult<WeightedGlmFit>,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    let fit = match fit {
        Ok(f) => f,
        Err(e) => return fail(out_error, error_to_code(&e), &e.to_string()),
    };
    let n_coef = fit.coefficients.len();
    let coef_ptr = libc::malloc(n_coef.max(1) * std::mem::size_of::<f64>()) as *mut f64;
    if coef_ptr.is_null() {
        return fail(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate coefficients",
        );
    }
    std::ptr::copy_nonoverlapping(fit.coefficients.as_ptr(), coef_ptr, n_coef);
    let (vcov_ptr, vcov_dim) = match &fit.vcov {
        Some((m, dim)) => {
            let p = alloc_f64(m);
            if p.is_null() {
                (std::ptr::null_mut(), 0)
            } else {
                (p, *dim)
            }
        }
        None => (std::ptr::null_mut(), 0),
    };
    *out_result = GlmFitResultCore {
        coefficients: coef_ptr,
        coefficients_len: n_coef,
        intercept: fit.intercept.unwrap_or(f64::NAN),
        deviance: fit.deviance,
        null_deviance: fit.null_deviance,
        pseudo_r_squared: fit.pseudo_r_squared,
        aic: fit.aic,
        dispersion: fit.dispersion,
        n_observations: fit.n_observations,
        n_features: fit.n_features,
        iterations: fit.iterations as u32,
        converged: fit.converged,
        summary: ModelSummaryFFI::from(&fit.summary),
        vcov: vcov_ptr,
        vcov_dim,
    };
    if !out_inference.is_null() {
        if let Some(inf) = fit.inference {
            alloc_inference_arrays!(inf, out_inference, out_error, {
                libc::free(coef_ptr as *mut libc::c_void);
                if !vcov_ptr.is_null() {
                    libc::free(vcov_ptr as *mut libc::c_void);
                }
                *out_result = GlmFitResultCore::default();
            });
        } else {
            *out_inference = FitResultInference::default();
        }
    }
    true
}

/// `(y, x, w)` of a weighted fit.
type WeightedInputs = (Vec<f64>, Vec<Vec<f64>>, Vec<f64>);

/// `(y, x, w)` as vectors, or `None` when a pointer is NULL.
unsafe fn weighted_inputs(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    w: DataArray,
) -> Option<WeightedInputs> {
    if x.is_null() || x_count == 0 {
        return None;
    }
    let x_vecs = slice::from_raw_parts(x, x_count)
        .iter()
        .map(|arr| arr.to_vec())
        .collect();
    Some((y.to_vec(), x_vecs, w.to_vec()))
}

macro_rules! weighted_glm_entry {
    ($name:ident, $opts_ffi:ty, $convert:ident, $fit:ident) => {
        /// The `*_fit` of the same family with prior weights `w` (#305),
        /// as R's `glm(weights = w)`. A NULL (NaN) weight drops the row.
        ///
        /// # Safety
        /// `x` must point to `x_count` valid arrays of `y.len` values and `w`
        /// must hold `y.len` values; the out pointers must be valid. Free the
        /// result with `anofox_free_glm_result` / `anofox_free_result_inference`.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            y: DataArray,
            x: *const DataArray,
            x_count: usize,
            w: DataArray,
            options: $opts_ffi,
            out_result: *mut GlmFitResultCore,
            out_inference: *mut FitResultInference,
            out_error: *mut AnofoxError,
        ) -> bool {
            ffi_guard(out_error, false, || {
                if !out_error.is_null() {
                    *out_error = AnofoxError::success();
                }
                if !check_confidence_level(options.confidence_level, out_error) {
                    return false;
                }
                if out_result.is_null() {
                    return fail(out_error, ErrorCode::InvalidInput, "out_result is NULL");
                }
                let Some((y_vec, x_vecs, w_vec)) = weighted_inputs(y, x, x_count, w) else {
                    return fail(out_error, ErrorCode::InsufficientData, "x is NULL or empty");
                };
                let opts = $convert(&options);
                write_weighted_glm(
                    $fit(&y_vec, &x_vecs, &w_vec, &opts),
                    out_result,
                    out_inference,
                    out_error,
                )
            })
        }
    };
}

weighted_glm_entry!(
    anofox_poisson_fit_weighted,
    PoissonOptionsFFI,
    poisson_options_from_ffi,
    fit_poisson_weighted
);
weighted_glm_entry!(
    anofox_tweedie_fit_weighted,
    TweedieOptionsFFI,
    tweedie_options_from_ffi,
    fit_tweedie_weighted
);
weighted_glm_entry!(
    anofox_gamma_fit_weighted,
    GammaOptionsFFI,
    gamma_options_from_ffi,
    fit_gamma_weighted
);
