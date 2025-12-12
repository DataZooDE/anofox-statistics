//! C FFI boundary for anofox-statistics
//!
//! This crate provides C-compatible functions for calling from the C++ DuckDB extension layer.

mod types;

pub use types::*;

use anofox_stats_core::{
    diagnostics::{compute_aic, compute_bic, compute_residuals, compute_vif, jarque_bera},
    models::{
        compute_aid, compute_aid_anomalies, fit_alm, fit_binomial, fit_bls, fit_elasticnet,
        fit_negbinomial, fit_ols, fit_poisson, fit_ridge, fit_rls, fit_tweedie, fit_wls, predict,
        RlsOptions,
    },
    AidOptions, AlmDistribution, AlmLoss, AlmOptions, BinomialLink, BinomialOptions, BlsOptions,
    ElasticNetOptions, NegBinomialOptions, OlsOptions, OutlierMethod, PoissonLink, PoissonOptions,
    RidgeOptions, StatsError, TweedieOptions, WlsOptions,
};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::slice;

/// Convert StatsError to ErrorCode
fn error_to_code(err: &StatsError) -> ErrorCode {
    match err {
        StatsError::InvalidAlpha(_) => ErrorCode::InvalidAlpha,
        StatsError::InvalidL1Ratio(_) => ErrorCode::InvalidL1Ratio,
        StatsError::InsufficientData { .. } => ErrorCode::InsufficientData,
        StatsError::InsufficientDataMsg(_) => ErrorCode::InsufficientData,
        StatsError::NoValidData => ErrorCode::NoValidData,
        StatsError::DimensionMismatch { .. } => ErrorCode::DimensionMismatch,
        StatsError::DimensionMismatchMsg(_) => ErrorCode::DimensionMismatch,
        StatsError::EmptyInput { .. } => ErrorCode::InvalidInput,
        StatsError::InvalidInput(_) => ErrorCode::InvalidInput,
        StatsError::InvalidValue { .. } => ErrorCode::InvalidInput,
        StatsError::SingularMatrix => ErrorCode::SingularMatrix,
        StatsError::CholeskyFailed | StatsError::QrFailed => ErrorCode::SingularMatrix,
        StatsError::ConvergenceFailure { .. } => ErrorCode::ConvergenceFailure,
        StatsError::AllocationFailure => ErrorCode::AllocationFailure,
        StatsError::SerializationError(_) => ErrorCode::SerializationError,
        StatsError::RegressError(_) => ErrorCode::InternalError,
    }
}

/// Fit an OLS regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_core` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_ols_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: OlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    // Initialize error
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    // Validate inputs
    if out_core.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_core is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert options
    let opts = OlsOptions {
        fit_intercept: options.fit_intercept,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    // Call the core function with panic catching
    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_ols(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in OLS fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_core) = FitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                r_squared: result.core.r_squared,
                adj_r_squared: result.core.adj_r_squared,
                residual_std_error: result.core.residual_std_error,
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
            };

            // Fill inference results if requested and available
            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();

                    // Allocate arrays
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0
                        && (std_err_ptr.is_null()
                            || t_val_ptr.is_null()
                            || p_val_ptr.is_null()
                            || ci_lo_ptr.is_null()
                            || ci_hi_ptr.is_null())
                    {
                        // Free any allocated memory
                        if !std_err_ptr.is_null() {
                            libc::free(std_err_ptr as *mut libc::c_void);
                        }
                        if !t_val_ptr.is_null() {
                            libc::free(t_val_ptr as *mut libc::c_void);
                        }
                        if !p_val_ptr.is_null() {
                            libc::free(p_val_ptr as *mut libc::c_void);
                        }
                        if !ci_lo_ptr.is_null() {
                            libc::free(ci_lo_ptr as *mut libc::c_void);
                        }
                        if !ci_hi_ptr.is_null() {
                            libc::free(ci_hi_ptr as *mut libc::c_void);
                        }
                        libc::free(coef_ptr as *mut libc::c_void);

                        if !out_error.is_null() {
                            (*out_error).set(
                                ErrorCode::AllocationFailure,
                                "Failed to allocate inference arrays",
                            );
                        }
                        return false;
                    }

                    // Copy data
                    std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.t_values.as_ptr(), t_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: inf.f_statistic.unwrap_or(f64::NAN),
                        f_pvalue: inf.f_pvalue.unwrap_or(f64::NAN),
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by anofox_ols_fit for core results
///
/// # Safety
/// `result` must be a pointer to a FitResultCore previously filled by anofox_ols_fit
#[no_mangle]
pub unsafe extern "C" fn anofox_free_result_core(result: *mut FitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
        (*result).coefficients = std::ptr::null_mut();
    }
}

/// Free memory allocated by anofox_ols_fit for inference results
///
/// # Safety
/// `result` must be a pointer to a FitResultInference previously filled by anofox_ols_fit
#[no_mangle]
pub unsafe extern "C" fn anofox_free_result_inference(result: *mut FitResultInference) {
    if result.is_null() {
        return;
    }
    if !(*result).std_errors.is_null() {
        libc::free((*result).std_errors as *mut libc::c_void);
        (*result).std_errors = std::ptr::null_mut();
    }
    if !(*result).t_values.is_null() {
        libc::free((*result).t_values as *mut libc::c_void);
        (*result).t_values = std::ptr::null_mut();
    }
    if !(*result).p_values.is_null() {
        libc::free((*result).p_values as *mut libc::c_void);
        (*result).p_values = std::ptr::null_mut();
    }
    if !(*result).ci_lower.is_null() {
        libc::free((*result).ci_lower as *mut libc::c_void);
        (*result).ci_lower = std::ptr::null_mut();
    }
    if !(*result).ci_upper.is_null() {
        libc::free((*result).ci_upper as *mut libc::c_void);
        (*result).ci_upper = std::ptr::null_mut();
    }
}

/// Fit a Ridge regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_core` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_ridge_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: RidgeOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    // Initialize error
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    // Validate inputs
    if out_core.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_core is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert options
    let opts = RidgeOptions {
        alpha: options.alpha,
        fit_intercept: options.fit_intercept,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    // Call the core function with panic catching for regress-rs issues
    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_ridge(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::InternalError,
                    "Internal panic in Ridge fit (possibly perfect fit or numerical issue)",
                );
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_core) = FitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                r_squared: result.core.r_squared,
                adj_r_squared: result.core.adj_r_squared,
                residual_std_error: result.core.residual_std_error,
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
            };

            // Fill inference results if requested and available
            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();

                    // Allocate arrays
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0
                        && (std_err_ptr.is_null()
                            || t_val_ptr.is_null()
                            || p_val_ptr.is_null()
                            || ci_lo_ptr.is_null()
                            || ci_hi_ptr.is_null())
                    {
                        // Free any allocated memory
                        if !std_err_ptr.is_null() {
                            libc::free(std_err_ptr as *mut libc::c_void);
                        }
                        if !t_val_ptr.is_null() {
                            libc::free(t_val_ptr as *mut libc::c_void);
                        }
                        if !p_val_ptr.is_null() {
                            libc::free(p_val_ptr as *mut libc::c_void);
                        }
                        if !ci_lo_ptr.is_null() {
                            libc::free(ci_lo_ptr as *mut libc::c_void);
                        }
                        if !ci_hi_ptr.is_null() {
                            libc::free(ci_hi_ptr as *mut libc::c_void);
                        }
                        libc::free(coef_ptr as *mut libc::c_void);

                        if !out_error.is_null() {
                            (*out_error).set(
                                ErrorCode::AllocationFailure,
                                "Failed to allocate inference arrays",
                            );
                        }
                        return false;
                    }

                    // Copy data
                    std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.t_values.as_ptr(), t_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: inf.f_statistic.unwrap_or(f64::NAN),
                        f_pvalue: inf.f_pvalue.unwrap_or(f64::NAN),
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit an Elastic Net regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_core` must be a valid pointer
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_elasticnet_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: ElasticNetOptionsFFI,
    out_core: *mut FitResultCore,
    out_error: *mut AnofoxError,
) -> bool {
    // Initialize error
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    // Validate inputs
    if out_core.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_core is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert options
    let opts = ElasticNetOptions {
        alpha: options.alpha,
        l1_ratio: options.l1_ratio,
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
    };

    // Call the core function with panic catching
    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_elasticnet(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::InternalError,
                    "Internal panic in Elastic Net fit",
                );
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_core) = FitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                r_squared: result.core.r_squared,
                adj_r_squared: result.core.adj_r_squared,
                residual_std_error: result.core.residual_std_error,
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit a Weighted Least Squares regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `weights` must be a valid DataArray with same length as y
/// - `out_core` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_wls_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    weights: DataArray,
    options: WlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    // Initialize error
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    // Validate inputs
    if out_core.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_core is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert weights to Vec
    let weights_vec = weights.to_vec();

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert options
    let opts = WlsOptions {
        fit_intercept: options.fit_intercept,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    // Call the core function with panic catching
    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_wls(&y_vec, &x_vecs, &weights_vec, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in WLS fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_core) = FitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                r_squared: result.core.r_squared,
                adj_r_squared: result.core.adj_r_squared,
                residual_std_error: result.core.residual_std_error,
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
            };

            // Fill inference results if requested and available
            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();

                    // Allocate arrays
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0
                        && (std_err_ptr.is_null()
                            || t_val_ptr.is_null()
                            || p_val_ptr.is_null()
                            || ci_lo_ptr.is_null()
                            || ci_hi_ptr.is_null())
                    {
                        // Free any allocated memory
                        if !std_err_ptr.is_null() {
                            libc::free(std_err_ptr as *mut libc::c_void);
                        }
                        if !t_val_ptr.is_null() {
                            libc::free(t_val_ptr as *mut libc::c_void);
                        }
                        if !p_val_ptr.is_null() {
                            libc::free(p_val_ptr as *mut libc::c_void);
                        }
                        if !ci_lo_ptr.is_null() {
                            libc::free(ci_lo_ptr as *mut libc::c_void);
                        }
                        if !ci_hi_ptr.is_null() {
                            libc::free(ci_hi_ptr as *mut libc::c_void);
                        }
                        libc::free(coef_ptr as *mut libc::c_void);

                        if !out_error.is_null() {
                            (*out_error).set(
                                ErrorCode::AllocationFailure,
                                "Failed to allocate inference arrays",
                            );
                        }
                        return false;
                    }

                    // Copy data
                    std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.t_values.as_ptr(), t_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: inf.f_statistic.unwrap_or(f64::NAN),
                        f_pvalue: inf.f_pvalue.unwrap_or(f64::NAN),
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Make predictions using fitted model coefficients
///
/// # Safety
/// - `x` must point to `x_count` valid DataArray structs
/// - `coefficients` must be a valid array of `coefficients_len` doubles
/// - `out_predictions` must be a valid pointer
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_predict(
    x: *const DataArray,
    x_count: usize,
    coefficients: *const f64,
    coefficients_len: usize,
    intercept: f64, // Use NaN if no intercept
    out_predictions: *mut *mut f64,
    out_predictions_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    // Initialize error
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    // Validate inputs
    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    if coefficients.is_null() || coefficients_len == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "coefficients is NULL or empty");
        }
        return false;
    }

    if out_predictions.is_null() || out_predictions_len.is_null() {
        if !out_error.is_null() {
            (*out_error).set(
                ErrorCode::InvalidInput,
                "out_predictions or out_predictions_len is NULL",
            );
        }
        return false;
    }

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert coefficients to slice
    let coef_slice = slice::from_raw_parts(coefficients, coefficients_len);
    let coef_vec: Vec<f64> = coef_slice.to_vec();

    // Handle intercept (NaN means no intercept)
    let intercept_opt = if intercept.is_nan() {
        None
    } else {
        Some(intercept)
    };

    // Call the core function
    let result = predict(&x_vecs, &coef_vec, intercept_opt);

    match result {
        Ok(predictions) => {
            let n = predictions.len();

            // Allocate output array
            let pred_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
            if pred_ptr.is_null() && n > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate predictions",
                    );
                }
                return false;
            }

            // Copy predictions
            std::ptr::copy_nonoverlapping(predictions.as_ptr(), pred_ptr, n);

            *out_predictions = pred_ptr;
            *out_predictions_len = n;

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by anofox_predict
///
/// # Safety
/// `predictions` must be a pointer previously returned by anofox_predict
#[no_mangle]
pub unsafe extern "C" fn anofox_free_predictions(predictions: *mut f64) {
    if !predictions.is_null() {
        libc::free(predictions as *mut libc::c_void);
    }
}

/// Get library version string
#[no_mangle]
pub extern "C" fn anofox_version() -> *const libc::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const libc::c_char
}

// ============================================================================
// Diagnostics FFI Functions
// ============================================================================

/// Compute VIF (Variance Inflation Factor) for each feature
///
/// # Safety
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_vif` must be a valid pointer to receive results
/// - `out_vif_len` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_compute_vif(
    x: *const DataArray,
    x_count: usize,
    out_vif: *mut *mut f64,
    out_vif_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    if out_vif.is_null() || out_vif_len.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_vif or out_vif_len is NULL");
        }
        return false;
    }

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    match compute_vif(&x_vecs) {
        Ok(vif_values) => {
            let n = vif_values.len();
            let ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
            if ptr.is_null() && n > 0 {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::AllocationFailure, "Failed to allocate VIF array");
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(vif_values.as_ptr(), ptr, n);
            *out_vif = ptr;
            *out_vif_len = n;
            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by anofox_compute_vif
///
/// # Safety
/// - `vif` must have been allocated by `anofox_compute_vif`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_vif(vif: *mut f64) {
    if !vif.is_null() {
        libc::free(vif as *mut libc::c_void);
    }
}

/// Result structure for residuals computation
#[repr(C)]
pub struct ResidualsResult {
    pub raw: *mut f64,
    pub standardized: *mut f64,
    pub studentized: *mut f64,
    pub leverage: *mut f64,
    pub len: usize,
    pub has_standardized: bool,
    pub has_studentized: bool,
    pub has_leverage: bool,
}

impl Default for ResidualsResult {
    fn default() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            standardized: std::ptr::null_mut(),
            studentized: std::ptr::null_mut(),
            leverage: std::ptr::null_mut(),
            len: 0,
            has_standardized: false,
            has_studentized: false,
            has_leverage: false,
        }
    }
}

/// Compute residuals from y and predictions
///
/// # Safety
/// - All array parameters must be valid
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_compute_residuals(
    y: DataArray,
    y_hat: DataArray,
    x: *const DataArray,
    x_count: usize,
    residual_std_error: f64, // Use NaN if not available
    include_studentized: bool,
    out_result: *mut ResidualsResult,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let y_hat_vec = y_hat.to_vec();

    // Convert x if provided
    let x_vecs: Option<Vec<Vec<f64>>> = if !x.is_null() && x_count > 0 {
        let x_arrays = slice::from_raw_parts(x, x_count);
        Some(x_arrays.iter().map(|arr| arr.to_vec()).collect())
    } else {
        None
    };

    let x_ref = x_vecs.as_deref();
    let rse = if residual_std_error.is_nan() {
        None
    } else {
        Some(residual_std_error)
    };

    match compute_residuals(&y_vec, &y_hat_vec, x_ref, rse, include_studentized) {
        Ok(result) => {
            let n = result.raw.len();

            // Allocate and copy raw residuals
            let raw_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
            if raw_ptr.is_null() && n > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate raw residuals",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.raw.as_ptr(), raw_ptr, n);

            // Handle optional arrays
            let (std_ptr, has_std) = if let Some(ref std_resid) = result.standardized {
                let ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                if !ptr.is_null() {
                    std::ptr::copy_nonoverlapping(std_resid.as_ptr(), ptr, n);
                }
                (ptr, true)
            } else {
                (std::ptr::null_mut(), false)
            };

            let (stud_ptr, has_stud) = if let Some(ref stud_resid) = result.studentized {
                let ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                if !ptr.is_null() {
                    std::ptr::copy_nonoverlapping(stud_resid.as_ptr(), ptr, n);
                }
                (ptr, true)
            } else {
                (std::ptr::null_mut(), false)
            };

            let (lev_ptr, has_lev) = if let Some(ref leverage) = result.leverage {
                let ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                if !ptr.is_null() {
                    std::ptr::copy_nonoverlapping(leverage.as_ptr(), ptr, n);
                }
                (ptr, true)
            } else {
                (std::ptr::null_mut(), false)
            };

            *out_result = ResidualsResult {
                raw: raw_ptr,
                standardized: std_ptr,
                studentized: stud_ptr,
                leverage: lev_ptr,
                len: n,
                has_standardized: has_std,
                has_studentized: has_stud,
                has_leverage: has_lev,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by anofox_compute_residuals
///
/// # Safety
/// - `result` must have been allocated by `anofox_compute_residuals`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_residuals(result: *mut ResidualsResult) {
    if result.is_null() {
        return;
    }
    if !(*result).raw.is_null() {
        libc::free((*result).raw as *mut libc::c_void);
    }
    if !(*result).standardized.is_null() {
        libc::free((*result).standardized as *mut libc::c_void);
    }
    if !(*result).studentized.is_null() {
        libc::free((*result).studentized as *mut libc::c_void);
    }
    if !(*result).leverage.is_null() {
        libc::free((*result).leverage as *mut libc::c_void);
    }
}

/// Compute AIC (Akaike Information Criterion)
///
/// # Arguments
/// * `rss` - Residual sum of squares
/// * `n` - Number of observations
/// * `k` - Number of parameters (including intercept)
/// * `out_aic` - Output AIC value
/// * `out_error` - Error information
///
/// # Safety
/// - `out_aic` must be a valid pointer
/// - `out_error` must be a valid pointer or null
#[no_mangle]
pub unsafe extern "C" fn anofox_compute_aic(
    rss: f64,
    n: usize,
    k: usize,
    out_aic: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_aic.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_aic is NULL");
        }
        return false;
    }

    match compute_aic(rss, n, k) {
        Ok(aic) => {
            *out_aic = aic;
            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Compute BIC (Bayesian Information Criterion)
///
/// # Arguments
/// * `rss` - Residual sum of squares
/// * `n` - Number of observations
/// * `k` - Number of parameters (including intercept)
/// * `out_bic` - Output BIC value
/// * `out_error` - Error information
///
/// # Safety
/// - `out_bic` must be a valid pointer
/// - `out_error` must be a valid pointer or null
#[no_mangle]
pub unsafe extern "C" fn anofox_compute_bic(
    rss: f64,
    n: usize,
    k: usize,
    out_bic: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_bic.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_bic is NULL");
        }
        return false;
    }

    match compute_bic(rss, n, k) {
        Ok(bic) => {
            *out_bic = bic;
            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

// ============================================================================
// RLS (Recursive Least Squares) FFI Functions
// ============================================================================

/// RLS options for FFI
#[repr(C)]
pub struct RlsOptionsFFI {
    /// Forgetting factor (λ), typically 0.95-1.0
    pub forgetting_factor: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Initial value for diagonal of P matrix
    pub initial_p_diagonal: f64,
}

impl Default for RlsOptionsFFI {
    fn default() -> Self {
        Self {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0,
        }
    }
}

/// Fit an RLS (Recursive Least Squares) regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_core` must be a valid pointer
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_rls_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: RlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_core.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_core is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert x arrays to Vec<Vec<f64>>
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert options
    let opts = RlsOptions {
        forgetting_factor: options.forgetting_factor,
        fit_intercept: options.fit_intercept,
        initial_p_diagonal: options.initial_p_diagonal,
    };

    // Call the core function
    let result = fit_rls(&y_vec, &x_vecs, &opts);

    match result {
        Ok(state) => {
            let coefficients = state.get_coefficients();
            let n_coef = coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_core) = FitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: state.get_intercept().unwrap_or(f64::NAN),
                r_squared: f64::NAN, // RLS doesn't compute R² during fitting
                adj_r_squared: f64::NAN,
                residual_std_error: f64::NAN,
                n_observations: state.n_observations,
                n_features: state.n_features,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

// ============================================================================
// Jarque-Bera Test FFI Functions
// ============================================================================

/// Result structure for Jarque-Bera test
#[repr(C)]
pub struct JarqueBeraResultFFI {
    /// JB test statistic
    pub statistic: f64,
    /// p-value for the test
    pub p_value: f64,
    /// Sample skewness
    pub skewness: f64,
    /// Sample kurtosis (excess)
    pub kurtosis: f64,
    /// Number of observations
    pub n: usize,
}

impl Default for JarqueBeraResultFFI {
    fn default() -> Self {
        Self {
            statistic: f64::NAN,
            p_value: f64::NAN,
            skewness: f64::NAN,
            kurtosis: f64::NAN,
            n: 0,
        }
    }
}

/// Compute the Jarque-Bera test for normality
///
/// # Safety
/// - `data` must be a valid DataArray
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_jarque_bera(
    data: DataArray,
    out_result: *mut JarqueBeraResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    let data_vec = data.to_vec();

    match jarque_bera(&data_vec) {
        Ok(result) => {
            *out_result = JarqueBeraResultFFI {
                statistic: result.statistic,
                p_value: result.p_value,
                skewness: result.skewness,
                kurtosis: result.kurtosis,
                n: result.n,
            };
            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

// ============================================================================
// Prediction Interval FFI Functions
// ============================================================================

/// Get the critical value from the t-distribution for a given confidence level and degrees of freedom
///
/// # Arguments
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% CI)
/// * `df` - Degrees of freedom (n - p - 1 for regression)
///
/// # Returns
/// The t-critical value, or NaN if invalid inputs
#[no_mangle]
pub extern "C" fn anofox_t_critical(confidence_level: f64, df: usize) -> f64 {
    if df == 0 || confidence_level <= 0.0 || confidence_level >= 1.0 {
        return f64::NAN;
    }

    // Create t-distribution with given degrees of freedom
    let t_dist = match StudentsT::new(0.0, 1.0, df as f64) {
        Ok(dist) => dist,
        Err(_) => return f64::NAN,
    };

    // Two-tailed critical value: need quantile at (1 + confidence_level) / 2
    let alpha = (1.0 + confidence_level) / 2.0;
    t_dist.inverse_cdf(alpha)
}

/// Prediction result with confidence interval
#[repr(C)]
pub struct PredictionResult {
    /// Predicted value
    pub yhat: f64,
    /// Lower bound of prediction interval
    pub yhat_lower: f64,
    /// Upper bound of prediction interval
    pub yhat_upper: f64,
}

impl Default for PredictionResult {
    fn default() -> Self {
        Self {
            yhat: f64::NAN,
            yhat_lower: f64::NAN,
            yhat_upper: f64::NAN,
        }
    }
}

/// Compute prediction with confidence interval for a single new observation
///
/// For OLS, the prediction interval is: yhat ± t_critical * se_pred
/// where se_pred = residual_std_error * sqrt(1 + 1/n + distance_from_mean)
///
/// For simplicity, this function uses a simplified formula assuming average leverage.
///
/// # Safety
/// - `coefficients` must point to `coefficients_len` valid doubles
/// - `x_new` must point to `x_len` valid doubles (same length as coefficients)
/// - `out_result` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_predict_with_interval(
    coefficients: *const f64,
    coefficients_len: usize,
    intercept: f64, // NaN if no intercept
    x_new: *const f64,
    x_len: usize,
    residual_std_error: f64,
    n_observations: usize,
    confidence_level: f64,
    out_result: *mut PredictionResult,
) -> bool {
    if out_result.is_null() {
        return false;
    }

    // Initialize with NaN
    *out_result = PredictionResult::default();

    // Validate inputs
    if coefficients.is_null() || coefficients_len == 0 {
        return false;
    }
    if x_new.is_null() || x_len != coefficients_len {
        return false;
    }

    // Compute prediction: yhat = intercept + sum(coefficients[i] * x_new[i])
    let coef_slice = slice::from_raw_parts(coefficients, coefficients_len);
    let x_slice = slice::from_raw_parts(x_new, x_len);

    let intercept_val = if intercept.is_nan() { 0.0 } else { intercept };
    let mut yhat = intercept_val;
    for (coef, x_val) in coef_slice.iter().zip(x_slice.iter()) {
        yhat += coef * x_val;
    }

    (*out_result).yhat = yhat;

    // Compute prediction interval if we have valid std error
    if residual_std_error.is_nan()
        || residual_std_error <= 0.0
        || n_observations <= coefficients_len + 1
    {
        // No valid interval, just return yhat with same bounds
        (*out_result).yhat_lower = yhat;
        (*out_result).yhat_upper = yhat;
        return true;
    }

    // Degrees of freedom
    let has_intercept = !intercept.is_nan();
    let df = if has_intercept {
        n_observations.saturating_sub(coefficients_len + 1)
    } else {
        n_observations.saturating_sub(coefficients_len)
    };

    if df == 0 {
        (*out_result).yhat_lower = yhat;
        (*out_result).yhat_upper = yhat;
        return true;
    }

    // Get t-critical value
    let t_crit = anofox_t_critical(confidence_level, df);
    if t_crit.is_nan() {
        (*out_result).yhat_lower = yhat;
        (*out_result).yhat_upper = yhat;
        return true;
    }

    // Simplified prediction interval formula: yhat ± t * se * sqrt(1 + 1/n)
    // This ignores the leverage term for simplicity (assumes average leverage)
    let n = n_observations as f64;
    let se_pred = residual_std_error * (1.0 + 1.0 / n).sqrt();
    let margin = t_crit * se_pred;

    (*out_result).yhat_lower = yhat - margin;
    (*out_result).yhat_upper = yhat + margin;

    true
}

// =============================================================================
// GLM (Generalized Linear Models) FFI Functions
// =============================================================================

/// Fit a Poisson regression model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_poisson_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: PoissonOptionsFFI,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert FFI options to core options
    let link = match options.link {
        PoissonLinkFFI::Log => PoissonLink::Log,
        PoissonLinkFFI::Identity => PoissonLink::Identity,
        PoissonLinkFFI::Sqrt => PoissonLink::Sqrt,
    };

    let opts = PoissonOptions {
        fit_intercept: options.fit_intercept,
        link,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_poisson(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in Poisson fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.core.coefficients.len();
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_result) = GlmFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                deviance: result.core.residual_deviance,
                null_deviance: result.core.null_deviance,
                pseudo_r_squared: result.core.pseudo_r_squared,
                aic: result.core.aic,
                dispersion: result.core.dispersion.unwrap_or(f64::NAN),
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
                iterations: result.core.iterations,
            };

            // Fill inference if available
            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0 && !std_err_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.z_values.as_ptr(), t_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);
                    }

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: f64::NAN,
                        f_pvalue: f64::NAN,
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit a Binomial (logistic) regression model
///
/// # Safety
/// - `y` must be a valid DataArray with binary (0/1) values
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_binomial_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: BinomialOptionsFFI,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    let link = match options.link {
        BinomialLinkFFI::Logit => BinomialLink::Logit,
        BinomialLinkFFI::Probit => BinomialLink::Probit,
        BinomialLinkFFI::Cloglog => BinomialLink::Cloglog,
    };

    let opts = BinomialOptions {
        fit_intercept: options.fit_intercept,
        link,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_binomial(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in Binomial fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.core.coefficients.len();
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_result) = GlmFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                deviance: result.core.residual_deviance,
                null_deviance: result.core.null_deviance,
                pseudo_r_squared: result.core.pseudo_r_squared,
                aic: result.core.aic,
                dispersion: result.core.dispersion.unwrap_or(f64::NAN),
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
                iterations: result.core.iterations,
            };

            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0 && !std_err_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.z_values.as_ptr(), t_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);
                    }

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: f64::NAN,
                        f_pvalue: f64::NAN,
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit a Negative Binomial regression model
///
/// # Safety
/// - `y` must be a valid DataArray with count values
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_negbinomial_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: NegBinomialOptionsFFI,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    let opts = NegBinomialOptions {
        fit_intercept: options.fit_intercept,
        alpha: None, // Let the algorithm estimate alpha
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_negbinomial(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::InternalError,
                    "Internal panic in NegBinomial fit",
                );
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.core.coefficients.len();
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_result) = GlmFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                deviance: result.core.residual_deviance,
                null_deviance: result.core.null_deviance,
                pseudo_r_squared: result.core.pseudo_r_squared,
                aic: result.core.aic,
                dispersion: result.core.dispersion.unwrap_or(f64::NAN),
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
                iterations: result.core.iterations,
            };

            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0 && !std_err_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.z_values.as_ptr(), t_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);
                    }

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: f64::NAN,
                        f_pvalue: f64::NAN,
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit a Tweedie regression model
///
/// # Safety
/// - `y` must be a valid DataArray with non-negative values
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_tweedie_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: TweedieOptionsFFI,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    let opts = TweedieOptions {
        fit_intercept: options.fit_intercept,
        power: options.power,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_tweedie(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in Tweedie fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.core.coefficients.len();
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_result) = GlmFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                deviance: result.core.residual_deviance,
                null_deviance: result.core.null_deviance,
                pseudo_r_squared: result.core.pseudo_r_squared,
                aic: result.core.aic,
                dispersion: result.core.dispersion.unwrap_or(f64::NAN),
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
                iterations: result.core.iterations,
            };

            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0 && !std_err_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.z_values.as_ptr(), t_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);
                    }

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: f64::NAN,
                        f_pvalue: f64::NAN,
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by GLM fit functions
///
/// # Safety
/// - `result` must be NULL or a valid pointer to a GlmFitResultCore
/// - Must only be called once per result (double-free is undefined behavior)
#[no_mangle]
pub unsafe extern "C" fn anofox_free_glm_result(result: *mut GlmFitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
        (*result).coefficients = std::ptr::null_mut();
    }
}

// =============================================================================
// ALM (Augmented Linear Models) FFI Functions
// =============================================================================

/// Convert FFI distribution code to core AlmDistribution
fn convert_alm_distribution(dist: AlmDistributionFFI) -> AlmDistribution {
    match dist {
        AlmDistributionFFI::Normal => AlmDistribution::Normal,
        AlmDistributionFFI::Laplace => AlmDistribution::Laplace,
        AlmDistributionFFI::StudentT => AlmDistribution::StudentT,
        AlmDistributionFFI::Logistic => AlmDistribution::Logistic,
        AlmDistributionFFI::AsymmetricLaplace => AlmDistribution::AsymmetricLaplace,
        AlmDistributionFFI::GeneralisedNormal => AlmDistribution::GeneralisedNormal,
        AlmDistributionFFI::S => AlmDistribution::S,
        AlmDistributionFFI::LogNormal => AlmDistribution::LogNormal,
        AlmDistributionFFI::LogLaplace => AlmDistribution::LogLaplace,
        AlmDistributionFFI::LogS => AlmDistribution::LogS,
        AlmDistributionFFI::LogGeneralisedNormal => AlmDistribution::LogGeneralisedNormal,
        AlmDistributionFFI::FoldedNormal => AlmDistribution::FoldedNormal,
        AlmDistributionFFI::RectifiedNormal => AlmDistribution::RectifiedNormal,
        AlmDistributionFFI::BoxCoxNormal => AlmDistribution::BoxCoxNormal,
        AlmDistributionFFI::Gamma => AlmDistribution::Gamma,
        AlmDistributionFFI::InverseGaussian => AlmDistribution::InverseGaussian,
        AlmDistributionFFI::Exponential => AlmDistribution::Exponential,
        AlmDistributionFFI::Beta => AlmDistribution::Beta,
        AlmDistributionFFI::LogitNormal => AlmDistribution::LogitNormal,
        AlmDistributionFFI::Poisson => AlmDistribution::Poisson,
        AlmDistributionFFI::NegativeBinomial => AlmDistribution::NegativeBinomial,
        AlmDistributionFFI::Binomial => AlmDistribution::Binomial,
        AlmDistributionFFI::Geometric => AlmDistribution::Geometric,
        AlmDistributionFFI::CumulativeLogistic => AlmDistribution::CumulativeLogistic,
        AlmDistributionFFI::CumulativeNormal => AlmDistribution::CumulativeNormal,
    }
}

/// Convert FFI loss code to core AlmLoss
fn convert_alm_loss(loss: AlmLossFFI) -> AlmLoss {
    match loss {
        AlmLossFFI::Likelihood => AlmLoss::Likelihood,
        AlmLossFFI::MSE => AlmLoss::MSE,
        AlmLossFFI::MAE => AlmLoss::MAE,
        AlmLossFFI::HAM => AlmLoss::HAM,
        AlmLossFFI::ROLE => AlmLoss::ROLE,
    }
}

/// Fit an Augmented Linear Model (ALM)
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_inference` can be NULL if not needed
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_alm_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: AlmOptionsFFI,
    out_result: *mut AlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    let opts = AlmOptions {
        fit_intercept: options.fit_intercept,
        distribution: convert_alm_distribution(options.distribution),
        loss: convert_alm_loss(options.loss),
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        quantile: options.quantile,
        role_trim: options.role_trim,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_alm(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in ALM fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.core.coefficients.len();
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);

            (*out_result) = AlmFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.core.intercept.unwrap_or(f64::NAN),
                log_likelihood: result.core.log_likelihood,
                aic: result.core.aic,
                bic: result.core.bic,
                scale: result.core.scale,
                n_observations: result.core.n_observations,
                n_features: result.core.n_features,
                iterations: result.core.iterations,
            };

            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.standard_errors.len();
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0 && !std_err_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(inf.standard_errors.as_ptr(), std_err_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.t_values.as_ptr(), t_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.conf_int_lower.as_ptr(), ci_lo_ptr, n);
                        std::ptr::copy_nonoverlapping(inf.conf_int_upper.as_ptr(), ci_hi_ptr, n);
                    }

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: opts.confidence_level,
                        f_statistic: f64::NAN,
                        f_pvalue: f64::NAN,
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by ALM fit function
///
/// # Safety
/// - `result` must be NULL or a valid pointer to an AlmFitResultCore
/// - Must only be called once per result (double-free is undefined behavior)
#[no_mangle]
pub unsafe extern "C" fn anofox_free_alm_result(result: *mut AlmFitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
        (*result).coefficients = std::ptr::null_mut();
    }
}

// =============================================================================
// BLS (Bounded Least Squares) FFI Functions
// =============================================================================

/// Fit a Bounded Least Squares / NNLS model
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `options.lower_bounds` must be NULL or point to `options.lower_bounds_len` valid f64 values
/// - `options.upper_bounds` must be NULL or point to `options.upper_bounds_len` valid f64 values
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_bls_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: BlsOptionsFFI,
    out_result: *mut BlsFitResultCore,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    if x.is_null() || x_count == 0 {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "x is NULL or empty");
        }
        return false;
    }

    let y_vec = y.to_vec();
    let x_arrays = slice::from_raw_parts(x, x_count);
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    // Convert bounds
    let lower_bounds = if options.lower_bounds.is_null() || options.lower_bounds_len == 0 {
        None
    } else {
        Some(slice::from_raw_parts(options.lower_bounds, options.lower_bounds_len).to_vec())
    };

    let upper_bounds = if options.upper_bounds.is_null() || options.upper_bounds_len == 0 {
        None
    } else {
        Some(slice::from_raw_parts(options.upper_bounds, options.upper_bounds_len).to_vec())
    };

    let opts = BlsOptions {
        fit_intercept: options.fit_intercept,
        lower_bounds,
        upper_bounds,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
    };

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_bls(&y_vec, &x_vecs, &opts)
    }));

    let fit_result = match fit_result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in BLS fit");
            }
            return false;
        }
    };

    match fit_result {
        Ok(result) => {
            let n_coef = result.coefficients.len();

            // Allocate coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate coefficients",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(result.coefficients.as_ptr(), coef_ptr, n_coef);

            // Allocate bound flags
            let lower_ptr = libc::malloc(n_coef * std::mem::size_of::<bool>()) as *mut bool;
            let upper_ptr = libc::malloc(n_coef * std::mem::size_of::<bool>()) as *mut bool;

            if n_coef > 0 && !lower_ptr.is_null() && !upper_ptr.is_null() {
                std::ptr::copy_nonoverlapping(result.at_lower_bound.as_ptr(), lower_ptr, n_coef);
                std::ptr::copy_nonoverlapping(result.at_upper_bound.as_ptr(), upper_ptr, n_coef);
            }

            (*out_result) = BlsFitResultCore {
                coefficients: coef_ptr,
                coefficients_len: n_coef,
                intercept: result.intercept.unwrap_or(f64::NAN),
                ssr: result.ssr,
                r_squared: result.r_squared,
                n_observations: result.n_observations,
                n_features: result.n_features,
                n_active_constraints: result.n_active_constraints,
                at_lower_bound: lower_ptr,
                at_upper_bound: upper_ptr,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Fit NNLS (Non-Negative Least Squares) - convenience function
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `x` must point to `x_count` valid DataArray structs
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_nnls_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    out_result: *mut BlsFitResultCore,
    out_error: *mut AnofoxError,
) -> bool {
    // Use BLS with default NNLS options (all lower bounds = 0)
    let options = BlsOptionsFFI {
        fit_intercept: false,
        lower_bounds: std::ptr::null(),
        lower_bounds_len: 0,
        upper_bounds: std::ptr::null(),
        upper_bounds_len: 0,
        max_iterations: 1000,
        tolerance: 1e-10,
    };
    anofox_bls_fit(y, x, x_count, options, out_result, out_error)
}

/// Free memory allocated by BLS fit function
///
/// # Safety
/// - `result` must be NULL or a valid pointer to a BlsFitResultCore
/// - Must only be called once per result (double-free is undefined behavior)
#[no_mangle]
pub unsafe extern "C" fn anofox_free_bls_result(result: *mut BlsFitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
        (*result).coefficients = std::ptr::null_mut();
    }
    if !(*result).at_lower_bound.is_null() {
        libc::free((*result).at_lower_bound as *mut libc::c_void);
        (*result).at_lower_bound = std::ptr::null_mut();
    }
    if !(*result).at_upper_bound.is_null() {
        libc::free((*result).at_upper_bound as *mut libc::c_void);
        (*result).at_upper_bound = std::ptr::null_mut();
    }
}

// =============================================================================
// AID (Automatic Identification of Demand) Functions
// =============================================================================

/// Compute AID (Automatic Identification of Demand) classification
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_aid(
    y: DataArray,
    options: AidOptionsFFI,
    out_result: *mut AidResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert options
    let opts = AidOptions {
        intermittent_threshold: options.intermittent_threshold,
        outlier_method: match options.outlier_method {
            OutlierMethodFFI::ZScore => OutlierMethod::ZScore,
            OutlierMethodFFI::Iqr => OutlierMethod::Iqr,
        },
    };

    // Call the core function with panic catching
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_aid(&y_vec, &opts)
    }));

    let result = match result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in AID");
            }
            return false;
        }
    };

    match result {
        Ok(aid_result) => {
            // Allocate and copy demand_type string
            let demand_type_bytes = aid_result.demand_type.as_bytes();
            let demand_type_ptr =
                libc::malloc(demand_type_bytes.len() + 1) as *mut libc::c_char;
            if demand_type_ptr.is_null() {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate demand_type",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(
                demand_type_bytes.as_ptr(),
                demand_type_ptr as *mut u8,
                demand_type_bytes.len(),
            );
            *demand_type_ptr.add(demand_type_bytes.len()) = 0;

            // Allocate and copy distribution string
            let distribution_bytes = aid_result.distribution.as_bytes();
            let distribution_ptr =
                libc::malloc(distribution_bytes.len() + 1) as *mut libc::c_char;
            if distribution_ptr.is_null() {
                libc::free(demand_type_ptr as *mut libc::c_void);
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate distribution",
                    );
                }
                return false;
            }
            std::ptr::copy_nonoverlapping(
                distribution_bytes.as_ptr(),
                distribution_ptr as *mut u8,
                distribution_bytes.len(),
            );
            *distribution_ptr.add(distribution_bytes.len()) = 0;

            (*out_result) = AidResultFFI {
                demand_type: demand_type_ptr,
                is_intermittent: aid_result.is_intermittent,
                distribution: distribution_ptr,
                mean: aid_result.mean,
                variance: aid_result.variance,
                zero_proportion: aid_result.zero_proportion,
                n_observations: aid_result.n_observations,
                has_stockouts: aid_result.has_stockouts,
                is_new_product: aid_result.is_new_product,
                is_obsolete_product: aid_result.is_obsolete_product,
                stockout_count: aid_result.stockout_count,
                new_product_count: aid_result.new_product_count,
                obsolete_product_count: aid_result.obsolete_product_count,
                high_outlier_count: aid_result.high_outlier_count,
                low_outlier_count: aid_result.low_outlier_count,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Compute AID per-observation anomaly flags
///
/// # Safety
/// - `y` must be a valid DataArray
/// - `out_result` must be a valid pointer
/// - `out_error` must be a valid pointer
///
/// # Returns
/// `true` on success, `false` on error (check `out_error` for details)
#[no_mangle]
pub unsafe extern "C" fn anofox_aid_anomaly(
    y: DataArray,
    options: AidOptionsFFI,
    out_result: *mut AidAnomalyResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }

    if out_result.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
        }
        return false;
    }

    // Convert y to Vec
    let y_vec = y.to_vec();

    // Convert options
    let opts = AidOptions {
        intermittent_threshold: options.intermittent_threshold,
        outlier_method: match options.outlier_method {
            OutlierMethodFFI::ZScore => OutlierMethod::ZScore,
            OutlierMethodFFI::Iqr => OutlierMethod::Iqr,
        },
    };

    // Call the core function with panic catching
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_aid_anomalies(&y_vec, &opts)
    }));

    let result = match result {
        Ok(r) => r,
        Err(_) => {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InternalError, "Internal panic in AID anomaly");
            }
            return false;
        }
    };

    match result {
        Ok(flags) => {
            let n = flags.len();

            // Allocate array for anomaly flags
            let flags_ptr =
                libc::malloc(n * std::mem::size_of::<AidAnomalyFlagsFFI>()) as *mut AidAnomalyFlagsFFI;
            if flags_ptr.is_null() && n > 0 {
                if !out_error.is_null() {
                    (*out_error).set(
                        ErrorCode::AllocationFailure,
                        "Failed to allocate anomaly flags",
                    );
                }
                return false;
            }

            // Copy flags
            for (i, flag) in flags.iter().enumerate() {
                *flags_ptr.add(i) = AidAnomalyFlagsFFI {
                    stockout: flag.stockout,
                    new_product: flag.new_product,
                    obsolete_product: flag.obsolete_product,
                    high_outlier: flag.high_outlier,
                    low_outlier: flag.low_outlier,
                };
            }

            (*out_result) = AidAnomalyResultFFI {
                flags: flags_ptr,
                len: n,
            };

            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

/// Free memory allocated by AID function
///
/// # Safety
/// - `result` must be NULL or a valid pointer to an AidResultFFI
/// - Must only be called once per result (double-free is undefined behavior)
#[no_mangle]
pub unsafe extern "C" fn anofox_free_aid_result(result: *mut AidResultFFI) {
    if result.is_null() {
        return;
    }
    if !(*result).demand_type.is_null() {
        libc::free((*result).demand_type as *mut libc::c_void);
        (*result).demand_type = std::ptr::null_mut();
    }
    if !(*result).distribution.is_null() {
        libc::free((*result).distribution as *mut libc::c_void);
        (*result).distribution = std::ptr::null_mut();
    }
}

/// Free memory allocated by AID anomaly function
///
/// # Safety
/// - `result` must be NULL or a valid pointer to an AidAnomalyResultFFI
/// - Must only be called once per result (double-free is undefined behavior)
#[no_mangle]
pub unsafe extern "C" fn anofox_free_aid_anomaly_result(result: *mut AidAnomalyResultFFI) {
    if result.is_null() {
        return;
    }
    if !(*result).flags.is_null() {
        libc::free((*result).flags as *mut libc::c_void);
        (*result).flags = std::ptr::null_mut();
    }
}
