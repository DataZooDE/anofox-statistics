//! C FFI boundary for anofox-statistics
//!
//! This crate provides C-compatible functions for calling from the C++ DuckDB extension layer.

mod types;

pub use types::*;

use anofox_stats_core::{models::{fit_ols, fit_ridge, fit_elasticnet}, OlsOptions, RidgeOptions, ElasticNetOptions, StatsError};
use std::slice;

/// Convert StatsError to ErrorCode
fn error_to_code(err: &StatsError) -> ErrorCode {
    match err {
        StatsError::InvalidAlpha(_) => ErrorCode::InvalidAlpha,
        StatsError::InvalidL1Ratio(_) => ErrorCode::InvalidL1Ratio,
        StatsError::InsufficientData { .. } => ErrorCode::InsufficientData,
        StatsError::NoValidData => ErrorCode::NoValidData,
        StatsError::DimensionMismatch { .. } => ErrorCode::DimensionMismatch,
        StatsError::EmptyInput { .. } => ErrorCode::InvalidInput,
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

    // Call the core function
    match fit_ols(&y_vec, &x_vecs, &opts) {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::AllocationFailure, "Failed to allocate coefficients");
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
                            (*out_error)
                                .set(ErrorCode::AllocationFailure, "Failed to allocate inference arrays");
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

    // Call the core function
    match fit_ridge(&y_vec, &x_vecs, &opts) {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::AllocationFailure, "Failed to allocate coefficients");
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
                            (*out_error)
                                .set(ErrorCode::AllocationFailure, "Failed to allocate inference arrays");
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

    // Call the core function
    match fit_elasticnet(&y_vec, &x_vecs, &opts) {
        Ok(result) => {
            // Fill core results
            let n_coef = result.core.coefficients.len();

            // Allocate and copy coefficients
            let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
            if coef_ptr.is_null() && n_coef > 0 {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::AllocationFailure, "Failed to allocate coefficients");
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

/// Get library version string
#[no_mangle]
pub extern "C" fn anofox_version() -> *const libc::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const libc::c_char
}
