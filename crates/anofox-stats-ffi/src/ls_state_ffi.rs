//! FFI for the compact least-squares states (#266): build, merge, describe and
//! fit `ols` / `wls` / `ridge` from a state's bytes.
//!
//! Byte buffers returned to C are `libc::malloc`ed and freed with
//! `anofox_free_bytes`.

use super::*;
use anofox_stats_core::models::StateFit;
use anofox_stats_core::models::{
    fit_ols_from_state, fit_ridge_from_state, fit_wls_from_state, ls_state_from_rows,
    ls_state_info, ls_state_merge,
};
use anofox_stats_core::StatsResult;

unsafe fn fail(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
    false
}

/// Hand `bytes` to C as a malloc'ed buffer.
unsafe fn write_bytes(
    bytes: Vec<u8>,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    let n = bytes.len();
    let ptr = libc::malloc(n.max(1)) as *mut u8;
    if ptr.is_null() {
        return fail(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate state bytes",
        );
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, n);
    *out_bytes = ptr;
    *out_len = n;
    true
}

/// Write a fit result the way `anofox_ols_fit` does, plus the covariance (a
/// malloc'ed `dim × dim` matrix, freed with `anofox_free_interval_matrix`; NULL
/// and 0 without inference) when `out_vcov` / `out_dim` are given.
unsafe fn write_fit_result(
    fit: StatsResult<StateFit>,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    let StateFit { fit: result, vcov } = match fit {
        Ok(r) => r,
        Err(e) => return fail(out_error, error_to_code(&e), &e.to_string()),
    };
    let n_coef = result.core.coefficients.len();
    let coef_ptr = libc::malloc(n_coef.max(1) * std::mem::size_of::<f64>()) as *mut f64;
    if coef_ptr.is_null() {
        return fail(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate coefficients",
        );
    }
    std::ptr::copy_nonoverlapping(result.core.coefficients.as_ptr(), coef_ptr, n_coef);
    *out_core = FitResultCore {
        coefficients: coef_ptr,
        coefficients_len: n_coef,
        intercept: result.core.intercept.unwrap_or(f64::NAN),
        r_squared: result.core.r_squared,
        adj_r_squared: result.core.adj_r_squared,
        residual_std_error: result.core.residual_std_error,
        n_observations: result.core.n_observations,
        n_features: result.core.n_features,
        summary: ModelSummaryFFI::from(&result.summary),
    };
    if !out_inference.is_null() {
        if let Some(inf) = result.inference {
            alloc_inference_arrays!(inf, out_inference, out_error, {
                libc::free(coef_ptr as *mut libc::c_void);
                *out_core = FitResultCore::default();
            });
        } else {
            *out_inference = FitResultInference::default();
        }
    }
    if !out_vcov.is_null() && !out_dim.is_null() {
        *out_vcov = std::ptr::null_mut();
        *out_dim = 0;
        if let Some((v, dim)) = vcov {
            let ptr = libc::malloc(v.len().max(1) * std::mem::size_of::<f64>()) as *mut f64;
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(v.as_ptr(), ptr, v.len());
                *out_vcov = ptr;
                *out_dim = dim;
            }
        }
    }
    true
}

unsafe fn state_slice<'a>(bytes: *const u8, len: usize) -> Option<&'a [u8]> {
    if bytes.is_null() {
        return None;
    }
    Some(slice::from_raw_parts(bytes, len))
}

/// Build a state from rows. `x` holds `x_count` feature columns of `y.len`
/// values; `weights.data` NULL means unweighted. Rows with a NULL / non-finite
/// value are skipped.
///
/// # Safety
/// `x` must point to `x_count` DataArrays; `out_bytes` / `out_len` must be valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_ls_state_from_rows(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    weights: DataArray,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if out_bytes.is_null() || out_len.is_null() || (x.is_null() && x_count > 0) {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        }
        let y_vec = y.to_vec();
        let x_vecs: Vec<Vec<f64>> = if x_count == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(x, x_count)
                .iter()
                .map(|a| a.to_vec())
                .collect()
        };
        let w_vec = (!weights.data.is_null()).then(|| weights.to_vec());
        match ls_state_from_rows(&y_vec, &x_vecs, w_vec.as_deref()) {
            Ok(bytes) => write_bytes(bytes, out_bytes, out_len, out_error),
            Err(e) => fail(out_error, error_to_code(&e), &e.to_string()),
        }
    })
}

/// Merge `n_states` states (`states[i]` has `lens[i]` bytes).
///
/// # Safety
/// `states` and `lens` must point to `n_states` entries.
#[no_mangle]
pub unsafe extern "C" fn anofox_ls_state_merge(
    states: *const *const u8,
    lens: *const usize,
    n_states: usize,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if out_bytes.is_null() || out_len.is_null() || states.is_null() || lens.is_null() {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        }
        let ptrs = slice::from_raw_parts(states, n_states);
        let lens = slice::from_raw_parts(lens, n_states);
        let mut parts = Vec::with_capacity(n_states);
        for (&p, &n) in ptrs.iter().zip(lens) {
            match state_slice(p, n) {
                Some(s) => parts.push(s),
                None => return fail(out_error, ErrorCode::InvalidInput, "NULL state pointer"),
            }
        }
        match ls_state_merge(&parts) {
            Ok(bytes) => write_bytes(bytes, out_bytes, out_len, out_error),
            Err(e) => fail(out_error, error_to_code(&e), &e.to_string()),
        }
    })
}

/// Row count, feature count and weighting of a state.
///
/// # Safety
/// `bytes` must point to `len` bytes; the outputs must be valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_ls_state_info(
    bytes: *const u8,
    len: usize,
    out_n_observations: *mut usize,
    out_n_features: *mut usize,
    out_weighted: *mut bool,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        let (Some(b), false) = (
            state_slice(bytes, len),
            out_n_observations.is_null() || out_n_features.is_null() || out_weighted.is_null(),
        ) else {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        };
        match ls_state_info(b) {
            Ok(info) => {
                *out_n_observations = info.n_observations;
                *out_n_features = info.n_features;
                *out_weighted = info.weighted;
                true
            }
            Err(e) => fail(out_error, error_to_code(&e), &e.to_string()),
        }
    })
}

/// OLS from a state; output as `anofox_ols_fit`.
///
/// # Safety
/// `bytes` must point to `len` bytes; `out_core` must be valid, `out_inference`,
/// `out_vcov` and `out_dim` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_ols_fit_from_state(
    bytes: *const u8,
    len: usize,
    options: OlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        let (Some(b), false) = (state_slice(bytes, len), out_core.is_null()) else {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        };
        let opts = OlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: convert_hc_type_ffi(options.hc_type),
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
            compute_loo: options.compute_loo,
        };
        write_fit_result(
            fit_ols_from_state(b, &opts),
            out_core,
            out_inference,
            out_vcov,
            out_dim,
            out_error,
        )
    })
}

/// WLS from a state; output as `anofox_wls_fit`.
///
/// # Safety
/// As `anofox_ols_fit_from_state`.
#[no_mangle]
pub unsafe extern "C" fn anofox_wls_fit_from_state(
    bytes: *const u8,
    len: usize,
    options: WlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        let (Some(b), false) = (state_slice(bytes, len), out_core.is_null()) else {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        };
        let opts = WlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: convert_hc_type_ffi(options.hc_type),
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
            compute_loo: options.compute_loo,
        };
        write_fit_result(
            fit_wls_from_state(b, &opts),
            out_core,
            out_inference,
            out_vcov,
            out_dim,
            out_error,
        )
    })
}

/// Ridge from a state; output as `anofox_ridge_fit`.
///
/// # Safety
/// As `anofox_ols_fit_from_state`.
#[no_mangle]
pub unsafe extern "C" fn anofox_ridge_fit_from_state(
    bytes: *const u8,
    len: usize,
    options: RidgeOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        let (Some(b), false) = (state_slice(bytes, len), out_core.is_null()) else {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        };
        let opts = RidgeOptions {
            alpha: options.alpha,
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            lambda_scaling: convert_lambda_scaling_ffi(options.lambda_scaling),
            compute_loo: options.compute_loo,
        };
        write_fit_result(
            fit_ridge_from_state(b, &opts),
            out_core,
            out_inference,
            out_vcov,
            out_dim,
            out_error,
        )
    })
}

/// Free a buffer returned by the state functions.
///
/// # Safety
/// `bytes` must come from an `anofox_ls_state_*` call (or be NULL).
#[no_mangle]
pub unsafe extern "C" fn anofox_free_bytes(bytes: *mut u8) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if !bytes.is_null() {
            libc::free(bytes as *mut libc::c_void);
        }
    })
}
