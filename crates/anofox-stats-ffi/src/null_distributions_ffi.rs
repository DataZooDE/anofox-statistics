//! C FFI for the AUC null distribution and the studentized range
//! distribution (#304). Declarations mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::null_distributions;

unsafe fn write_scalar(
    result: anofox_stats_core::StatsResult<f64>,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_error.is_null() {
        *out_error = AnofoxError::success();
    }
    if out_value.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "out_value is NULL");
        }
        return false;
    }
    match result {
        Ok(v) => {
            *out_value = v;
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

/// Class sizes arrive as signed SQL integers; a negative size is an input error.
fn class_sizes(n_pos: i64, n_neg: i64) -> anofox_stats_core::StatsResult<(usize, usize)> {
    if n_pos < 0 || n_neg < 0 {
        return Err(anofox_stats_core::StatsError::InvalidInput(format!(
            "n_pos and n_neg must be non-negative, got {} and {}",
            n_pos, n_neg
        )));
    }
    Ok((n_pos as usize, n_neg as usize))
}

/// Quantile of the ROC AUC of a random classifier
/// (R `qwilcox(p, n_pos, n_neg) / (n_pos * n_neg)`).
///
/// # Safety
/// `out_value` must be a valid pointer; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_auc_null_quantile(
    p: f64,
    n_pos: i64,
    n_neg: i64,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let r = class_sizes(n_pos, n_neg)
            .and_then(|(a, b)| null_distributions::auc_null_quantile(p, a, b));
        write_scalar(r, out_value, out_error)
    })
}

/// One-sided p-value `P(AUC_random >= auc)` of an observed AUC.
///
/// # Safety
/// `out_value` must be a valid pointer; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_auc_null_p_value(
    auc: f64,
    n_pos: i64,
    n_neg: i64,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let r = class_sizes(n_pos, n_neg)
            .and_then(|(a, b)| null_distributions::auc_null_p_value(auc, a, b));
        write_scalar(r, out_value, out_error)
    })
}

/// Studentized range distribution function (R `ptukey`).
///
/// # Safety
/// `out_value` must be a valid pointer; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_ptukey(
    q: f64,
    n_means: f64,
    df: f64,
    n_ranges: f64,
    lower_tail: bool,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let r = null_distributions::ptukey(q, n_means, df, n_ranges, lower_tail);
        write_scalar(r, out_value, out_error)
    })
}

/// Studentized range quantile function (R `qtukey`).
///
/// # Safety
/// `out_value` must be a valid pointer; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_qtukey(
    p: f64,
    n_means: f64,
    df: f64,
    n_ranges: f64,
    lower_tail: bool,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        let r = null_distributions::qtukey(p, n_means, df, n_ranges, lower_tail);
        write_scalar(r, out_value, out_error)
    })
}
