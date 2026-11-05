//! C FFI for the generalized VIF (#307). The struct is mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::diagnostics::compute_gvif;

/// GVIF of one term.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GvifTermFFI {
    /// Number of columns of the term.
    pub df: u64,
    pub gvif: f64,
    /// GVIF^(1/(2 df)).
    pub gvif_adj: f64,
}

/// Generalized VIF (Fox & Monette 1992, `car::vif`) per term, on the rows
/// where every column is finite.
///
/// # Safety
/// `x` and `term_of_column` must point to `x_count` valid entries;
/// `out_terms` to `n_terms` writable entries. `out_n_observations` is
/// required.
#[no_mangle]
pub unsafe extern "C" fn anofox_compute_gvif(
    x: *const DataArray,
    x_count: usize,
    term_of_column: *const u64,
    n_terms: usize,
    out_terms: *mut GvifTermFFI,
    out_n_observations: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if x.is_null()
            || term_of_column.is_null()
            || x_count == 0
            || out_n_observations.is_null()
            || (out_terms.is_null() && n_terms > 0)
        {
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::InvalidInput,
                    "x, terms or an output pointer is NULL",
                );
            }
            return false;
        }
        let columns: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
            .iter()
            .map(|arr| arr.to_vec())
            .collect();
        let terms: Vec<usize> = slice::from_raw_parts(term_of_column, x_count)
            .iter()
            .map(|&t| t as usize)
            .collect();
        match compute_gvif(&columns, &terms, n_terms) {
            Ok(r) => {
                let out = slice::from_raw_parts_mut(out_terms, n_terms);
                for (dst, t) in out.iter_mut().zip(&r.terms) {
                    *dst = GvifTermFFI {
                        df: t.df as u64,
                        gvif: t.gvif,
                        gvif_adj: t.gvif_adj,
                    };
                }
                *out_n_observations = r.n_observations;
                true
            }
            Err(e) => {
                if !out_error.is_null() {
                    (*out_error).set(error_to_code(&e), &e.to_string());
                }
                false
            }
        }
    })
}
