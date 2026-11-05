//! C FFI for partial / semi-partial correlation (#296). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::partial_correlation::{
    partial_correlation, PartialCorKind, PartialCorMethod, PartialCorrelationResult,
};

/// Options of `anofox_partial_cor`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PartialCorOptionsFFI {
    /// 0 partial (controls removed from x and y), 1 semi-partial (from y only).
    pub kind: i32,
    /// 0 pearson, 1 spearman.
    pub method: i32,
}

/// Result of `anofox_partial_cor`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PartialCorResultFFI {
    pub r: f64,
    pub statistic: f64,
    pub p_value: f64,
    pub df: f64,
    pub n: usize,
    pub n_controls: usize,
    /// Unified test fields (#159), method name included; keep last.
    pub unified: TestSummaryFFI,
}

impl From<&PartialCorrelationResult> for PartialCorResultFFI {
    fn from(r: &PartialCorrelationResult) -> Self {
        Self {
            r: r.r,
            statistic: r.statistic,
            p_value: r.p_value,
            df: r.df,
            n: r.n,
            n_controls: r.n_controls,
            unified: TestSummaryFFI::from(&r.unified),
        }
    }
}

/// Partial (`kind = 0`) or semi-partial (`kind = 1`) correlation of `x` and `y`
/// controlling for `controls[0..n_controls]` (upstream
/// `correlation::partial_cor` / `semi_partial_cor`, R `ppcor`). NaN entries
/// drop their row.
///
/// # Safety
/// - `x`, `y` and each of the `n_controls` arrays at `controls` must be valid
///   DataArrays of equal length
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_partial_cor(
    x: DataArray,
    y: DataArray,
    controls: *const DataArray,
    n_controls: usize,
    options: PartialCorOptionsFFI,
    out_result: *mut PartialCorResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        let fail = |msg: &str| {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, msg);
            }
            false
        };
        if out_result.is_null() || (controls.is_null() && n_controls > 0) {
            return fail("NULL pointer argument");
        }
        let kind = match options.kind {
            0 => PartialCorKind::Partial,
            1 => PartialCorKind::SemiPartial,
            _ => return fail("invalid partial correlation kind"),
        };
        let method = match options.method {
            0 => PartialCorMethod::Pearson,
            1 => PartialCorMethod::Spearman,
            _ => return fail("invalid partial correlation method"),
        };
        let z: Vec<Vec<f64>> = if n_controls == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(controls, n_controls)
                .iter()
                .map(|c| c.to_vec())
                .collect()
        };
        match partial_correlation(&x.to_vec(), &y.to_vec(), &z, kind, method) {
            Ok(r) => {
                *out_result = PartialCorResultFFI::from(&r);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn layout_matches_header() {
        assert_eq!(offset_of!(PartialCorResultFFI, n), 4 * size_of::<f64>());
        assert_eq!(
            size_of::<PartialCorResultFFI>(),
            offset_of!(PartialCorResultFFI, unified) + size_of::<TestSummaryFFI>()
        );
    }

    #[test]
    fn rejects_bad_kind_and_reports_degenerate() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 1.0, 4.0, 3.0, 5.0];
        let zs = [1.0, 1.0, 1.0, 1.0, 1.0];
        let arr = |v: &[f64]| DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        };
        let controls = [arr(&zs)];
        let mut out = std::mem::MaybeUninit::<PartialCorResultFFI>::zeroed();
        let mut err = AnofoxError::success();
        let opts = PartialCorOptionsFFI { kind: 7, method: 0 };
        let ok = unsafe {
            anofox_partial_cor(
                arr(&xs),
                arr(&ys),
                controls.as_ptr(),
                1,
                opts,
                out.as_mut_ptr(),
                &mut err,
            )
        };
        assert!(!ok);
        assert_eq!(err.code, ErrorCode::InvalidInput);
        let opts = PartialCorOptionsFFI { kind: 0, method: 0 };
        let ok = unsafe {
            anofox_partial_cor(
                arr(&xs),
                arr(&ys),
                controls.as_ptr(),
                1,
                opts,
                out.as_mut_ptr(),
                &mut err,
            )
        };
        assert!(!ok);
        assert_eq!(err.code, ErrorCode::InsufficientData);
    }
}
