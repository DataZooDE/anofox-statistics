//! C FFI for the P-spline smoother (#316). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`; the layout is pinned by the test below and
//! the static_assert in `src/include/model_struct.cpp`.

use super::*;
use anofox_stats_core::models::{fit_predict_pspline, IntervalType, PSplineOptions};

/// Options of `anofox_pspline_fit_predict`. `n_basis` 0 lets upstream choose;
/// a NaN `lambda` selects the smoothing parameter by GCV.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PSplineOptionsFFI {
    pub n_basis: u32,
    pub penalty_order: u32,
    pub lambda: f64,
}

/// A fitted P-spline and its predictions at the requested points.
/// `coefficients` has `coefficients_len` values; `fitted`, `se_fit`, `lower`
/// and `upper` have `n_points` (NaN where the point is not finite or, for the
/// bounds, when no interval was requested). Released with
/// `anofox_free_pspline_result`.
#[repr(C)]
#[derive(Debug)]
pub struct PSplineResultFFI {
    pub lambda: f64,
    pub edf: f64,
    pub sigma2: f64,
    pub r_squared: f64,
    pub coefficients: *mut f64,
    pub fitted: *mut f64,
    pub se_fit: *mut f64,
    pub lower: *mut f64,
    pub upper: *mut f64,
    pub coefficients_len: usize,
    pub n_points: usize,
    pub n_observations: usize,
    pub penalty_order: usize,
    pub summary: ModelSummaryFFI,
}

impl Default for PSplineResultFFI {
    fn default() -> Self {
        Self {
            lambda: f64::NAN,
            edf: f64::NAN,
            sigma2: f64::NAN,
            r_squared: f64::NAN,
            coefficients: std::ptr::null_mut(),
            fitted: std::ptr::null_mut(),
            se_fit: std::ptr::null_mut(),
            lower: std::ptr::null_mut(),
            upper: std::ptr::null_mut(),
            coefficients_len: 0,
            n_points: 0,
            n_observations: 0,
            penalty_order: 0,
            summary: ModelSummaryFFI::default(),
        }
    }
}

/// Fit upstream's `PSplineRegressor` on the complete rows of (x, y) and
/// predict at the `n_points` values of `x_new`. `interval`: -1 none,
/// 0 prediction, 1 confidence, at `level`.
///
/// # Safety
/// `x` and `y` must be valid DataArrays of equal length, `x_new` must point to
/// `n_points` doubles (or be NULL when `n_points` is 0). `out` is required and
/// is released with `anofox_free_pspline_result`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_pspline_fit_predict(
    x: DataArray,
    y: DataArray,
    x_new: *const f64,
    n_points: usize,
    options: PSplineOptionsFFI,
    interval: i32,
    level: f64,
    out: *mut PSplineResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out, out_error) {
            return false;
        }
        *out = PSplineResultFFI::default();
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let points: &[f64] = if x_new.is_null() || n_points == 0 {
            &[]
        } else {
            slice::from_raw_parts(x_new, n_points)
        };
        let opts = PSplineOptions {
            n_basis: options.n_basis as usize,
            penalty_order: options.penalty_order as usize,
            lambda: (!options.lambda.is_nan()).then_some(options.lambda),
        };
        let kind = match interval {
            0 => Some(IntervalType::Prediction),
            1 => Some(IntervalType::Confidence),
            _ => None,
        };
        let (fit, pred) = match fit_predict_pspline(&x_vec, &y_vec, points, &opts, kind, level) {
            Ok(r) => r,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let mut res = PSplineResultFFI {
            lambda: fit.lambda,
            edf: fit.edf,
            sigma2: fit.sigma2,
            r_squared: fit.r_squared,
            coefficients: alloc_f64(&fit.coefficients),
            fitted: alloc_f64(&pred.fit),
            se_fit: alloc_f64(&pred.se_fit),
            lower: alloc_f64(&pred.lower),
            upper: alloc_f64(&pred.upper),
            coefficients_len: fit.coefficients.len(),
            n_points: points.len(),
            n_observations: fit.n_observations,
            penalty_order: fit.penalty_order,
            summary: ModelSummaryFFI::from(&fit.summary),
        };
        let failed = (res.coefficients_len > 0 && res.coefficients.is_null())
            || (res.n_points > 0
                && (res.fitted.is_null()
                    || res.se_fit.is_null()
                    || res.lower.is_null()
                    || res.upper.is_null()));
        if failed {
            anofox_free_pspline_result(&mut res);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate P-spline result",
                );
            }
            return false;
        }
        *out = res;
        true
    })
}

/// Release the arrays of a `PSplineResultFFI`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_pspline_fit_predict`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_pspline_result(result: *mut PSplineResultFFI) {
    if result.is_null() {
        return;
    }
    let r = &mut *result;
    for p in [
        &mut r.coefficients,
        &mut r.fitted,
        &mut r.se_fit,
        &mut r.lower,
        &mut r.upper,
    ] {
        if !p.is_null() {
            libc::free(*p as *mut libc::c_void);
            *p = std::ptr::null_mut();
        }
    }
    r.coefficients_len = 0;
    r.n_points = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Mirrors the static_asserts in src/include/model_struct.cpp.
    #[test]
    fn pspline_layout() {
        let d = size_of::<f64>();
        assert_eq!(offset_of!(PSplineOptionsFFI, lambda), d);
        assert_eq!(size_of::<PSplineOptionsFFI>(), 2 * d);
        assert_eq!(offset_of!(PSplineResultFFI, coefficients), 4 * d);
        assert_eq!(
            size_of::<PSplineResultFFI>(),
            offset_of!(PSplineResultFFI, summary) + size_of::<ModelSummaryFFI>()
        );
    }

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    #[test]
    fn pspline_fit_predict_roundtrip() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 / 4.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let new = [0.5, f64::NAN];
        let opts = PSplineOptionsFFI {
            n_basis: 10,
            penalty_order: 2,
            lambda: f64::NAN,
        };
        unsafe {
            let mut out = PSplineResultFFI::default();
            let mut err = AnofoxError::success();
            let ok = anofox_pspline_fit_predict(
                arr(&x),
                arr(&y),
                new.as_ptr(),
                new.len(),
                opts,
                0,
                0.95,
                &mut out,
                &mut err,
            );
            assert!(ok);
            assert_eq!(out.coefficients_len, 10);
            assert_eq!(out.n_points, 2);
            assert_eq!(out.n_observations, 40);
            assert!((*out.fitted - 0.5f64.sin()).abs() < 0.05);
            assert!((*out.lower) < *out.fitted && *out.fitted < *out.upper);
            assert!((*out.fitted.add(1)).is_nan());
            anofox_free_pspline_result(&mut out);
            assert!(out.coefficients.is_null() && out.fitted.is_null());

            let bad = PSplineOptionsFFI {
                lambda: -1.0,
                ..opts
            };
            assert!(!anofox_pspline_fit_predict(
                arr(&x),
                arr(&y),
                std::ptr::null(),
                0,
                bad,
                -1,
                0.95,
                &mut out,
                &mut err,
            ));
            assert_ne!(err.code, ErrorCode::Success);
        }
    }
}
