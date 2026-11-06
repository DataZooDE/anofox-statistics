//! C FFI for the P-spline smoother (#316). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`; the layout is pinned by the test below and
//! the static_assert in `src/include/model_struct.cpp`.

use super::*;
use anofox_stats_core::models::{
    fit_predict_pspline, pspline_predict, IntervalType, PSplineOptions, StoredPSpline,
};

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
/// bounds, when no interval was requested); `knots` has `coefficients_len + 4`
/// and `covariance` (row-major) `coefficients_len²`. Released with
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
    pub knots: *mut f64,
    pub covariance: *mut f64,
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
            knots: std::ptr::null_mut(),
            covariance: std::ptr::null_mut(),
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
            knots: alloc_f64(&fit.knots),
            covariance: alloc_f64(&fit.covariance),
            coefficients_len: fit.coefficients.len(),
            n_points: points.len(),
            n_observations: fit.n_observations,
            penalty_order: fit.penalty_order,
            summary: ModelSummaryFFI::from(&fit.summary),
        };
        let failed = (res.coefficients_len > 0
            && (res.coefficients.is_null() || res.knots.is_null() || res.covariance.is_null()))
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

/// Predict at the `n_points` values of `x_new` from a stored P-spline (its
/// `n_coefficients` coefficients, `n_knots` knots, row-major
/// `n_covariance`-value covariance, `sigma2`, `edf` and `n_observations`),
/// rebuilt upstream (`FittedPSpline::from_parts`). `interval` and `level` as
/// `anofox_pspline_fit_predict`. Writes `n_points` values to each of
/// `out_fit`, `out_se`, `out_lower` and `out_upper` (NaN where the point is
/// not finite or, for the bounds, without an interval).
///
/// # Safety
/// The input pointers must point to their lengths (or be NULL when the
/// length is 0); the outputs to `n_points` writable doubles; `out_error`
/// must be valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_pspline_predict(
    knots: *const f64,
    n_knots: usize,
    coefficients: *const f64,
    n_coefficients: usize,
    covariance: *const f64,
    n_covariance: usize,
    sigma2: f64,
    edf: f64,
    n_observations: usize,
    x_new: *const f64,
    n_points: usize,
    interval: i32,
    level: f64,
    out_fit: *mut f64,
    out_se: *mut f64,
    out_lower: *mut f64,
    out_upper: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if n_points > 0
            && (!check_out(out_fit, out_error)
                || !check_out(out_se, out_error)
                || !check_out(out_lower, out_error)
                || !check_out(out_upper, out_error))
        {
            return false;
        }
        let view = |p: *const f64, len: usize| -> &[f64] {
            if p.is_null() || len == 0 {
                &[]
            } else {
                slice::from_raw_parts(p, len)
            }
        };
        let model = StoredPSpline {
            knots: view(knots, n_knots),
            coefficients: view(coefficients, n_coefficients),
            covariance: view(covariance, n_covariance),
            sigma2,
            edf,
            n_observations,
        };
        let kind = match interval {
            0 => Some(IntervalType::Prediction),
            1 => Some(IntervalType::Confidence),
            _ => None,
        };
        match pspline_predict(&model, view(x_new, n_points), kind, level) {
            Ok(p) => {
                for (out, v) in [
                    (out_fit, &p.fit),
                    (out_se, &p.se_fit),
                    (out_lower, &p.lower),
                    (out_upper, &p.upper),
                ] {
                    std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                }
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
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
        &mut r.knots,
        &mut r.covariance,
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
            offset_of!(PSplineResultFFI, coefficients_len),
            4 * d + 7 * size_of::<*mut f64>()
        );
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
            let (mut f, mut se, mut lo, mut hi) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            assert!(anofox_pspline_predict(
                out.knots,
                out.coefficients_len + 4,
                out.coefficients,
                out.coefficients_len,
                out.covariance,
                out.coefficients_len * out.coefficients_len,
                out.sigma2,
                out.edf,
                out.n_observations,
                new.as_ptr(),
                2,
                0,
                0.95,
                f.as_mut_ptr(),
                se.as_mut_ptr(),
                lo.as_mut_ptr(),
                hi.as_mut_ptr(),
                &mut err,
            ));
            assert_eq!(f[0], *out.fitted);
            assert!((hi[0] - *out.upper).abs() < 1e-9 && f[1].is_nan());
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
