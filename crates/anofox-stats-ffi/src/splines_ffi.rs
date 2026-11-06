//! C FFI for the regression-spline bases (#317). Declarations mirrored in
//! `src/include/anofox_stats_ffi.h`.
//!
//! Variable-length results are written into caller buffers: every function
//! sets `*out_len` to the number of values of the result and writes them only
//! when `*out_len <= capacity`, so the caller can grow its buffer and call
//! again.

use super::*;
use anofox_stats_core::splines::{self, Extrapolation, SplineBasis, SplineKind, SplineSpec};

/// Spline specification. Must match `AnofoxSplineSpec`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SplineSpecFFI {
    /// 0 rcs, 1 lsp, 2 bspline
    pub kind: i32,
    /// rcs: `norm` of `rcspline.eval` (0, 1, 2)
    pub norm: i32,
    /// bspline: 0 constant, 1 linear, 2 continue, 3 periodic, 4 error
    pub extrapolation: i32,
    /// bspline: keep the last spline
    pub include_bias: bool,
    /// bspline: degree
    pub degree: usize,
    /// Knots (rcs, lsp) or base knots (bspline); NaN for a NULL element
    pub knots: *const f64,
    pub n_knots: usize,
}

fn invalid(msg: &str) -> anofox_stats_core::StatsError {
    anofox_stats_core::StatsError::InvalidInput(msg.to_string())
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn read_spec(spec: *const SplineSpecFFI) -> anofox_stats_core::StatsResult<SplineSpec> {
    if spec.is_null() {
        return Err(invalid("spec is NULL"));
    }
    let s = &*spec;
    let kind = match s.kind {
        0 => SplineKind::Rcs,
        1 => SplineKind::Lsp,
        2 => SplineKind::BSpline,
        _ => return Err(invalid("invalid spline kind")),
    };
    let extrapolation = match s.extrapolation {
        0 => Extrapolation::Constant,
        1 => Extrapolation::Linear,
        2 => Extrapolation::Continue,
        3 => Extrapolation::Periodic,
        4 => Extrapolation::Error,
        _ => return Err(invalid("invalid extrapolation")),
    };
    Ok(SplineSpec {
        kind,
        knots: slice(s.knots, s.n_knots).to_vec(),
        norm: s.norm,
        degree: s.degree,
        extrapolation,
        include_bias: s.include_bias,
    })
}

/// Write `values` to `out` when they fit; always report the length.
unsafe fn write_values<T: Copy>(values: &[T], out: *mut T, capacity: usize, out_len: *mut usize) {
    *out_len = values.len();
    if values.len() <= capacity && !values.is_empty() && !out.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    }
}

unsafe fn finish<T>(
    result: anofox_stats_core::StatsResult<T>,
    out_error: *mut AnofoxError,
    write: impl FnOnce(T),
) -> bool {
    match result {
        Ok(v) => {
            write(v);
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

/// The basis row of the spline `spec` at `x` (a NaN row for a non-finite `x`).
///
/// # Safety
/// `spec` must be valid; `out_values` must hold `capacity` doubles;
/// `out_len` must be valid; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_spline_basis(
    spec: *const SplineSpecFFI,
    x: f64,
    out_values: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = read_spec(spec)
            .and_then(|s| SplineBasis::new(&s))
            .and_then(|b| b.row(x));
        finish(r, out_error, |row| {
            write_values(&row, out_values, capacity, out_len)
        })
    })
}

/// Validate `spec` and describe its basis: the knots as kept upstream
/// (`out_knots`, `out_n_knots`; capacity `knots_capacity`) and the
/// nonlinearity flag of every column (`out_nonlinear`, `out_n_columns`;
/// capacity `columns_capacity`).
///
/// # Safety
/// All pointers must be valid for their capacities; `out_error` may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_spline_info(
    spec: *const SplineSpecFFI,
    out_knots: *mut f64,
    knots_capacity: usize,
    out_n_knots: *mut usize,
    out_nonlinear: *mut bool,
    columns_capacity: usize,
    out_n_columns: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_n_knots, out_error) || !check_out(out_n_columns, out_error) {
            return false;
        }
        let r = read_spec(spec)
            .and_then(|s| SplineBasis::new(&s))
            .and_then(|b| Ok((b.knots(), b.nonlinear()?)));
        finish(r, out_error, |(knots, nonlinear)| {
            write_values(&knots, out_knots, knots_capacity, out_n_knots);
            write_values(&nonlinear, out_nonlinear, columns_capacity, out_n_columns);
        })
    })
}

/// Default knots of a restricted cubic spline with `nk` knots placed on the
/// finite values of `x` (`Hmisc::rcspline.eval(x, nk, knots.only = TRUE,
/// fractied)`).
///
/// # Safety
/// `x` must be a valid DataArray; `out_knots` must hold `capacity` doubles;
/// `out_len` must be valid; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_rcs_default_knots(
    x: DataArray,
    nk: usize,
    fractied: f64,
    out_knots: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = splines::rcs_default_knots(&x.to_vec(), nk, fractied);
        finish(r, out_error, |k| {
            write_values(&k, out_knots, capacity, out_len)
        })
    })
}

/// Base knots of a B-spline placed on the finite values of `x`:
/// `placement` 0 uniform, 1 quantile (sklearn `knots`).
///
/// # Safety
/// As `anofox_rcs_default_knots`.
#[no_mangle]
pub unsafe extern "C" fn anofox_bspline_data_knots(
    x: DataArray,
    n_knots: usize,
    placement: i32,
    out_knots: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = match placement {
            0 | 1 => splines::bspline_data_knots(&x.to_vec(), n_knots, placement == 1),
            _ => Err(invalid("invalid knot placement")),
        };
        finish(r, out_error, |k| {
            write_values(&k, out_knots, capacity, out_len)
        })
    })
}

/// Knot quantiles of RMS Table 2.3 for `nk` knots.
///
/// # Safety
/// `out_values` must hold `capacity` doubles; `out_len` must be valid;
/// `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_default_knot_quantiles(
    nk: usize,
    out_values: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = splines::default_knot_quantiles(nk);
        finish(r, out_error, |q| {
            write_values(&q, out_values, capacity, out_len)
        })
    })
}

/// Restricted interaction (`rms::%ia%`) of two basis rows `a` (`n_a` values,
/// flags `a_nonlinear`) and `b`; the product row goes to `out_values`.
///
/// # Safety
/// The inputs must hold `n_a` / `n_b` values; `out_values` must hold
/// `capacity` doubles; `out_len` must be valid; `out_error` may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_restricted_interaction(
    a: *const f64,
    a_nonlinear: *const bool,
    n_a: usize,
    b: *const f64,
    b_nonlinear: *const bool,
    n_b: usize,
    out_values: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = splines::restricted_interaction(
            slice(a, n_a),
            slice(a_nonlinear, n_a),
            slice(b, n_b),
            slice(b_nonlinear, n_b),
        );
        finish(r, out_error, |(row, _)| {
            write_values(&row, out_values, capacity, out_len)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: i32, knots: &[f64]) -> SplineSpecFFI {
        SplineSpecFFI {
            kind,
            norm: 2,
            extrapolation: 0,
            include_bias: true,
            degree: 3,
            knots: knots.as_ptr(),
            n_knots: knots.len(),
        }
    }

    #[test]
    fn basis_with_growing_buffer() {
        let knots = [0.0, 1.0, 3.0, 4.0, 6.0];
        let s = spec(0, &knots);
        let mut buf = [0.0; 4];
        let mut len = 0usize;
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_spline_basis(
                &s,
                2.5,
                buf.as_mut_ptr(),
                0,
                &mut len,
                &mut err
            ));
            assert_eq!(len, 4);
            assert_eq!(buf, [0.0; 4]);
            assert!(anofox_spline_basis(
                &s,
                2.5,
                buf.as_mut_ptr(),
                4,
                &mut len,
                &mut err
            ));
            assert_eq!(buf[0], 2.5);
            let bad = spec(0, &knots[..2]);
            assert!(!anofox_spline_basis(
                &bad,
                2.5,
                buf.as_mut_ptr(),
                4,
                &mut len,
                &mut err
            ));
            assert_eq!(err.code, ErrorCode::InvalidInput);
        }
    }

    #[test]
    fn info_and_default_knots() {
        let knots = [3.0, 1.0, 2.0, 2.0];
        let s = spec(0, &knots);
        let (mut k, mut nl) = ([0.0; 4], [true; 4]);
        let (mut nk, mut nc) = (0usize, 0usize);
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_spline_info(
                &s,
                k.as_mut_ptr(),
                4,
                &mut nk,
                nl.as_mut_ptr(),
                4,
                &mut nc,
                &mut err
            ));
            assert_eq!(
                (&k[..nk], &nl[..nc]),
                (&[1.0, 2.0, 3.0][..], &[false, true][..])
            );

            let x: Vec<f64> = (0..5).map(f64::from).collect();
            let arr = DataArray {
                data: x.as_ptr(),
                validity: std::ptr::null(),
                len: x.len(),
            };
            let mut out = [0.0; 8];
            let mut len = 0usize;
            assert!(!anofox_rcs_default_knots(
                arr,
                3,
                0.05,
                out.as_mut_ptr(),
                8,
                &mut len,
                &mut err
            ));
            assert_eq!(err.code, ErrorCode::InsufficientData);
            assert!(anofox_default_knot_quantiles(
                3,
                out.as_mut_ptr(),
                8,
                &mut len,
                &mut err
            ));
            assert_eq!(&out[..len], &[0.1, 0.5, 0.9]);
        }
    }
}
