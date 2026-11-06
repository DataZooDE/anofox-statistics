//! C FFI for additive regression with optimal transformations (`Hmisc::areg`,
//! #320 follow-up): upstream anofox-regression `transform::areg`, and the
//! stored model rebuilt with `Areg::from_parts` to predict, transform and
//! back-transform. The struct is mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::reduction::{
    areg_inverse_y, areg_predict, areg_transform_y, fit_areg, parse_areg_type, AregType,
    StoredAreg, StoredAregBasis,
};

/// The `AregType` of the FFI code: 0 spline, 1 linear, 2 categorical.
fn areg_type_of(code: i32) -> Option<AregType> {
    match code {
        0 => parse_areg_type("s"),
        1 => parse_areg_type("l"),
        2 => parse_areg_type("c"),
        _ => None,
    }
}

unsafe fn malloc_values<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn read<T: Copy>(p: *const T, len: usize) -> Vec<T> {
    if p.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(p, len).to_vec()
    }
}

/// A fitted `areg` model. Arrays allocated by `anofox_areg_fit`, released
/// with `anofox_free_areg_result`.
///
/// `x_coefficients` (`n_x_coefficients`, intercept first), `y_coefficients`
/// (`n_y_coefficients`, intercept first); per predictor (`n_predictors`):
/// `x_df` basis columns, `x_kinds` (0 linear, 1 spline, 2 categorical) and
/// `x_values_len` parameters, whose values (knots or levels) are
/// concatenated in `x_values`; the response's parameters in `y_values`
/// (`n_y_values`) of kind `y_kind`. `mad` / `med` are NaN when the response
/// transformation is not monotone.
#[repr(C)]
#[derive(Debug)]
pub struct AregResultFFI {
    pub x_coefficients: *mut f64,
    pub y_coefficients: *mut f64,
    pub x_df: *mut i64,
    pub x_kinds: *mut i64,
    pub x_values: *mut f64,
    pub x_values_len: *mut i64,
    pub y_values: *mut f64,
    pub n_x_coefficients: usize,
    pub n_y_coefficients: usize,
    pub n_predictors: usize,
    pub n_y_values: usize,
    pub y_df: usize,
    pub y_kind: usize,
    pub n_observations: usize,
    pub r_squared: f64,
    pub mad: f64,
    pub med: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// `Hmisc::areg(x, y, xtype, ytype, nk)` for `n` responses `y` and
/// `n_features` predictor columns `x`. `x_types` holds one code per predictor
/// (0 spline, 1 linear, 2 categorical) or is NULL (all splines); `y_type`
/// likewise.
///
/// # Safety
/// `y` must point to `n` doubles, `x` to `n_features` arrays of `n` values,
/// `x_types` to `n_features` codes or be NULL; `out_result` and `out_error`
/// valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_areg_fit(
    y: *const f64,
    n: usize,
    x: *const DataArray,
    n_features: usize,
    nk: u32,
    x_types: *const i32,
    y_type: i32,
    out_result: *mut AregResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let invalid = |msg: String| {
            set_error(out_error, &StatsError::InvalidInput(msg));
            false
        };
        let Some(yt) = areg_type_of(y_type) else {
            return invalid(format!("unknown areg type code {y_type}"));
        };
        let xt = if x_types.is_null() {
            None
        } else {
            let mut v = Vec::with_capacity(n_features);
            for &c in slice::from_raw_parts(x_types, n_features) {
                match areg_type_of(c) {
                    Some(t) => v.push(t),
                    None => return invalid(format!("unknown areg type code {c}")),
                }
            }
            Some(v)
        };
        let yv = read(y, n);
        let cols: Vec<Vec<f64>> = (0..n_features).map(|j| (*x.add(j)).to_vec()).collect();
        match fit_areg(&yv, &cols, nk as usize, xt, yt) {
            Ok(f) => {
                let kinds: Vec<i64> = f.x_bases.iter().map(|b| i64::from(b.kind)).collect();
                let lens: Vec<i64> = f.x_bases.iter().map(|b| b.values.len() as i64).collect();
                let values: Vec<f64> = f.x_bases.iter().flat_map(|b| b.values.clone()).collect();
                let df: Vec<i64> = f.x_df.iter().map(|&d| d as i64).collect();
                let out = AregResultFFI {
                    x_coefficients: malloc_values(&f.x_coefficients),
                    y_coefficients: malloc_values(&f.y_coefficients),
                    x_df: malloc_values(&df),
                    x_kinds: malloc_values(&kinds),
                    x_values: malloc_values(&values),
                    x_values_len: malloc_values(&lens),
                    y_values: malloc_values(&f.y_basis.values),
                    n_x_coefficients: f.x_coefficients.len(),
                    n_y_coefficients: f.y_coefficients.len(),
                    n_predictors: f.x_bases.len(),
                    n_y_values: f.y_basis.values.len(),
                    y_df: f.y_df,
                    y_kind: usize::from(f.y_basis.kind),
                    n_observations: f.n_observations,
                    r_squared: f.r_squared,
                    mad: f.mad.unwrap_or(f64::NAN),
                    med: f.med.unwrap_or(f64::NAN),
                    y_min: f.y_range.0,
                    y_max: f.y_range.1,
                };
                let failed = out.x_coefficients.is_null()
                    || out.y_coefficients.is_null()
                    || (out.n_predictors > 0
                        && (out.x_df.is_null()
                            || out.x_kinds.is_null()
                            || out.x_values_len.is_null()))
                    || (!values.is_empty() && out.x_values.is_null())
                    || (out.n_y_values > 0 && out.y_values.is_null());
                *out_result = out;
                if failed {
                    anofox_free_areg_result(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate areg result",
                        );
                    }
                    return false;
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

/// Free the arrays of a result filled by `anofox_areg_fit`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_areg_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_areg_result(result: *mut AregResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.x_coefficients,
            &mut r.y_coefficients,
            &mut r.x_values,
            &mut r.y_values,
        ] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
        for p in [&mut r.x_df, &mut r.x_kinds, &mut r.x_values_len] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
    })
}

/// Apply a stored `areg` model, rebuilt upstream (`Areg::from_parts`), to
/// one input: `what` 0 the linear predictor (transformed-`y` scale) of the
/// predictor row `input` (`n_input` = `n_predictors` values), 1 the response
/// transformation of `input[0]`, 2 its back-transformation (NaN when the
/// transformation is not monotone or the response categorical). The model
/// parts are laid out as `AregResultFFI`.
///
/// # Safety
/// Every pointer must point to its length (`x_kinds` / `x_values_len` to
/// `n_predictors`, `x_values` to their sum); `out_value` and `out_error`
/// valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_areg_apply(
    what: u32,
    x_kinds: *const i64,
    x_values: *const f64,
    x_values_len: *const i64,
    n_predictors: usize,
    x_coefficients: *const f64,
    n_x_coefficients: usize,
    y_kind: i64,
    y_values: *const f64,
    n_y_values: usize,
    y_coefficients: *const f64,
    n_y_coefficients: usize,
    y_min: f64,
    y_max: f64,
    input: *const f64,
    n_input: usize,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_value, out_error) {
            return false;
        }
        let kind = |k: i64| -> Option<u8> { u8::try_from(k).ok() };
        let kinds = read(x_kinds, n_predictors);
        let lens = read(x_values_len, n_predictors);
        if lens.iter().any(|&l| l < 0) {
            set_error(
                out_error,
                &StatsError::InvalidInput("negative parameter count".into()),
            );
            return false;
        }
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let values = read(x_values, total);
        let mut start = 0;
        let mut x_bases = Vec::with_capacity(n_predictors);
        for (k, l) in kinds.iter().zip(&lens) {
            let l = *l as usize;
            x_bases.push(StoredAregBasis {
                kind: kind(*k).unwrap_or(u8::MAX),
                values: values[start..start + l].to_vec(),
            });
            start += l;
        }
        let model = StoredAreg {
            x_bases,
            x_coefficients: read(x_coefficients, n_x_coefficients),
            y_basis: StoredAregBasis {
                kind: kind(y_kind).unwrap_or(u8::MAX),
                values: read(y_values, n_y_values),
            },
            y_coefficients: read(y_coefficients, n_y_coefficients),
            y_range: (y_min, y_max),
        };
        let fitted = match model.rebuild() {
            Ok(m) => m,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let input = read(input, n_input);
        let value = match what {
            0 => match areg_predict(&fitted, &[input]) {
                Ok(v) => v[0],
                Err(e) => {
                    set_error(out_error, &e);
                    return false;
                }
            },
            1 | 2 if n_input != 1 => {
                set_error(
                    out_error,
                    &StatsError::InvalidInput("one value is transformed at a time".into()),
                );
                return false;
            }
            1 => areg_transform_y(&fitted, &input)[0],
            2 => areg_inverse_y(&fitted, &input).map_or(f64::NAN, |v| v[0]),
            _ => {
                set_error(
                    out_error,
                    &StatsError::InvalidInput(format!("unknown areg operation {what}")),
                );
                return false;
            }
        };
        *out_value = value;
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn areg_fit_and_apply_roundtrip() {
        let n = 60;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (0.5 * x1[i] + 0.3 * x2[i] + 0.1 * ((i * 7 % 5) as f64)).exp())
            .collect();
        let cols = [
            DataArray {
                data: x1.as_ptr(),
                validity: std::ptr::null(),
                len: n,
            },
            DataArray {
                data: x2.as_ptr(),
                validity: std::ptr::null(),
                len: n,
            },
        ];
        let types = [0i32, 2];
        let mut err = AnofoxError::success();
        unsafe {
            let mut r = std::mem::MaybeUninit::<AregResultFFI>::uninit();
            assert!(anofox_areg_fit(
                y.as_ptr(),
                n,
                cols.as_ptr(),
                2,
                4,
                types.as_ptr(),
                0,
                r.as_mut_ptr(),
                &mut err
            ));
            let mut r = r.assume_init();
            assert_eq!(r.n_predictors, 2);
            assert_eq!(*r.x_kinds.add(1), 2);
            let mut lp = 0.0;
            assert!(anofox_areg_apply(
                0,
                r.x_kinds,
                r.x_values,
                r.x_values_len,
                r.n_predictors,
                r.x_coefficients,
                r.n_x_coefficients,
                r.y_kind as i64,
                r.y_values,
                r.n_y_values,
                r.y_coefficients,
                r.n_y_coefficients,
                r.y_min,
                r.y_max,
                [x1[0], x2[0]].as_ptr(),
                2,
                &mut lp,
                &mut err
            ));
            assert!(lp.is_finite());
            let mut ty = 0.0;
            assert!(anofox_areg_apply(
                1,
                r.x_kinds,
                r.x_values,
                r.x_values_len,
                r.n_predictors,
                r.x_coefficients,
                r.n_x_coefficients,
                r.y_kind as i64,
                r.y_values,
                r.n_y_values,
                r.y_coefficients,
                r.n_y_coefficients,
                r.y_min,
                r.y_max,
                [y[0]].as_ptr(),
                1,
                &mut ty,
                &mut err
            ));
            assert!(ty.is_finite());
            assert!(!anofox_areg_apply(
                0,
                r.x_kinds,
                r.x_values,
                r.x_values_len,
                r.n_predictors,
                r.x_coefficients,
                r.n_x_coefficients,
                r.y_kind as i64,
                r.y_values,
                r.n_y_values,
                r.y_coefficients,
                r.n_y_coefficients,
                r.y_min,
                r.y_max,
                [1.0].as_ptr(),
                1,
                &mut ty,
                &mut err
            ));
            anofox_free_areg_result(&mut r);
            assert!(r.x_coefficients.is_null() && r.x_kinds.is_null());
        }
    }
}
