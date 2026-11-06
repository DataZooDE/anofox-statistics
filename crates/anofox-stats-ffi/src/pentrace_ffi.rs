//! C FFI for penalized maximum likelihood à la rms (#323): `rms::pentrace`
//! and `rms::ols(penalty = )` through upstream's `solvers::pentrace`. The
//! structs are mirrored in `src/include/anofox_stats_ffi.h`; arrays are
//! `libc::malloc`ed and released by the matching `anofox_free_*` function.

use super::*;
use anofox_stats_core::models::{
    penalized_ols, pentrace, OrdinalLink, Penalty, PenaltyGrid, PenaltyTerms, PentraceCriterion,
    PentraceModel, TermSpec,
};
use libc::c_char;
use std::ffi::CStr;

/// The term specification of the columns (as `anofox_wald_anova`'s inputs):
/// `term_of_column[j]` is the 0-based term of column `j` (every column needs
/// one); `term_names` has `n_terms` names; `interaction_factors` (NULL: no
/// interactions) holds two 0-based main-effect terms per term, negative for a
/// main effect; `nonlinear` (NULL: all linear) flags the nonlinear columns;
/// `categorical` (NULL: none) flags the categorical terms (`n_terms` flags).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PenaltyTermsFFI {
    pub term_of_column: *const i64,
    pub nonlinear: *const bool,
    pub term_names: *const *const c_char,
    pub n_terms: usize,
    pub interaction_factors: *const i64,
    pub categorical: *const bool,
}

/// The penalties to trace: every combination of the given values; a
/// component with no values falls back as in rms.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PenaltyGridFFI {
    pub simple: *const f64,
    pub n_simple: usize,
    pub nonlinear: *const f64,
    pub n_nonlinear: usize,
    pub interaction: *const f64,
    pub n_interaction: usize,
    pub nonlinear_interaction: *const f64,
    pub n_nonlinear_interaction: usize,
}

/// One row of `anofox_pentrace`; the penalty components are resolved.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PentraceRowFFI {
    pub simple: f64,
    pub nonlinear: f64,
    pub interaction: f64,
    pub nonlinear_interaction: f64,
    pub df: f64,
    pub lr_chisq: f64,
    pub aic: f64,
    pub bic: f64,
    pub aic_c: f64,
}

/// Result of `anofox_pentrace`.
#[repr(C)]
#[derive(Debug)]
pub struct PentraceResultFFI {
    pub rows: *mut PentraceRowFFI,
    pub n_rows: usize,
    /// 0-based index of the best row.
    pub best: usize,
    pub n_observations: usize,
}

impl Default for PentraceResultFFI {
    fn default() -> Self {
        Self {
            rows: std::ptr::null_mut(),
            n_rows: 0,
            best: 0,
            n_observations: 0,
        }
    }
}

/// Result of `anofox_penalized_ols`. `vcov` / `vcov_sandwich` are row-major
/// `(n_features + 1)²`, intercept first; `effective_df_diagonal` has
/// `n_features + 1` entries (intercept first), `effective_df_by_term`
/// `n_terms`.
#[repr(C)]
#[derive(Debug)]
pub struct PenalizedOlsResultFFI {
    pub coefficients: *mut f64,
    pub n_features: usize,
    pub intercept: f64,
    pub vcov: *mut f64,
    pub vcov_sandwich: *mut f64,
    pub effective_df_diagonal: *mut f64,
    pub effective_df_by_term: *mut f64,
    pub n_terms: usize,
    pub sigma: f64,
    pub r_squared: f64,
    pub lr_chisq: f64,
    pub effective_df: f64,
    pub simple: f64,
    pub nonlinear: f64,
    pub interaction: f64,
    pub nonlinear_interaction: f64,
    pub n_observations: usize,
}

impl Default for PenalizedOlsResultFFI {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            n_features: 0,
            intercept: f64::NAN,
            vcov: std::ptr::null_mut(),
            vcov_sandwich: std::ptr::null_mut(),
            effective_df_diagonal: std::ptr::null_mut(),
            effective_df_by_term: std::ptr::null_mut(),
            n_terms: 0,
            sigma: f64::NAN,
            r_squared: f64::NAN,
            lr_chisq: f64::NAN,
            effective_df: f64::NAN,
            simple: f64::NAN,
            nonlinear: f64::NAN,
            interaction: f64::NAN,
            nonlinear_interaction: f64::NAN,
            n_observations: 0,
        }
    }
}

unsafe fn fail(out_error: *mut AnofoxError, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
    false
}

unsafe fn slice_or_empty<'a, T>(ptr: *const T, n: usize) -> &'a [T] {
    if n == 0 || ptr.is_null() {
        &[]
    } else {
        slice::from_raw_parts(ptr, n)
    }
}

unsafe fn read_columns(x: *const DataArray, x_count: usize) -> Vec<Vec<f64>> {
    slice_or_empty(x, x_count)
        .iter()
        .map(|arr| arr.to_vec())
        .collect()
}

/// The term specification of `p` columns; `Err` names what is wrong.
unsafe fn read_terms(spec: *const PenaltyTermsFFI, p: usize) -> Result<PenaltyTerms, String> {
    let Some(s) = spec.as_ref() else {
        return Ok(PenaltyTerms::columns(p));
    };
    if s.term_of_column.is_null() {
        return Ok(PenaltyTerms::columns(p));
    }
    let n_terms = s.n_terms;
    let mut term_of_column = Vec::with_capacity(p);
    for (j, &t) in slice_or_empty(s.term_of_column, p).iter().enumerate() {
        if t < 0 || t as usize >= n_terms {
            return Err(format!("column {} has no term", j + 1));
        }
        term_of_column.push(t as usize);
    }
    if n_terms > 0 && s.term_names.is_null() {
        return Err("NULL term names".into());
    }
    let names: Vec<String> = slice_or_empty(s.term_names, n_terms)
        .iter()
        .map(|&c| {
            if c.is_null() {
                String::new()
            } else {
                CStr::from_ptr(c).to_string_lossy().into_owned()
            }
        })
        .collect();
    let factors: Vec<Option<(usize, usize)>> = if s.interaction_factors.is_null() {
        vec![None; n_terms]
    } else {
        slice_or_empty(s.interaction_factors, 2 * n_terms)
            .chunks(2)
            .map(|f| (f[0] >= 0 && f[1] >= 0).then(|| (f[0] as usize, f[1] as usize)))
            .collect()
    };
    let nonlinear = if s.nonlinear.is_null() {
        vec![false; p]
    } else {
        slice_or_empty(s.nonlinear, p).to_vec()
    };
    let categorical = slice_or_empty(s.categorical, n_terms)
        .iter()
        .enumerate()
        .filter(|(_, &c)| c)
        .map(|(t, _)| t)
        .collect();
    Ok(PenaltyTerms {
        term_of_column,
        terms: names
            .into_iter()
            .zip(factors)
            .map(|(name, factors)| TermSpec { name, factors })
            .collect(),
        nonlinear,
        categorical,
    })
}

fn link_of(link: i32) -> Option<OrdinalLink> {
    Some(match link {
        0 => OrdinalLink::Logistic,
        1 => OrdinalLink::Probit,
        2 => OrdinalLink::LogLog,
        3 => OrdinalLink::CLogLog,
        4 => OrdinalLink::Cauchit,
        _ => return None,
    })
}

/// `rms::pentrace`: fit the model at every penalty of `grid` and score it by
/// the effective d.f. `model` 0 is least squares (`ols`), 1 binary logistic,
/// 2 ordinal (cumulative link `link`: 0 logistic, 1 probit, 2 loglog, 3
/// cloglog, 4 cauchit); `criterion` 0 is corrected AIC, 1 AIC, 2 BIC. Rows
/// with a non-finite value are dropped.
///
/// # Safety
/// `x` must point to `x_count` DataArrays of the length of `y`; `terms` (or
/// NULL: every column its own linear term) and the grid arrays must hold the
/// stated number of entries. `out_result` is released with
/// `anofox_free_pentrace`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_pentrace(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    terms: *const PenaltyTermsFFI,
    grid: PenaltyGridFFI,
    model: i32,
    link: i32,
    criterion: i32,
    out_result: *mut PentraceResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = PentraceResultFFI::default();
        if x.is_null() || x_count == 0 {
            return fail(out_error, "x is NULL or empty");
        }
        let spec = match read_terms(terms, x_count) {
            Ok(s) => s,
            Err(msg) => return fail(out_error, &msg),
        };
        let Some(link) = link_of(link) else {
            return fail(out_error, "unknown link");
        };
        let model = match model {
            0 => PentraceModel::Ols,
            1 => PentraceModel::Binary(link),
            2 => PentraceModel::Ordinal(link),
            _ => return fail(out_error, "unknown model"),
        };
        let criterion = match criterion {
            0 => PentraceCriterion::CorrectedAic,
            1 => PentraceCriterion::Aic,
            2 => PentraceCriterion::Bic,
            _ => return fail(out_error, "unknown criterion"),
        };
        let grid = PenaltyGrid {
            simple: slice_or_empty(grid.simple, grid.n_simple).to_vec(),
            nonlinear: slice_or_empty(grid.nonlinear, grid.n_nonlinear).to_vec(),
            interaction: slice_or_empty(grid.interaction, grid.n_interaction).to_vec(),
            nonlinear_interaction: slice_or_empty(
                grid.nonlinear_interaction,
                grid.n_nonlinear_interaction,
            )
            .to_vec(),
        };
        let y_vec = y.to_vec();
        let x_vecs = read_columns(x, x_count);
        match pentrace(&y_vec, &x_vecs, &spec, &grid, model, criterion) {
            Ok(r) => {
                let rows: Vec<PentraceRowFFI> = r
                    .rows
                    .iter()
                    .map(|row| PentraceRowFFI {
                        simple: row.penalty[0],
                        nonlinear: row.penalty[1],
                        interaction: row.penalty[2],
                        nonlinear_interaction: row.penalty[3],
                        df: row.df,
                        lr_chisq: row.lr_chisq,
                        aic: row.aic,
                        bic: row.bic,
                        aic_c: row.aic_c,
                    })
                    .collect();
                let ptr = libc::malloc(rows.len().max(1) * std::mem::size_of::<PentraceRowFFI>())
                    as *mut PentraceRowFFI;
                if ptr.is_null() {
                    if !out_error.is_null() {
                        (*out_error).set(ErrorCode::AllocationFailure, "allocation failed");
                    }
                    return false;
                }
                std::ptr::copy_nonoverlapping(rows.as_ptr(), ptr, rows.len());
                *out_result = PentraceResultFFI {
                    rows: ptr,
                    n_rows: rows.len(),
                    best: r.best,
                    n_observations: r.n_observations,
                };
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Free the rows of a `PentraceResultFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_pentrace`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_pentrace(result: *mut PentraceResultFFI) {
    if let Some(r) = result.as_mut() {
        if !r.rows.is_null() {
            libc::free(r.rows as *mut libc::c_void);
            r.rows = std::ptr::null_mut();
        }
        r.n_rows = 0;
    }
}

/// `rms::ols(penalty = )`: least squares with rms's penalty for the
/// components `simple`, `nonlinear`, `interaction`, `nonlinear_interaction`
/// (NaN: fall back as in rms; `simple` NaN is 0) on the columns described by
/// `terms` (NULL: every column its own linear term).
///
/// # Safety
/// As [`anofox_pentrace`]; `out_result` is released with
/// `anofox_free_penalized_ols`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_penalized_ols(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    terms: *const PenaltyTermsFFI,
    simple: f64,
    nonlinear: f64,
    interaction: f64,
    nonlinear_interaction: f64,
    out_result: *mut PenalizedOlsResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = PenalizedOlsResultFFI::default();
        if x.is_null() || x_count == 0 {
            return fail(out_error, "x is NULL or empty");
        }
        let spec = match read_terms(terms, x_count) {
            Ok(s) => s,
            Err(msg) => return fail(out_error, &msg),
        };
        let opt = |v: f64| (!v.is_nan()).then_some(v);
        let penalty = Penalty {
            simple: opt(simple).unwrap_or(0.0),
            nonlinear: opt(nonlinear),
            interaction: opt(interaction),
            nonlinear_interaction: opt(nonlinear_interaction),
        };
        let y_vec = y.to_vec();
        let x_vecs = read_columns(x, x_count);
        match penalized_ols(&y_vec, &x_vecs, &spec, &penalty) {
            Ok(r) => {
                let mut res = PenalizedOlsResultFFI {
                    coefficients: alloc_f64(&r.coefficients),
                    n_features: r.coefficients.len(),
                    intercept: r.intercept,
                    vcov: alloc_f64(&r.vcov),
                    vcov_sandwich: alloc_f64(&r.vcov_sandwich),
                    effective_df_diagonal: alloc_f64(&r.effective_df_diagonal),
                    effective_df_by_term: alloc_f64(&r.effective_df_by_term),
                    n_terms: r.effective_df_by_term.len(),
                    sigma: r.sigma,
                    r_squared: r.r_squared,
                    lr_chisq: r.lr_chisq,
                    effective_df: r.effective_df,
                    simple: r.penalty[0],
                    nonlinear: r.penalty[1],
                    interaction: r.penalty[2],
                    nonlinear_interaction: r.penalty[3],
                    n_observations: r.n_observations,
                };
                if res.coefficients.is_null()
                    || res.vcov.is_null()
                    || res.vcov_sandwich.is_null()
                    || res.effective_df_diagonal.is_null()
                    || (res.n_terms > 0 && res.effective_df_by_term.is_null())
                {
                    anofox_free_penalized_ols(&mut res);
                    if !out_error.is_null() {
                        (*out_error).set(ErrorCode::AllocationFailure, "allocation failed");
                    }
                    return false;
                }
                *out_result = res;
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Free the arrays of a `PenalizedOlsResultFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_penalized_ols`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_penalized_ols(result: *mut PenalizedOlsResultFFI) {
    if let Some(r) = result.as_mut() {
        for p in [
            &mut r.coefficients,
            &mut r.vcov,
            &mut r.vcov_sandwich,
            &mut r.effective_df_diagonal,
            &mut r.effective_df_by_term,
        ] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
        r.n_features = 0;
        r.n_terms = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pentrace_round_trip() {
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x1[i] + (i as f64 * 1.3).cos())
            .collect();
        let ya = DataArray {
            data: y.as_ptr(),
            validity: std::ptr::null(),
            len: n,
        };
        let xa = [DataArray {
            data: x1.as_ptr(),
            validity: std::ptr::null(),
            len: n,
        }];
        let simple = [0.0, 1.0, 5.0];
        let grid = PenaltyGridFFI {
            simple: simple.as_ptr(),
            n_simple: 3,
            nonlinear: std::ptr::null(),
            n_nonlinear: 0,
            interaction: std::ptr::null(),
            n_interaction: 0,
            nonlinear_interaction: std::ptr::null(),
            n_nonlinear_interaction: 0,
        };
        let mut out = PentraceResultFFI::default();
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_pentrace(
                ya,
                xa.as_ptr(),
                1,
                std::ptr::null(),
                grid,
                0,
                0,
                0,
                &mut out,
                &mut err
            ));
            assert_eq!(out.n_rows, 3);
            assert!(((*out.rows).df - 1.0).abs() < 1e-8);
            anofox_free_pentrace(&mut out);

            let mut fit = PenalizedOlsResultFFI::default();
            let ya = DataArray {
                data: y.as_ptr(),
                validity: std::ptr::null(),
                len: n,
            };
            assert!(anofox_penalized_ols(
                ya,
                xa.as_ptr(),
                1,
                std::ptr::null(),
                2.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                &mut fit,
                &mut err
            ));
            assert_eq!(fit.n_features, 1);
            assert!(fit.effective_df.is_finite() && fit.sigma > 0.0);
            anofox_free_penalized_ols(&mut fit);
        }
    }
}
