//! C FFI for the rms-style Wald ANOVA and fast backward step-down (#311):
//! `rms::anova.rms` / `rms::fastbw` through upstream's `inference::anova` and
//! `inference::fastbw`. The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`; arrays and strings are `libc::malloc`ed
//! and released by the matching `anofox_free_*` function.

use super::*;
use anofox_stats_core::models::{
    fastbw, wald_anova, FastbwRule, ModelCovariance, TermSpec, WaldAnovaRow,
};
use libc::c_char;
use std::ffi::CStr;

/// One row of `anofox_wald_anova`. NaN marks a value that does not apply.
#[repr(C)]
#[derive(Debug)]
pub struct WaldAnovaRowFFI {
    /// Label as `anova.rms` prints it.
    pub label: *mut c_char,
    /// What the row tests (`main_effect`, `nonlinear`, `total`, ...).
    pub kind: *mut c_char,
    /// 0-based term (caller's numbering), -1 for the totals.
    pub term: i64,
    /// 0-based slope columns tested (model order).
    pub columns: *mut u64,
    pub n_columns: usize,
    pub chi_square: f64,
    pub df: f64,
    pub f_statistic: f64,
    pub df_residual: f64,
    pub p_value: f64,
    pub partial_ss: f64,
}

/// Result of `anofox_wald_anova`. The error row fields are NaN unless an F
/// test with σ² was requested.
#[repr(C)]
#[derive(Debug)]
pub struct WaldAnovaResultFFI {
    pub rows: *mut WaldAnovaRowFFI,
    pub n_rows: usize,
    pub error_df: f64,
    pub error_ss: f64,
    pub error_ms: f64,
}

impl Default for WaldAnovaResultFFI {
    fn default() -> Self {
        Self {
            rows: std::ptr::null_mut(),
            n_rows: 0,
            error_df: f64::NAN,
            error_ss: f64::NAN,
            error_ms: f64::NAN,
        }
    }
}

/// One deletion of `anofox_fastbw`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FastbwStepFFI {
    /// 0-based term (caller's numbering).
    pub term: u64,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
    pub residual_chi_square: f64,
    pub residual_df: f64,
    pub residual_p_value: f64,
    pub aic: f64,
}

/// Result of `anofox_fastbw`.
#[repr(C)]
#[derive(Debug)]
pub struct FastbwResultFFI {
    pub deleted: *mut FastbwStepFFI,
    pub n_deleted: usize,
    /// 0-based kept terms, ascending.
    pub kept_terms: *mut u64,
    pub n_kept: usize,
    /// Approximate slopes of the reduced model (model order, 0 when deleted).
    pub coefficients: *mut f64,
    pub n_coefficients: usize,
    /// Approximate intercept, NaN without one.
    pub intercept: f64,
    /// Approximate reduced-model covariance (`fastbw`'s `var`), `vcov_dim`
    /// square, row-major in the layout of the input covariance; zero rows and
    /// columns for deleted parameters.
    pub vcov: *mut f64,
    pub vcov_dim: usize,
    /// 0-based kept parameters in that layout (`parms.kept`), ascending.
    pub kept_parameters: *mut u64,
    pub n_kept_parameters: usize,
}

impl Default for FastbwResultFFI {
    fn default() -> Self {
        Self {
            deleted: std::ptr::null_mut(),
            n_deleted: 0,
            kept_terms: std::ptr::null_mut(),
            n_kept: 0,
            coefficients: std::ptr::null_mut(),
            n_coefficients: 0,
            intercept: f64::NAN,
            vcov: std::ptr::null_mut(),
            vcov_dim: 0,
            kept_parameters: std::ptr::null_mut(),
            n_kept_parameters: 0,
        }
    }
}

unsafe fn free_ptr<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn alloc_array<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn fail(out_error: *mut AnofoxError, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
    false
}

/// The model's coefficients and covariance; `None` on NULL pointers.
unsafe fn read_model<'a>(
    coefficients: *const f64,
    p: usize,
    intercept: f64,
    vcov: *const f64,
    dim: usize,
) -> Option<ModelCovariance<'a>> {
    if (p > 0 && coefficients.is_null()) || (dim > 0 && vcov.is_null()) {
        return None;
    }
    let slice_or_empty = |ptr: *const f64, n: usize| {
        if n == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(ptr, n)
        }
    };
    Some(ModelCovariance {
        coefficients: slice_or_empty(coefficients, p),
        intercept,
        vcov: slice_or_empty(vcov, dim * dim),
        dim,
    })
}

/// `term_of_column` (negative: no term) as options.
unsafe fn read_terms(term_of_column: *const i64, p: usize) -> Option<Vec<Option<usize>>> {
    if p > 0 && term_of_column.is_null() {
        return None;
    }
    if p == 0 {
        return Some(Vec::new());
    }
    Some(
        slice::from_raw_parts(term_of_column, p)
            .iter()
            .map(|&t| (t >= 0).then_some(t as usize))
            .collect(),
    )
}

unsafe fn write_row(r: &WaldAnovaRow) -> WaldAnovaRowFFI {
    WaldAnovaRowFFI {
        label: alloc_string(&r.label),
        kind: alloc_string(r.kind),
        term: r.term.map_or(-1, |t| t as i64),
        columns: alloc_u64(&r.columns),
        n_columns: r.columns.len(),
        chi_square: r.chi_square,
        df: r.df as f64,
        f_statistic: r.f,
        df_residual: r.df_residual,
        p_value: r.p_value,
        partial_ss: r.partial_ss,
    }
}

/// `rms::anova.rms` table of a fitted model from its slopes (`p`, feature
/// order) and row-major covariance (`dim = p` or `p + 1`, intercept first).
/// `term_of_column[j]` is the 0-based term of slope `j` (negative: none);
/// `term_names` has `n_terms` names; `interaction_factors` (NULL: no
/// interactions) holds two 0-based main-effect terms per term, negative for a
/// main effect; `nonlinear` (NULL: all linear) flags the nonlinear slopes.
/// `f_test` refers every row to `F(df, df_residual)`; a finite `sigma2` adds
/// partial sums of squares (and the error row for an F test).
///
/// # Safety
/// The arrays must hold the stated number of entries; `out_result` must be
/// valid and is released with `anofox_free_wald_anova`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_wald_anova(
    coefficients: *const f64,
    p: usize,
    vcov: *const f64,
    dim: usize,
    term_of_column: *const i64,
    nonlinear: *const bool,
    term_names: *const *const c_char,
    n_terms: usize,
    interaction_factors: *const i64,
    f_test: bool,
    df_residual: f64,
    sigma2: f64,
    out_result: *mut WaldAnovaResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = WaldAnovaResultFFI::default();
        let Some(model) = read_model(coefficients, p, f64::NAN, vcov, dim) else {
            return fail(out_error, "NULL coefficients or vcov");
        };
        let Some(terms) = read_terms(term_of_column, p) else {
            return fail(out_error, "NULL terms");
        };
        if n_terms > 0 && term_names.is_null() {
            return fail(out_error, "NULL term names");
        }
        let names: Vec<String> = if n_terms == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(term_names, n_terms)
                .iter()
                .map(|&s| {
                    if s.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(s).to_string_lossy().into_owned()
                    }
                })
                .collect()
        };
        let factors: Vec<Option<(usize, usize)>> = if interaction_factors.is_null() || n_terms == 0
        {
            vec![None; n_terms]
        } else {
            slice::from_raw_parts(interaction_factors, 2 * n_terms)
                .chunks(2)
                .map(|f| (f[0] >= 0 && f[1] >= 0).then(|| (f[0] as usize, f[1] as usize)))
                .collect()
        };
        let specs: Vec<TermSpec> = names
            .into_iter()
            .zip(factors)
            .map(|(name, factors)| TermSpec { name, factors })
            .collect();
        let nl: Vec<bool> = if nonlinear.is_null() || p == 0 {
            vec![false; p]
        } else {
            slice::from_raw_parts(nonlinear, p).to_vec()
        };
        let sigma2 = sigma2.is_finite().then_some(sigma2);
        match wald_anova(
            &model,
            &terms,
            &specs,
            &nl,
            f_test.then_some(df_residual),
            sigma2,
        ) {
            Ok(r) => {
                let rows: Vec<WaldAnovaRowFFI> = r.rows.iter().map(|row| write_row(row)).collect();
                let n_rows = rows.len();
                let ptr = libc::malloc(n_rows * std::mem::size_of::<WaldAnovaRowFFI>())
                    as *mut WaldAnovaRowFFI;
                if ptr.is_null() {
                    if !out_error.is_null() {
                        (*out_error).set(ErrorCode::AllocationFailure, "allocation failed");
                    }
                    return false;
                }
                for (i, row) in rows.into_iter().enumerate() {
                    std::ptr::write(ptr.add(i), row);
                }
                let (error_df, error_ss, error_ms) =
                    r.error.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                *out_result = WaldAnovaResultFFI {
                    rows: ptr,
                    n_rows,
                    error_df,
                    error_ss,
                    error_ms,
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

/// Free the rows of a `WaldAnovaResultFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_wald_anova`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_wald_anova(result: *mut WaldAnovaResultFFI) {
    if let Some(r) = result.as_mut() {
        if !r.rows.is_null() {
            for i in 0..r.n_rows {
                let row = &mut *r.rows.add(i);
                free_ptr(&mut row.label);
                free_ptr(&mut row.kind);
                free_ptr(&mut row.columns);
            }
        }
        free_ptr(&mut r.rows);
        r.n_rows = 0;
    }
}

/// `rms::fastbw` of a fitted model (slopes, intercept, covariance as for
/// `anofox_wald_anova`; slopes without a term are never deleted). `rule` 0 is
/// AIC (`χ² − k_aic·d.f. ≤ aics`), 1 is P (`p > sls`); `individual` applies it
/// to the deleted term's own test instead of the residual test.
///
/// # Safety
/// The arrays must hold the stated number of entries; `out_result` must be
/// valid and is released with `anofox_free_fastbw`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_fastbw(
    coefficients: *const f64,
    p: usize,
    intercept: f64,
    vcov: *const f64,
    dim: usize,
    term_of_column: *const i64,
    n_terms: usize,
    rule: i32,
    individual: bool,
    k_aic: f64,
    aics: f64,
    sls: f64,
    out_result: *mut FastbwResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = FastbwResultFFI::default();
        let Some(model) = read_model(coefficients, p, intercept, vcov, dim) else {
            return fail(out_error, "NULL coefficients or vcov");
        };
        let Some(terms) = read_terms(term_of_column, p) else {
            return fail(out_error, "NULL terms");
        };
        let rule = match rule {
            0 => FastbwRule::Aic { k: k_aic, aics },
            1 => FastbwRule::P { sls },
            other => return fail(out_error, &format!("unknown rule {other}")),
        };
        let names: Vec<String> = (0..n_terms).map(|t| format!("t{t}")).collect();
        match fastbw(&model, &terms, &names, rule, individual) {
            Ok(r) => {
                let steps: Vec<FastbwStepFFI> = r
                    .deleted
                    .iter()
                    .map(|d| FastbwStepFFI {
                        term: d.term as u64,
                        chi_square: d.chi_square,
                        df: d.df as f64,
                        p_value: d.p_value,
                        residual_chi_square: d.residual_chi_square,
                        residual_df: d.residual_df as f64,
                        residual_p_value: d.residual_p_value,
                        aic: d.aic,
                    })
                    .collect();
                *out_result = FastbwResultFFI {
                    deleted: alloc_array(&steps),
                    n_deleted: steps.len(),
                    kept_terms: alloc_u64(&r.kept_terms),
                    n_kept: r.kept_terms.len(),
                    coefficients: alloc_array(&r.coefficients),
                    n_coefficients: r.coefficients.len(),
                    intercept: r.intercept,
                    vcov: alloc_array(&r.vcov),
                    vcov_dim: dim,
                    kept_parameters: alloc_u64(&r.kept_parameters),
                    n_kept_parameters: r.kept_parameters.len(),
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

/// Free the arrays of a `FastbwResultFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_fastbw`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_fastbw(result: *mut FastbwResultFFI) {
    if let Some(r) = result.as_mut() {
        free_ptr(&mut r.deleted);
        free_ptr(&mut r.kept_terms);
        free_ptr(&mut r.coefficients);
        free_ptr(&mut r.vcov);
        free_ptr(&mut r.kept_parameters);
        r.n_deleted = 0;
        r.n_kept = 0;
        r.n_coefficients = 0;
        r.vcov_dim = 0;
        r.n_kept_parameters = 0;
    }
}
