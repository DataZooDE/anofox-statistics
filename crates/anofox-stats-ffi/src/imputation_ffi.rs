//! C FFI for multiple imputation (`aregImpute`) and Rubin pooling (#321).
//! The structs are mirrored in `src/include/anofox_stats_ffi.h`. Every output
//! buffer is allocated by the caller, who knows its size.

use super::*;
use anofox_stats_core::imputation::{
    areg_impute, pool_rubin, AregImputeConfig, DonorMatch, ImputeVariable,
};

/// Options of `anofox_areg_impute` (`Hmisc::aregImpute` names in brackets).
/// Fixed-width fields only, so the layout is the same on every target.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AregImputeOptionsFFI {
    /// Tricube scale of weighted donor matching (`fweighted`).
    pub fweighted: f64,
    /// Iterations before the first kept imputation (`burnin`); negative: Hmisc's default.
    pub burnin: i64,
    /// Number of imputations (`n.impute`).
    pub n_impute: u64,
    /// Spline knots (`nk`).
    pub nk: u64,
    /// Seed of R's generator (`set.seed`).
    pub seed: i32,
    /// Non-zero: `match = "closest"`; zero: `"weighted"`.
    pub closest: i32,
    /// Non-zero: `tlinear = TRUE`.
    pub tlinear: i32,
    pub _reserved: i32,
}

/// One pooled coefficient. Mirrors `AnofoxPooledTerm`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PooledTermFFI {
    pub estimate: f64,
    pub std_error: f64,
    pub statistic: f64,
    pub df: f64,
    pub p_value: f64,
    pub conf_low: f64,
    pub conf_high: f64,
    pub riv: f64,
    pub lambda: f64,
    pub fmi: f64,
}

/// The D1 pooled Wald test; NaN when not computed. Mirrors `AnofoxPooledWald`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PooledWaldFFI {
    pub f: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    pub riv: f64,
}

/// Multiple imputation of `n_rows` rows of `n_cols` values (row-major in
/// `data`, NaN = missing) by `Hmisc::aregImpute`. `kinds` holds one entry per
/// column: 0 numeric (spline), 1 linear, 2 categorical. Writes the completed
/// data sets to `out_completed` (`n_impute * n_rows * n_cols` values,
/// imputation-major, then row-major) and the R² of each column's last
/// imputation model to `out_r_squared` (`n_cols` values, NaN for complete
/// columns).
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` doubles, `kinds` to `n_cols` ints
/// - `out_completed` / `out_r_squared` must be writable for the sizes above
/// - `out_error` must be a valid pointer
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_areg_impute(
    data: *const f64,
    n_rows: usize,
    n_cols: usize,
    kinds: *const i32,
    options: AregImputeOptionsFFI,
    out_completed: *mut f64,
    out_r_squared: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_completed, out_error) || !check_out(out_r_squared, out_error) {
            return false;
        }
        if kinds.is_null() || (data.is_null() && n_rows * n_cols > 0) {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "data or kinds is NULL");
            }
            return false;
        }
        let mut kind_list = Vec::with_capacity(n_cols);
        for &k in slice::from_raw_parts(kinds, n_cols) {
            kind_list.push(match k {
                0 => ImputeVariable::Numeric,
                1 => ImputeVariable::Linear,
                2 => ImputeVariable::Categorical,
                _ => {
                    if !out_error.is_null() {
                        (*out_error).set(ErrorCode::InvalidInput, "unknown variable kind");
                    }
                    return false;
                }
            });
        }
        let rows: Vec<Vec<f64>> = if n_rows == 0 || n_cols == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(data, n_rows * n_cols)
                .chunks(n_cols)
                .map(|r| r.to_vec())
                .collect()
        };
        let config = AregImputeConfig {
            n_impute: options.n_impute as usize,
            nk: options.nk as usize,
            tlinear: options.tlinear != 0,
            burnin: (options.burnin >= 0).then_some(options.burnin as usize),
            matching: if options.closest != 0 {
                DonorMatch::Closest
            } else {
                DonorMatch::Weighted(options.fweighted)
            },
            seed: options.seed,
        };
        match areg_impute(&rows, &kind_list, &config) {
            Ok(r) => {
                let out =
                    slice::from_raw_parts_mut(out_completed, config.n_impute * n_rows * n_cols);
                for (dst, v) in out.iter_mut().zip(r.completed.iter().flatten().flatten()) {
                    *dst = *v;
                }
                std::ptr::copy_nonoverlapping(r.r_squared.as_ptr(), out_r_squared, n_cols);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Rubin's rules over `m` fits with `dim` coefficients each: `estimates`
/// (`m * dim`, fit-major) and `vcovs` (`m * dim * dim`, each row-major).
/// `df_complete` is the complete-data d.f. (NaN: none, Rubin's d.f.). The D1
/// Wald test covers the coefficients `wald_from..dim` (NaN when empty or
/// singular). Writes `dim` terms to `out_terms`, the pooled covariance
/// (`dim * dim`, row-major) to `out_vcov` and the test to `out_wald`.
///
/// # Safety
/// - `estimates` / `vcovs` must point to the sizes above
/// - `out_terms`, `out_vcov`, `out_wald` must be writable for the sizes above
/// - `out_error` must be a valid pointer
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_pool_rubin(
    estimates: *const f64,
    vcovs: *const f64,
    m: usize,
    dim: usize,
    df_complete: f64,
    confidence_level: f64,
    wald_from: usize,
    out_terms: *mut PooledTermFFI,
    out_vcov: *mut f64,
    out_wald: *mut PooledWaldFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_terms, out_error)
            || !check_out(out_vcov, out_error)
            || !check_out(out_wald, out_error)
        {
            return false;
        }
        if m * dim > 0 && (estimates.is_null() || vcovs.is_null()) {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "estimates or vcovs is NULL");
            }
            return false;
        }
        let (est, cov): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if m * dim == 0 {
            (vec![Vec::new(); m], vec![Vec::new(); m])
        } else {
            (
                slice::from_raw_parts(estimates, m * dim)
                    .chunks(dim)
                    .map(|c| c.to_vec())
                    .collect(),
                slice::from_raw_parts(vcovs, m * dim * dim)
                    .chunks(dim * dim)
                    .map(|c| c.to_vec())
                    .collect(),
            )
        };
        let dfc = (!df_complete.is_nan()).then_some(df_complete);
        match pool_rubin(&est, &cov, dfc, confidence_level, wald_from) {
            Ok(r) => {
                let terms = slice::from_raw_parts_mut(out_terms, dim);
                for (dst, t) in terms.iter_mut().zip(&r.terms) {
                    *dst = PooledTermFFI {
                        estimate: t.estimate,
                        std_error: t.std_error,
                        statistic: t.statistic,
                        df: t.df,
                        p_value: t.p_value,
                        conf_low: t.conf_low,
                        conf_high: t.conf_high,
                        riv: t.riv,
                        lambda: t.lambda,
                        fmi: t.fmi,
                    };
                }
                std::ptr::copy_nonoverlapping(r.vcov.as_ptr(), out_vcov, dim * dim);
                *out_wald = match r.wald {
                    Some(w) => PooledWaldFFI {
                        f: w.f,
                        df1: w.df1,
                        df2: w.df2,
                        p_value: w.p_value,
                        riv: w.riv,
                    },
                    None => PooledWaldFFI {
                        f: f64::NAN,
                        df1: f64::NAN,
                        df2: f64::NAN,
                        p_value: f64::NAN,
                        riv: f64::NAN,
                    },
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
