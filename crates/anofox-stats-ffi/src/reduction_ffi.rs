//! C FFI for data reduction and transforms (#320): generalized Spearman ρ²,
//! Hoeffding's D, Box–Cox, variable clustering and redundancy analysis
//! (upstream anofox-regression `reduction` / `transform`). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::reduction::{
    box_cox, box_cox_inverse, box_cox_transform, hoeffding_d, parse_linkage, parse_redun_kind,
    parse_similarity, parse_similarity_transform, redun, spearman2, spearman2_categorical, varclus,
    BoxCoxSettings, RedunSettings,
};
use std::ffi::CStr;

unsafe fn malloc_doubles(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn malloc_i64(values: &[i64]) -> *mut i64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut i64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn free_ptr<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn doubles(p: *const f64, len: usize) -> Vec<f64> {
    if p.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(p, len).to_vec()
    }
}

unsafe fn columns(x: *const DataArray, x_count: usize) -> Vec<Vec<f64>> {
    if x.is_null() || x_count == 0 {
        return Vec::new();
    }
    slice::from_raw_parts(x, x_count)
        .iter()
        .map(|a| a.to_vec())
        .collect()
}

unsafe fn name(p: *const c_char, default: &str) -> String {
    if p.is_null() {
        default.to_string()
    } else {
        CStr::from_ptr(p).to_string_lossy().to_lowercase()
    }
}

unsafe fn set_alloc_error(out_error: *mut AnofoxError, what: &str) {
    if !out_error.is_null() {
        (*out_error).set(
            ErrorCode::AllocationFailure,
            &format!("Failed to allocate {what} result"),
        );
    }
}

unsafe fn set_invalid(out_error: *mut AnofoxError, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
}

fn to_i64(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

// ------------------------------------------------------------- spearman2 --

/// Generalized Spearman ρ² (`Hmisc::spearman2`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Spearman2ResultFFI {
    pub rho2: f64,
    pub adjusted_rho2: f64,
    pub statistic: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    pub n: i64,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

unsafe fn write_spearman2(
    r: anofox_stats_core::StatsResult<anofox_stats_core::reduction::Spearman2Fit>,
    out_result: *mut Spearman2ResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    match r {
        Ok(s) => {
            *out_result = Spearman2ResultFFI {
                rho2: s.rho2,
                adjusted_rho2: s.adjusted_rho2,
                statistic: s.f_statistic,
                df1: s.df1,
                df2: s.df2,
                p_value: s.p_value,
                n: to_i64(s.n),
                unified: TestSummaryFFI {
                    statistic: s.f_statistic,
                    df1: s.df1,
                    df2: s.df2,
                    p_value: s.p_value,
                    estimate: s.rho2,
                    n: to_i64(s.n),
                    method: types::fixed_buffer("Spearman rho^2"),
                    effect_name: types::fixed_buffer("rho2"),
                    effect_size: s.rho2,
                    ..TestSummaryFFI::default()
                },
            };
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

/// `spearman2(x, y, p)`: `y` and `x` hold `n` values each, `p` is 1 or 2.
///
/// # Safety
/// `y` and `x` must point to `n` doubles; `out_result`, `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_spearman2(
    y: *const f64,
    x: *const f64,
    n: usize,
    p: usize,
    out_result: *mut Spearman2ResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (y, x) = (doubles(y, n), doubles(x, n));
        write_spearman2(spearman2(&y, &x, p), out_result, out_error)
    })
}

/// `spearman2(factor(x), y)`: `codes` holds the level code of each row.
///
/// # Safety
/// `y` and `codes` must point to `n` values; `out_result`, `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_spearman2_categorical(
    y: *const f64,
    codes: *const u64,
    n: usize,
    out_result: *mut Spearman2ResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let y = doubles(y, n);
        let codes: Vec<usize> = if codes.is_null() || n == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(codes, n)
                .iter()
                .map(|&c| c as usize)
                .collect()
        };
        write_spearman2(spearman2_categorical(&y, &codes), out_result, out_error)
    })
}

// ---------------------------------------------------------- Hoeffding's D --

/// Hoeffding's D of one pair (`Hmisc::hoeffd`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HoeffdingResultFFI {
    /// `30 ×` Hoeffding's D.
    pub d: f64,
    pub p_value: f64,
    pub n: i64,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

/// Hoeffding's D of `x` and `y` (`n` values each).
///
/// # Safety
/// `x` and `y` must point to `n` doubles; `out_result`, `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_hoeffding_d(
    x: *const f64,
    y: *const f64,
    n: usize,
    out_result: *mut HoeffdingResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        match hoeffding_d(&doubles(x, n), &doubles(y, n)) {
            Ok(h) => {
                *out_result = HoeffdingResultFFI {
                    d: h.d,
                    p_value: h.p_value,
                    n: to_i64(h.n),
                    unified: TestSummaryFFI {
                        statistic: h.d,
                        p_value: h.p_value,
                        estimate: h.d,
                        n: to_i64(h.n),
                        method: types::fixed_buffer("Hoeffding's D"),
                        ..TestSummaryFFI::default()
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

// ---------------------------------------------------------------- Box–Cox --

/// Options of `anofox_box_cox`. A NaN number or a NULL / empty grid keeps
/// MASS's default.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BoxCoxOptionsFFI {
    pub lambdas: *const f64,
    pub n_lambdas: usize,
    pub eps: f64,
    pub fit_intercept: bool,
    pub confidence_level: f64,
}

/// A Box–Cox profile. The arrays are allocated by `anofox_box_cox` and
/// released with `anofox_free_box_cox_result`.
#[repr(C)]
#[derive(Debug)]
pub struct BoxCoxResultFFI {
    pub lambda: f64,
    pub max_log_likelihood: f64,
    /// NaN when the profile stays above the cut to the grid's end.
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub confidence_level: f64,
    pub lambdas: *mut f64,
    pub log_likelihood: *mut f64,
    pub n_lambdas: usize,
    pub n_observations: usize,
}

/// `MASS::boxcox(lm(y ~ x))` for `n` responses `y` and `x_count` predictor
/// columns `x` (none: intercept only).
///
/// # Safety
/// `y` must point to `n` doubles, `x` to `x_count` arrays of `n` values,
/// `options.lambdas` to `options.n_lambdas` doubles (or be NULL);
/// `out_result`, `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_box_cox(
    y: *const f64,
    n: usize,
    x: *const DataArray,
    x_count: usize,
    options: BoxCoxOptionsFFI,
    out_result: *mut BoxCoxResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let y = doubles(y, n);
        let x = columns(x, x_count);
        let finite = |v: f64| (!v.is_nan()).then_some(v);
        let settings = BoxCoxSettings {
            lambdas: (!options.lambdas.is_null() && options.n_lambdas > 0)
                .then(|| doubles(options.lambdas, options.n_lambdas)),
            eps: finite(options.eps),
            with_intercept: Some(options.fit_intercept),
            confidence_level: finite(options.confidence_level),
        };
        match box_cox(&y, &x, &settings) {
            Ok(f) => {
                let out = BoxCoxResultFFI {
                    lambda: f.lambda,
                    max_log_likelihood: f.max_log_likelihood,
                    ci_lower: f.ci_lower,
                    ci_upper: f.ci_upper,
                    confidence_level: settings.confidence_level.unwrap_or(0.95),
                    lambdas: malloc_doubles(&f.lambdas),
                    log_likelihood: malloc_doubles(&f.log_likelihood),
                    n_lambdas: f.lambdas.len(),
                    n_observations: f.n_observations,
                };
                let failed =
                    out.n_lambdas > 0 && (out.lambdas.is_null() || out.log_likelihood.is_null());
                *out_result = out;
                if failed {
                    anofox_free_box_cox_result(out_result);
                    set_alloc_error(out_error, "Box-Cox");
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

/// Free the arrays of a result filled by `anofox_box_cox`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_box_cox`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_box_cox_result(result: *mut BoxCoxResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_ptr(&mut r.lambdas);
        free_ptr(&mut r.log_likelihood);
    })
}

/// The Box–Cox transform `(y^λ − 1)/λ` (`log y` at `λ = 0`).
#[no_mangle]
pub extern "C" fn anofox_box_cox_transform(y: f64, lambda: f64) -> f64 {
    box_cox_transform(y, lambda)
}

/// The inverse Box–Cox transform `(λz + 1)^{1/λ}` (`exp z` at `λ = 0`).
#[no_mangle]
pub extern "C" fn anofox_box_cox_inverse(z: f64, lambda: f64) -> f64 {
    box_cox_inverse(z, lambda)
}

// ---------------------------------------------------------------- varclus --

/// A variable clustering. The arrays are allocated by `anofox_varclus` and
/// released with `anofox_free_varclus_result`.
#[repr(C)]
#[derive(Debug)]
pub struct VarclusResultFFI {
    /// Similarity matrix, row-major, `n_variables²` values.
    pub similarity: *mut f64,
    /// Merges in R's `hclust` convention, `2 (n_variables − 1)` values: the
    /// pair of merge `k` at `2k`, `2k + 1`; `-j` is variable `j` (1-based),
    /// `k` the cluster formed at step `k` (1-based).
    pub merge: *mut i64,
    /// `n_variables − 1` merge heights.
    pub height: *mut f64,
    /// Dendrogram order, 1-based variable numbers, `n_variables` values.
    pub order: *mut i64,
    /// Cluster (1-based) of each variable after the cut, or NULL without one.
    pub clusters: *mut i64,
    pub n_variables: usize,
    pub n_observations: usize,
}

/// `Hmisc::varclus` of `x_count` columns of `n` values (complete rows):
/// `similarity` is `spearman` / `pearson` / `hoeffding`, `transform`
/// `square` / `abs` / `none`, `linkage` an `hclust` method; NULL names keep
/// the defaults. A non-NaN `cut_height` also cuts the tree there.
///
/// # Safety
/// `x` must point to `x_count` arrays of equal length; the names must be
/// NULL or NUL-terminated; `out_result`, `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_varclus(
    x: *const DataArray,
    x_count: usize,
    similarity: *const c_char,
    transform: *const c_char,
    linkage: *const c_char,
    cut_height: f64,
    out_result: *mut VarclusResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (sim_name, trans_name, link_name) = (
            name(similarity, "spearman"),
            name(transform, "square"),
            name(linkage, "complete"),
        );
        let (Some(sim), Some(trans), Some(link)) = (
            parse_similarity(&sim_name),
            parse_similarity_transform(&trans_name),
            parse_linkage(&link_name),
        ) else {
            set_invalid(
                out_error,
                &format!("unknown similarity '{sim_name}', transform '{trans_name}' or method '{link_name}'"),
            );
            return false;
        };
        let cols = columns(x, x_count);
        let cut = (!cut_height.is_nan()).then_some(cut_height);
        match varclus(&cols, sim, trans, link, cut) {
            Ok(v) => {
                let p = v.similarity.len();
                let sim_flat: Vec<f64> = v.similarity.concat();
                let merge: Vec<i64> = v.merge.iter().flat_map(|m| m.iter().copied()).collect();
                let order: Vec<i64> = v.order.iter().map(|&o| to_i64(o + 1)).collect();
                let clusters: Option<Vec<i64>> = v
                    .clusters
                    .as_ref()
                    .map(|c| c.iter().map(|&k| to_i64(k + 1)).collect());
                let out = VarclusResultFFI {
                    similarity: malloc_doubles(&sim_flat),
                    merge: malloc_i64(&merge),
                    height: malloc_doubles(&v.height),
                    order: malloc_i64(&order),
                    clusters: clusters
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |c| malloc_i64(c)),
                    n_variables: p,
                    n_observations: v.n_observations,
                };
                let failed = out.similarity.is_null()
                    || out.merge.is_null()
                    || out.height.is_null()
                    || out.order.is_null()
                    || (clusters.is_some() && out.clusters.is_null());
                *out_result = out;
                if failed {
                    anofox_free_varclus_result(out_result);
                    set_alloc_error(out_error, "varclus");
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

/// Free the arrays of a result filled by `anofox_varclus`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_varclus`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_varclus_result(result: *mut VarclusResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_ptr(&mut r.similarity);
        free_ptr(&mut r.merge);
        free_ptr(&mut r.height);
        free_ptr(&mut r.order);
        free_ptr(&mut r.clusters);
    })
}

// ------------------------------------------------------------------ redun --

/// Options of `anofox_redun`. A NaN number, a negative `nk` or a negative
/// flag (-1) keeps Hmisc's default; flags are 0 / 1 otherwise.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RedunOptionsFFI {
    pub r2: f64,
    pub nk: i64,
    pub adjusted: i32,
    pub tlinear: i32,
    pub rank: i32,
    pub qrank: i32,
    pub allcat: i32,
}

/// A redundancy analysis; variable numbers are 1-based columns of `x`. The
/// arrays are allocated by `anofox_redun` and released with
/// `anofox_free_redun_result`.
#[repr(C)]
#[derive(Debug)]
pub struct RedunResultFFI {
    pub kept: *mut i64,
    pub n_kept: usize,
    /// In the order removed.
    pub removed: *mut i64,
    /// R² of each removed variable when removed, `n_removed` values.
    pub r_squared: *mut f64,
    pub n_removed: usize,
    /// R² of the earlier removed variables at each later removal, flat:
    /// removal `k` (0-based, from the second) contributes `k` values.
    pub r_squared_later: *mut f64,
    pub n_r_squared_later: usize,
    /// Variables analysed (all but `too_few`), with their first-step R².
    pub analysed: *mut i64,
    pub r_squared_first: *mut f64,
    pub n_analysed: usize,
    /// Constant variables left out.
    pub too_few: *mut i64,
    pub n_too_few: usize,
    pub n_observations: usize,
}

/// `Hmisc::redun` of `x_count` columns of `n` values; `kinds` holds one
/// NUL-terminated name per column (`spline` / `numeric`, `linear`,
/// `categorical`) or is NULL (all numeric).
///
/// # Safety
/// `x` must point to `x_count` arrays of equal length, `kinds` to
/// `x_count` NUL-terminated strings or be NULL; `out_result`, `out_error`
/// valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_redun(
    x: *const DataArray,
    x_count: usize,
    kinds: *const *const c_char,
    options: RedunOptionsFFI,
    out_result: *mut RedunResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let mut parsed = Vec::with_capacity(x_count);
        for j in 0..x_count {
            let k = if kinds.is_null() {
                "spline".to_string()
            } else {
                name(*kinds.add(j), "spline")
            };
            match parse_redun_kind(&k) {
                Some(v) => parsed.push(v),
                None => {
                    set_invalid(
                        out_error,
                        &format!("unknown variable type '{k}' (spline, linear or categorical)"),
                    );
                    return false;
                }
            }
        }
        let flag = |v: i32| (v >= 0).then_some(v != 0);
        let settings = RedunSettings {
            r2: (!options.r2.is_nan()).then_some(options.r2),
            adjusted: flag(options.adjusted),
            nk: usize::try_from(options.nk).ok(),
            tlinear: flag(options.tlinear),
            rank: flag(options.rank),
            qrank: flag(options.qrank),
            allcat: flag(options.allcat),
        };
        let cols = columns(x, x_count);
        match redun(&cols, &parsed, &settings) {
            Ok(r) => {
                let one_based =
                    |v: &[usize]| -> Vec<i64> { v.iter().map(|&j| to_i64(j + 1)).collect() };
                let later: Vec<f64> = r.r_squared_later.concat();
                let out = RedunResultFFI {
                    kept: malloc_i64(&one_based(&r.kept)),
                    n_kept: r.kept.len(),
                    removed: malloc_i64(&one_based(&r.removed)),
                    r_squared: malloc_doubles(&r.r_squared),
                    n_removed: r.removed.len(),
                    r_squared_later: malloc_doubles(&later),
                    n_r_squared_later: later.len(),
                    analysed: malloc_i64(&one_based(&r.analysed)),
                    r_squared_first: malloc_doubles(&r.r_squared_first),
                    n_analysed: r.analysed.len(),
                    too_few: malloc_i64(&one_based(&r.too_few)),
                    n_too_few: r.too_few.len(),
                    n_observations: r.n_observations,
                };
                let failed = (out.n_kept > 0 && out.kept.is_null())
                    || (out.n_removed > 0 && (out.removed.is_null() || out.r_squared.is_null()))
                    || (out.n_r_squared_later > 0 && out.r_squared_later.is_null())
                    || (out.n_analysed > 0 && out.analysed.is_null())
                    || (!r.r_squared_first.is_empty() && out.r_squared_first.is_null())
                    || (out.n_too_few > 0 && out.too_few.is_null());
                *out_result = out;
                if failed {
                    anofox_free_redun_result(out_result);
                    set_alloc_error(out_error, "redun");
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

/// Free the arrays of a result filled by `anofox_redun`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_redun`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_redun_result(result: *mut RedunResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_ptr(&mut r.kept);
        free_ptr(&mut r.removed);
        free_ptr(&mut r.r_squared);
        free_ptr(&mut r.r_squared_later);
        free_ptr(&mut r.analysed);
        free_ptr(&mut r.r_squared_first);
        free_ptr(&mut r.too_few);
    })
}
