//! C FFI for the predictive accuracy indexes (#308): concordance, somers2,
//! rcorr.cens, Gini's mean difference, R2Measures, the lrm stats and val.prob.
//! The structs are mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::diagnostics::accuracy::{
    concordance, gini_md, lrm_stats, r2_measures, rcorr_cens, somers2, val_prob,
    CensoredConcordance, Concordance, LrmStats, R2Measures, Somers2, ValProb,
};

/// Pair counts and rank indexes of `anofox_concordance`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConcordanceFFI {
    pub concordant: f64,
    pub discordant: f64,
    pub tied_x: f64,
    pub tied_y: f64,
    pub tied_xy: f64,
    pub c: f64,
    pub dxy: f64,
    pub gamma: f64,
    pub tau_a: f64,
}

impl From<&Concordance> for ConcordanceFFI {
    fn from(r: &Concordance) -> Self {
        Self {
            concordant: r.concordant,
            discordant: r.discordant,
            tied_x: r.tied_x,
            tied_y: r.tied_y,
            tied_xy: r.tied_xy,
            c: r.c(),
            dxy: r.dxy(),
            gamma: r.gamma(),
            tau_a: r.tau_a(),
        }
    }
}

/// Result of `anofox_somers2`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Somers2FFI {
    pub c: f64,
    pub dxy: f64,
    /// Observations (sum of weights when weighted).
    pub n: f64,
}

impl From<&Somers2> for Somers2FFI {
    fn from(r: &Somers2) -> Self {
        Self {
            c: r.c,
            dxy: r.dxy,
            n: r.n,
        }
    }
}

/// Result of `anofox_rcorr_cens`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RcorrCensFFI {
    pub c: f64,
    pub dxy: f64,
    pub sd: f64,
    pub relevant_pairs: f64,
    pub concordant: f64,
    pub uncertain: f64,
    pub n: usize,
    pub missing: usize,
    pub uncensored: usize,
}

impl From<&CensoredConcordance> for RcorrCensFFI {
    fn from(r: &CensoredConcordance) -> Self {
        Self {
            c: r.c,
            dxy: r.dxy,
            sd: r.sd,
            relevant_pairs: r.relevant_pairs,
            concordant: r.concordant,
            uncertain: r.uncertain,
            n: r.n,
            missing: r.missing,
            uncensored: r.uncensored,
        }
    }
}

/// `Hmisc::R2Measures`; the effective-sample-size fields are NaN when no
/// effective sample size was given.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct R2MeasuresFFI {
    pub r2: f64,
    pub r2_adj: f64,
    pub r2_ess: f64,
    pub r2_adj_ess: f64,
}

impl From<&R2Measures> for R2MeasuresFFI {
    fn from(r: &R2Measures) -> Self {
        Self {
            r2: r.r2,
            r2_adj: r.r2_adj,
            r2_ess: r.r2_ess.unwrap_or(f64::NAN),
            r2_adj_ess: r.r2_adj_ess.unwrap_or(f64::NAN),
        }
    }
}

/// The `stats` vector of a binary `rms::lrm` fit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LrmStatsFFI {
    pub sum_weights: f64,
    pub model_lr: f64,
    pub df: f64,
    pub p_value: f64,
    pub c: f64,
    pub dxy: f64,
    pub gamma: f64,
    pub tau_a: f64,
    pub r2: f64,
    /// R2Measures with n = the sum of weights and m = the effective sample size.
    pub r2_measures: R2MeasuresFFI,
    /// Effective sample size `3 n1 n0 / n` of the R2(m) measures.
    pub ess: f64,
    pub brier: f64,
    pub g: f64,
    pub gr: f64,
    pub gp: f64,
    pub n: usize,
}

impl From<&LrmStats> for LrmStatsFFI {
    fn from(r: &LrmStats) -> Self {
        Self {
            sum_weights: r.sum_weights,
            model_lr: r.model_lr,
            df: r.df,
            p_value: r.p_value,
            c: r.c,
            dxy: r.dxy,
            gamma: r.gamma,
            tau_a: r.tau_a,
            r2: r.r2,
            r2_measures: R2MeasuresFFI::from(&r.r2_measures),
            ess: r.effective_sample_size,
            brier: r.brier,
            g: r.g,
            gr: r.gr,
            gp: r.gp,
            n: r.obs,
        }
    }
}

/// Result of `anofox_val_prob` (`rms::val.prob`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ValProbFFI {
    pub dxy: f64,
    pub c: f64,
    pub r2: f64,
    pub d: f64,
    pub d_chisq: f64,
    pub d_p: f64,
    pub u: f64,
    pub u_chisq: f64,
    pub u_p: f64,
    pub q: f64,
    pub brier: f64,
    pub intercept: f64,
    pub slope: f64,
    pub emax: f64,
    pub e90: f64,
    pub eavg: f64,
    pub s_z: f64,
    pub s_p: f64,
}

impl From<&ValProb> for ValProbFFI {
    fn from(r: &ValProb) -> Self {
        Self {
            dxy: r.dxy,
            c: r.c,
            r2: r.r2,
            d: r.d,
            d_chisq: r.d_chisq,
            d_p: r.d_p,
            u: r.u,
            u_chisq: r.u_chisq,
            u_p: r.u_p,
            q: r.q,
            brier: r.brier,
            intercept: r.intercept,
            slope: r.slope,
            emax: r.emax,
            e90: r.e90,
            eavg: r.eavg,
            s_z: r.s_z,
            s_p: r.s_p,
        }
    }
}

/// `Some(weights)` unless the array is empty.
unsafe fn optional_weights(weights: &DataArray) -> Option<Vec<f64>> {
    if weights.len == 0 {
        None
    } else {
        Some(weights.to_vec())
    }
}

unsafe fn write_out<T, R>(
    result: anofox_stats_core::StatsResult<T>,
    convert: impl FnOnce(T) -> R,
    out_result: *mut R,
    out_error: *mut AnofoxError,
) -> bool {
    match result {
        Ok(r) => {
            *out_result = convert(r);
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

/// Concordance of the prediction `x` with the ordered response `y`, weighted
/// by `w_i w_j` when `weights` is non-empty (`survival::concordance`).
///
/// # Safety
/// - `x`, `y` and `weights` must be valid DataArrays (`weights` may be empty)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_concordance(
    x: DataArray,
    y: DataArray,
    weights: DataArray,
    out_result: *mut ConcordanceFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let w = optional_weights(&weights);
        let r = concordance(&x.to_vec(), &y.to_vec(), w.as_deref());
        write_out(r, |r| ConcordanceFFI::from(&r), out_result, out_error)
    })
}

/// ROC area and Somers' Dxy of `p` for the 0/1 response `y`
/// (`Hmisc::somers2`; case weights when `weights` is non-empty).
///
/// # Safety
/// - `p`, `y` and `weights` must be valid DataArrays (`weights` may be empty)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_somers2(
    p: DataArray,
    y: DataArray,
    weights: DataArray,
    out_result: *mut Somers2FFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let w = optional_weights(&weights);
        let r = somers2(&p.to_vec(), &y.to_vec(), w.as_deref());
        write_out(r, |r| Somers2FFI::from(&r), out_result, out_error)
    })
}

/// Harrell's C of `x` for right-censored `time` (`event` non-zero = observed),
/// as `Hmisc::rcorr.cens`.
///
/// # Safety
/// - `x`, `time` and `event` must be valid DataArrays
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_rcorr_cens(
    x: DataArray,
    time: DataArray,
    event: DataArray,
    out_result: *mut RcorrCensFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let event: Vec<bool> = event.to_vec().iter().map(|&e| e != 0.0).collect();
        let r = rcorr_cens(&x.to_vec(), &time.to_vec(), &event);
        write_out(r, |r| RcorrCensFFI::from(&r), out_result, out_error)
    })
}

/// Gini's mean difference of `x` (`Hmisc::GiniMd`); NaN for fewer than 2
/// values.
///
/// # Safety
/// - `x` must be a valid DataArray
/// - `out_value` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_gini_md(
    x: DataArray,
    out_value: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_value, out_error) {
            return false;
        }
        *out_value = gini_md(&x.to_vec());
        true
    })
}

/// `Hmisc::R2Measures(lr, p, n, ess)`; `ess` NaN for none.
///
/// # Safety
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_r2_measures(
    lr: f64,
    p: f64,
    n: f64,
    ess: f64,
    out_result: *mut R2MeasuresFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let ess = if ess.is_nan() { None } else { Some(ess) };
        *out_result = R2MeasuresFFI::from(&r2_measures(lr, p, n, ess));
        true
    })
}

/// `rms::lrm` stats of a binary model from its 0/1 response `y`, linear
/// predictor `lp` (at the MLE) and model d.f.; case weights when `weights`
/// is non-empty.
///
/// # Safety
/// - `y`, `lp` and `weights` must be valid DataArrays (`weights` may be empty)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_lrm_stats(
    y: DataArray,
    lp: DataArray,
    weights: DataArray,
    df: f64,
    out_result: *mut LrmStatsFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let w = optional_weights(&weights);
        let r = lrm_stats(&y.to_vec(), &lp.to_vec(), w.as_deref(), df);
        write_out(r, |s| LrmStatsFFI::from(&s), out_result, out_error)
    })
}

/// External validation of the probabilities `p` against the 0/1 outcomes
/// `y`, as `rms::val.prob(p, y, pl = FALSE)`.
///
/// # Safety
/// - `p` and `y` must be valid DataArrays
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_val_prob(
    p: DataArray,
    y: DataArray,
    out_result: *mut ValProbFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let r = val_prob(&p.to_vec(), &y.to_vec());
        write_out(r, |r| ValProbFFI::from(&r), out_result, out_error)
    })
}
