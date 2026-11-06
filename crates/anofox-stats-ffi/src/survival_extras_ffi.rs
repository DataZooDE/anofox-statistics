//! C FFI for the AFT residuals and the extended survfit / survdiff (#322).
//! The structs are mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::aft_augment;
use anofox_stats_core::tests::survival::{
    survdiff_test, survfit_curve, HazardType, SurvConfType, SurvfitRows, SurvfitSpec,
};

unsafe fn sx_malloc<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn sx_free<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn sx_view<'a>(p: *const f64, n: usize) -> Option<&'a [f64]> {
    if p.is_null() {
        None
    } else if n == 0 {
        Some(&[])
    } else {
        Some(slice::from_raw_parts(p, n))
    }
}

// =============================================================================
// aft_augment
// =============================================================================

/// Per-input-row results of an AFT fit (NaN for rows left out). `score_mat`
/// is `n_rows * n_params`, row-major, in the order intercept, coefficients,
/// `log(scale)`. Arrays are `libc::malloc`ed; free with
/// `anofox_free_aft_augment_result`.
#[repr(C)]
#[derive(Debug)]
pub struct AftAugmentResultFFI {
    pub linear_predictor: *mut f64,
    pub standardized_residual: *mut f64,
    pub cox_snell_residual: *mut f64,
    pub score_mat: *mut f64,
    pub is_training: *mut bool,
    pub n_rows: usize,
    pub n_params: usize,
}

/// Fit an AFT model as `anofox_aft_fit` does and return its per-row
/// standardized and Cox–Snell residuals and score contributions.
///
/// # Safety
/// As `anofox_aft_fit`; `out_result` is required.
#[no_mangle]
pub unsafe extern "C" fn anofox_aft_augment(
    time: DataArray,
    x: *const DataArray,
    x_count: usize,
    event: DataArray,
    options: AftOptionsFFI,
    out_result: *mut AftAugmentResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        if x.is_null() || x_count == 0 {
            if !out_error.is_null() {
                (*out_error).set(
                    if x_count == 0 {
                        ErrorCode::InsufficientData
                    } else {
                        ErrorCode::InvalidInput
                    },
                    "x is NULL or empty",
                );
            }
            return false;
        }
        let x_vecs: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
            .iter()
            .map(|a| a.to_vec())
            .collect();
        let dist = match options.dist {
            AftDistributionFFI::Weibull => AftDistribution::Weibull,
            AftDistributionFFI::LogNormal => AftDistribution::LogNormal,
            AftDistributionFFI::LogLogistic => AftDistribution::LogLogistic,
            AftDistributionFFI::Exponential => AftDistribution::Exponential,
        };
        let opts = AftOptions::default()
            .dist(dist)
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations)
            .tolerance(options.tolerance)
            .compute_inference(false)
            .confidence_level(options.confidence_level)
            .priors(priors_from_ffi(options.priors, options.priors_len))
            .vcov(vcov_from_ffi(options.vcov));
        match aft_augment(&time.to_vec(), &x_vecs, &event.to_vec(), &opts) {
            Ok(a) => {
                let n = a.is_training.len();
                let out = AftAugmentResultFFI {
                    linear_predictor: sx_malloc(&a.linear_predictor),
                    standardized_residual: sx_malloc(&a.standardized_residuals),
                    cox_snell_residual: sx_malloc(&a.cox_snell_residuals),
                    score_mat: sx_malloc(&a.score),
                    is_training: sx_malloc(&a.is_training),
                    n_rows: n,
                    n_params: a.n_params,
                };
                let failed = n > 0
                    && (out.linear_predictor.is_null()
                        || out.standardized_residual.is_null()
                        || out.cox_snell_residual.is_null()
                        || out.is_training.is_null()
                        || (a.n_params > 0 && out.score_mat.is_null()));
                *out_result = out;
                if failed {
                    anofox_free_aft_augment_result(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate AFT augment result",
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

/// Free the arrays of a result filled by `anofox_aft_augment`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_aft_augment`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_aft_augment_result(result: *mut AftAugmentResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.linear_predictor,
            &mut r.standardized_residual,
            &mut r.cox_snell_residual,
            &mut r.score_mat,
        ] {
            sx_free(p);
        }
        sx_free(&mut r.is_training);
        r.n_rows = 0;
    })
}

// =============================================================================
// survfit (Kaplan-Meier with weights, entry times, robust SEs, cumulative hazard)
// =============================================================================

/// Options of `anofox_survfit`. `conf_type`: 0 log, 1 log-log, 2 plain;
/// `hazard`: 0 Nelson-Aalen, 1 Fleming-Harrington; `robust`: -1 survfit's
/// default, 0 Greenwood, 1 infinitesimal jackknife.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SurvfitOptionsFFI {
    pub conf_level: f64,
    pub conf_type: i32,
    pub hazard: i32,
    pub robust: i32,
    pub timefix: bool,
}

/// The rows of `anofox_survfit`: `time` and `event` (non-zero = event) are
/// required, the optional columns are NULL when absent. Each holds `n_rows`
/// values; rows with a non-finite value in any column are left out.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SurvfitDataFFI {
    pub time: *const f64,
    pub event: *const f64,
    pub entry: *const f64,
    pub weights: *const f64,
    pub cluster: *const f64,
    pub n_rows: usize,
}

/// One survival curve, `len` points (the distinct times). Counts are weighted
/// sums. `std_err` is the SE of `surv`, `std_chaz` of `cumhaz`; NaN where R
/// gives `NA`. Arrays are `libc::malloc`ed; free with
/// `anofox_free_survfit_result`.
#[repr(C)]
#[derive(Debug)]
pub struct SurvfitResultFFI {
    pub time: *mut f64,
    pub n_risk: *mut f64,
    pub n_event: *mut f64,
    pub n_censor: *mut f64,
    pub surv: *mut f64,
    pub std_err: *mut f64,
    pub conf_low: *mut f64,
    pub conf_high: *mut f64,
    pub cumhaz: *mut f64,
    pub std_chaz: *mut f64,
    pub len: usize,
    pub n: usize,
}

/// Kaplan-Meier curve and cumulative hazard, R
/// `survfit(Surv(entry, time, event) ~ 1, weights, id, conf.type, ctype, robust)`
/// (anofox-regression's `survival::survfit`).
///
/// # Safety
/// `data` must point to a valid `SurvfitDataFFI`; `out_result` is required.
#[no_mangle]
pub unsafe extern "C" fn anofox_survfit(
    data: *const SurvfitDataFFI,
    options: SurvfitOptionsFFI,
    out_result: *mut SurvfitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if !check_confidence_level(options.conf_level, out_error) {
            return false;
        }
        if data.is_null() {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "data is NULL");
            }
            return false;
        }
        let d = &*data;
        let n = d.n_rows;
        let (Some(time), Some(event)) = (sx_view(d.time, n), sx_view(d.event, n)) else {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "time or event is NULL");
            }
            return false;
        };
        let conf_type = match options.conf_type {
            0 => SurvConfType::Log,
            1 => SurvConfType::LogLog,
            2 => SurvConfType::Plain,
            _ => {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::InvalidInput, "unknown conf_type");
                }
                return false;
            }
        };
        let hazard = match options.hazard {
            0 => HazardType::NelsonAalen,
            1 => HazardType::FlemingHarrington,
            _ => {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::InvalidInput, "unknown hazard");
                }
                return false;
            }
        };
        let spec = SurvfitSpec {
            conf_level: options.conf_level,
            conf_type,
            hazard,
            timefix: options.timefix,
            robust: match options.robust {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            },
        };
        let rows = SurvfitRows {
            time,
            event,
            entry: sx_view(d.entry, n),
            weights: sx_view(d.weights, n),
            cluster: sx_view(d.cluster, n),
        };
        match survfit_curve(&rows, &spec) {
            Ok(r) => {
                let c = &r.curve;
                let len = c.time.len();
                let out = SurvfitResultFFI {
                    time: sx_malloc(&c.time),
                    n_risk: sx_malloc(&c.n_risk),
                    n_event: sx_malloc(&c.n_event),
                    n_censor: sx_malloc(&c.n_censor),
                    surv: sx_malloc(&c.surv),
                    std_err: sx_malloc(&r.std_err),
                    conf_low: sx_malloc(&c.lower),
                    conf_high: sx_malloc(&c.upper),
                    cumhaz: sx_malloc(&c.cumhaz),
                    std_chaz: sx_malloc(&c.std_chaz),
                    len,
                    n: c.records,
                };
                let failed = len > 0
                    && [
                        out.time,
                        out.n_risk,
                        out.n_event,
                        out.n_censor,
                        out.surv,
                        out.std_err,
                        out.conf_low,
                        out.conf_high,
                        out.cumhaz,
                        out.std_chaz,
                    ]
                    .iter()
                    .any(|p| p.is_null());
                *out_result = out;
                if failed {
                    anofox_free_survfit_result(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate survfit curve",
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

/// Free the arrays of a result filled by `anofox_survfit`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_survfit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_survfit_result(result: *mut SurvfitResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.time,
            &mut r.n_risk,
            &mut r.n_event,
            &mut r.n_censor,
            &mut r.surv,
            &mut r.std_err,
            &mut r.conf_low,
            &mut r.conf_high,
            &mut r.cumhaz,
            &mut r.std_chaz,
        ] {
            sx_free(p);
        }
        r.len = 0;
    })
}

// =============================================================================
// survdiff (G-rho family, stratified)
// =============================================================================

/// G-rho test of equal survival across the groups coded by `groups`
/// (`time.len` codes), R `survdiff(Surv(time, event) ~ group + strata(s), rho)`
/// (anofox-regression's `survival::survdiff`). `strata` is NULL or holds
/// `time.len` stratum values.
///
/// # Safety
/// `time` and `event` must be valid arrays of equal length, `groups` (and
/// `strata` unless NULL) must hold `time.len` values; `out_result` is required.
#[no_mangle]
pub unsafe extern "C" fn anofox_survdiff(
    time: DataArray,
    event: DataArray,
    groups: *const u64,
    strata: *const f64,
    rho: f64,
    out_result: *mut TestSummaryFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let n = time.len;
        if groups.is_null() && n > 0 {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "groups is NULL");
            }
            return false;
        }
        let codes: &[u64] = if n == 0 {
            &[]
        } else {
            slice::from_raw_parts(groups, n)
        };
        let events: Vec<bool> = event.to_vec().iter().map(|&e| e != 0.0).collect();
        let r = survdiff_test(&time.to_vec(), &events, codes, sx_view(strata, n), rho);
        write_summary(r, out_result, out_error)
    })
}

#[cfg(test)]
mod layout_tests {
    use super::{AftAugmentResultFFI, SurvfitDataFFI, SurvfitOptionsFFI, SurvfitResultFFI};
    use std::mem::{offset_of, size_of};

    /// Mirrors the static_asserts in src/aggregate_functions/survival_aggregate.cpp
    /// and src/aggregate_functions/aft_aggregate.cpp.
    #[test]
    fn layouts() {
        let p = size_of::<*mut f64>();
        assert_eq!(offset_of!(AftAugmentResultFFI, n_rows), 5 * p);
        assert_eq!(offset_of!(SurvfitDataFFI, n_rows), 5 * p);
        assert_eq!(offset_of!(SurvfitResultFFI, len), 10 * p);
        assert_eq!(
            offset_of!(SurvfitOptionsFFI, timefix),
            size_of::<f64>() + 12
        );
    }
}
