//! C FFI for the data-drift tests (#245-#255). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::drift::{
    anderson_darling_ksamp, drift_test, Alternative, Binning, CvmMethod, DriftKind, DriftOptions,
    DriftResult, ZeroFill,
};

/// Options of `anofox_drift_test`; each kind reads only its own.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DriftOptionsFFI {
    /// Epps-Singleton evaluation points; NULL / 0 for the default.
    pub t: *const f64,
    pub t_len: usize,
    /// Wasserstein order.
    pub p: f64,
    /// Logarithm base of KL / Jensen-Shannon; NaN for e.
    pub base: f64,
    /// Zero fill of PSI / KL; NaN for evidently's rule.
    pub epsilon: f64,
    /// Permutations of Wasserstein / TVD (0: none).
    pub n_permutations: usize,
    pub seed: u64,
    /// Bins of `binning = 3` (quantile).
    pub n_bins: usize,
    /// 0 two_sided, 1 less, 2 greater.
    pub alternative: i32,
    /// KS exact p-value: -1 auto, 0 no, 1 yes.
    pub exact: i32,
    /// Cramér-von Mises: 0 auto, 1 exact, 2 asymptotic.
    pub cvm_method: i32,
    /// 0 auto, 1 sturges, 2 categorical, 3 quantile.
    pub binning: i32,
    pub has_seed: bool,
}

/// Result of a drift test. A field that does not apply is NaN (numbers) or 0
/// (`n_bins`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DriftResultFFI {
    pub statistic: f64,
    /// Normed Wasserstein distance, Jensen-Shannon divergence, or raw AD.
    pub statistic2: f64,
    pub p_value: f64,
    pub df: f64,
    /// Reference size (Anderson-Darling: total size).
    pub n1: usize,
    /// Current size (Anderson-Darling: number of groups).
    pub n2: usize,
    pub n_bins: usize,
    pub exact: bool,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

impl From<&DriftResult> for DriftResultFFI {
    fn from(r: &DriftResult) -> Self {
        Self {
            statistic: r.statistic,
            statistic2: r.statistic2,
            p_value: r.p_value,
            df: r.df,
            n1: r.n1,
            n2: r.n2,
            n_bins: r.n_bins,
            exact: r.exact,
            unified: TestSummaryFFI::from(&r.unified),
        }
    }
}

unsafe fn invalid_input(out_error: *mut AnofoxError, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
}

fn drift_kind(kind: i32) -> Option<DriftKind> {
    Some(match kind {
        0 => DriftKind::Ks,
        1 => DriftKind::Cvm,
        2 => DriftKind::EppsSingleton,
        3 => DriftKind::Wasserstein,
        4 => DriftKind::Psi,
        5 => DriftKind::KlDivergence,
        6 => DriftKind::JensenShannon,
        7 => DriftKind::Hellinger,
        8 => DriftKind::Tvd,
        _ => return None,
    })
}

unsafe fn drift_options(o: &DriftOptionsFFI) -> Result<DriftOptions, &'static str> {
    Ok(DriftOptions {
        alternative: match o.alternative {
            0 => Alternative::TwoSided,
            1 => Alternative::Less,
            2 => Alternative::Greater,
            _ => return Err("invalid alternative"),
        },
        exact: match o.exact {
            -1 => None,
            0 => Some(false),
            _ => Some(true),
        },
        cvm_method: match o.cvm_method {
            0 => CvmMethod::Auto,
            1 => CvmMethod::Exact,
            2 => CvmMethod::Asymptotic,
            _ => return Err("invalid Cramer-von Mises method"),
        },
        t: if o.t.is_null() || o.t_len == 0 {
            None
        } else {
            Some(slice::from_raw_parts(o.t, o.t_len).to_vec())
        },
        p: o.p,
        n_permutations: o.n_permutations,
        seed: if o.has_seed { Some(o.seed) } else { None },
        binning: match o.binning {
            0 => Binning::Auto,
            1 => Binning::Sturges,
            2 => Binning::Categorical,
            3 => Binning::Quantile(o.n_bins),
            _ => return Err("invalid binning"),
        },
        base: if o.base.is_nan() { None } else { Some(o.base) },
        zero_fill: if o.epsilon.is_nan() {
            ZeroFill::Evidently
        } else {
            ZeroFill::Constant(o.epsilon)
        },
    })
}

/// Two-sample drift test of `current` against `reference` (NaN entries are
/// dropped). `kind`: 0 KS, 1 Cramér-von Mises, 2 Epps-Singleton,
/// 3 Wasserstein, 4 PSI, 5 KL, 6 Jensen-Shannon, 7 Hellinger, 8 TVD.
///
/// # Safety
/// - `reference` and `current` must be valid DataArrays
/// - `options.t` must point to `options.t_len` doubles or be NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_drift_test(
    kind: i32,
    reference: DataArray,
    current: DataArray,
    options: DriftOptionsFFI,
    out_result: *mut DriftResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let Some(kind) = drift_kind(kind) else {
            invalid_input(out_error, "invalid drift test kind");
            return false;
        };
        let opts = match drift_options(&options) {
            Ok(o) => o,
            Err(msg) => {
                invalid_input(out_error, msg);
                return false;
            }
        };
        let finite = |a: &DataArray| -> Vec<f64> {
            a.to_vec().into_iter().filter(|v| !v.is_nan()).collect()
        };
        match drift_test(kind, &finite(&reference), &finite(&current), &opts) {
            Ok(r) => {
                *out_result = DriftResultFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// k-sample Anderson-Darling test of `values` grouped by the numeric codes in
/// `groups` (rows with a NaN value or code are dropped).
///
/// # Safety
/// - `values` and `groups` must be valid DataArrays of equal length
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_ad_test_ksamp(
    values: DataArray,
    groups: DataArray,
    midrank: bool,
    out_result: *mut DriftResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let groups = split_groups_canonically(&values.to_vec(), &groups.to_vec());
        match anderson_darling_ksamp(&groups, midrank) {
            Ok(r) => {
                *out_result = DriftResultFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}
