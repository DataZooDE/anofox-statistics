//! C FFI for the data-drift tests (#245-#255). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::drift::{
    anderson_darling_ksamp, column_drift, dataset_drift, drift_test, Alternative, Binning,
    ColumnDrift, ColumnDriftOptions, CvmMethod, DatasetDrift, DriftKind, DriftOptions, DriftResult,
    DriftTest, ZeroFill,
};
use libc::c_char;
use std::ffi::CStr;

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
    /// z test: shares of the non-smallest category in the reference and the
    /// current sample.
    pub estimate_reference: f64,
    pub estimate_current: f64,
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
            estimate_reference: r.estimate_reference,
            estimate_current: r.estimate_current,
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
        9 => DriftKind::Z,
        10 => DriftKind::ChiSquare,
        11 => DriftKind::GTest,
        _ => return None,
    })
}

fn binning(code: i32, n_bins: usize) -> Result<Binning, &'static str> {
    Ok(match code {
        0 => Binning::Auto,
        1 => Binning::Sturges,
        2 => Binning::Categorical,
        3 => Binning::Quantile(n_bins),
        _ => return Err("invalid binning"),
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
        binning: binning(o.binning, o.n_bins)?,
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
/// 3 Wasserstein, 4 PSI, 5 KL, 6 Jensen-Shannon, 7 Hellinger, 8 TVD,
/// 9 z test, 10 chi-squared drift, 11 G drift.
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

/// Options of `anofox_column_drift` (#300).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ColumnDriftOptionsFFI {
    /// evidently's test name (`"ks"`, `"psi"`, ...); NULL or `"auto"` for
    /// evidently's default choice.
    pub test: *const c_char,
    /// Drift threshold; NaN for the test's default.
    pub threshold: f64,
    /// Zero fill of PSI / KL; NaN for evidently's rule.
    pub epsilon: f64,
    /// Permutations of `TVD` / `empirical_mmd`.
    pub n_permutations: usize,
    pub seed: u64,
    /// Bins of `binning = 3` (quantile).
    pub n_bins: usize,
    /// 0 auto, 1 sturges, 2 categorical, 3 quantile.
    pub binning: i32,
    pub has_seed: bool,
}

/// Result of `anofox_column_drift`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ColumnDriftResultFFI {
    /// The drift score: a p-value or a distance (`is_p_value`).
    pub score: f64,
    pub threshold: f64,
    pub n1: usize,
    pub n2: usize,
    pub drifted: bool,
    pub is_p_value: bool,
    /// evidently's name of the test that was run.
    pub test: [c_char; TEST_NAME_LEN],
}

impl ColumnDriftResultFFI {
    fn new(r: &ColumnDrift, n1: usize, n2: usize) -> Self {
        Self {
            score: r.score,
            threshold: r.threshold,
            n1,
            n2,
            drifted: r.drifted,
            is_p_value: r.test.is_p_value(),
            test: fixed_buffer::<TEST_NAME_LEN>(r.test.name()),
        }
    }
}

unsafe fn column_drift_options(o: &ColumnDriftOptionsFFI) -> Result<ColumnDriftOptions, String> {
    let test = if o.test.is_null() {
        None
    } else {
        let name = CStr::from_ptr(o.test).to_string_lossy();
        if name == "auto" {
            None
        } else {
            Some(DriftTest::from_name(&name).ok_or_else(|| {
                let names: Vec<&str> = DriftTest::ALL.iter().map(|t| t.name()).collect();
                format!(
                    "unknown drift test '{}' (expected 'auto' or one of {})",
                    name,
                    names.join(", ")
                )
            })?)
        }
    };
    Ok(ColumnDriftOptions {
        test,
        threshold: if o.threshold.is_nan() {
            None
        } else {
            Some(o.threshold)
        },
        binning: binning(o.binning, o.n_bins)?,
        zero_fill: if o.epsilon.is_nan() {
            ZeroFill::Evidently
        } else {
            ZeroFill::Constant(o.epsilon)
        },
        n_permutations: o.n_permutations,
        seed: if o.has_seed { Some(o.seed) } else { None },
    })
}

/// evidently's drift of one column of `current` against `reference` (NaN
/// entries are dropped). `categorical`: the values are category codes in
/// label order.
///
/// # Safety
/// - `reference` and `current` must be valid DataArrays
/// - `options.test` must be a NUL-terminated string or NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_column_drift(
    reference: DataArray,
    current: DataArray,
    categorical: bool,
    options: ColumnDriftOptionsFFI,
    out_result: *mut ColumnDriftResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let opts = match column_drift_options(&options) {
            Ok(o) => o,
            Err(msg) => {
                invalid_input(out_error, &msg);
                return false;
            }
        };
        let finite = |a: &DataArray| -> Vec<f64> {
            a.to_vec().into_iter().filter(|v| !v.is_nan()).collect()
        };
        let (r, c) = (finite(&reference), finite(&current));
        match column_drift(&r, &c, categorical, &opts) {
            Ok(d) => {
                *out_result = ColumnDriftResultFFI::new(&d, r.len(), c.len());
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Result of `anofox_dataset_drift`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DatasetDriftResultFFI {
    pub share_drifted: f64,
    pub n_columns: usize,
    pub n_drifted: usize,
    pub dataset_drift: bool,
}

impl From<&DatasetDrift> for DatasetDriftResultFFI {
    fn from(d: &DatasetDrift) -> Self {
        Self {
            share_drifted: d.share_drifted,
            n_columns: d.n_columns,
            n_drifted: d.n_drifted,
            dataset_drift: d.dataset_drift,
        }
    }
}

/// evidently's dataset drift of the per-column flags `drifted`: drifted when
/// the share of drifted columns is at least `drift_share`.
///
/// # Safety
/// - `drifted` must point to `len` bools (or be NULL with `len == 0`)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_dataset_drift(
    drifted: *const bool,
    len: usize,
    drift_share: f64,
    out_result: *mut DatasetDriftResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let flags: &[bool] = if drifted.is_null() || len == 0 {
            &[]
        } else {
            slice::from_raw_parts(drifted, len)
        };
        match dataset_drift(flags, drift_share) {
            Ok(d) => {
                *out_result = DatasetDriftResultFFI::from(&d);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}
