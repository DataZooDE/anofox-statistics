//! C FFI for the multiple-model forecast comparisons (#295): the SPA test, the
//! MSPE-adjusted SPA test and the Model Confidence Set. The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::forecast::{
    model_confidence_set, mspe_adjusted_spa, spa_test, McsOptions, McsStatistic, SpaResult,
    SpaTestOptions,
};

/// Options of `anofox_spa_test`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpaOptionsFFI {
    pub n_bootstrap: usize,
    /// Expected stationary-bootstrap block length; <= 0 for `T^(1/3)`.
    pub block_length: f64,
    pub seed: u64,
    pub has_seed: bool,
}

/// Result of `anofox_spa_test`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpaResultFFI {
    pub statistic: f64,
    pub p_value_consistent: f64,
    pub p_value_upper: f64,
    /// Block length actually used.
    pub block_length: f64,
    /// 0-based index of the best competing model, -1 for none.
    pub best_model: i64,
    pub n_bootstrap: usize,
    pub n: usize,
    pub n_models: usize,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

impl From<&SpaResult> for SpaResultFFI {
    fn from(r: &SpaResult) -> Self {
        Self {
            statistic: r.statistic,
            p_value_consistent: r.p_value_consistent,
            p_value_upper: r.p_value_upper,
            block_length: r.block_length,
            best_model: r.best_model_idx.map_or(-1, |i| i as i64),
            n_bootstrap: r.n_bootstrap,
            n: r.n,
            n_models: r.n_models,
            unified: TestSummaryFFI::from(&r.unified),
        }
    }
}

/// Options of `anofox_model_confidence_set`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McsOptionsFFI {
    pub alpha: f64,
    pub n_bootstrap: usize,
    /// Expected stationary-bootstrap block length; <= 0 for `T^(1/3)`.
    pub block_length: f64,
    pub seed: u64,
    /// 0 range (T_R), 1 max (T_max).
    pub statistic: i32,
    pub has_seed: bool,
}

/// One elimination step of the MCS.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McsStepFFI {
    /// 0-based model index.
    pub model: u64,
    pub p_value: f64,
    pub eliminated: bool,
}

/// Result of `anofox_model_confidence_set`; the sets are written to the
/// caller's buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McsResultFFI {
    pub mcs_p_value: f64,
    /// Block length actually used.
    pub block_length: f64,
    pub n_bootstrap: usize,
    pub n: usize,
    pub n_models: usize,
    /// Entries written to `out_included`.
    pub n_included: usize,
    /// Entries written to `out_steps`.
    pub n_steps: usize,
}

unsafe fn invalid_input(out_error: *mut AnofoxError, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
}

unsafe fn model_columns(models: *const DataArray, n_models: usize) -> Vec<Vec<f64>> {
    if models.is_null() || n_models == 0 {
        return vec![];
    }
    slice::from_raw_parts(models, n_models)
        .iter()
        .map(|a| a.to_vec())
        .collect()
}

/// SPA-type test of `n_models` competing models against a benchmark. `kind`:
/// 0 the SPA test on losses, 1 the MSPE-adjusted SPA test on forecast errors.
///
/// # Safety
/// - `benchmark` and the `n_models` DataArrays at `models` must be valid
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_spa_test(
    kind: i32,
    benchmark: DataArray,
    models: *const DataArray,
    n_models: usize,
    options: SpaOptionsFFI,
    out_result: *mut SpaResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let opts = SpaTestOptions {
            n_bootstrap: options.n_bootstrap,
            block_length: options.block_length,
            seed: options.has_seed.then_some(options.seed),
        };
        let bench = benchmark.to_vec();
        let columns = model_columns(models, n_models);
        let result = match kind {
            0 => spa_test(&bench, &columns, &opts),
            1 => mspe_adjusted_spa(&bench, &columns, &opts),
            _ => {
                invalid_input(out_error, "invalid SPA test kind");
                return false;
            }
        };
        match result {
            Ok(r) => {
                *out_result = SpaResultFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Model Confidence Set of `n_models` loss series. `out_included` and
/// `out_steps` must each hold `n_models` entries.
///
/// # Safety
/// - the `n_models` DataArrays at `models` must be valid
/// - `out_included` and `out_steps` must point to `n_models` entries
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_model_confidence_set(
    models: *const DataArray,
    n_models: usize,
    options: McsOptionsFFI,
    out_included: *mut u64,
    out_steps: *mut McsStepFFI,
    out_result: *mut McsResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if n_models > 0 && (out_included.is_null() || out_steps.is_null()) {
            invalid_input(out_error, "out_included or out_steps is NULL");
            return false;
        }
        let statistic = match options.statistic {
            0 => McsStatistic::Range,
            1 => McsStatistic::Max,
            _ => {
                invalid_input(out_error, "invalid MCS statistic");
                return false;
            }
        };
        let opts = McsOptions {
            alpha: options.alpha,
            n_bootstrap: options.n_bootstrap,
            statistic,
            block_length: options.block_length,
            seed: options.has_seed.then_some(options.seed),
        };
        let columns = model_columns(models, n_models);
        match model_confidence_set(&columns, &opts) {
            Ok(r) => {
                let n_included = r.included_models.len().min(n_models);
                for (i, &m) in r.included_models.iter().take(n_included).enumerate() {
                    *out_included.add(i) = m as u64;
                }
                let n_steps = r.elimination_sequence.len().min(n_models);
                for (i, s) in r.elimination_sequence.iter().take(n_steps).enumerate() {
                    *out_steps.add(i) = McsStepFFI {
                        model: s.model_idx as u64,
                        p_value: s.p_value,
                        eliminated: s.eliminated,
                    };
                }
                *out_result = McsResultFFI {
                    mcs_p_value: r.mcs_p_value,
                    block_length: r.block_length,
                    n_bootstrap: r.n_bootstrap,
                    n: r.n,
                    n_models: r.n_models,
                    n_included,
                    n_steps,
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
