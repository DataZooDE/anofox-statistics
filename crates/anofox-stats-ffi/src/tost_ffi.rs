//! C FFI for the TOST equivalence variants (#293): one-sample t, one and two
//! proportions, paired and two-sample Wilcoxon, Yuen and bootstrap. The structs
//! are mirrored in `src/include/anofox_stats_ffi.h`; the result is the
//! `TostResultFFI` of `anofox_tost_t_test`.

use super::*;
use anofox_stats_core::tests::equivalence::{
    tost_bootstrap, tost_prop_one, tost_prop_two, tost_t_test_one_sample, tost_wilcoxon_paired,
    tost_wilcoxon_two_sample, tost_yuen, TostBootstrapOptions, TostBounds, TostPropOptions,
    TostResult, TostTTestOptions, TostWilcoxonOptions, TostYuenOptions,
};

/// Options of `anofox_tost_variant` / `anofox_tost_prop`; each kind reads only
/// its own fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TostVariantOptionsFFI {
    pub bound_lower: f64,
    pub bound_upper: f64,
    pub alpha: f64,
    /// One-sample t: hypothesised mean. One proportion: hypothesised p0.
    pub mu: f64,
    /// Yuen: proportion trimmed from each tail.
    pub trim: f64,
    /// Bootstrap: number of resamples.
    pub n_bootstrap: usize,
    pub seed: u64,
    pub has_seed: bool,
}

fn tost_result_ffi(r: &TostResult) -> TostResultFFI {
    TostResultFFI {
        t_lower: r.statistic_lower,
        t_upper: r.statistic_upper,
        p_lower: r.p_value_lower,
        p_upper: r.p_value_upper,
        p_value: r.p_value,
        df: r.df,
        estimate: r.estimate,
        ci_lower: r.ci_lower,
        ci_upper: r.ci_upper,
        bound_lower: r.bounds_lower,
        bound_upper: r.bounds_upper,
        equivalent: r.equivalent,
        n: r.n,
        // SAFETY: alloc_string returns an owned C string freed by anofox_free_tost_result.
        method: unsafe { alloc_string(&r.method) },
        unified: TestSummaryFFI::from(&r.unified),
    }
}

unsafe fn finish(
    result: Result<TostResult, StatsError>,
    out_result: *mut TostResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    match result {
        Ok(r) => {
            *out_result = tost_result_ffi(&r);
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

/// TOST variant on samples (NaN entries are dropped by the core). `kind`:
/// 0 one-sample t (`x` against `options.mu`, `y` ignored), 1 paired Wilcoxon
/// signed-rank (`x`, `y` paired), 2 two-sample Wilcoxon rank-sum, 3 Yuen
/// trimmed means, 4 bootstrap of the mean difference.
///
/// # Safety
/// - `x` and `y` must be valid DataArrays (`y` may be empty for kind 0)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_tost_variant(
    kind: i32,
    x: DataArray,
    y: DataArray,
    options: TostVariantOptionsFFI,
    out_result: *mut TostResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let bounds = TostBounds::Raw {
            lower: options.bound_lower,
            upper: options.bound_upper,
        };
        let alpha = options.alpha;
        let x = x.to_vec();
        let y = y.to_vec();
        let result = match kind {
            0 => tost_t_test_one_sample(
                &x,
                options.mu,
                &TostTTestOptions {
                    bounds,
                    alpha,
                    pooled: false,
                },
            ),
            1 => tost_wilcoxon_paired(&x, &y, &TostWilcoxonOptions { bounds, alpha }),
            2 => tost_wilcoxon_two_sample(&x, &y, &TostWilcoxonOptions { bounds, alpha }),
            3 => tost_yuen(
                &x,
                &y,
                &TostYuenOptions {
                    bounds,
                    trim: options.trim,
                    alpha,
                },
            ),
            4 => tost_bootstrap(
                &x,
                &y,
                &TostBootstrapOptions {
                    bounds,
                    n_bootstrap: options.n_bootstrap,
                    alpha,
                    seed: options.has_seed.then_some(options.seed),
                },
            ),
            _ => {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::InvalidInput, "invalid TOST variant kind");
                }
                return false;
            }
        };
        finish(result, out_result, out_error)
    })
}

/// TOST for one proportion (`two_sample` false: `successes1` of `trials1`
/// against `options.mu` as p0) or the difference of two proportions
/// (`two_sample` true: group 1 minus group 2).
///
/// # Safety
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_tost_prop(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
    two_sample: bool,
    options: TostVariantOptionsFFI,
    out_result: *mut TostResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let opts = TostPropOptions {
            bounds: TostBounds::Raw {
                lower: options.bound_lower,
                upper: options.bound_upper,
            },
            alpha: options.alpha,
        };
        let result = if two_sample {
            tost_prop_two(successes1, trials1, successes2, trials2, &opts)
        } else {
            tost_prop_one(successes1, trials1, options.mu, &opts)
        };
        finish(result, out_result, out_error)
    })
}
