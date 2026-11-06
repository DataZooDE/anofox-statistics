//! C FFI for ordinal regression (#313): upstream `OrdinalRegressor`
//! (`rms::orm` / `MASS::polr`) and `ContinuationRatioRegressor`. The structs
//! are mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::ordinal::{
    continuation_ratio_predict, fit_continuation_ratio, fit_ordinal, ordinal_link_from_code,
    ordinal_predict, OrdinalOptions, OrdinalPrediction,
};

/// Fit options of `anofox_ordinal_fit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrdinalOptionsFFI {
    /// `orm`'s `penalty = λ` (>= 0; 0: none).
    pub penalty: f64,
    /// Convergence tolerance on the largest score (> 0).
    pub tolerance: f64,
    /// 0 logistic, 1 probit, 2 loglog, 3 cloglog, 4 cauchit.
    pub link: u32,
    pub max_iterations: u32,
    /// 1-based column of `x` used as an offset; 0: none.
    pub offset_column: u32,
    /// Proportional-odds score test: -1 auto (at most 50 levels), 0 no, 1 yes.
    pub po_test: i32,
}

/// A fitted cumulative link model. The arrays are allocated by
/// `anofox_ordinal_fit` and released with `anofox_free_ordinal_result`.
#[repr(C)]
#[derive(Debug)]
pub struct OrdinalFitResultFFI {
    /// `n_levels` values each.
    pub levels: *mut f64,
    pub counts: *mut f64,
    /// `n_levels - 1` values each; `intercept_std_errors` is NULL when not
    /// computed (more than 1000 levels).
    pub intercepts: *mut f64,
    pub intercept_std_errors: *mut f64,
    /// `n_features` values each.
    pub coefficients: *mut f64,
    pub std_errors: *mut f64,
    pub z_values: *mut f64,
    pub p_values: *mut f64,
    pub n_levels: usize,
    pub n_features: usize,
    pub n_observations: usize,
    pub log_likelihood: f64,
    pub penalized_log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub lr_chisq: f64,
    pub lr_df: f64,
    pub lr_p_value: f64,
    pub aic: f64,
    /// NaN when the proportional-odds test was not computed.
    pub po_statistic: f64,
    pub po_df: f64,
    pub po_p_value: f64,
    pub iterations: u32,
    pub converged: bool,
}

/// A fitted continuation-ratio model. The arrays are allocated by
/// `anofox_continuation_ratio_fit` and released with
/// `anofox_free_continuation_ratio_result`.
#[repr(C)]
#[derive(Debug)]
pub struct ContinuationRatioFitResultFFI {
    /// `n_levels` values.
    pub levels: *mut f64,
    /// `n_coefficients` values each.
    pub coefficients: *mut f64,
    pub std_errors: *mut f64,
    pub z_values: *mut f64,
    pub p_values: *mut f64,
    pub n_levels: usize,
    pub n_coefficients: usize,
    pub n_features: usize,
    pub n_observations: usize,
    pub n_records: usize,
    pub log_likelihood: f64,
    pub aic: f64,
    pub iterations: u32,
    pub converged: bool,
}

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

unsafe fn free_doubles(ptrs: &mut [&mut *mut f64]) {
    for p in ptrs.iter_mut() {
        if !p.is_null() {
            libc::free(**p as *mut libc::c_void);
            **p = std::ptr::null_mut();
        }
    }
}

unsafe fn view(p: *const f64, len: usize) -> Vec<f64> {
    if p.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(p, len).to_vec()
    }
}

unsafe fn columns(x: *const DataArray, n_features: usize) -> Vec<Vec<f64>> {
    if x.is_null() || n_features == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(x, n_features)
            .iter()
            .map(|a| a.to_vec())
            .collect()
    }
}

unsafe fn alloc_failed(out_error: *mut AnofoxError) {
    if !out_error.is_null() {
        (*out_error).set(
            ErrorCode::AllocationFailure,
            "Failed to allocate the ordinal regression result",
        );
    }
}

/// Fit a cumulative link model to the ordered response `y` (`n` values; its
/// distinct values are the levels) on `n_features` x columns of `n` values,
/// with optional case `weights` (NULL: unweighted). Fewer than two rows or
/// levels is `INSUFFICIENT_DATA`; an invalid option is `INVALID_INPUT`.
///
/// # Safety
/// - `y` must point to `n` doubles, `x` to `n_features` arrays of `n` values,
///   `weights` to `n` doubles or be NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ordinal_fit(
    y: *const f64,
    n: usize,
    x: *const DataArray,
    n_features: usize,
    weights: *const f64,
    options: OrdinalOptionsFFI,
    out_result: *mut OrdinalFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let Some(link) = ordinal_link_from_code(options.link) else {
            set_error(
                out_error,
                &StatsError::InvalidInput(format!("unknown ordinal link code {}", options.link)),
            );
            return false;
        };
        let y = view(y, n);
        let x_columns = columns(x, n_features);
        let w = (!weights.is_null() && n > 0).then(|| view(weights, n));
        let opts = OrdinalOptions {
            link,
            penalty: options.penalty,
            offset_column: (options.offset_column > 0).then(|| options.offset_column as usize - 1),
            max_iterations: options.max_iterations as usize,
            tolerance: options.tolerance,
            po_test: match options.po_test {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            },
        };
        match fit_ordinal(&y, &x_columns, w.as_deref(), &opts) {
            Ok(fit) => {
                let (po_statistic, po_df, po_p_value) =
                    fit.po_test.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                let ise = fit.intercept_std_errors.as_deref().unwrap_or(&[]);
                let out = OrdinalFitResultFFI {
                    levels: malloc_doubles(&fit.levels),
                    counts: malloc_doubles(&fit.counts),
                    intercepts: malloc_doubles(&fit.intercepts),
                    intercept_std_errors: malloc_doubles(ise),
                    coefficients: malloc_doubles(&fit.coefficients),
                    std_errors: malloc_doubles(&fit.std_errors),
                    z_values: malloc_doubles(&fit.z_values),
                    p_values: malloc_doubles(&fit.p_values),
                    n_levels: fit.levels.len(),
                    n_features: fit.n_features,
                    n_observations: fit.n_observations,
                    log_likelihood: fit.log_likelihood,
                    penalized_log_likelihood: fit.penalized_log_likelihood,
                    null_log_likelihood: fit.null_log_likelihood,
                    lr_chisq: fit.lr_chisq,
                    lr_df: fit.lr_df,
                    lr_p_value: fit.lr_p_value,
                    aic: fit.aic,
                    po_statistic,
                    po_df,
                    po_p_value,
                    iterations: fit.iterations as u32,
                    converged: fit.converged,
                };
                let failed = out.levels.is_null()
                    || out.counts.is_null()
                    || out.intercepts.is_null()
                    || (!ise.is_empty() && out.intercept_std_errors.is_null())
                    || (fit.n_features > 0
                        && (out.coefficients.is_null()
                            || out.std_errors.is_null()
                            || out.z_values.is_null()
                            || out.p_values.is_null()));
                *out_result = out;
                if failed {
                    anofox_free_ordinal_result(out_result);
                    alloc_failed(out_error);
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

/// Free the arrays of a result filled by `anofox_ordinal_fit`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_ordinal_fit`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_ordinal_result(result: *mut OrdinalFitResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_doubles(&mut [
            &mut r.levels,
            &mut r.counts,
            &mut r.intercepts,
            &mut r.intercept_std_errors,
            &mut r.coefficients,
            &mut r.std_errors,
            &mut r.z_values,
            &mut r.p_values,
        ]);
    })
}

/// Fit the continuation-ratio model to the ordered response `y` (at least
/// three levels) on `n_features` x columns, with optional case `weights` and
/// the 0-based `cohort_specific` columns (`n_cohort_specific` values, NULL
/// when none). Fewer than three levels is `INSUFFICIENT_DATA`.
///
/// # Safety
/// - `y` must point to `n` doubles, `x` to `n_features` arrays of `n` values,
///   `weights` to `n` doubles or be NULL, `cohort_specific` to
///   `n_cohort_specific` values or be NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_continuation_ratio_fit(
    y: *const f64,
    n: usize,
    x: *const DataArray,
    n_features: usize,
    weights: *const f64,
    cohort_specific: *const u64,
    n_cohort_specific: usize,
    out_result: *mut ContinuationRatioFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let y = view(y, n);
        let x_columns = columns(x, n_features);
        let w = (!weights.is_null() && n > 0).then(|| view(weights, n));
        let cs: Vec<usize> = if cohort_specific.is_null() || n_cohort_specific == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(cohort_specific, n_cohort_specific)
                .iter()
                .map(|&c| c as usize)
                .collect()
        };
        match fit_continuation_ratio(&y, &x_columns, w.as_deref(), &cs) {
            Ok(fit) => {
                let out = ContinuationRatioFitResultFFI {
                    levels: malloc_doubles(&fit.levels),
                    coefficients: malloc_doubles(&fit.coefficients),
                    std_errors: malloc_doubles(&fit.std_errors),
                    z_values: malloc_doubles(&fit.z_values),
                    p_values: malloc_doubles(&fit.p_values),
                    n_levels: fit.levels.len(),
                    n_coefficients: fit.coefficients.len(),
                    n_features: fit.n_features,
                    n_observations: fit.n_observations,
                    n_records: fit.n_records,
                    log_likelihood: fit.log_likelihood,
                    aic: fit.aic,
                    iterations: fit.iterations as u32,
                    converged: fit.converged,
                };
                let failed = out.levels.is_null()
                    || out.coefficients.is_null()
                    || out.std_errors.is_null()
                    || out.z_values.is_null()
                    || out.p_values.is_null();
                *out_result = out;
                if failed {
                    anofox_free_continuation_ratio_result(out_result);
                    alloc_failed(out_error);
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

/// Free the arrays of a result filled by `anofox_continuation_ratio_fit`.
///
/// # Safety
/// - `result` must be NULL or a result filled by
///   `anofox_continuation_ratio_fit`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_continuation_ratio_result(
    result: *mut ContinuationRatioFitResultFFI,
) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_doubles(&mut [
            &mut r.levels,
            &mut r.coefficients,
            &mut r.std_errors,
            &mut r.z_values,
            &mut r.p_values,
        ]);
    })
}

unsafe fn write_prediction(
    p: &OrdinalPrediction,
    out_proba: *mut f64,
    out_exceedance: *mut f64,
    out_mean: *mut f64,
    out_class: *mut usize,
) {
    std::ptr::copy_nonoverlapping(p.probabilities.as_ptr(), out_proba, p.probabilities.len());
    std::ptr::copy_nonoverlapping(p.exceedance.as_ptr(), out_exceedance, p.exceedance.len());
    *out_mean = p.mean;
    *out_class = p.class;
}

/// Predict one row `x` (`x_len` values: the columns of the fit, including an
/// offset column) under a stored cumulative link model with `link` (code as
/// `OrdinalOptionsFFI`), `n_levels` levels, `n_levels - 1` intercepts and
/// `n_coefficients` slopes; `offset_column` is 1-based (0: none). Writes the
/// `n_levels` category probabilities, the `n_levels - 1` exceedance
/// probabilities, the mean and the index of the most probable level.
///
/// # Safety
/// - `levels` must point to `n_levels` doubles, `intercepts` to
///   `n_levels - 1`, `coefficients` to `n_coefficients`, `x` to `x_len`
/// - `out_proba` must point to `n_levels` writable doubles, `out_exceedance`
///   to `n_levels - 1`; `out_mean`, `out_class` and `out_error` must be valid
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ordinal_predict(
    link: u32,
    levels: *const f64,
    n_levels: usize,
    intercepts: *const f64,
    coefficients: *const f64,
    n_coefficients: usize,
    offset_column: u64,
    x: *const f64,
    x_len: usize,
    out_proba: *mut f64,
    out_exceedance: *mut f64,
    out_mean: *mut f64,
    out_class: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_proba, out_error)
            || !check_out(out_mean, out_error)
            || !check_out(out_class, out_error)
            || (n_levels > 1 && !check_out(out_exceedance, out_error))
        {
            return false;
        }
        let Some(link) = ordinal_link_from_code(link) else {
            set_error(
                out_error,
                &StatsError::InvalidInput(format!("unknown ordinal link code {link}")),
            );
            return false;
        };
        let lv = view(levels, n_levels);
        let ic = view(intercepts, n_levels.saturating_sub(1));
        let co = view(coefficients, n_coefficients);
        let row = view(x, x_len);
        let offset = (offset_column > 0).then(|| offset_column as usize - 1);
        match ordinal_predict(link, &lv, &ic, &co, offset, &row) {
            Ok(p) => {
                write_prediction(&p, out_proba, out_exceedance, out_mean, out_class);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Predict one row `x` (`x_len` values) under a stored continuation-ratio
/// model: `n_levels` levels, `n_coefficients` coefficients in the layout of
/// `ContinuationRatioFitResultFFI` and the 0-based `cohort_specific` columns.
/// Outputs as `anofox_ordinal_predict`.
///
/// # Safety
/// - as `anofox_ordinal_predict`; `cohort_specific` must point to
///   `n_cohort_specific` values or be NULL
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_continuation_ratio_predict(
    levels: *const f64,
    n_levels: usize,
    coefficients: *const f64,
    n_coefficients: usize,
    cohort_specific: *const u64,
    n_cohort_specific: usize,
    x: *const f64,
    x_len: usize,
    out_proba: *mut f64,
    out_exceedance: *mut f64,
    out_mean: *mut f64,
    out_class: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_proba, out_error)
            || !check_out(out_mean, out_error)
            || !check_out(out_class, out_error)
            || (n_levels > 1 && !check_out(out_exceedance, out_error))
        {
            return false;
        }
        let lv = view(levels, n_levels);
        let co = view(coefficients, n_coefficients);
        let row = view(x, x_len);
        let cs: Vec<usize> = if cohort_specific.is_null() || n_cohort_specific == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(cohort_specific, n_cohort_specific)
                .iter()
                .map(|&c| c as usize)
                .collect()
        };
        match continuation_ratio_predict(&lv, &co, &cs, &row) {
            Ok(p) => {
                write_prediction(&p, out_proba, out_exceedance, out_mean, out_class);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> OrdinalOptionsFFI {
        OrdinalOptionsFFI {
            penalty: 0.0,
            tolerance: 1e-10,
            link: 0,
            max_iterations: 100,
            offset_column: 0,
            po_test: -1,
        }
    }

    fn data() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..40).map(|i| ((i * 7) as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s = v + 0.7 * ((i * 11) as f64).sin();
                if s < -0.4 {
                    1.0
                } else if s < 0.4 {
                    2.0
                } else {
                    3.0
                }
            })
            .collect();
        (y, x)
    }

    #[test]
    fn ordinal_fit_predict_and_errors() {
        let (y, x) = data();
        let col = [DataArray {
            data: x.as_ptr(),
            validity: std::ptr::null(),
            len: x.len(),
        }];
        let mut err = AnofoxError::success();
        unsafe {
            let mut fit = std::mem::MaybeUninit::<OrdinalFitResultFFI>::uninit();
            assert!(anofox_ordinal_fit(
                y.as_ptr(),
                y.len(),
                col.as_ptr(),
                1,
                std::ptr::null(),
                options(),
                fit.as_mut_ptr(),
                &mut err,
            ));
            let mut fit = fit.assume_init();
            assert_eq!((fit.n_levels, fit.n_features), (3, 1));
            assert!(!fit.intercept_std_errors.is_null());
            assert!(fit.po_df == 1.0);
            let (mut proba, mut exceed, mut mean, mut class) = ([0.0; 3], [0.0; 2], 0.0, 0usize);
            assert!(anofox_ordinal_predict(
                0,
                fit.levels,
                fit.n_levels,
                fit.intercepts,
                fit.coefficients,
                fit.n_features,
                0,
                [2.0].as_ptr(),
                1,
                proba.as_mut_ptr(),
                exceed.as_mut_ptr(),
                &mut mean,
                &mut class,
                &mut err,
            ));
            assert!((proba.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!((exceed[1] - proba[2]).abs() < 1e-12);
            assert_eq!(class, 2);
            assert!(!anofox_ordinal_predict(
                0,
                fit.levels,
                fit.n_levels,
                fit.intercepts,
                fit.coefficients,
                fit.n_features,
                0,
                [2.0, 1.0].as_ptr(),
                2,
                proba.as_mut_ptr(),
                exceed.as_mut_ptr(),
                &mut mean,
                &mut class,
                &mut err,
            ));
            assert!(matches!(err.code, ErrorCode::DimensionMismatch));
            anofox_free_ordinal_result(&mut fit);
            assert!(fit.levels.is_null() && fit.coefficients.is_null());

            let one = [1.0; 40];
            let mut fit2 = std::mem::MaybeUninit::<OrdinalFitResultFFI>::uninit();
            assert!(!anofox_ordinal_fit(
                one.as_ptr(),
                one.len(),
                col.as_ptr(),
                1,
                std::ptr::null(),
                options(),
                fit2.as_mut_ptr(),
                &mut err,
            ));
            assert!(matches!(err.code, ErrorCode::InsufficientData));
            let mut bad = options();
            bad.link = 9;
            assert!(!anofox_ordinal_fit(
                y.as_ptr(),
                y.len(),
                col.as_ptr(),
                1,
                std::ptr::null(),
                bad,
                fit2.as_mut_ptr(),
                &mut err,
            ));
            assert!(matches!(err.code, ErrorCode::InvalidInput));
        }
    }

    #[test]
    fn continuation_ratio_fit_and_predict() {
        let (y, x) = data();
        let col = [DataArray {
            data: x.as_ptr(),
            validity: std::ptr::null(),
            len: x.len(),
        }];
        let mut err = AnofoxError::success();
        unsafe {
            let mut fit = std::mem::MaybeUninit::<ContinuationRatioFitResultFFI>::uninit();
            let cs = [0u64];
            assert!(anofox_continuation_ratio_fit(
                y.as_ptr(),
                y.len(),
                col.as_ptr(),
                1,
                std::ptr::null(),
                cs.as_ptr(),
                1,
                fit.as_mut_ptr(),
                &mut err,
            ));
            let mut fit = fit.assume_init();
            assert_eq!((fit.n_levels, fit.n_coefficients), (3, 4));
            let (mut proba, mut exceed, mut mean, mut class) = ([0.0; 3], [0.0; 2], 0.0, 0usize);
            assert!(anofox_continuation_ratio_predict(
                fit.levels,
                fit.n_levels,
                fit.coefficients,
                fit.n_coefficients,
                cs.as_ptr(),
                1,
                [0.5].as_ptr(),
                1,
                proba.as_mut_ptr(),
                exceed.as_mut_ptr(),
                &mut mean,
                &mut class,
                &mut err,
            ));
            assert!((proba.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            anofox_free_continuation_ratio_result(&mut fit);
            assert!(fit.levels.is_null());
        }
    }
}
