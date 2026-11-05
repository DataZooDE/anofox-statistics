//! C FFI for multinomial logistic regression (#314): upstream
//! `MultinomialLogisticRegressor` (sklearn `LogisticRegression`). The structs
//! are mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::multinomial::{
    fit_multinomial, most_probable_class, multinomial_predict_proba, MultinomialOptions,
};

/// Fit options of `anofox_multinomial_fit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MultinomialOptionsFFI {
    /// Penalty strength `lambda = 1 / C` (>= 0).
    pub lambda: f64,
    /// Elastic-net mixing in `[0, 1]` (0 ridge, 1 lasso).
    pub l1_ratio: f64,
    /// Convergence tolerance (> 0).
    pub tolerance: f64,
    /// Maximum proximal Newton iterations.
    pub max_iterations: u32,
    /// Fit unpenalized intercepts.
    pub fit_intercept: bool,
}

/// A fitted multinomial logistic regression. The arrays are allocated by
/// `anofox_multinomial_fit` and released with `anofox_free_multinomial_result`.
#[repr(C)]
#[derive(Debug)]
pub struct MultinomialFitResultFFI {
    /// Coefficients, row-major `n_rows x n_features`: one row for two
    /// classes, `n_classes` rows otherwise.
    pub coefficients: *mut f64,
    /// Intercepts, `n_rows` values.
    pub intercepts: *mut f64,
    pub n_rows: usize,
    pub n_features: usize,
    pub n_classes: usize,
    pub n_observations: usize,
    pub log_likelihood: f64,
    pub objective: f64,
    pub accuracy: f64,
    pub iterations: u32,
    pub converged: bool,
}

unsafe fn malloc_values(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

/// Fit to the class indices `y` (`0..K`, every class present) of `n` rows,
/// with the features as `n_features` column arrays of `n` values and optional
/// sample `weights` (`n` values, NULL: unweighted). Fewer than two rows or
/// classes is `INSUFFICIENT_DATA`; an invalid option or weight is
/// `INVALID_INPUT`.
///
/// # Safety
/// - `y` must point to `n` doubles, `x` to `n_features` arrays of `n` values
///   each, `weights` to `n` doubles or be NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_multinomial_fit(
    y: *const f64,
    n: usize,
    x: *const DataArray,
    n_features: usize,
    weights: *const f64,
    options: MultinomialOptionsFFI,
    out_result: *mut MultinomialFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let y: Vec<f64> = if y.is_null() || n == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(y, n).to_vec()
        };
        let x_columns: Vec<Vec<f64>> = if x.is_null() || n_features == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(x, n_features)
                .iter()
                .map(|a| a.to_vec())
                .collect()
        };
        let w: Option<Vec<f64>> = if weights.is_null() || n == 0 {
            None
        } else {
            Some(slice::from_raw_parts(weights, n).to_vec())
        };
        let opts = MultinomialOptions {
            lambda: options.lambda,
            l1_ratio: options.l1_ratio,
            fit_intercept: options.fit_intercept,
            max_iterations: options.max_iterations as usize,
            tolerance: options.tolerance,
        };
        match fit_multinomial(&y, &x_columns, w.as_deref(), &opts) {
            Ok(fit) => {
                let flat: Vec<f64> = fit.coefficients.iter().flatten().copied().collect();
                let out = MultinomialFitResultFFI {
                    coefficients: malloc_values(&flat),
                    intercepts: malloc_values(&fit.intercepts),
                    n_rows: fit.coefficients.len(),
                    n_features: fit.n_features,
                    n_classes: fit.n_classes,
                    n_observations: fit.n_observations,
                    log_likelihood: fit.log_likelihood,
                    objective: fit.objective,
                    accuracy: fit.accuracy,
                    iterations: fit.iterations as u32,
                    converged: fit.converged,
                };
                let failed = (!flat.is_empty() && out.coefficients.is_null())
                    || (!fit.intercepts.is_empty() && out.intercepts.is_null());
                *out_result = out;
                if failed {
                    anofox_free_multinomial_result(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate the multinomial result",
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

/// Free the arrays of a result filled by `anofox_multinomial_fit`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_multinomial_fit`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_multinomial_result(result: *mut MultinomialFitResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [&mut r.coefficients, &mut r.intercepts] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
    })
}

/// Class probabilities of one row `x` (`x_len` values) under a model of
/// `n_classes` classes with `n_rows` coefficient rows of `n_features` values
/// (row-major, as `MultinomialFitResultFFI`) and `n_rows` intercepts. Writes
/// `n_classes` probabilities to `out_proba` and the index of the most
/// probable class to `out_class`.
///
/// # Safety
/// - `coefficients` must point to `n_rows * n_features` doubles, `intercepts`
///   to `n_rows` and `x` to `x_len`
/// - `out_proba` must point to `n_classes` writable doubles; `out_class` and
///   `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_multinomial_predict_proba(
    coefficients: *const f64,
    intercepts: *const f64,
    n_rows: usize,
    n_features: usize,
    n_classes: usize,
    x: *const f64,
    x_len: usize,
    out_proba: *mut f64,
    out_class: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_proba, out_error) || !check_out(out_class, out_error) {
            return false;
        }
        let view = |p: *const f64, len: usize| -> Vec<f64> {
            if p.is_null() || len == 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(p, len).to_vec()
            }
        };
        let flat = view(coefficients, n_rows * n_features);
        let rows: Vec<Vec<f64>> = if n_features == 0 {
            vec![Vec::new(); n_rows]
        } else {
            flat.chunks(n_features).map(|r| r.to_vec()).collect()
        };
        let b = view(intercepts, n_rows);
        let row = view(x, x_len);
        match multinomial_predict_proba(&rows, &b, n_classes, &row) {
            Ok(p) => {
                std::ptr::copy_nonoverlapping(p.as_ptr(), out_proba, p.len());
                *out_class = most_probable_class(&p);
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

    fn options() -> MultinomialOptionsFFI {
        MultinomialOptionsFFI {
            lambda: 1.0,
            l1_ratio: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            fit_intercept: true,
        }
    }

    #[test]
    fn fit_predict_and_error_codes() {
        let y = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let x = [0.1, 0.9, 2.1, 0.3, 1.2, 1.8, 0.7, -0.2];
        let col = [DataArray {
            data: x.as_ptr(),
            validity: std::ptr::null(),
            len: x.len(),
        }];
        let mut fit = std::mem::MaybeUninit::<MultinomialFitResultFFI>::uninit();
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_multinomial_fit(
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
            assert_eq!((fit.n_rows, fit.n_features, fit.n_classes), (3, 1, 3));
            let mut proba = [0.0; 3];
            let mut class = 0usize;
            assert!(anofox_multinomial_predict_proba(
                fit.coefficients,
                fit.intercepts,
                fit.n_rows,
                fit.n_features,
                fit.n_classes,
                [2.0].as_ptr(),
                1,
                proba.as_mut_ptr(),
                &mut class,
                &mut err,
            ));
            assert!((proba.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert_eq!(class, 2);
            // x of the wrong length
            assert!(!anofox_multinomial_predict_proba(
                fit.coefficients,
                fit.intercepts,
                fit.n_rows,
                fit.n_features,
                fit.n_classes,
                [2.0, 1.0].as_ptr(),
                2,
                proba.as_mut_ptr(),
                &mut class,
                &mut err,
            ));
            assert!(matches!(err.code, ErrorCode::DimensionMismatch));
            anofox_free_multinomial_result(&mut fit);
            assert!(fit.coefficients.is_null() && fit.intercepts.is_null());

            // a single class is degenerate (NULL in SQL)
            let one = [1.0; 8];
            let mut fit2 = std::mem::MaybeUninit::<MultinomialFitResultFFI>::uninit();
            assert!(!anofox_multinomial_fit(
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
            // an invalid option is an input error
            let mut bad = options();
            bad.l1_ratio = 2.0;
            assert!(!anofox_multinomial_fit(
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
}
