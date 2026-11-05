//! C FFI for the collinearity and separation diagnostics (#306): BKW condition
//! indices with variance-decomposition proportions, the per-column separation /
//! count-sparsity checks and the exact separation LP. The structs are mirrored
//! in `src/include/anofox_stats_ffi.h`; arrays and strings are `libc::malloc`ed
//! and released by the matching `anofox_free_*` function.

use super::*;
use anofox_stats_core::diagnostics::{
    check_binary_separation_diagnostic, check_count_sparsity_diagnostic,
    compute_condition_diagnostic, detect_linear_separation,
};
use anofox_stats_core::{ConditionSeverity, SeparationCheckResult, SeparationType};

/// Result of `anofox_condition_diagnostic`.
#[repr(C)]
#[derive(Debug)]
pub struct ConditionDiagnosticFFI {
    /// kappa of the unscaled design (R `kappa(X, exact = TRUE)`).
    pub condition_number: f64,
    pub condition_number_xtx: f64,
    /// Largest BKW condition index (unit-column-length design).
    pub scaled_condition_number: f64,
    /// 0 well_conditioned, 1 moderate, 2 high, 3 severe.
    pub severity: i32,
    /// `n_components` singular values of the scaled design, descending.
    pub singular_values: *mut f64,
    /// `n_components` condition indices.
    pub condition_indices: *mut f64,
    /// `n_components * n_terms` proportions, component-major: entry
    /// `k * n_terms + j` is coefficient j's share on component k.
    pub variance_proportions: *mut f64,
    pub n_components: usize,
    /// Coefficients (the intercept first when fitted).
    pub n_terms: usize,
    /// Warning text or NULL.
    pub warning: *mut c_char,
}

impl Default for ConditionDiagnosticFFI {
    fn default() -> Self {
        Self {
            condition_number: f64::NAN,
            condition_number_xtx: f64::NAN,
            scaled_condition_number: f64::NAN,
            severity: 0,
            singular_values: std::ptr::null_mut(),
            condition_indices: std::ptr::null_mut(),
            variance_proportions: std::ptr::null_mut(),
            n_components: 0,
            n_terms: 0,
            warning: std::ptr::null_mut(),
        }
    }
}

/// Result of `anofox_separation_check`.
#[repr(C)]
#[derive(Debug)]
pub struct SeparationCheckFFI {
    pub has_separation: bool,
    /// 0-based indices into x of the flagged predictors.
    pub separated_predictors: *mut u64,
    /// Kind per flagged predictor: 1 complete, 2 quasi, 3 monotonic_response.
    pub separation_types: *mut i32,
    pub n_separated: usize,
    /// Warning text or NULL.
    pub warning: *mut c_char,
}

impl Default for SeparationCheckFFI {
    fn default() -> Self {
        Self {
            has_separation: false,
            separated_predictors: std::ptr::null_mut(),
            separation_types: std::ptr::null_mut(),
            n_separated: 0,
            warning: std::ptr::null_mut(),
        }
    }
}

/// Result of `anofox_detect_separation`.
#[repr(C)]
#[derive(Debug)]
pub struct LinearSeparationFFI {
    pub separated: bool,
    /// `n_terms` values: +inf / -inf for diverging coefficients, 0 otherwise.
    pub infinite: *mut f64,
    /// `n_terms` values of the separating direction.
    pub direction: *mut f64,
    /// Coefficients (the intercept first when fitted).
    pub n_terms: usize,
}

impl Default for LinearSeparationFFI {
    fn default() -> Self {
        Self {
            separated: false,
            infinite: std::ptr::null_mut(),
            direction: std::ptr::null_mut(),
            n_terms: 0,
        }
    }
}

fn severity_code(s: ConditionSeverity) -> i32 {
    match s {
        ConditionSeverity::WellConditioned => 0,
        ConditionSeverity::Moderate => 1,
        ConditionSeverity::High => 2,
        ConditionSeverity::Severe => 3,
    }
}

fn separation_code(t: SeparationType) -> i32 {
    match t {
        SeparationType::None => 0,
        SeparationType::Complete => 1,
        SeparationType::Quasi => 2,
        SeparationType::MonotonicResponse => 3,
    }
}

unsafe fn alloc_warning(w: &Option<String>) -> *mut c_char {
    match w {
        Some(s) => alloc_string(s),
        None => std::ptr::null_mut(),
    }
}

unsafe fn free_ptr<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

/// BKW collinearity diagnostics of the design `[1, x]` (or `x`) of `x_count`
/// columns: R's `kappa(X, exact = TRUE)`, the condition indices and the
/// variance-decomposition proportions of the unit-column-length design
/// (`perturb::colldiag`, `olsrr::ols_eigen_cindex`).
///
/// # Safety
/// - `x` must point to `x_count` valid DataArrays of equal length
/// - `out_result` and `out_error` must be valid pointers; free the result
///   with `anofox_free_condition_diagnostic`
#[no_mangle]
pub unsafe extern "C" fn anofox_condition_diagnostic(
    x: *const DataArray,
    x_count: usize,
    fit_intercept: bool,
    out_result: *mut ConditionDiagnosticFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = ConditionDiagnosticFFI::default();
        match compute_condition_diagnostic(&columns(x, x_count), fit_intercept) {
            Ok(r) => {
                let n_terms = r.variance_proportions.first().map_or(0, |row| row.len());
                let flat: Vec<f64> = r.variance_proportions.concat();
                *out_result = ConditionDiagnosticFFI {
                    condition_number: r.condition_number,
                    condition_number_xtx: r.condition_number_xtx,
                    scaled_condition_number: r.scaled_condition_number,
                    severity: severity_code(r.severity),
                    singular_values: alloc_f64(&r.singular_values),
                    condition_indices: alloc_f64(&r.condition_indices),
                    variance_proportions: alloc_f64(&flat),
                    n_components: r.singular_values.len(),
                    n_terms,
                    warning: alloc_warning(&r.warning),
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

/// Free the arrays of a `ConditionDiagnosticFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_condition_diagnostic`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_condition_diagnostic(result: *mut ConditionDiagnosticFFI) {
    if let Some(r) = result.as_mut() {
        free_ptr(&mut r.singular_values);
        free_ptr(&mut r.condition_indices);
        free_ptr(&mut r.variance_proportions);
        free_ptr(&mut r.warning);
    }
}

unsafe fn write_separation_check(
    result: anofox_stats_core::StatsResult<SeparationCheckResult>,
    out_result: *mut SeparationCheckFFI,
    out_error: *mut AnofoxError,
) -> bool {
    match result {
        Ok(r) => {
            let types: Vec<i32> = r
                .separation_types
                .iter()
                .map(|&t| separation_code(t))
                .collect();
            *out_result = SeparationCheckFFI {
                has_separation: r.has_separation,
                separated_predictors: alloc_u64(&r.separated_predictors),
                separation_types: alloc_i32(&types),
                n_separated: r.separated_predictors.len().min(types.len()),
                warning: alloc_warning(&r.warning),
            };
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

/// Per-predictor separation heuristics of upstream. `kind` 0: binary response
/// (`check_binary_separation`, `y` must be 0/1); 1: count response
/// (`check_count_sparsity`, `y` must be a non-negative count).
///
/// # Safety
/// - `x` must point to `x_count` valid DataArrays of `y.len` values
/// - `out_result` and `out_error` must be valid pointers; free the result
///   with `anofox_free_separation_check`
#[no_mangle]
pub unsafe extern "C" fn anofox_separation_check(
    kind: i32,
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    out_result: *mut SeparationCheckFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = SeparationCheckFFI::default();
        let x_vecs = columns(x, x_count);
        let y_vec = y.to_vec();
        let result = match kind {
            0 => check_binary_separation_diagnostic(&x_vecs, &y_vec),
            1 => check_count_sparsity_diagnostic(&x_vecs, &y_vec),
            _ => {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::InvalidInput, "invalid separation check kind");
                }
                return false;
            }
        };
        write_separation_check(result, out_result, out_error)
    })
}

/// Free the arrays of a `SeparationCheckFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_separation_check`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_separation_check(result: *mut SeparationCheckFFI) {
    if let Some(r) = result.as_mut() {
        free_ptr(&mut r.separated_predictors);
        free_ptr(&mut r.separation_types);
        free_ptr(&mut r.warning);
    }
}

/// Exact (quasi-)separation check of a binary response by upstream's linear
/// program (Konis 2007, R `detectseparation::detect_separation`).
///
/// # Safety
/// - `x` must point to `x_count` valid DataArrays of `y.len` values
/// - `out_result` and `out_error` must be valid pointers; free the result
///   with `anofox_free_linear_separation`
#[no_mangle]
pub unsafe extern "C" fn anofox_detect_separation(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    fit_intercept: bool,
    out_result: *mut LinearSeparationFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        *out_result = LinearSeparationFFI::default();
        match detect_linear_separation(&columns(x, x_count), &y.to_vec(), fit_intercept) {
            Ok(r) => {
                *out_result = LinearSeparationFFI {
                    separated: r.separated,
                    infinite: alloc_f64(&r.infinite),
                    direction: alloc_f64(&r.direction),
                    n_terms: r.infinite.len(),
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

/// Free the arrays of a `LinearSeparationFFI`.
///
/// # Safety
/// `result` must be NULL or filled by `anofox_detect_separation`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_linear_separation(result: *mut LinearSeparationFFI) {
    if let Some(r) = result.as_mut() {
        free_ptr(&mut r.infinite);
        free_ptr(&mut r.direction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    #[test]
    fn condition_diagnostic_round_trip() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let x = [arr(&a), arr(&b)];
        let mut out = ConditionDiagnosticFFI::default();
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_condition_diagnostic(
                x.as_ptr(),
                2,
                true,
                &mut out,
                &mut err
            ));
            assert_eq!(out.n_components, 3);
            assert_eq!(out.n_terms, 3);
            let pi = slice::from_raw_parts(out.variance_proportions, 9);
            for j in 0..3 {
                let total: f64 = (0..3).map(|k| pi[k * 3 + j]).sum();
                assert!((total - 1.0).abs() < 1e-10);
            }
            anofox_free_condition_diagnostic(&mut out);
            assert!(out.variance_proportions.is_null());
        }
    }

    #[test]
    fn separation_check_and_lp() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = [arr(&a)];
        let mut err = AnofoxError::success();
        unsafe {
            let mut chk = SeparationCheckFFI::default();
            assert!(anofox_separation_check(
                0,
                arr(&y),
                x.as_ptr(),
                1,
                &mut chk,
                &mut err
            ));
            assert!(chk.has_separation);
            assert_eq!(chk.n_separated, 1);
            assert_eq!(*chk.separation_types, 1);
            anofox_free_separation_check(&mut chk);

            let mut lin = LinearSeparationFFI::default();
            assert!(anofox_detect_separation(
                arr(&y),
                x.as_ptr(),
                1,
                true,
                &mut lin,
                &mut err
            ));
            assert!(lin.separated);
            assert_eq!(lin.n_terms, 2);
            assert_eq!(*lin.infinite.add(1), f64::INFINITY);
            anofox_free_linear_separation(&mut lin);

            let bad = [0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
            assert!(!anofox_detect_separation(
                arr(&bad),
                x.as_ptr(),
                1,
                true,
                &mut lin,
                &mut err
            ));
            assert_eq!(err.code, ErrorCode::InvalidInput);
        }
    }
}
