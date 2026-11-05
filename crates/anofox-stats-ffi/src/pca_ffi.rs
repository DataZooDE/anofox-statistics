//! C FFI for principal component analysis (#303). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::multivariate::{pca_fit, pca_transform};

/// A fitted PCA. The arrays are allocated by `anofox_pca_fit` and released
/// with `anofox_free_pca_result`.
#[repr(C)]
#[derive(Debug)]
pub struct PcaResultFFI {
    /// Column means, `n_features` values.
    pub center: *mut f64,
    /// Component standard deviations, `n_components` values.
    pub sdev: *mut f64,
    /// Share of the total variance per component, `n_components` values.
    pub proportion_of_variance: *mut f64,
    /// Loadings, component-major: `loadings[j * n_features + i]` is variable
    /// `i` of component `j`.
    pub loadings: *mut f64,
    pub n_features: usize,
    pub n_components: usize,
    pub n_observations: usize,
}

unsafe fn malloc_copy(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

/// PCA (R's `prcomp(x, center = TRUE, scale. = FALSE)`) of `n_rows` rows of
/// `n_features` values each, row-major in `data`, keeping the first
/// `n_components` components (0: all).
///
/// # Safety
/// - `data` must point to `n_rows * n_features` doubles
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_pca_fit(
    data: *const f64,
    n_rows: usize,
    n_features: usize,
    n_components: usize,
    out_result: *mut PcaResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let values: &[f64] = if data.is_null() || n_rows == 0 || n_features == 0 {
            &[]
        } else {
            slice::from_raw_parts(data, n_rows * n_features)
        };
        let rows: Vec<Vec<f64>> = if n_features == 0 {
            vec![Vec::new(); n_rows]
        } else {
            values.chunks(n_features).map(|r| r.to_vec()).collect()
        };
        let k = if n_components == 0 {
            None
        } else {
            Some(n_components)
        };
        match pca_fit(&rows, k) {
            Ok(fit) => {
                let k = fit.sdev.len();
                let d = fit.center.len();
                let loadings: Vec<f64> = (0..k)
                    .flat_map(|j| fit.rotation.iter().map(move |row| row[j]))
                    .collect();
                let out = PcaResultFFI {
                    center: malloc_copy(&fit.center),
                    sdev: malloc_copy(&fit.sdev),
                    proportion_of_variance: malloc_copy(&fit.proportion_of_variance),
                    loadings: malloc_copy(&loadings),
                    n_features: d,
                    n_components: k,
                    n_observations: fit.n_observations,
                };
                let failed = (d > 0 && out.center.is_null())
                    || (k > 0
                        && (out.sdev.is_null()
                            || out.proportion_of_variance.is_null()
                            || out.loadings.is_null()));
                *out_result = out;
                if failed {
                    anofox_free_pca_result(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate PCA result",
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

/// Free the arrays of a result filled by `anofox_pca_fit`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_pca_fit`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_pca_result(result: *mut PcaResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.center,
            &mut r.sdev,
            &mut r.proportion_of_variance,
            &mut r.loadings,
        ] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
    })
}

/// Scores of one row `x` (`x_len` values) on the `n_components` components
/// of a fitted PCA (`predict.prcomp`): `center` has `n_features` values,
/// `sdev` `n_components`, and `loadings` is component-major as in
/// `PcaResultFFI`. Writes `n_components` scores to `out_scores`.
///
/// # Safety
/// - `center` must point to `n_features` doubles, `sdev` to `n_components`,
///   `loadings` to `n_components * n_features` and `x` to `x_len`
/// - `out_scores` must point to `n_components` writable doubles
/// - `out_error` must be a valid pointer
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_pca_transform(
    center: *const f64,
    sdev: *const f64,
    loadings: *const f64,
    n_features: usize,
    n_components: usize,
    x: *const f64,
    x_len: usize,
    out_scores: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_scores, out_error) {
            return false;
        }
        let view = |p: *const f64, len: usize| -> Vec<f64> {
            if p.is_null() || len == 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(p, len).to_vec()
            }
        };
        let center = view(center, n_features);
        let sdev = view(sdev, n_components);
        let flat = view(loadings, n_components * n_features);
        let rotation: Vec<Vec<f64>> = (0..n_features)
            .map(|i| {
                (0..n_components)
                    .map(|j| flat[j * n_features + i])
                    .collect()
            })
            .collect();
        let row = view(x, x_len);
        match pca_transform(&center, &sdev, &rotation, &[row]) {
            Ok(scores) => {
                std::ptr::copy_nonoverlapping(scores[0].as_ptr(), out_scores, n_components);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}
