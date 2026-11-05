//! FFI for the cluster-robust (sandwich) covariance of OLS / WLS fits (#309):
//! `sandwich::vcovCL` through upstream's `cluster_vcov`.
//!
//! Outputs are those of `anofox_ols_fit_from_state` (core, inference and the
//! covariance), plus the number of clusters per dimension.

use super::*;
use anofox_stats_core::models::{
    fit_ols_cluster, fit_wls_cluster, ClusterFit, ClusterSpec, StateFit,
};
use anofox_stats_core::StatsResult;

unsafe fn fail(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
    false
}

/// Read `n_dims` code arrays of `n` entries each.
unsafe fn read_clusters(
    clusters: *const *const i32,
    n_dims: usize,
    n: usize,
) -> Result<Vec<Vec<usize>>, &'static str> {
    if clusters.is_null() || n_dims == 0 {
        return Err("cluster: at least one clustering dimension is needed");
    }
    let dims = slice::from_raw_parts(clusters, n_dims);
    let mut out = Vec::with_capacity(n_dims);
    for &d in dims {
        if d.is_null() && n > 0 {
            return Err("cluster: NULL code array");
        }
        let codes = if n == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(d, n)
        };
        if codes.iter().any(|&c| c < 0) {
            return Err("cluster: codes must be non-negative");
        }
        out.push(codes.iter().map(|&c| c as usize).collect());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
unsafe fn write_cluster_fit(
    fit: StatsResult<ClusterFit>,
    n_dims: usize,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_n_clusters: *mut i64,
    out_error: *mut AnofoxError,
) -> bool {
    if !out_n_clusters.is_null() {
        let out = slice::from_raw_parts_mut(out_n_clusters, n_dims);
        out.fill(0);
        if let Ok(f) = &fit {
            for (o, &g) in out.iter_mut().zip(&f.n_clusters) {
                *o = g as i64;
            }
        }
    }
    let fit = fit.map(|f| StateFit {
        fit: f.fit,
        vcov: f.vcov,
    });
    ls_state_ffi::write_fit_result(fit, out_core, out_inference, out_vcov, out_dim, out_error)
}

/// OLS with the cluster-robust covariance (`vcovCL(type = hc_type or HC1,
/// cadjust)`). `clusters` holds `n_dims` arrays of `y.len` non-negative codes
/// (two or more dimensions: multiway clustering). Outputs as
/// `anofox_ols_fit_from_state`; `out_n_clusters` (NULL or `n_dims` entries)
/// receives the clusters per dimension among the fitted rows (0 without
/// inference).
///
/// # Safety
/// `x` must point to `x_count` DataArrays and `clusters` to `n_dims` arrays of
/// `y.len` codes; `out_core` must be valid, the other outputs may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_ols_cluster_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    clusters: *const *const i32,
    n_dims: usize,
    cadjust: bool,
    options: OlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_n_clusters: *mut i64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        if out_core.is_null() || x.is_null() || x_count == 0 {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        }
        let y_vec = y.to_vec();
        let x_vecs: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
            .iter()
            .map(|a| a.to_vec())
            .collect();
        let codes = match read_clusters(clusters, n_dims, y_vec.len()) {
            Ok(c) => c,
            Err(msg) => return fail(out_error, ErrorCode::InvalidInput, msg),
        };
        let spec = ClusterSpec { codes, cadjust };
        let opts = OlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: convert_hc_type_ffi(options.hc_type),
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
            compute_loo: options.compute_loo,
        };
        write_cluster_fit(
            fit_ols_cluster(&y_vec, &x_vecs, &spec, &opts),
            n_dims,
            out_core,
            out_inference,
            out_vcov,
            out_dim,
            out_n_clusters,
            out_error,
        )
    })
}

/// WLS with the cluster-robust covariance; as `anofox_ols_cluster_fit` with
/// the observation `weights`.
///
/// # Safety
/// As `anofox_ols_cluster_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_wls_cluster_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    weights: DataArray,
    clusters: *const *const i32,
    n_dims: usize,
    cadjust: bool,
    options: WlsOptionsFFI,
    out_core: *mut FitResultCore,
    out_inference: *mut FitResultInference,
    out_vcov: *mut *mut f64,
    out_dim: *mut usize,
    out_n_clusters: *mut i64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        if out_core.is_null() || x.is_null() || x_count == 0 {
            return fail(out_error, ErrorCode::InvalidInput, "NULL input pointer");
        }
        let y_vec = y.to_vec();
        let w_vec = weights.to_vec();
        let x_vecs: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
            .iter()
            .map(|a| a.to_vec())
            .collect();
        let codes = match read_clusters(clusters, n_dims, y_vec.len()) {
            Ok(c) => c,
            Err(msg) => return fail(out_error, ErrorCode::InvalidInput, msg),
        };
        let spec = ClusterSpec { codes, cadjust };
        let opts = WlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            solver: convert_solver_ffi(options.solver),
            hc_type: convert_hc_type_ffi(options.hc_type),
            alias_tolerance: alias_tolerance_ffi(options.alias_tolerance),
            compute_loo: options.compute_loo,
        };
        write_cluster_fit(
            fit_wls_cluster(&y_vec, &x_vecs, &w_vec, &spec, &opts),
            n_dims,
            out_core,
            out_inference,
            out_vcov,
            out_dim,
            out_n_clusters,
            out_error,
        )
    })
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
    fn ols_cluster_fit_ffi_round_trip() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 1.0 + 0.5 * v + if i % 2 == 0 { 0.3 } else { -0.2 })
            .collect();
        let g: Vec<i32> = (0..10).map(|i| i / 2).collect();
        let ya = arr(&y);
        let xa = [arr(&x)];
        let dims = [g.as_ptr()];
        let mut core = FitResultCore::default();
        let mut inf = FitResultInference::default();
        let mut vcov: *mut f64 = std::ptr::null_mut();
        let mut dim = 0usize;
        let mut ng = [0i64; 1];
        let mut err = AnofoxError::success();
        let opts = OlsOptionsFFI {
            compute_inference: true,
            ..Default::default()
        };
        unsafe {
            assert!(anofox_ols_cluster_fit(
                ya,
                xa.as_ptr(),
                1,
                dims.as_ptr(),
                1,
                true,
                opts,
                &mut core,
                &mut inf,
                &mut vcov,
                &mut dim,
                ng.as_mut_ptr(),
                &mut err,
            ));
            assert_eq!(ng[0], 5);
            assert_eq!(dim, 2);
            assert!(!vcov.is_null());
            let se = *inf.std_errors;
            assert!((se - (*vcov.add(3)).sqrt()).abs() < 1e-12);
            anofox_free_result_core(&mut core);
            anofox_free_result_inference(&mut inf);
            libc::free(vcov as *mut libc::c_void);

            let bad = [-1i32; 10];
            let dims = [bad.as_ptr()];
            assert!(!anofox_ols_cluster_fit(
                arr(&y),
                xa.as_ptr(),
                1,
                dims.as_ptr(),
                1,
                true,
                OlsOptionsFFI {
                    compute_inference: true,
                    ..Default::default()
                },
                &mut core,
                &mut inf,
                &mut vcov,
                &mut dim,
                ng.as_mut_ptr(),
                &mut err,
            ));
        }
    }
}
