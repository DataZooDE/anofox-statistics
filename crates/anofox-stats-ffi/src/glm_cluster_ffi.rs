//! FFI for the cluster-robust (sandwich) covariance of the GLM and AFT fits:
//! `sandwich::vcovCL(glm)` / `survreg(cluster =)` through upstream's
//! `GlmResult::cluster_vcov` / `AftResult::cluster_vcov`
//! (`anofox_stats_core::models::{glm_cluster_inference, aft_cluster_inference}`).
//!
//! Each `anofox_<family>_cluster_fit` takes the options of `anofox_<family>_fit`
//! plus the cluster codes, and fills the same outputs with the cluster-robust
//! inference and covariance, and the clusters per dimension.

use super::cluster_ffi::read_clusters;
use super::*;
use anofox_stats_core::models::{
    aft_cluster_inference, glm_cluster_inference, AftResult, ClusterSpec, GlmResult,
};
use anofox_stats_core::StatsResult;

unsafe fn fail(out_error: *mut AnofoxError, code: ErrorCode, msg: &str) -> bool {
    if !out_error.is_null() {
        (*out_error).set(code, msg);
    }
    false
}

unsafe fn write_n_clusters(out: *mut i64, n_dims: usize, n_clusters: &[usize]) {
    if out.is_null() {
        return;
    }
    let out = slice::from_raw_parts_mut(out, n_dims);
    out.fill(0);
    for (o, &g) in out.iter_mut().zip(n_clusters) {
        *o = g as i64;
    }
}

/// Inputs shared by the GLM entry points.
struct Inputs {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    spec: ClusterSpec,
}

unsafe fn read_inputs(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    clusters: *const *const i32,
    n_dims: usize,
    cadjust: bool,
    out_error: *mut AnofoxError,
) -> Option<Inputs> {
    if x.is_null() || x_count == 0 {
        fail(
            out_error,
            if x_count == 0 {
                ErrorCode::InsufficientData
            } else {
                ErrorCode::InvalidInput
            },
            "x is NULL or empty",
        );
        return None;
    }
    let y = y.to_vec();
    let x: Vec<Vec<f64>> = slice::from_raw_parts(x, x_count)
        .iter()
        .map(|a| a.to_vec())
        .collect();
    match read_clusters(clusters, n_dims, y.len()) {
        Ok(codes) => Some(Inputs {
            y,
            x,
            spec: ClusterSpec { codes, cadjust },
        }),
        Err(msg) => {
            fail(out_error, ErrorCode::InvalidInput, msg);
            None
        }
    }
}

/// Write a GLM result as `anofox_poisson_fit` does (z values in `t_values`).
unsafe fn write_glm(
    result: GlmResult,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_error: *mut AnofoxError,
) -> bool {
    let n_coef = result.core.coefficients.len();
    let coef_ptr = alloc_f64(&result.core.coefficients);
    if coef_ptr.is_null() && n_coef > 0 {
        return fail(
            out_error,
            ErrorCode::AllocationFailure,
            "Failed to allocate coefficients",
        );
    }
    let (vcov_ptr, vcov_dim) = glm_vcov_ffi(&result);
    *out_result = GlmFitResultCore {
        coefficients: coef_ptr,
        coefficients_len: n_coef,
        intercept: result.core.intercept.unwrap_or(f64::NAN),
        deviance: result.core.residual_deviance,
        null_deviance: result.core.null_deviance,
        pseudo_r_squared: result.core.pseudo_r_squared,
        aic: result.core.aic,
        dispersion: result.core.dispersion.unwrap_or(f64::NAN),
        n_observations: result.core.n_observations,
        n_features: result.core.n_features,
        iterations: result.core.iterations,
        converged: result.core.converged,
        summary: ModelSummaryFFI::from(&anofox_stats_core::ModelSummary::from_glm(&result)),
        vcov: vcov_ptr,
        vcov_dim,
    };
    if !out_inference.is_null() {
        *out_inference = match result.inference {
            Some(inf) => FitResultInference {
                std_errors: alloc_f64(&inf.std_errors),
                t_values: alloc_f64(&inf.z_values),
                p_values: alloc_f64(&inf.p_values),
                ci_lower: alloc_f64(&inf.ci_lower),
                ci_upper: alloc_f64(&inf.ci_upper),
                len: inf.std_errors.len(),
                confidence_level: inf.confidence_level,
                f_statistic: f64::NAN,
                f_pvalue: f64::NAN,
            },
            None => FitResultInference::default(),
        };
    }
    true
}

/// Fit with `fit`, replace its inference by the cluster-robust one and write
/// it out.
#[allow(clippy::too_many_arguments)]
unsafe fn glm_cluster_fit(
    inputs: Inputs,
    n_dims: usize,
    hc_type: HcTypeFFI,
    fit: impl FnOnce(&[f64], &[Vec<f64>]) -> StatsResult<GlmResult>,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_n_clusters: *mut i64,
    out_error: *mut AnofoxError,
) -> bool {
    let fitted = fit(&inputs.y, &inputs.x).and_then(|mut r| {
        let n = glm_cluster_inference(
            &mut r,
            &inputs.y,
            &inputs.x,
            &inputs.spec,
            convert_hc_type_ffi(hc_type),
        )?;
        Ok((r, n))
    });
    match fitted {
        Ok((result, n_clusters)) => {
            write_n_clusters(out_n_clusters, n_dims, &n_clusters);
            write_glm(result, out_result, out_inference, out_error)
        }
        Err(e) => fail(out_error, error_to_code(&e), &e.to_string()),
    }
}

macro_rules! glm_cluster_entry {
    ($(#[$doc:meta])* $name:ident, $opts_ty:ty, $opts_fn:ident, $fit:expr) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// `x` must point to `x_count` DataArrays and `clusters` to `n_dims`
        /// arrays of `y.len` codes; `out_result` must be valid, `out_inference`
        /// and `out_n_clusters` (`n_dims` entries) may be NULL.
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            y: DataArray,
            x: *const DataArray,
            x_count: usize,
            options: $opts_ty,
            clusters: *const *const i32,
            n_dims: usize,
            hc_type: HcTypeFFI,
            cadjust: bool,
            out_result: *mut GlmFitResultCore,
            out_inference: *mut FitResultInference,
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
                if out_result.is_null() {
                    return fail(out_error, ErrorCode::InvalidInput, "out_result is NULL");
                }
                let Some(inputs) = read_inputs(y, x, x_count, clusters, n_dims, cadjust, out_error)
                else {
                    return false;
                };
                let opts = $opts_fn(&options);
                glm_cluster_fit(
                    inputs,
                    n_dims,
                    hc_type,
                    |y, x| ($fit)(y, x, &opts),
                    out_result,
                    out_inference,
                    out_n_clusters,
                    out_error,
                )
            })
        }
    };
}

glm_cluster_entry!(
    /// Poisson regression with the cluster-robust covariance
    /// (`sandwich::vcovCL(glm(..., family = poisson), cluster, type = hc_type
    /// (HC0), cadjust)`); outputs as `anofox_poisson_fit`.
    anofox_poisson_cluster_fit,
    PoissonOptionsFFI,
    poisson_options_from_ffi,
    fit_poisson
);
glm_cluster_entry!(
    /// Binomial regression with the cluster-robust covariance.
    anofox_binomial_cluster_fit,
    BinomialOptionsFFI,
    binomial_options_from_ffi,
    fit_binomial
);
glm_cluster_entry!(
    /// Negative binomial regression with the cluster-robust covariance
    /// (`theta` held at its estimate, as `vcovCL(glm.nb)`).
    anofox_negbinomial_cluster_fit,
    NegBinomialOptionsFFI,
    negbinomial_options_from_ffi,
    fit_negbinomial
);
glm_cluster_entry!(
    /// Tweedie regression with the cluster-robust covariance.
    anofox_tweedie_cluster_fit,
    TweedieOptionsFFI,
    tweedie_options_from_ffi,
    fit_tweedie
);
glm_cluster_entry!(
    /// Gamma regression with the cluster-robust covariance.
    anofox_gamma_cluster_fit,
    GammaOptionsFFI,
    gamma_options_from_ffi,
    fit_gamma
);

/// Logistic regression with the cluster-robust covariance; outputs as
/// `anofox_logistic_fit`.
///
/// # Safety
/// As `anofox_poisson_cluster_fit`; `out_extras` may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_logistic_cluster_fit(
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    options: LogisticOptionsFFI,
    clusters: *const *const i32,
    n_dims: usize,
    hc_type: HcTypeFFI,
    cadjust: bool,
    out_result: *mut GlmFitResultCore,
    out_inference: *mut FitResultInference,
    out_extras: *mut LogisticFitExtras,
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
        if out_result.is_null() {
            return fail(out_error, ErrorCode::InvalidInput, "out_result is NULL");
        }
        let Some(inputs) = read_inputs(y, x, x_count, clusters, n_dims, cadjust, out_error) else {
            return false;
        };
        let opts = logistic_options_from_ffi(&options);
        let mut extras = None;
        let ok = glm_cluster_fit(
            inputs,
            n_dims,
            hc_type,
            |y, x| {
                fit_logistic(y, x, &opts).map(|l| {
                    extras = Some(LogisticFitExtras {
                        accuracy: l.accuracy,
                        threshold: l.threshold,
                    });
                    l.fit
                })
            },
            out_result,
            out_inference,
            out_n_clusters,
            out_error,
        );
        if ok && !out_extras.is_null() {
            if let Some(e) = extras {
                *out_extras = e;
            }
        }
        ok
    })
}

/// AFT regression with the cluster-robust covariance (`survreg(...,
/// cluster = id)` with `hc_type` HC0 (default) and `cadjust` false; HC1 and
/// `cadjust` as `vcovCL`). Outputs as `anofox_aft_fit`, plus the clustered
/// covariance of (intercept, coefficients, log scale) in `out_vcov`
/// (`out_dim` square, row-major; NULL when undefined, e.g. one cluster;
/// free with `anofox_free_interval_matrix`) and the clusters per dimension.
///
/// # Safety
/// As `anofox_aft_fit`; `clusters` must point to `n_dims` arrays of
/// `time.len` codes; `out_vcov`, `out_dim` and `out_n_clusters` may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_aft_cluster_fit(
    time: DataArray,
    x: *const DataArray,
    x_count: usize,
    event: DataArray,
    options: AftOptionsFFI,
    clusters: *const *const i32,
    n_dims: usize,
    hc_type: HcTypeFFI,
    cadjust: bool,
    out_core: *mut AftFitResultCore,
    out_inference: *mut AftInferenceFFI,
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
        if out_core.is_null() {
            return fail(out_error, ErrorCode::InvalidInput, "out_core is NULL");
        }
        let Some(inputs) = read_inputs(time, x, x_count, clusters, n_dims, cadjust, out_error)
        else {
            return false;
        };
        let event = event.to_vec();
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
            .compute_inference(options.compute_inference)
            .confidence_level(options.confidence_level)
            .priors(priors_from_ffi(options.priors, options.priors_len))
            .vcov(vcov_from_ffi(options.vcov));
        let fitted: StatsResult<(AftResult, Vec<usize>)> =
            fit_aft(&inputs.y, &inputs.x, &event, &opts).and_then(|mut r| {
                let n = aft_cluster_inference(
                    &mut r,
                    &inputs.y,
                    &inputs.x,
                    &event,
                    &inputs.spec,
                    convert_hc_type_ffi(hc_type),
                )?;
                Ok((r, n))
            });
        let (result, n_clusters) = match fitted {
            Ok(r) => r,
            Err(e) => return fail(out_error, error_to_code(&e), &e.to_string()),
        };
        write_n_clusters(out_n_clusters, n_dims, &n_clusters);
        if !out_vcov.is_null() && !out_dim.is_null() {
            *out_vcov = std::ptr::null_mut();
            *out_dim = 0;
            if let Some(v) = result.inference.as_ref().and_then(|i| i.vcov.as_ref()) {
                let k = v.nrows();
                let flat: Vec<f64> = (0..k * k).map(|i| v[(i / k, i % k)]).collect();
                *out_vcov = alloc_f64(&flat);
                *out_dim = if (*out_vcov).is_null() { 0 } else { k };
            }
        }
        let coef_ptr = alloc_f64(&result.core.coefficients);
        if coef_ptr.is_null() && !result.core.coefficients.is_empty() {
            return fail(
                out_error,
                ErrorCode::AllocationFailure,
                "Failed to allocate AFT coefficients",
            );
        }
        *out_core = AftFitResultCore {
            coefficients: coef_ptr,
            coefficients_len: result.core.coefficients.len(),
            intercept: result.core.intercept.unwrap_or(f64::NAN),
            scale: result.core.scale,
            log_likelihood: result.core.log_likelihood,
            null_log_likelihood: result.core.null_log_likelihood,
            aic: result.core.aic,
            bic: result.core.bic,
            n_observations: result.core.n_observations,
            n_events: result.core.n_events,
            n_censored: result.core.n_censored,
            n_features: result.core.n_features,
            iterations: result.core.iterations,
            converged: result.core.converged,
            summary: ModelSummaryFFI::from(&anofox_stats_core::ModelSummary::from_aft(&result)),
        };
        if !out_inference.is_null() {
            if let Some(inf) = result.inference {
                *out_inference = AftInferenceFFI {
                    std_errors: alloc_f64(&inf.std_errors),
                    z_values: alloc_f64(&inf.z_values),
                    p_values: alloc_f64(&inf.p_values),
                    ci_lower: alloc_f64(&inf.ci_lower),
                    ci_upper: alloc_f64(&inf.ci_upper),
                    len: inf.std_errors.len(),
                    confidence_level: inf.confidence_level,
                    intercept_std_error: inf.intercept_std_error.unwrap_or(f64::NAN),
                    log_scale_std_error: inf.log_scale_std_error.unwrap_or(f64::NAN),
                };
            }
        }
        true
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
    fn poisson_cluster_fit_round_trip() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| ((0.5 + v).exp() + (i % 3) as f64).round())
            .collect();
        let g: Vec<i32> = (0..20).map(|i| i / 4).collect();
        let xa = [arr(&x)];
        let dims = [g.as_ptr()];
        let opts = || PoissonOptionsFFI {
            compute_inference: true,
            ..Default::default()
        };
        let mut core = GlmFitResultCore::default();
        let mut inf = FitResultInference::default();
        let mut ng = [0i64; 1];
        let mut err = AnofoxError::success();
        unsafe {
            assert!(anofox_poisson_cluster_fit(
                arr(&y),
                xa.as_ptr(),
                1,
                opts(),
                dims.as_ptr(),
                1,
                HcTypeFFI::None,
                true,
                &mut core,
                &mut inf,
                ng.as_mut_ptr(),
                &mut err,
            ));
            assert_eq!(ng[0], 5);
            assert_eq!(core.vcov_dim, 2);
            assert!((*inf.std_errors - (*core.vcov.add(3)).sqrt()).abs() < 1e-12);
            anofox_free_glm_result(&mut core);
            anofox_free_result_inference(&mut inf);
            // HC3 is for linear models only.
            assert!(!anofox_poisson_cluster_fit(
                arr(&y),
                xa.as_ptr(),
                1,
                opts(),
                dims.as_ptr(),
                1,
                HcTypeFFI::HC3,
                true,
                &mut core,
                &mut inf,
                ng.as_mut_ptr(),
                &mut err,
            ));
        }
    }
}
