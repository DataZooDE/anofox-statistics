//! Cluster-robust (sandwich) covariance of the GLM and AFT fits, the
//! counterpart of the OLS / WLS one (#309).
//!
//! The covariance is upstream's: `GlmResult::cluster_vcov`
//! (`sandwich::vcovCL(glm(...), cluster, type, cadjust)`) for the
//! `penalized_glm::fit_*` results and `AftResult::cluster_vcov`
//! (`survreg(..., cluster = id)` / `rms::robcov(psm(...))`) for `aft::fit_aft`.
//! With Gaussian priors or a ridge `lambda` both use the penalized bread
//! `(XᵀWX + P)⁻¹` (the sandwich of the penalized estimating equation,
//! `rms::robcov` of a penalized fit), whatever `vcov` type the fit reports;
//! a coefficient at an L1 prior's kink has no sandwich, which upstream
//! reports as an error and this module passes on as an input error.
//!
//! The Wald tests and intervals built from the clustered covariance are
//! normal (z), as `lmtest::coeftest(glm, vcovCL(...))` (`coeftest.glm` uses
//! `df = Inf`) and `summary(survreg(..., cluster = id))`; they use upstream's
//! `CoefficientInference` with an infinite residual d.f. This module only
//! maps the cluster codes onto the fit and the results onto its inference.

use crate::errors::{StatsError, StatsResult};
use crate::models::cluster::ClusterSpec;
use crate::types::HcType;
use anofox_regression::inference::{
    ClusterError, ClusterOptions, ClusterType, CoefficientInference,
};
use anofox_regression::solvers::aft::AftResult;
use anofox_regression::solvers::penalized_glm::GlmResult;
use faer::{Col, Mat};

/// `vcovCL(type = ...)` of a GLM / AFT fit: HC0 (the default of both) or
/// HC1; HC2 / HC3 are for linear models only.
fn cluster_options(hc_type: Option<HcType>, cadjust: bool) -> StatsResult<ClusterOptions> {
    let kind =
        match hc_type.unwrap_or(HcType::HC0) {
            HcType::HC0 => ClusterType::HC0,
            HcType::HC1 => ClusterType::HC1,
            HcType::HC2 | HcType::HC3 => return Err(StatsError::InvalidInput(
                "cluster: hc_type HC2 / HC3 are available for linear models only; use HC0 or HC1"
                    .into(),
            )),
        };
    Ok(ClusterOptions::default().kind(kind).cadjust(cadjust))
}

/// Codes of the rows `rows` per dimension, and the clusters per dimension.
fn codes_of(spec: &ClusterSpec, n: usize, rows: &[usize]) -> StatsResult<Vec<usize>> {
    if spec.codes.is_empty() {
        return Err(StatsError::InvalidInput(
            "cluster: at least one clustering dimension is needed".into(),
        ));
    }
    if let Some(c) = spec.codes.iter().find(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatch {
            y_len: n,
            x_rows: c.len(),
        });
    }
    Ok(spec
        .codes
        .iter()
        .map(|c| {
            let mut u: Vec<usize> = rows.iter().map(|&i| c[i]).collect();
            u.sort_unstable();
            u.dedup();
            u.len()
        })
        .collect())
}

/// The clustered covariance, `None` when it is undefined for the data (a
/// dimension with fewer than two clusters, a singular information); an
/// unsupported variant (an L1 kink) is an error.
fn clustered(
    n_clusters: &[usize],
    vcov: impl FnOnce() -> Result<Mat<f64>, ClusterError>,
) -> StatsResult<Option<Mat<f64>>> {
    if n_clusters.iter().any(|&g| g < 2) {
        return Ok(None);
    }
    match vcov() {
        Ok(v) => Ok(Some(v)),
        Err(ClusterError::Singular) => Ok(None),
        Err(ClusterError::Unsupported(msg)) => Err(StatsError::InvalidInput(format!(
            "cluster-robust covariance: {msg}"
        ))),
        Err(e) => Err(StatsError::InvalidInput(format!(
            "cluster-robust covariance: {e}"
        ))),
    }
}

/// Normal Wald inference of `beta` with standard errors `se`.
struct Wald {
    z: Col<f64>,
    p: Col<f64>,
    lo: Col<f64>,
    hi: Col<f64>,
}

fn wald(beta: &Col<f64>, se: &Col<f64>, level: f64) -> Wald {
    let z = CoefficientInference::t_statistics(beta, se);
    let p = CoefficientInference::p_values(&z, f64::INFINITY);
    let (lo, hi) = CoefficientInference::confidence_intervals(beta, se, f64::INFINITY, level);
    Wald { z, p, lo, hi }
}

fn sd(v: &Mat<f64>, k: usize) -> f64 {
    let d = v[(k, k)];
    if d >= 0.0 {
        d.sqrt()
    } else {
        f64::NAN
    }
}

/// Replace the inference of a GLM fit by the cluster-robust one
/// (`sandwich::vcovCL(glm, cluster, type = hc_type (HC0), cadjust)`). `y` and
/// `x` are the data the fit was given (column-major, offset column
/// included); `spec` has one code per input row. Returns the clusters per
/// dimension among the fitted rows. Without a clustered covariance (see
/// [`clustered`]) the inference is NaN and the covariance `None`.
pub fn glm_cluster_inference(
    result: &mut GlmResult,
    y: &[f64],
    x: &[Vec<f64>],
    spec: &ClusterSpec,
    hc_type: Option<HcType>,
) -> StatsResult<Vec<usize>> {
    let opts = cluster_options(hc_type, spec.cadjust)?;
    let n_clusters = codes_of(spec, y.len(), &result.valid_rows)?;
    if result.inference.is_none() {
        return Ok(n_clusters);
    }
    let refs: Vec<&[usize]> = spec.codes.iter().map(Vec::as_slice).collect();
    let v = clustered(&n_clusters, || result.cluster_vcov(y, x, &refs, &opts))?;
    let coefficients = result.core.coefficients.clone();
    let intercept = result.core.intercept;
    let Some(inf) = result.inference.as_mut() else {
        return Ok(n_clusters);
    };
    let p = coefficients.len();
    let mut se = vec![f64::NAN; p];
    let mut se_int = f64::NAN;
    if let Some(v) = &v {
        for (k, param) in inf.matrix_parameters.iter().enumerate() {
            match param {
                None => se_int = sd(v, k),
                Some(j) => se[*j] = sd(v, k),
            }
        }
    }
    let w = wald(
        &Col::from_fn(p, |j| coefficients[j]),
        &Col::from_fn(p, |j| se[j]),
        inf.confidence_level,
    );
    inf.std_errors = se;
    inf.z_values = w.z.iter().copied().collect();
    inf.p_values = w.p.iter().copied().collect();
    inf.ci_lower = w.lo.iter().copied().collect();
    inf.ci_upper = w.hi.iter().copied().collect();
    if let Some(b0) = intercept {
        let wi = wald(
            &Col::from_fn(1, |_| b0),
            &Col::from_fn(1, |_| se_int),
            inf.confidence_level,
        );
        inf.intercept_std_error = Some(se_int);
        inf.intercept_z_value = Some(wi.z[0]);
        inf.intercept_p_value = Some(wi.p[0]);
        inf.intercept_ci_lower = Some(wi.lo[0]);
        inf.intercept_ci_upper = Some(wi.hi[0]);
    }
    inf.vcov = v;
    Ok(n_clusters)
}

/// Replace the inference of an AFT fit by the cluster-robust one
/// (`survreg(..., cluster = id)` with `hc_type` HC0 and `cadjust = false`,
/// its convention; `rms::robcov`). `time`, `x`, `event` are the data of the
/// fit. The covariance covers intercept, coefficients and `log σ` (when the
/// scale is estimated). Returns the clusters per dimension among the fitted
/// rows; without a clustered covariance the inference is NaN and the
/// covariance `None`.
pub fn aft_cluster_inference(
    result: &mut AftResult,
    time: &[f64],
    x: &[Vec<f64>],
    event: &[f64],
    spec: &ClusterSpec,
    hc_type: Option<HcType>,
) -> StatsResult<Vec<usize>> {
    let opts = cluster_options(hc_type, spec.cadjust)?;
    let used: Vec<usize> = (0..time.len())
        .filter(|&i| {
            time[i].is_finite()
                && event.get(i).is_some_and(|e| e.is_finite())
                && x.iter().all(|c| c[i].is_finite())
        })
        .collect();
    let n_clusters = codes_of(spec, time.len(), &used)?;
    if result.inference.is_none() {
        return Ok(n_clusters);
    }
    let refs: Vec<&[usize]> = spec.codes.iter().map(Vec::as_slice).collect();
    let v = clustered(&n_clusters, || {
        result.cluster_vcov(time, x, event, &refs, &opts)
    })?;
    let coefficients = result.core.coefficients.clone();
    let intercept = result.core.intercept;
    let Some(inf) = result.inference.as_mut() else {
        return Ok(n_clusters);
    };
    let p = coefficients.len();
    let off = usize::from(intercept.is_some());
    let k = v.as_ref().map_or(0, |m| m.nrows());
    let se_at = |i: usize| v.as_ref().filter(|_| i < k).map_or(f64::NAN, |m| sd(m, i));
    let se: Vec<f64> = (0..p).map(|j| se_at(off + j)).collect();
    let w = wald(
        &Col::from_fn(p, |j| coefficients[j]),
        &Col::from_fn(p, |j| se[j]),
        inf.confidence_level,
    );
    inf.std_errors = se;
    inf.z_values = w.z.iter().copied().collect();
    inf.p_values = w.p.iter().copied().collect();
    inf.ci_lower = w.lo.iter().copied().collect();
    inf.ci_upper = w.hi.iter().copied().collect();
    if let Some(b0) = intercept {
        let s0 = se_at(0);
        let wi = wald(
            &Col::from_fn(1, |_| b0),
            &Col::from_fn(1, |_| s0),
            inf.confidence_level,
        );
        inf.intercept_std_error = Some(s0);
        inf.intercept_z_value = Some(wi.z[0]);
        inf.intercept_p_value = Some(wi.p[0]);
        inf.intercept_ci_lower = Some(wi.lo[0]);
        inf.intercept_ci_upper = Some(wi.hi[0]);
    }
    if inf.log_scale_std_error.is_some() {
        inf.log_scale_std_error = Some(se_at(off + p));
    }
    inf.vcov = v;
    Ok(n_clusters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::AftOptions;
    use crate::types::PoissonOptions;

    fn counts() -> (Vec<f64>, Vec<Vec<f64>>, Vec<usize>) {
        let x: Vec<f64> = (0..24).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| ((1.0 + 0.6 * v).exp() + (i % 3) as f64).round())
            .collect();
        let g: Vec<usize> = (0..24).map(|i| i / 4).collect();
        (y, vec![x], g)
    }

    #[test]
    fn glm_cluster_replaces_inference() {
        let (y, x, g) = counts();
        let opts = PoissonOptions::default().compute_inference(true);
        let mut r = crate::models::fit_poisson(&y, &x, &opts).unwrap();
        let classical = r.inference.as_ref().unwrap().std_errors[0];
        let spec = ClusterSpec {
            codes: vec![g],
            cadjust: true,
        };
        let n = glm_cluster_inference(&mut r, &y, &x, &spec, None).unwrap();
        assert_eq!(n, vec![6]);
        let inf = r.inference.as_ref().unwrap();
        let v = inf.vcov.as_ref().unwrap();
        assert!((inf.std_errors[0] - v[(1, 1)].sqrt()).abs() < 1e-14);
        assert!((inf.intercept_std_error.unwrap() - v[(0, 0)].sqrt()).abs() < 1e-14);
        assert!(inf.std_errors[0] != classical);
        assert!((inf.z_values[0] - r.core.coefficients[0] / inf.std_errors[0]).abs() < 1e-12);
        // HC2 / HC3 are rejected; one cluster leaves the inference undefined.
        assert!(glm_cluster_inference(&mut r, &y, &x, &spec, Some(HcType::HC3)).is_err());
        let one = ClusterSpec {
            codes: vec![vec![0; 24]],
            cadjust: true,
        };
        let n = glm_cluster_inference(&mut r, &y, &x, &one, None).unwrap();
        assert_eq!(n, vec![1]);
        let inf = r.inference.as_ref().unwrap();
        assert!(inf.std_errors[0].is_nan() && inf.vcov.is_none());
    }

    #[test]
    fn aft_cluster_covers_log_scale() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.41).cos()).collect();
        let time: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| (1.0 + 0.5 * v + 0.1 * (i % 5) as f64).exp())
            .collect();
        let event: Vec<f64> = (0..30)
            .map(|i| if i % 4 == 0 { 0.0 } else { 1.0 })
            .collect();
        let xs = vec![x];
        let opts = AftOptions::default().compute_inference(true);
        let mut r = crate::models::fit_aft(&time, &xs, &event, &opts).unwrap();
        let spec = ClusterSpec {
            codes: vec![(0..30).map(|i| i / 3).collect()],
            cadjust: false,
        };
        let n = aft_cluster_inference(&mut r, &time, &xs, &event, &spec, None).unwrap();
        assert_eq!(n, vec![10]);
        let inf = r.inference.as_ref().unwrap();
        let v = inf.vcov.as_ref().unwrap();
        assert_eq!(v.nrows(), 3);
        assert!((inf.log_scale_std_error.unwrap() - v[(2, 2)].sqrt()).abs() < 1e-14);
    }
}
