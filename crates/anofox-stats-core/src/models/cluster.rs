//! Cluster-robust (sandwich) covariance of OLS / WLS fits (#309).
//!
//! The covariance is upstream's `FittedOls::cluster_vcov` /
//! `FittedWls::cluster_vcov` (`anofox_regression::inference::cluster`, i.e.
//! `sandwich::vcovCL`), and the coefficient tests built from it use upstream's
//! `CoefficientInference` (t with the residual degrees of freedom, as
//! `lmtest::coeftest(fit, vcovCL(...))`) and `wald_chunk` (the overall F test,
//! as `lmtest::waldtest(fit, vcov = vcovCL(...))`). This module only maps the
//! cluster codes onto the rows upstream fitted and the results onto
//! `FitResultInference`.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultInference, HcType, ModelSummary, OlsOptions, WlsOptions};
use anofox_regression::core::RegressionResult;
use anofox_regression::inference::{
    wald_chunk, ClusterError, ClusterOptions, ClusterType, CoefficientInference, WaldDistribution,
};
use faer::{Col, Mat};

/// Clustering of the input rows: one code vector per clustering dimension,
/// each with one entry per input row (codes need not be dense).
#[derive(Debug, Clone)]
pub struct ClusterSpec {
    /// One code vector per dimension (two or more: multiway clustering).
    pub codes: Vec<Vec<usize>>,
    /// `vcovCL`'s `cadjust`: multiply each dimension's meat by `G/(G − 1)`.
    pub cadjust: bool,
}

/// A clustered fit: the fit with cluster-robust inference, the cluster-robust
/// coefficient covariance (row-major, intercept first, zero rows / columns for
/// columns that are not estimated; `None` without inference) and the number
/// of clusters per dimension among the fitted rows.
#[derive(Debug, Clone)]
pub struct ClusterFit {
    /// Fit; its inference (and intercept inference) are the cluster-robust ones.
    pub fit: FitResult,
    /// `(vcov, dim)`
    pub vcov: Option<(Vec<f64>, usize)>,
    /// Clusters per dimension.
    pub n_clusters: Vec<usize>,
}

impl ClusterFit {
    pub(crate) fn plain(fit: FitResult) -> Self {
        Self {
            fit,
            vcov: None,
            n_clusters: Vec::new(),
        }
    }
}

/// Which `vcovCL(type = ...)`: `hc_type` (HC1 when unset, `vcovCL`'s default
/// for linear models).
fn cluster_options(hc_type: Option<HcType>, cadjust: bool) -> ClusterOptions {
    let kind = match hc_type.unwrap_or(HcType::HC1) {
        HcType::HC0 => ClusterType::HC0,
        HcType::HC1 => ClusterType::HC1,
        HcType::HC2 => ClusterType::HC2,
        HcType::HC3 => ClusterType::HC3,
    };
    ClusterOptions::default().kind(kind).cadjust(cadjust)
}

pub(crate) fn validate_spec(spec: &ClusterSpec, n_obs: usize) -> StatsResult<()> {
    if spec.codes.is_empty() {
        return Err(StatsError::InvalidInput(
            "cluster: at least one clustering dimension is needed".into(),
        ));
    }
    if let Some(c) = spec.codes.iter().find(|c| c.len() != n_obs) {
        return Err(StatsError::DimensionMismatch {
            y_len: n_obs,
            x_rows: c.len(),
        });
    }
    Ok(())
}

/// A covariance matrix, row-major, with its dimension.
pub(crate) type Vcov = (Vec<f64>, usize);

/// What `fit_ols` / `fit_wls` hand to [`lm_cluster_inference`].
pub(crate) struct LmClusterInput<'a> {
    pub spec: &'a ClusterSpec,
    /// Input rows that were fitted, in fit order.
    pub valid_indices: &'a [usize],
    /// Input columns that are in the design, in design order.
    pub kept: &'a [usize],
    pub n_features: usize,
    pub result: &'a RegressionResult,
    pub hc_type: Option<HcType>,
    pub confidence_level: f64,
}

/// Cluster-robust inference of a linear fit. `vcov` computes upstream's
/// covariance (`[intercept,] kept` square) from the per-dimension codes of the
/// fitted rows. Returns the inference (full feature width, NaN for columns not
/// estimated), the expanded covariance and the clusters per dimension; the
/// intercept inference is written into `summary`.
///
/// When the clustered covariance is not available (fewer than two clusters in
/// a dimension, or upstream rejects the variant for this data, e.g. a cluster
/// with leverage 1 for HC2 / HC3) the inference is NaN (NULL in SQL) and there
/// is no covariance: substituting classical standard errors would hand back
/// numbers the caller did not ask for (as the `hc_type` path does).
pub(crate) fn lm_cluster_inference(
    input: &LmClusterInput<'_>,
    summary: &mut ModelSummary,
    vcov: impl FnOnce(&[&[usize]], &ClusterOptions) -> Result<Mat<f64>, ClusterError>,
) -> (FitResultInference, Option<Vcov>, Vec<usize>) {
    let codes: Vec<Vec<usize>> = input
        .spec
        .codes
        .iter()
        .map(|c| input.valid_indices.iter().map(|&i| c[i]).collect())
        .collect();
    let n_clusters: Vec<usize> = codes
        .iter()
        .map(|c| {
            let mut u = c.clone();
            u.sort_unstable();
            u.dedup();
            u.len()
        })
        .collect();
    let unavailable = || FitResultInference {
        std_errors: vec![f64::NAN; input.n_features],
        t_values: vec![f64::NAN; input.n_features],
        p_values: vec![f64::NAN; input.n_features],
        ci_lower: vec![f64::NAN; input.n_features],
        ci_upper: vec![f64::NAN; input.n_features],
        confidence_level: input.confidence_level,
        f_statistic: Some(f64::NAN),
        f_pvalue: Some(f64::NAN),
    };
    if n_clusters.iter().any(|&g| g < 2) {
        return (unavailable(), None, n_clusters);
    }
    let refs: Vec<&[usize]> = codes.iter().map(|c| c.as_slice()).collect();
    let opts = cluster_options(input.hc_type, input.spec.cadjust);
    match vcov(&refs, &opts) {
        Ok(v) => {
            let (inf, full) = inference_from_vcov(input, summary, &v);
            (inf, Some(full), n_clusters)
        }
        Err(_) => (unavailable(), None, n_clusters),
    }
}

fn inference_from_vcov(
    input: &LmClusterInput<'_>,
    summary: &mut ModelSummary,
    v: &Mat<f64>,
) -> (FitResultInference, (Vec<f64>, usize)) {
    let r = input.result;
    let off = usize::from(r.intercept.is_some());
    let p = r.coefficients.nrows();
    let n = input.valid_indices.len();
    let n_params = r.aliased.iter().filter(|&&a| !a).count() + off;
    let df = n as f64 - n_params as f64;

    // Standard errors from the diagonal; NaN for aliased columns (their rows
    // in upstream's covariance are zero).
    let se = |k: usize| -> f64 {
        let d = v[(k, k)];
        if d >= 0.0 {
            d.sqrt()
        } else {
            f64::NAN
        }
    };
    let slope_se = Col::from_fn(p, |j| if r.aliased[j] { f64::NAN } else { se(off + j) });
    let t = CoefficientInference::t_statistics(&r.coefficients, &slope_se);
    let pv = CoefficientInference::p_values(&t, df);
    let (lo, hi) = CoefficientInference::confidence_intervals(
        &r.coefficients,
        &slope_se,
        df,
        input.confidence_level,
    );

    if let Some(b0) = r.intercept {
        let b = Col::from_fn(1, |_| b0);
        let s = Col::from_fn(1, |_| se(0));
        let ti = CoefficientInference::t_statistics(&b, &s);
        let pi = CoefficientInference::p_values(&ti, df);
        let (l, h) = CoefficientInference::confidence_intervals(&b, &s, df, input.confidence_level);
        summary.set_intercept_inference(Some(s[0]), Some(ti[0]), Some(pi[0]), Some((l[0], h[0])));
    }

    // Overall Wald F test of the estimated slopes with the clustered covariance.
    let beta = Col::from_fn(p + off, |k| {
        if k < off {
            r.intercept.unwrap_or(0.0)
        } else {
            r.coefficients[k - off]
        }
    });
    let slopes: Vec<usize> = (0..p).filter(|&j| !r.aliased[j]).map(|j| off + j).collect();
    let (f_statistic, f_pvalue) =
        wald_chunk(&beta, v, &slopes, WaldDistribution::F { df_residual: df })
            .ok()
            .and_then(|w| w.f.map(|f| (f, w.p_value)))
            .unwrap_or((f64::NAN, f64::NAN));

    let expand = |c: &Col<f64>| -> Vec<f64> {
        let mut full = vec![f64::NAN; input.n_features];
        for (k, &j) in input.kept.iter().enumerate() {
            full[j] = c[k];
        }
        full
    };
    let inference = FitResultInference {
        std_errors: expand(&slope_se),
        t_values: expand(&t),
        p_values: expand(&pv),
        ci_lower: expand(&lo),
        ci_upper: expand(&hi),
        confidence_level: input.confidence_level,
        f_statistic: Some(f_statistic),
        f_pvalue: Some(f_pvalue),
    };

    // Covariance in the full feature layout, zero for columns not in the design.
    let dim = input.n_features + off;
    let pos = |k: usize| {
        if k < off {
            k
        } else {
            off + input.kept[k - off]
        }
    };
    let mut full = vec![0.0; dim * dim];
    for a in 0..v.nrows() {
        for b in 0..v.ncols() {
            full[pos(a) * dim + pos(b)] = v[(a, b)];
        }
    }
    (inference, (full, dim))
}

/// OLS with a cluster-robust covariance (`sandwich::vcovCL(lm(...), cluster,
/// type = hc_type (HC1), cadjust)`). `options.compute_inference` decides whether
/// inference and the covariance are computed; `options.hc_type` picks the type.
pub fn fit_ols_cluster(
    y: &[f64],
    x: &[Vec<f64>],
    spec: &ClusterSpec,
    options: &OlsOptions,
) -> StatsResult<ClusterFit> {
    validate_spec(spec, y.len())?;
    crate::models::ols::fit_ols_impl(y, x, options, Some(spec))
}

/// WLS with a cluster-robust covariance (`sandwich::vcovCL(lm(..., weights), ...)`).
pub fn fit_wls_cluster(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    spec: &ClusterSpec,
    options: &WlsOptions,
) -> StatsResult<ClusterFit> {
    validate_spec(spec, y.len())?;
    crate::models::wls::fit_wls_impl(y, x, weights, options, Some(spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>, Vec<usize>) {
        let x: Vec<f64> = (0..12)
            .map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.1)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 1.0 + 2.0 * v + ((i * 7 % 5) as f64 - 2.0) * 0.3)
            .collect();
        let g: Vec<usize> = (0..12).map(|i| i / 3).collect();
        (y, vec![x], g)
    }

    #[test]
    fn singleton_clusters_hc0_without_adjust_is_white_hc0() {
        let (y, x, _) = data();
        let spec = ClusterSpec {
            codes: vec![(0..12).collect()],
            cadjust: false,
        };
        let opts = OlsOptions {
            compute_inference: true,
            hc_type: Some(HcType::HC0),
            ..Default::default()
        };
        let cl = fit_ols_cluster(&y, &x, &spec, &opts).unwrap();
        let hc = crate::models::fit_ols(&y, &x, &opts).unwrap();
        let a = cl.fit.inference.unwrap().std_errors[0];
        let b = hc.inference.unwrap().std_errors[0];
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        assert_eq!(cl.n_clusters, vec![12]);
        let (v, dim) = cl.vcov.unwrap();
        assert_eq!(dim, 2);
        assert!((v[3].sqrt() - a).abs() < 1e-12);
    }

    #[test]
    fn errors_and_no_inference() {
        let (y, x, g) = data();
        let one = ClusterSpec {
            codes: vec![vec![0; 12]],
            cadjust: true,
        };
        let opts = OlsOptions {
            compute_inference: true,
            ..Default::default()
        };
        // One cluster: the fit stands, its clustered inference is undefined.
        let single = fit_ols_cluster(&y, &x, &one, &opts).unwrap();
        assert!(single.vcov.is_none());
        assert_eq!(single.n_clusters, vec![1]);
        assert!(single.fit.inference.unwrap().std_errors[0].is_nan());
        assert!(single.fit.summary.intercept_std_error.is_nan());
        let empty = ClusterSpec {
            codes: vec![],
            cadjust: true,
        };
        assert!(fit_ols_cluster(&y, &x, &empty, &opts).is_err());
        let short = ClusterSpec {
            codes: vec![vec![0, 1]],
            cadjust: true,
        };
        assert!(fit_ols_cluster(&y, &x, &short, &opts).is_err());
        let spec = ClusterSpec {
            codes: vec![g],
            cadjust: true,
        };
        let plain = fit_ols_cluster(&y, &x, &spec, &OlsOptions::default()).unwrap();
        assert!(plain.fit.inference.is_none() && plain.vcov.is_none());
        let w = vec![1.0; 12];
        let wls = fit_wls_cluster(
            &y,
            &x,
            &w,
            &spec,
            &WlsOptions {
                compute_inference: true,
                ..Default::default()
            },
        )
        .unwrap();
        let ols = fit_ols_cluster(&y, &x, &spec, &opts).unwrap();
        let (a, b) = (
            wls.fit.inference.unwrap().std_errors[0],
            ols.fit.inference.unwrap().std_errors[0],
        );
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}
