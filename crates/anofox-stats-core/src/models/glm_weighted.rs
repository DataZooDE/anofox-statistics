//! Prior (case) weights for the Poisson, Tweedie and Gamma GLMs (#305).
//!
//! The unweighted fits go through `anofox_regression::solvers::penalized_glm`
//! (see [`super::glm`]), whose options carry no weights. A weighted fit is
//! `anofox_regression::solvers::{PoissonRegressor, TweedieRegressor}` with
//! `weights(...)` (upstream #308, R's `glm(weights = )`); this module only
//! marshals the rows and reshapes the fitted model into the fields the
//! `*_fit_agg` result carries. The unweighted path is untouched.
//!
//! The penalized engine's extras (an L2 `lambda`, explicit priors, the
//! covariance type) have no weighted counterpart and are rejected.

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    FitResultInference, GammaOptions, ModelSummary, PoissonOptions, TweedieOptions,
};
use anofox_regression::core::RegressionResult;
use anofox_regression::solvers::penalized_glm::{GlmPriorOptions, PenalizedGlmFamily};
use anofox_regression::solvers::{FittedRegressor, PoissonRegressor, Regressor, TweedieRegressor};
use faer::{Col, Mat};

/// A weighted GLM fit, in the shape of the unweighted [`super::GlmResult`]
/// fields the SQL result reports.
#[derive(Debug, Clone)]
pub struct WeightedGlmFit {
    /// Feature coefficients (offset column excluded; NaN where aliased).
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    /// `1 - deviance / null_deviance`.
    pub pseudo_r_squared: f64,
    pub aic: f64,
    pub dispersion: f64,
    pub n_observations: usize,
    pub n_features: usize,
    pub iterations: usize,
    pub converged: bool,
    pub inference: Option<FitResultInference>,
    pub summary: ModelSummary,
    /// Coefficient covariance, row-major, intercept first; `None` without
    /// inference.
    pub vcov: Option<(Vec<f64>, usize)>,
}

/// The options a weighted fit honours, common to the three families.
struct Common {
    fit_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
    offset_column: Option<usize>,
}

fn reject_penalty(lambda: f64, prior_opts: &GlmPriorOptions) -> StatsResult<()> {
    if lambda != 0.0 || !prior_opts.is_empty() {
        return Err(StatsError::InvalidInput(
            "prior weights cannot be combined with a penalty ('glm_lambda' or 'prior')".to_string(),
        ));
    }
    Ok(())
}

/// The design (features without the offset column), the response, the
/// weights and the offset over the rows where all of them are finite.
struct Rows {
    x: Mat<f64>,
    y: Col<f64>,
    w: Col<f64>,
    offset: Option<Col<f64>>,
    n_features: usize,
}

fn rows(y: &[f64], x: &[Vec<f64>], w: &[f64], offset_column: Option<usize>) -> StatsResult<Rows> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    crate::validation::validate_x_columns(y.len(), x)?;
    if w.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: y.len(),
            x_rows: w.len(),
        });
    }
    let offset = match offset_column {
        Some(c) if c == 0 || c > x.len() => {
            return Err(StatsError::InvalidValue {
                field: "offset",
                message: format!(
                    "offset must be a 1-based index into x (1..={}), got {c}",
                    x.len()
                ),
            })
        }
        Some(c) => Some(c - 1),
        None => None,
    };
    let features: Vec<usize> = (0..x.len()).filter(|&j| Some(j) != offset).collect();
    let used: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && w[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if used.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n = used.len();
    Ok(Rows {
        x: Mat::from_fn(n, features.len(), |i, j| x[features[j]][used[i]]),
        y: Col::from_fn(n, |i| y[used[i]]),
        w: Col::from_fn(n, |i| w[used[i]]),
        offset: offset.map(|o| Col::from_fn(n, |i| x[o][used[i]])),
        n_features: features.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    family: PenalizedGlmFamily,
    result: &RegressionResult,
    vcov: Option<Mat<f64>>,
    common: &Common,
    n_features: usize,
    deviance: f64,
    null_deviance: f64,
    dispersion: f64,
    iterations: usize,
    converged: bool,
) -> WeightedGlmFit {
    let vcov = vcov.map(|m| {
        let dim = m.nrows();
        let mut out = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                out[i * dim + j] = m[(i, j)];
            }
        }
        (out, dim)
    });
    WeightedGlmFit {
        coefficients: result.coefficients.iter().copied().collect(),
        intercept: if common.fit_intercept {
            result.intercept
        } else {
            None
        },
        deviance,
        null_deviance,
        pseudo_r_squared: result.r_squared,
        aic: result.aic,
        dispersion,
        n_observations: result.n_observations,
        n_features,
        iterations,
        converged,
        // Wald z / t tests; a GLM has no overall F test (NULL, as unweighted).
        inference: common.compute_inference.then(|| FitResultInference {
            f_statistic: None,
            f_pvalue: None,
            ..super::inference_from_result(result, n_features, common.confidence_level)
        }),
        summary: ModelSummary::from_result(family.model_info(), result, common.compute_inference),
        vcov,
    }
}

/// Weighted Poisson regression (`glm(y ~ x, family = poisson(link), weights = w)`).
pub fn fit_poisson_weighted(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[f64],
    options: &PoissonOptions,
) -> StatsResult<WeightedGlmFit> {
    reject_penalty(options.lambda, &options.prior_opts)?;
    let common = Common {
        fit_intercept: options.with_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        offset_column: options.offset_column,
    };
    let r = rows(y, x, w, common.offset_column)?;
    let mut builder = PoissonRegressor::builder()
        .link(options.link)
        .with_intercept(common.fit_intercept)
        .max_iterations(common.max_iterations as usize)
        .tolerance(common.tolerance)
        .compute_inference(common.compute_inference)
        .confidence_level(common.confidence_level)
        .estimate_dispersion(options.estimate_dispersion)
        .error_on_non_convergence(false)
        .weights(r.w);
    if let Some(o) = r.offset {
        builder = builder.offset(o);
    }
    let fitted = builder.build().fit(&r.x, &r.y).map_err(StatsError::from)?;
    Ok(assemble(
        PenalizedGlmFamily::Poisson(options.link),
        fitted.result(),
        fitted.vcov(),
        &common,
        r.n_features,
        fitted.deviance,
        fitted.null_deviance,
        fitted.dispersion,
        fitted.iterations,
        fitted.converged,
    ))
}

/// Weighted Tweedie (log link) or Gamma (`power = 2`) regression.
fn fit_tweedie_family_weighted(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[f64],
    power: f64,
    family: PenalizedGlmFamily,
    common: Common,
) -> StatsResult<WeightedGlmFit> {
    let r = rows(y, x, w, common.offset_column)?;
    let mut builder = TweedieRegressor::builder()
        .var_power(power)
        .link_power(0.0)
        .with_intercept(common.fit_intercept)
        .max_iterations(common.max_iterations as usize)
        .tolerance(common.tolerance)
        .compute_inference(common.compute_inference)
        .confidence_level(common.confidence_level)
        .error_on_non_convergence(false)
        .weights(r.w);
    if let Some(o) = r.offset {
        builder = builder.offset(o);
    }
    let fitted = builder.build().fit(&r.x, &r.y).map_err(StatsError::from)?;
    Ok(assemble(
        family,
        fitted.result(),
        fitted.vcov(),
        &common,
        r.n_features,
        fitted.deviance,
        fitted.null_deviance,
        fitted.dispersion,
        fitted.iterations,
        fitted.converged,
    ))
}

/// Weighted Tweedie regression with log link (`statmod::tweedie(var.power,
/// link.power = 0)` with `weights = w`).
pub fn fit_tweedie_weighted(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[f64],
    options: &TweedieOptions,
) -> StatsResult<WeightedGlmFit> {
    reject_penalty(options.lambda, &options.prior_opts)?;
    if !(1.0..=2.0).contains(&options.power) {
        return Err(StatsError::InvalidValue {
            field: "power",
            message: "Tweedie power parameter must be in [1, 2]".to_string(),
        });
    }
    if y.iter().any(|v| v.is_finite() && *v < 0.0) {
        return Err(StatsError::InvalidValue {
            field: "y",
            message: "Tweedie regression requires non-negative response values".to_string(),
        });
    }
    fit_tweedie_family_weighted(
        y,
        x,
        w,
        options.power,
        PenalizedGlmFamily::Tweedie {
            power: options.power,
        },
        Common {
            fit_intercept: options.with_intercept,
            max_iterations: options.max_iterations,
            tolerance: options.tolerance,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            offset_column: options.offset_column,
        },
    )
}

/// Weighted Gamma regression with log link (`glm(family = Gamma("log"),
/// weights = w)`).
pub fn fit_gamma_weighted(
    y: &[f64],
    x: &[Vec<f64>],
    w: &[f64],
    options: &GammaOptions,
) -> StatsResult<WeightedGlmFit> {
    reject_penalty(options.lambda, &options.prior_opts)?;
    if y.iter().any(|v| v.is_finite() && *v <= 0.0) {
        return Err(StatsError::InvalidValue {
            field: "y",
            message: "Gamma regression requires strictly positive response values".to_string(),
        });
    }
    fit_tweedie_family_weighted(
        y,
        x,
        w,
        2.0,
        PenalizedGlmFamily::Gamma,
        Common {
            fit_intercept: options.with_intercept,
            max_iterations: options.max_iterations,
            tolerance: options.tolerance,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            offset_column: options.offset_column,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let x: Vec<f64> = (0..20).map(|i| i as f64 / 4.0).collect();
        let y: Vec<f64> = (0..20).map(|i| ((i * 7) % 5 + i / 4) as f64).collect();
        (y, vec![x])
    }

    #[test]
    fn unit_weights_match_the_unweighted_engine_fit() {
        let (y, x) = data();
        let w = vec![1.0; y.len()];
        let opts = PoissonOptions::default().compute_inference(true);
        let weighted = fit_poisson_weighted(&y, &x, &w, &opts).unwrap();
        let plain = super::super::glm::fit_poisson(&y, &x, &opts).unwrap();
        assert!((weighted.coefficients[0] - plain.core.coefficients[0]).abs() < 1e-8);
        assert!((weighted.deviance - plain.core.residual_deviance).abs() < 1e-8);
        assert!((weighted.aic - plain.core.aic).abs() < 1e-6);
    }

    #[test]
    fn integer_weights_equal_replicated_rows() {
        let (y, x) = data();
        let w: Vec<f64> = (0..y.len()).map(|i| (1 + i % 3) as f64).collect();
        let mut yr = Vec::new();
        let mut xr = Vec::new();
        for i in 0..y.len() {
            for _ in 0..(w[i] as usize) {
                yr.push(y[i] + 1.0);
                xr.push(x[0][i]);
            }
        }
        let yp: Vec<f64> = y.iter().map(|v| v + 1.0).collect();
        let weighted = fit_gamma_weighted(&yp, &x, &w, &GammaOptions::default()).unwrap();
        let replicated =
            super::super::glm::fit_gamma(&yr, &[xr], &GammaOptions::default()).unwrap();
        assert!((weighted.coefficients[0] - replicated.core.coefficients[0]).abs() < 1e-7);
    }

    #[test]
    fn penalties_are_rejected() {
        let (y, x) = data();
        let w = vec![1.0; y.len()];
        let opts = PoissonOptions::default().lambda(0.5);
        assert!(fit_poisson_weighted(&y, &x, &w, &opts).is_err());
    }
}
