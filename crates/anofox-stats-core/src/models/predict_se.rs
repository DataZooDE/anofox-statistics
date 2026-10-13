//! Coefficient covariance on the model STRUCT and predictions with standard
//! errors from it (#158).
//!
//! The model STRUCT carries the coefficient covariance `vcov` (row-major,
//! intercept first). It is computed upstream: for OLS / WLS / Ridge it is
//! `σ² · M` with the variance factor `M` of
//! `anofox_regression::inference::compute_variance_factor` (the matrix the
//! leverage-aware intervals use), for the GLMs it is the covariance of
//! `GlmInferenceResult::vcov` scaled by the fit's `DispersionRule`, i.e. the
//! covariance behind the reported standard errors.
//!
//! [`predict_se_linear`] evaluates `intervals_from_variance_factor` (R
//! `predict.lm(se.fit = TRUE, interval = ...)`); [`predict_se_glm`] rebuilds a
//! `GlmResult` around the covariance and calls its `predict_se` (R
//! `predict.glm(se.fit = TRUE)`, intervals built on the link scale).

use crate::errors::{StatsError, StatsResult};
use anofox_regression::core::{BinomialLink, IntervalType, ModelInfo, PoissonLink, PredictionType};
use anofox_regression::inference::{intervals_from_variance_factor, PredictSeOptions};
use anofox_regression::solvers::penalized_glm::{
    DispersionRule, GlmFitResult, GlmInferenceResult, GlmResult, PenalizedGlmFamily,
};
use faer::{Col, Mat};

/// Which interval [`predict_se_linear`] / [`predict_se_glm`] compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictInterval {
    /// Standard error only; the bounds are NaN.
    None,
    /// For the mean response.
    Confidence,
    /// For a new observation (linear models only).
    Prediction,
}

/// One prediction with its standard error and interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictSe {
    pub fit: f64,
    pub se_fit: f64,
    pub lower: f64,
    pub upper: f64,
}

/// `σ² · M` for OLS (no weights, `ridge_lambda = 0`), WLS (`weights`) and
/// Ridge (the effective `ridge_lambda`), from the training design; see
/// [`crate::models::interval_matrix`]. Row-major `dim x dim`, intercept first.
pub fn coefficient_vcov(
    x: &[Vec<f64>],
    coefficients: &[f64],
    fit_intercept: bool,
    weights: Option<&[f64]>,
    ridge_lambda: f64,
    residual_std_error: f64,
) -> StatsResult<(Vec<f64>, usize)> {
    let (m, dim) =
        crate::models::interval_matrix(x, coefficients, fit_intercept, weights, ridge_lambda)?;
    let s2 = residual_std_error * residual_std_error;
    Ok((m.into_iter().map(|v| v * s2).collect(), dim))
}

/// The covariance behind a GLM's standard errors, expanded to the feature
/// order (intercept first; rows / columns of dropped columns are 0), or
/// `None` without inference.
pub fn glm_vcov(fit: &GlmResult) -> Option<(Vec<f64>, usize)> {
    let inf = fit.inference.as_ref()?;
    let vcov = inf.vcov.as_ref()?;
    let off = usize::from(fit.core.intercept.is_some());
    let dim = fit.core.coefficients.len() + off;
    let scale = fit
        .dispersion_rule
        .covariance_scale(fit.core.dispersion.unwrap_or(1.0));
    let pos = |p: Option<usize>| match p {
        None => 0,
        Some(j) => j + off,
    };
    let mut out = vec![0.0; dim * dim];
    for (a, &pa) in inf.matrix_parameters.iter().enumerate() {
        for (b, &pb) in inf.matrix_parameters.iter().enumerate() {
            out[pos(pa) * dim + pos(pb)] = scale * vcov[(a, b)];
        }
    }
    Some((out, dim))
}

fn check_layout(
    coefficients: &[f64],
    x0: &[f64],
    has_intercept: bool,
    vcov: &[f64],
    dim: usize,
) -> StatsResult<()> {
    if coefficients.len() != x0.len() {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "the model has {} coefficients but x has {} values",
            coefficients.len(),
            x0.len()
        )));
    }
    let expected = x0.len() + usize::from(has_intercept);
    if dim != expected || dim.checked_mul(dim) != Some(vcov.len()) {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "vcov is {} values for dimension {dim}, expected {expected}x{expected}",
            vcov.len()
        )));
    }
    Ok(())
}

fn check_level(level: f64) -> StatsResult<()> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidValue {
            field: "level",
            message: format!("level must be in (0, 1), got {level}"),
        })
    }
}

/// Prediction for a linear model (OLS, WLS, Ridge) from its covariance:
/// `se_fit = sqrt(x0' V x0)`, a confidence interval `fit ± t_df · se_fit` and
/// a prediction interval `fit ± t_df · sqrt(se_fit² + σ²)`.
///
/// `intercept` is NaN without an intercept; NaN coefficients (aliased
/// columns) contribute nothing.
#[allow(clippy::too_many_arguments)]
pub fn predict_se_linear(
    coefficients: &[f64],
    intercept: f64,
    x0: &[f64],
    vcov: &[f64],
    dim: usize,
    residual_std_error: f64,
    df_residual: f64,
    interval: PredictInterval,
    level: f64,
) -> StatsResult<PredictSe> {
    check_level(level)?;
    let has_intercept = !intercept.is_nan();
    check_layout(coefficients, x0, has_intercept, vcov, dim)?;
    let p = x0.len();
    let fit = coefficients
        .iter()
        .zip(x0)
        .filter(|(c, _)| !c.is_nan())
        .fold(if has_intercept { intercept } else { 0.0 }, |a, (c, v)| {
            a + c * v
        });
    // Aliased columns have zero rows in V; zero their x so a NaN cannot leak in.
    let xm = Mat::from_fn(
        1,
        p,
        |_, j| if coefficients[j].is_nan() { 0.0 } else { x0[j] },
    );
    let v = Mat::from_fn(dim, dim, |a, b| vcov[a * dim + b]);
    let yhat = Col::from_fn(1, |_| fit);
    let conf = intervals_from_variance_factor(
        &xm,
        &v,
        &yhat,
        1.0,
        df_residual,
        level,
        IntervalType::Confidence,
        has_intercept,
    );
    let (lower, upper) = match interval {
        PredictInterval::None => (f64::NAN, f64::NAN),
        PredictInterval::Confidence => (conf.lower[0], conf.upper[0]),
        PredictInterval::Prediction => {
            let s2 = residual_std_error * residual_std_error;
            if s2 > 0.0 && s2.is_finite() {
                let m = Mat::from_fn(dim, dim, |a, b| vcov[a * dim + b] / s2);
                let pred = intervals_from_variance_factor(
                    &xm,
                    &m,
                    &yhat,
                    s2,
                    df_residual,
                    level,
                    IntervalType::Prediction,
                    has_intercept,
                );
                (pred.lower[0], pred.upper[0])
            } else {
                (f64::NAN, f64::NAN)
            }
        }
    };
    Ok(PredictSe {
        fit,
        se_fit: conf.se[0],
        lower,
        upper,
    })
}

/// The penalized-GLM family whose link is `link`; only the link enters
/// `GlmResult::predict_se`.
fn family_for_link(link: &str) -> StatsResult<PenalizedGlmFamily> {
    Ok(match link {
        "log" => PenalizedGlmFamily::Poisson(PoissonLink::Log),
        "identity" => PenalizedGlmFamily::Poisson(PoissonLink::Identity),
        "sqrt" => PenalizedGlmFamily::Poisson(PoissonLink::Sqrt),
        "logit" => PenalizedGlmFamily::Binomial(BinomialLink::Logit),
        "probit" => PenalizedGlmFamily::Binomial(BinomialLink::Probit),
        "cloglog" => PenalizedGlmFamily::Binomial(BinomialLink::Cloglog),
        other => {
            return Err(StatsError::InvalidValue {
                field: "link",
                message: format!("predict: unsupported link '{other}'"),
            })
        }
    })
}

/// Prediction for a GLM from its covariance, as `GlmResult::predict_se`:
/// the standard error on the link scale is `sqrt(x0' V x0)`, on the response
/// scale the delta-method value; intervals are `eta ± z · se` on the link
/// scale, mapped through the inverse link for the response scale. A
/// prediction interval is an error.
#[allow(clippy::too_many_arguments)]
pub fn predict_se_glm(
    coefficients: &[f64],
    intercept: f64,
    x0: &[f64],
    vcov: &[f64],
    dim: usize,
    link: &str,
    response_scale: bool,
    interval: PredictInterval,
    level: f64,
) -> StatsResult<PredictSe> {
    check_level(level)?;
    let has_intercept = !intercept.is_nan();
    check_layout(coefficients, x0, has_intercept, vcov, dim)?;
    if interval == PredictInterval::Prediction {
        return Err(StatsError::InvalidInput(
            "prediction intervals are only defined for linear models; use {'interval': 'confidence'}".into(),
        ));
    }
    let off = usize::from(has_intercept);
    // Fitted parameters in the order of the covariance: the intercept, then
    // the non-aliased features.
    let mut params: Vec<Option<usize>> = Vec::new();
    if has_intercept {
        params.push(None);
    }
    params.extend(
        (0..coefficients.len())
            .filter(|&j| !coefficients[j].is_nan())
            .map(Some),
    );
    let pos = |p: Option<usize>| p.map_or(0, |j| j + off);
    let compact = Mat::from_fn(params.len(), params.len(), |a, b| {
        vcov[pos(params[a]) * dim + pos(params[b])]
    });
    let family = family_for_link(link)?;
    let n_features = coefficients.len();
    let fit = GlmResult {
        core: GlmFitResult {
            coefficients: coefficients.to_vec(),
            intercept: has_intercept.then_some(intercept),
            null_deviance: f64::NAN,
            residual_deviance: f64::NAN,
            pseudo_r_squared: f64::NAN,
            aic: f64::NAN,
            n_observations: 0,
            n_features,
            iterations: 0,
            converged: true,
            dispersion: None,
        },
        inference: Some(GlmInferenceResult {
            std_errors: Vec::new(),
            z_values: Vec::new(),
            p_values: Vec::new(),
            ci_lower: Vec::new(),
            ci_upper: Vec::new(),
            confidence_level: level,
            intercept_std_error: None,
            intercept_z_value: None,
            intercept_p_value: None,
            intercept_ci_lower: None,
            intercept_ci_upper: None,
            vcov: Some(compact),
            information: None,
            matrix_parameters: params,
        }),
        log_likelihood: f64::NAN,
        bic: f64::NAN,
        fitted_values: Vec::new(),
        valid_rows: Vec::new(),
        model_info: ModelInfo::new("glm", None, "identity"),
        family,
        // `vcov` already carries the dispersion.
        dispersion_rule: DispersionRule::Fixed,
        offset_column: None,
        penalized: false,
    };
    let mut options = PredictSeOptions::new().level(level);
    options.scale = if response_scale {
        PredictionType::Response
    } else {
        PredictionType::Link
    };
    let x_cols: Vec<Vec<f64>> = x0.iter().map(|&v| vec![v]).collect();
    let r = fit
        .predict_se(&x_cols, &options)
        .map_err(|e| StatsError::InvalidInput(e.to_string()))?;
    let (lower, upper) = match interval {
        PredictInterval::None => (f64::NAN, f64::NAN),
        _ => (r.lower[0], r.upper[0]),
    };
    Ok(PredictSe {
        fit: r.fit[0],
        se_fit: r.se_fit[0],
        lower,
        upper,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_closed_form() {
        // y = 1 + 2x on x = 0..4 with residuals; V and sigma made up.
        let vcov = vec![0.5, -0.1, -0.1, 0.04];
        let r = predict_se_linear(
            &[2.0],
            1.0,
            &[3.0],
            &vcov,
            2,
            0.8,
            10.0,
            PredictInterval::Confidence,
            0.95,
        )
        .unwrap();
        let h: f64 = 0.5 - 2.0 * 0.1 * 3.0 + 0.04 * 9.0;
        assert!((r.fit - 7.0).abs() < 1e-12);
        assert!((r.se_fit - h.sqrt()).abs() < 1e-12);
        let t = 2.228138851986274; // qt(0.975, 10)
        assert!((r.upper - (7.0 + t * h.sqrt())).abs() < 1e-9);
        let p = predict_se_linear(
            &[2.0],
            1.0,
            &[3.0],
            &vcov,
            2,
            0.8,
            10.0,
            PredictInterval::Prediction,
            0.95,
        )
        .unwrap();
        assert!((p.upper - (7.0 + t * (h + 0.64).sqrt())).abs() < 1e-9);
        assert!((p.se_fit - h.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn glm_response_interval_is_mapped_link_interval() {
        let vcov = vec![0.04, 0.0, 0.0, 0.01];
        let link = predict_se_glm(
            &[0.5],
            0.2,
            &[2.0],
            &vcov,
            2,
            "log",
            false,
            PredictInterval::Confidence,
            0.95,
        )
        .unwrap();
        let resp = predict_se_glm(
            &[0.5],
            0.2,
            &[2.0],
            &vcov,
            2,
            "log",
            true,
            PredictInterval::Confidence,
            0.95,
        )
        .unwrap();
        assert!((link.fit - 1.2).abs() < 1e-12);
        assert!((link.se_fit - (0.04f64 + 0.04).sqrt()).abs() < 1e-12);
        assert!((resp.fit - 1.2f64.exp()).abs() < 1e-12);
        assert!((resp.lower - link.lower.exp()).abs() < 1e-12);
        assert!((resp.se_fit - link.se_fit * 1.2f64.exp()).abs() < 1e-12);
        assert!(predict_se_glm(
            &[0.5],
            0.2,
            &[2.0],
            &vcov,
            2,
            "log",
            true,
            PredictInterval::Prediction,
            0.95
        )
        .is_err());
    }
}
