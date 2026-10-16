//! Bayesian Ridge and ARD regression wrappers (#61).
//!
//! Both are `anofox_regression::solvers::{BayesianRidge, ArdRegression}`
//! (scikit-learn's `BayesianRidge` / `ARDRegression`): empirical-Bayes linear
//! models whose noise precision `alpha` and weight precision(s) `lambda` are
//! estimated by evidence maximisation. Prediction intervals come from the
//! posterior predictive distribution.

use crate::errors::{StatsError, StatsResult};
use crate::types::ModelSummary;
use anofox_regression::core::{HasModelInfo, IntervalType};
use anofox_regression::prelude::*;
use anofox_regression::solvers::{ArdRegression, BayesianRidge};
use faer::{Col, Mat};

/// Options shared by Bayesian Ridge and ARD (scikit-learn's defaults).
#[derive(Debug, Clone, Copy)]
pub struct BayesianOptions {
    pub fit_intercept: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    /// Gamma-prior shape / rate of the noise precision.
    pub alpha_1: f64,
    pub alpha_2: f64,
    /// Gamma-prior shape / rate of the weight precision(s).
    pub lambda_1: f64,
    pub lambda_2: f64,
    /// ARD only: features whose precision reaches this value are pruned.
    pub threshold_lambda: f64,
}

impl Default for BayesianOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 300,
            tolerance: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 10_000.0,
        }
    }
}

/// A fitted Bayesian Ridge or ARD model.
#[derive(Debug, Clone)]
pub struct BayesianFitResult {
    /// Posterior mean of the coefficients (0 for features ARD pruned).
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub n_features: usize,
    /// Estimated noise precision (scikit-learn `alpha_`).
    pub noise_precision: f64,
    /// Bayesian Ridge: the weight precision (`lambda_`); NaN for ARD.
    pub weight_precision: f64,
    /// ARD: the per-feature weight precisions (`lambda_`); empty for Bayesian Ridge.
    pub weight_precisions: Vec<f64>,
    /// Posterior standard deviation of each coefficient (0 for pruned features).
    pub posterior_sd: Vec<f64>,
    pub iterations: usize,
    /// Bayesian Ridge: log marginal likelihood (evidence); NaN for ARD.
    pub log_marginal_likelihood: f64,
    pub summary: ModelSummary,
}

/// Point predictions with posterior predictive bounds.
#[derive(Debug, Clone)]
pub struct BayesianPredictions {
    pub yhat: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

enum Fitted {
    Ridge(anofox_regression::solvers::FittedBayesianRidge),
    Ard(anofox_regression::solvers::FittedArd),
}

impl Fitted {
    fn regressor(&self) -> &dyn FittedRegressor {
        match self {
            Fitted::Ridge(f) => f,
            Fitted::Ard(f) => f,
        }
    }
}

fn validate_options(options: &BayesianOptions) -> StatsResult<()> {
    for (field, value) in [
        ("alpha_1", options.alpha_1),
        ("alpha_2", options.alpha_2),
        ("lambda_1", options.lambda_1),
        ("lambda_2", options.lambda_2),
    ] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(StatsError::InvalidValue {
                field,
                message: format!("{field} must be finite and >= 0, got {value}"),
            });
        }
    }
    if !(options.tolerance.is_finite() && options.tolerance > 0.0) {
        return Err(StatsError::InvalidValue {
            field: "tolerance",
            message: "tolerance must be > 0".to_string(),
        });
    }
    if options.max_iterations == 0 {
        return Err(StatsError::InvalidValue {
            field: "max_iterations",
            message: "max_iterations must be at least 1".to_string(),
        });
    }
    if options.threshold_lambda.is_nan() || options.threshold_lambda <= 0.0 {
        return Err(StatsError::InvalidValue {
            field: "threshold_lambda",
            message: "threshold_lambda must be > 0".to_string(),
        });
    }
    Ok(())
}

/// Complete rows of (y, x) as faer types; rows with a non-finite value are dropped.
fn design(y: &[f64], x: &[Vec<f64>]) -> StatsResult<(Col<f64>, Mat<f64>)> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    crate::validation::validate_x_columns(y.len(), x)?;
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.len() < 2 {
        return Err(StatsError::InsufficientData {
            rows: rows.len(),
            cols: x.len(),
        });
    }
    let y_col = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let x_mat = Mat::from_fn(rows.len(), x.len(), |i, j| x[j][rows[i]]);
    Ok((y_col, x_mat))
}

fn fit(y: &Col<f64>, x: &Mat<f64>, options: &BayesianOptions, ard: bool) -> StatsResult<Fitted> {
    validate_options(options)?;
    if ard {
        ArdRegression::builder()
            .fit_intercept(options.fit_intercept)
            .max_iter(options.max_iterations)
            .tolerance(options.tolerance)
            .alpha_1(options.alpha_1)
            .alpha_2(options.alpha_2)
            .lambda_1(options.lambda_1)
            .lambda_2(options.lambda_2)
            .threshold_lambda(options.threshold_lambda)
            .build()
            .fit(x, y)
            .map(Fitted::Ard)
            .map_err(StatsError::from)
    } else {
        BayesianRidge::builder()
            .fit_intercept(options.fit_intercept)
            .max_iter(options.max_iterations)
            .tolerance(options.tolerance)
            .alpha_1(options.alpha_1)
            .alpha_2(options.alpha_2)
            .lambda_1(options.lambda_1)
            .lambda_2(options.lambda_2)
            .build()
            .fit(x, y)
            .map(Fitted::Ridge)
            .map_err(StatsError::from)
    }
}

/// Fit Bayesian Ridge (`ard = false`) or ARD (`ard = true`) regression.
pub fn fit_bayesian(
    y: &[f64],
    x: &[Vec<f64>],
    options: &BayesianOptions,
    ard: bool,
) -> StatsResult<BayesianFitResult> {
    let (y_col, x_mat) = design(y, x)?;
    let fitted = fit(&y_col, &x_mat, options, ard)?;
    let result = fitted.regressor().result();
    let (
        noise_precision,
        weight_precision,
        weight_precisions,
        sigma_diag,
        iterations,
        evidence,
        info,
    ) = match &fitted {
        Fitted::Ridge(f) => (
            f.alpha(),
            f.lambda(),
            Vec::new(),
            f.sigma_diag(),
            f.n_iter(),
            f.log_marginal_likelihood(),
            f.model_info(),
        ),
        Fitted::Ard(f) => (
            f.alpha(),
            f64::NAN,
            f.lambdas().to_vec(),
            f.sigma_diag(),
            f.n_iter(),
            f64::NAN,
            f.model_info(),
        ),
    };
    Ok(BayesianFitResult {
        coefficients: result.coefficients.iter().copied().collect(),
        intercept: result.intercept,
        r_squared: result.r_squared,
        n_observations: result.n_observations,
        n_features: x.len(),
        noise_precision,
        weight_precision,
        weight_precisions,
        posterior_sd: sigma_diag.iter().map(|v| v.max(0.0).sqrt()).collect(),
        iterations,
        log_marginal_likelihood: evidence,
        summary: ModelSummary::from_result(info, result, false),
    })
}

/// Fit on the complete rows of (y, x) and predict `x_new` (row-major,
/// `x.len()` columns) with posterior predictive intervals at `level`. Rows of
/// `x_new` with a non-finite value predict NaN.
pub fn fit_predict_bayesian(
    y: &[f64],
    x: &[Vec<f64>],
    x_new: &[f64],
    options: &BayesianOptions,
    ard: bool,
    level: f64,
) -> StatsResult<BayesianPredictions> {
    if !(level > 0.0 && level < 1.0) {
        return Err(StatsError::InvalidValue {
            field: "confidence_level",
            message: format!("confidence_level must be in (0, 1), got {level}"),
        });
    }
    let (y_col, x_mat) = design(y, x)?;
    let fitted = fit(&y_col, &x_mat, options, ard)?;
    let p = x.len();
    let n_new = x_new.len().checked_div(p).unwrap_or(0);
    let complete: Vec<usize> = (0..n_new)
        .filter(|&i| x_new[i * p..(i + 1) * p].iter().all(|v| v.is_finite()))
        .collect();
    let mut out = BayesianPredictions {
        yhat: vec![f64::NAN; n_new],
        lower: vec![f64::NAN; n_new],
        upper: vec![f64::NAN; n_new],
    };
    if complete.is_empty() {
        return Ok(out);
    }
    let xq = Mat::from_fn(complete.len(), p, |i, j| x_new[complete[i] * p + j]);
    let pred = fitted
        .regressor()
        .predict_with_interval(&xq, Some(IntervalType::Prediction), level);
    for (k, &i) in complete.iter().enumerate() {
        out.yhat[i] = pred.fit[k];
        out.lower[i] = pred.lower[k];
        out.upper[i] = pred.upper[k];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let x1: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let x2: Vec<f64> = (0..40).map(|i| (i as f64 * 0.91).cos()).collect();
        let x3: Vec<f64> = (0..40).map(|i| ((i * 7 % 11) as f64) * 0.1).collect();
        let y = (0..40)
            .map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + 0.05 * (i as f64 * 1.3).sin())
            .collect();
        (y, vec![x1, x2, x3])
    }

    #[test]
    fn bayesian_ridge_recovers_coefficients() {
        let (y, x) = data();
        let r = fit_bayesian(&y, &x, &BayesianOptions::default(), false).unwrap();
        assert!((r.coefficients[0] - 2.0).abs() < 0.05);
        assert!((r.coefficients[1] + 0.5).abs() < 0.1);
        assert!(r.noise_precision > 0.0 && r.weight_precision > 0.0);
        assert!(r.weight_precisions.is_empty());
        assert_eq!(r.posterior_sd.len(), 3);
        assert!(r.log_marginal_likelihood.is_finite());
        assert!(r.summary.log_likelihood.is_nan());
    }

    #[test]
    fn ard_prunes_irrelevant_feature() {
        let (y, x) = data();
        let r = fit_bayesian(&y, &x, &BayesianOptions::default(), true).unwrap();
        assert_eq!(r.weight_precisions.len(), 3);
        assert!(r.weight_precision.is_nan());
        assert!(r.coefficients[2].abs() < 0.05);
    }

    #[test]
    fn fit_predict_gives_bands_and_nan_for_missing_rows() {
        let (y, x) = data();
        let x_new = vec![0.5, 0.2, 0.3, f64::NAN, 0.0, 0.0];
        let p =
            fit_predict_bayesian(&y, &x, &x_new, &BayesianOptions::default(), false, 0.9).unwrap();
        assert!(p.lower[0] < p.yhat[0] && p.yhat[0] < p.upper[0]);
        assert!(p.yhat[1].is_nan() && p.lower[1].is_nan());
    }

    #[test]
    fn invalid_options_are_rejected() {
        let (y, x) = data();
        let bad = BayesianOptions {
            alpha_1: -1.0,
            ..Default::default()
        };
        assert!(fit_bayesian(&y, &x, &bad, false).is_err());
        assert!(
            fit_predict_bayesian(&y, &x, &[], &BayesianOptions::default(), false, 1.5).is_err()
        );
    }
}
