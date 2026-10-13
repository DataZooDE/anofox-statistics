//! Accelerated failure time (AFT) regression with right censoring.
//!
//! The model (Newton-Raphson on `(beta, log sigma)`, Laplace inference, Gaussian
//! priors) lives in [`anofox_regression::solvers::aft`]. This module only
//! forwards the call and maps the error type onto [`StatsError`].

use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::aft;
use anofox_regression::ModelInfo;

pub use aft::{AftDistribution, AftFitResult, AftInference, AftOptions, AftResult};

/// Fit an AFT model.
///
/// * `time` — strictly positive event or censoring times.
/// * `x` — column-major features.
/// * `event` — 1.0 when the event was observed, 0.0 when the row is right-censored.
pub fn fit_aft(
    time: &[f64],
    x: &[Vec<f64>],
    event: &[f64],
    options: &AftOptions,
) -> StatsResult<AftResult> {
    aft::fit_aft(time, x, event, options).map_err(StatsError::from)
}

/// Model-based survival `S(t | x)` at each of `times` for one covariate row of a
/// fitted AFT model given by its pieces (as the model STRUCT carries them).
/// Delegates to [`AftResult::survival_curve`] (R `psurvreg(..., lower.tail = FALSE)`).
pub fn aft_survival_curve(
    coefficients: &[f64],
    intercept: Option<f64>,
    scale: f64,
    dist: AftDistribution,
    x_row: &[f64],
    times: &[f64],
) -> Vec<f64> {
    let result = AftResult {
        core: AftFitResult {
            coefficients: coefficients.to_vec(),
            intercept,
            scale,
            log_likelihood: f64::NAN,
            null_log_likelihood: f64::NAN,
            aic: f64::NAN,
            bic: f64::NAN,
            n_observations: 0,
            n_events: 0,
            n_censored: 0,
            n_features: coefficients.len(),
            iterations: 0,
            converged: true,
        },
        inference: None,
        model_info: ModelInfo::new("aft", Some(dist.as_str()), "log"),
    };
    result.survival_curve(x_row, times)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_stats_errors() {
        let err = fit_aft(&[], &[], &[], &AftOptions::default()).unwrap_err();
        assert!(matches!(err, StatsError::EmptyInput { field: "time" }));
    }

    #[test]
    fn survival_curve_is_one_at_zero_and_decreasing() {
        let s = aft_survival_curve(
            &[0.5],
            Some(1.0),
            0.8,
            AftDistribution::Weibull,
            &[1.0],
            &[0.0, 1.0, 5.0, 20.0],
        );
        assert_eq!(s[0], 1.0);
        assert!(s[1] > s[2] && s[2] > s[3] && s[3] > 0.0);
    }
}
