//! Accelerated failure time (AFT) regression with right censoring.
//!
//! The model (Newton-Raphson on `(beta, log sigma)`, Laplace inference, Gaussian
//! priors) lives in [`anofox_regression::solvers::aft`]. This module only
//! forwards the call and maps the error type onto [`StatsError`].

use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::aft;

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
/// fitted AFT model given by its pieces (as the model STRUCT carries them), as R's
/// `psurvreg(t, eta, sigma, dist, lower.tail = FALSE)` with `eta = intercept + x'beta`
/// (NaN coefficients, i.e. aliased columns, are skipped).
///
/// `AftResult` is `#[non_exhaustive]` upstream and can't be rebuilt from stored
/// pieces, so this mirrors `AftResult::survival_curve` on the public
/// [`AftDistribution::log_survival`]: `S = 1` for `t <= 0`, `0` at `+Inf`, `NaN` for a
/// `NaN` time, a row of the wrong length or an invalid scale.
pub fn aft_survival_curve(
    coefficients: &[f64],
    intercept: Option<f64>,
    scale: f64,
    dist: AftDistribution,
    x_row: &[f64],
    times: &[f64],
) -> Vec<f64> {
    let eta = if x_row.len() == coefficients.len() {
        intercept.unwrap_or(0.0)
            + coefficients
                .iter()
                .zip(x_row)
                .filter(|(b, _)| !b.is_nan())
                .map(|(b, x)| b * x)
                .sum::<f64>()
    } else {
        f64::NAN
    };
    let valid = eta.is_finite() && scale.is_finite() && scale > 0.0;
    times
        .iter()
        .map(|&t| {
            if t.is_nan() || !valid {
                f64::NAN
            } else if t <= 0.0 {
                1.0
            } else if t == f64::INFINITY {
                0.0
            } else {
                dist.log_survival((t.ln() - eta) / scale).exp()
            }
        })
        .collect()
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
