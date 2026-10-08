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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_stats_errors() {
        let err = fit_aft(&[], &[], &[], &AftOptions::default()).unwrap_err();
        assert!(matches!(err, StatsError::EmptyInput { field: "time" }));
    }
}
