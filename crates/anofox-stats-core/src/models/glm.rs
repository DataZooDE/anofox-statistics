//! Generalized Linear Models (GLM) — Poisson, Binomial, Negative Binomial,
//! Tweedie, Gamma, Logistic.
//!
//! The penalized IRLS engine (priors, Laplace curvature, dispersion rules,
//! per-family log-likelihoods) lives in
//! [`anofox_regression::solvers::penalized_glm`]. This module is the interface
//! layer only: it forwards the options and maps the engine's error type onto
//! [`StatsError`], so the FFI error codes and the error-vs-NULL policy are
//! unchanged.

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    BinomialOptions, GammaOptions, LogisticOptions, NegBinomialOptions, PoissonOptions,
    TweedieOptions,
};
use anofox_regression::solvers::penalized_glm as pg;

pub use pg::{GlmResult, LogisticResult};

/// Fit a Poisson regression model (count response).
pub fn fit_poisson(y: &[f64], x: &[Vec<f64>], options: &PoissonOptions) -> StatsResult<GlmResult> {
    pg::fit_poisson(y, x, options).map_err(StatsError::from)
}

/// Fit a Binomial regression model (response in `[0, 1]`).
pub fn fit_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &BinomialOptions,
) -> StatsResult<GlmResult> {
    pg::fit_binomial(y, x, options).map_err(StatsError::from)
}

/// Fit a Negative Binomial regression model (overdispersed counts).
pub fn fit_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &NegBinomialOptions,
) -> StatsResult<GlmResult> {
    pg::fit_negbinomial(y, x, options).map_err(StatsError::from)
}

/// Fit a Tweedie regression model.
pub fn fit_tweedie(y: &[f64], x: &[Vec<f64>], options: &TweedieOptions) -> StatsResult<GlmResult> {
    pg::fit_tweedie(y, x, options).map_err(StatsError::from)
}

/// Fit a Gamma regression model (strictly positive response).
pub fn fit_gamma(y: &[f64], x: &[Vec<f64>], options: &GammaOptions) -> StatsResult<GlmResult> {
    pg::fit_gamma(y, x, options).map_err(StatsError::from)
}

/// Fit a logistic regression model (binary response).
pub fn fit_logistic(
    y: &[f64],
    x: &[Vec<f64>],
    options: &LogisticOptions,
) -> StatsResult<LogisticResult> {
    pg::fit_logistic(y, x, options).map_err(StatsError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_stats_errors() {
        let err = fit_poisson(
            &[-1.0, 2.0, 3.0],
            &[vec![1.0, 2.0, 3.0]],
            &Default::default(),
        )
        .unwrap_err();
        assert!(matches!(err, StatsError::InvalidValue { field: "y", .. }));

        let err = fit_gamma(&[1.0, 2.0], &[vec![1.0, 2.0, 3.0]], &Default::default()).unwrap_err();
        assert!(matches!(err, StatsError::DimensionMismatch { .. }));
    }
}
