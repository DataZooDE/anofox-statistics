//! Passive-Aggressive online regression wrapper (#63).
//!
//! `anofox_regression::solvers::PassiveAggressiveRegressor` (Crammer et al.
//! 2006; scikit-learn's `PassiveAggressiveRegressor`). The rows are processed
//! in input order, without shuffling: one pass is the online (streaming)
//! update, equal to scikit-learn's `partial_fit`; more passes repeat it with
//! scikit-learn's early stopping.

use crate::errors::{StatsError, StatsResult};
use crate::types::ModelSummary;
use anofox_regression::core::HasModelInfo;
use anofox_regression::prelude::*;
use anofox_regression::solvers::{PaLoss, PassiveAggressiveRegressor};
use faer::{Col, Mat};

/// Options of the Passive-Aggressive regressor.
#[derive(Debug, Clone, Copy)]
pub struct PassiveAggressiveOptions {
    pub fit_intercept: bool,
    /// Aggressiveness (regularisation) parameter C > 0.
    pub c: f64,
    /// Width of the epsilon-insensitive zone, >= 0.
    pub epsilon: f64,
    /// PA-II (squared epsilon-insensitive loss) instead of PA-I.
    pub squared_loss: bool,
    /// Passes over the rows; 1 = a single online pass.
    pub max_epochs: usize,
    /// Early-stopping tolerance of multi-pass fits.
    pub tolerance: f64,
}

impl Default for PassiveAggressiveOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            c: 1.0,
            epsilon: 0.1,
            squared_loss: false,
            max_epochs: 1,
            tolerance: 1e-3,
        }
    }
}

/// A fitted Passive-Aggressive model.
#[derive(Debug, Clone)]
pub struct PassiveAggressiveResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub n_features: usize,
    /// Passes over the rows actually run.
    pub epochs: usize,
    pub summary: ModelSummary,
}

/// Fit on the complete rows of (y, x) in input order.
pub fn fit_passive_aggressive(
    y: &[f64],
    x: &[Vec<f64>],
    options: &PassiveAggressiveOptions,
) -> StatsResult<PassiveAggressiveResult> {
    if !(options.c.is_finite() && options.c > 0.0) {
        return Err(StatsError::InvalidValue {
            field: "c",
            message: format!("C must be finite and > 0, got {}", options.c),
        });
    }
    if !(options.epsilon.is_finite() && options.epsilon >= 0.0) {
        return Err(StatsError::InvalidValue {
            field: "epsilon",
            message: format!("epsilon must be finite and >= 0, got {}", options.epsilon),
        });
    }
    if options.max_epochs == 0 {
        return Err(StatsError::InvalidValue {
            field: "max_iterations",
            message: "max_iterations (passes) must be at least 1".to_string(),
        });
    }
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
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let y_col = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let x_mat = Mat::from_fn(rows.len(), x.len(), |i, j| x[j][rows[i]]);

    let fitted = PassiveAggressiveRegressor::builder()
        .c(options.c)
        .epsilon(options.epsilon)
        .with_intercept(options.fit_intercept)
        .loss(if options.squared_loss {
            PaLoss::SquaredEpsilonInsensitive
        } else {
            PaLoss::EpsilonInsensitive
        })
        .max_iter(options.max_epochs)
        .tolerance(options.tolerance)
        .shuffle(false)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let result = fitted.result();
    Ok(PassiveAggressiveResult {
        coefficients: result.coefficients.iter().copied().collect(),
        intercept: result.intercept,
        r_squared: result.r_squared,
        n_observations: rows.len(),
        n_features: x.len(),
        epochs: fitted.n_iter(),
        summary: ModelSummary::from_result(fitted.model_info(), result, false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pass_matches_streaming_updates() {
        // One pass over the rows in order equals the per-row PA-I update.
        let x1: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = x1.iter().map(|v| 0.5 + 1.5 * v).collect();
        let opts = PassiveAggressiveOptions::default();
        let r = fit_passive_aggressive(&y, std::slice::from_ref(&x1), &opts).unwrap();
        let (mut w, mut b) = (0.0_f64, 0.0_f64);
        for i in 0..30 {
            let resid = y[i] - (b + w * x1[i]);
            let loss = (resid.abs() - opts.epsilon).max(0.0);
            if loss > 0.0 && x1[i] != 0.0 {
                let tau = (loss / (x1[i] * x1[i])).min(opts.c) * resid.signum();
                w += tau * x1[i];
                b += tau;
            }
        }
        assert!((r.coefficients[0] - w).abs() < 1e-12);
        assert!((r.intercept.unwrap() - b).abs() < 1e-12);
        assert_eq!(r.epochs, 1);
        assert!(r.summary.log_likelihood.is_nan());
    }

    #[test]
    fn more_passes_get_closer_and_options_are_checked() {
        let x1: Vec<f64> = (0..50).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();
        let y: Vec<f64> = x1.iter().map(|v| 1.0 + 2.0 * v).collect();
        let one = fit_passive_aggressive(
            &y,
            std::slice::from_ref(&x1),
            &PassiveAggressiveOptions::default(),
        )
        .unwrap();
        let many = fit_passive_aggressive(
            &y,
            std::slice::from_ref(&x1),
            &PassiveAggressiveOptions {
                max_epochs: 200,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((many.coefficients[0] - 2.0).abs() <= (one.coefficients[0] - 2.0).abs());
        assert!(many.epochs > 1);
        for bad in [
            PassiveAggressiveOptions {
                c: 0.0,
                ..Default::default()
            },
            PassiveAggressiveOptions {
                epsilon: -1.0,
                ..Default::default()
            },
            PassiveAggressiveOptions {
                max_epochs: 0,
                ..Default::default()
            },
        ] {
            assert!(fit_passive_aggressive(&y, std::slice::from_ref(&x1), &bad).is_err());
        }
    }
}
