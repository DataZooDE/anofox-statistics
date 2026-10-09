//! Recursive Least Squares (RLS) wrapper.
//!
//! Wraps `anofox_regression::solvers::RlsRegressor`: filters non-finite rows,
//! drops constant columns (reported as NaN, as the other linear models do) and
//! reshapes the upstream result into the workspace-local `FitResult`.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore};
use anofox_regression::solvers::{FittedRegressor, Regressor, RlsRegressor};
use faer::{Col, Mat};

/// Options for RLS fitting.
#[derive(Debug, Clone)]
pub struct RlsOptions {
    /// Forgetting factor (λ) in (0, 1]; 1.0 = no forgetting (standard RLS),
    /// < 1.0 = exponential forgetting of old data.
    pub forgetting_factor: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Initial value for diagonal of P matrix (controls initial uncertainty)
    pub initial_p_diagonal: f64,
}

impl Default for RlsOptions {
    fn default() -> Self {
        Self {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0,
        }
    }
}

/// Fit RLS on a batch of data (observations processed in order).
///
/// R², adjusted R² and the residual standard error describe the final
/// coefficients on the rows used to fit them (upstream `RlsRegressor`
/// statistics).
pub fn fit_rls(y: &[f64], x: &[Vec<f64>], options: &RlsOptions) -> StatsResult<FitResult> {
    let n_features = x.len();
    if n_features == 0 {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    let n_obs = y.len();
    if n_obs == 0 {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    crate::validation::validate_x_columns(n_obs, x)?;
    if !(options.forgetting_factor > 0.0 && options.forgetting_factor <= 1.0) {
        return Err(StatsError::InvalidInput(
            "forgetting_factor must be in (0, 1]".into(),
        ));
    }
    if !(options.initial_p_diagonal.is_finite() && options.initial_p_diagonal > 0.0) {
        return Err(StatsError::InvalidInput(
            "initial_p_diagonal must be > 0".into(),
        ));
    }

    let valid: Vec<usize> = (0..n_obs)
        .filter(|&i| y[i].is_finite() && x.iter().all(|col| col[i].is_finite()))
        .collect();
    if valid.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let dropped = crate::validation::droppable_columns(x, &valid, options.fit_intercept);
    let kept: Vec<usize> = (0..n_features).filter(|&j| !dropped[j]).collect();
    if kept.is_empty() && !options.fit_intercept {
        return Err(StatsError::InsufficientData {
            rows: valid.len(),
            cols: n_features,
        });
    }

    let y_col = Col::from_fn(valid.len(), |i| y[valid[i]]);
    let x_mat = Mat::from_fn(valid.len(), kept.len(), |i, j| x[kept[j]][valid[i]]);
    let fitted = RlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .forgetting_factor(options.forgetting_factor)
        .initial_p_diagonal(options.initial_p_diagonal)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let result = fitted.result();

    let mut coefficients = vec![f64::NAN; n_features];
    for (r, &j) in kept.iter().enumerate() {
        coefficients[j] = result.coefficients[r];
    }
    Ok(FitResult {
        core: FitResultCore {
            coefficients,
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            residual_std_error: result.rmse,
            n_observations: valid.len(),
            n_features,
        },
        inference: None,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predict(r: &FitResult, x: &[f64]) -> f64 {
        r.core.intercept.unwrap_or(0.0)
            + r.core
                .coefficients
                .iter()
                .zip(x)
                .filter(|(c, _)| c.is_finite())
                .map(|(c, v)| c * v)
                .sum::<f64>()
    }

    #[test]
    fn test_rls_batch_fit() {
        let y: Vec<f64> = (1..=20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let x = vec![(1..=20).map(|i| i as f64).collect()];
        let r = fit_rls(&y, &x, &RlsOptions::default()).unwrap();
        assert_eq!(r.core.n_observations, 20);
        assert!((predict(&r, &[25.0]) - 51.0).abs() < 1.0);
    }

    #[test]
    fn test_rls_no_intercept_and_multivariate() {
        let opts = RlsOptions {
            fit_intercept: false,
            initial_p_diagonal: 1000.0,
            ..RlsOptions::default()
        };
        let r = fit_rls(&[3.0, 6.0, 9.0], &[vec![1.0, 2.0, 3.0]], &opts).unwrap();
        assert!(r.core.intercept.is_none());
        assert!((r.core.coefficients[0] - 3.0).abs() < 0.1);

        let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=20).map(|i| ((i * 7) % 11) as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| x1[i] + 2.0 * x2[i] + 0.5).collect();
        let r = fit_rls(&y, &[x1, x2], &RlsOptions::default()).unwrap();
        assert!((predict(&r, &[5.0, 10.0]) - 25.5).abs() < 2.0);
    }

    #[test]
    fn test_rls_invalid_options() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = [1.0, 2.0, 3.0];
        for (ff, p0) in [(0.0, 1.0), (1.5, 1.0), (1.0, 0.0)] {
            let o = RlsOptions {
                forgetting_factor: ff,
                initial_p_diagonal: p0,
                ..RlsOptions::default()
            };
            assert!(matches!(
                fit_rls(&y, &x, &o),
                Err(StatsError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn test_fit_rls_mismatched_columns_errors() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0]];
        let r = fit_rls(&y, &x, &RlsOptions::default());
        assert!(matches!(r, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_fit_rls_no_intercept_keeps_ones_column() {
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let y: Vec<f64> = x1.iter().map(|v| 2.0 * v + 1.0).collect();
        let opts = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: false,
            initial_p_diagonal: 1e6,
        };
        let r = fit_rls(&y, &[vec![1.0; 30], x1], &opts).unwrap();
        let c = &r.core.coefficients;
        assert!((c[0] - 1.0).abs() < 1e-2, "ones-column coef {}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-2);
    }

    /// With no forgetting and a diffuse prior, RLS is OLS: coefficients and the
    /// in-sample summary must match the batch least-squares solution
    /// (reference values from numpy.linalg.lstsq).
    #[test]
    fn test_rls_converges_to_ols_with_summary() {
        let x1: Vec<f64> = (0..40).map(|i| ((i * 37) % 17) as f64 / 3.0).collect();
        let x2: Vec<f64> = (0..40).map(|i| ((i * 11) % 13) as f64 * 0.5).collect();
        let y: Vec<f64> = (0..40)
            .map(|i| 1.5 + 2.0 * x1[i] - 0.7 * x2[i] + (((i * 7) % 5) as f64 - 2.0) * 0.3)
            .collect();
        let x = vec![x1, x2];
        let opts = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 1e8,
        };
        let r = fit_rls(&y, &x, &opts).unwrap();
        let c = &r.core.coefficients;
        assert!((r.core.intercept.unwrap() - 1.50559658).abs() < 1e-5);
        assert!((c[0] - 1.9836767).abs() < 1e-5, "slope {}", c[0]);
        assert!((c[1] - -0.68717123).abs() < 1e-5);
        assert!((r.core.r_squared - 0.9846591023871762).abs() < 1e-8);
        assert!((r.core.adj_r_squared - 0.983829864678375).abs() < 1e-8);
        assert!((r.core.residual_std_error - 0.4396778687186898).abs() < 1e-7);
    }
}
