//! Elastic Net Regression (combined L1+L2 regularization) wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{ElasticNetOptions, FitResult, FitResultCore};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Fit an Elastic Net regression model
///
/// Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization:
/// - L1 penalty encourages sparsity (feature selection)
/// - L2 penalty handles multicollinearity
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features)
/// * `options` - Fitting options including alpha (regularization strength) and l1_ratio
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and convergence info
pub fn fit_elasticnet(
    y: &[f64],
    x: &[Vec<f64>],
    options: &ElasticNetOptions,
) -> StatsResult<FitResult> {
    // Validate alpha parameter
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }

    // Validate l1_ratio
    if options.l1_ratio < 0.0 || options.l1_ratio > 1.0 {
        return Err(StatsError::InvalidL1Ratio(options.l1_ratio));
    }

    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    // Check all feature vectors have same length as y
    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

    // Check we have enough observations
    let min_obs = if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    };
    if n_obs <= min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_obs,
            cols: n_features,
        });
    }

    // Filter out rows with NaN values
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect();

    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();
    if n_valid <= min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    // Build and fit the model
    // Note: regress-rs uses different parameter names:
    // - lambda = regularization strength (our alpha)
    // - alpha = L1/L2 mix ratio (our l1_ratio)
    let fitted = ElasticNetRegressor::builder()
        .with_intercept(options.fit_intercept)
        .lambda(options.alpha) // our alpha is their lambda
        .alpha(options.l1_ratio) // our l1_ratio is their alpha
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract results
    let result = fitted.result();

    // Build core results
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    let core = FitResultCore {
        coefficients,
        intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: n_valid,
        n_features,
    };

    // Note: Elastic Net typically doesn't provide inference statistics
    // because the L1 penalty makes the standard errors non-standard
    Ok(FitResult {
        core,
        inference: None,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticnet_regression() {
        // Linear relationship with some noise
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = ElasticNetOptions {
            alpha: 0.1,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_elasticnet(&y, &x, &options).unwrap();

        // Should find coefficients close to true values
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
        assert!(result.core.r_squared > 0.90);
    }

    #[test]
    fn test_elasticnet_invalid_alpha() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = ElasticNetOptions {
            alpha: -1.0,
            ..Default::default()
        };

        let result = fit_elasticnet(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidAlpha(_))));
    }

    #[test]
    fn test_elasticnet_invalid_l1_ratio() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = ElasticNetOptions {
            l1_ratio: 1.5, // Invalid: > 1
            ..Default::default()
        };

        let result = fit_elasticnet(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidL1Ratio(_))));
    }

    #[test]
    fn test_elasticnet_lasso_like() {
        // l1_ratio = 1 is pure Lasso
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = ElasticNetOptions {
            alpha: 0.1,
            l1_ratio: 1.0, // Pure Lasso
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_elasticnet(&y, &x, &options).unwrap();
        assert!(result.core.r_squared > 0.90);
    }

    #[test]
    fn test_elasticnet_ridge_like() {
        // l1_ratio = 0 is pure Ridge
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = ElasticNetOptions {
            alpha: 0.1,
            l1_ratio: 0.0, // Pure Ridge
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_elasticnet(&y, &x, &options).unwrap();
        assert!(result.core.r_squared > 0.90);
    }
}
