//! Quantile regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{QuantileFitResult, QuantileOptions};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Fit a Quantile regression model
///
/// Quantile regression estimates conditional quantiles of the response variable,
/// rather than the conditional mean. It is robust to outliers and useful for
/// understanding the full distribution of the response.
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
///
/// # Returns
/// * `QuantileFitResult` containing coefficients for the specified quantile
pub fn fit_quantile(
    y: &[f64],
    x: &[Vec<f64>],
    options: &QuantileOptions,
) -> StatsResult<QuantileFitResult> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    // Validate tau
    if options.tau <= 0.0 || options.tau >= 1.0 {
        return Err(StatsError::InvalidValue {
            field: "tau",
            message: "tau must be in (0, 1)".to_string(),
        });
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

    // Check minimum observations
    let min_obs = if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    };

    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    // Build and fit the Quantile model
    let fitted = QuantileRegressor::new(options.tau)
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract results
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    Ok(QuantileFitResult {
        coefficients,
        intercept,
        tau: options.tau,
        n_observations: n_valid,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_median() {
        // Simple linear data
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let options = QuantileOptions {
            tau: 0.5, // Median
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_quantile(&y, &x, &options);
        assert!(result.is_ok());
        let result = result.unwrap();

        // Coefficient should be approximately 2 (y = 2*x)
        assert!((result.coefficients[0] - 2.0).abs() < 0.5);
        assert!((result.tau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_75th_percentile() {
        // Data with some spread
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        let options = QuantileOptions {
            tau: 0.75, // 75th percentile
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_quantile(&y, &x, &options);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!((result.tau - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let options = QuantileOptions::default();
        let result = fit_quantile(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::EmptyInput { .. })));
    }

    #[test]
    fn test_quantile_invalid_tau() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // tau = 0 is invalid
        let options = QuantileOptions {
            tau: 0.0,
            ..Default::default()
        };
        let result = fit_quantile(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));

        // tau = 1 is invalid
        let options = QuantileOptions {
            tau: 1.0,
            ..Default::default()
        };
        let result = fit_quantile(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn test_quantile_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0]; // Wrong length

        let options = QuantileOptions::default();
        let result = fit_quantile(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }
}
