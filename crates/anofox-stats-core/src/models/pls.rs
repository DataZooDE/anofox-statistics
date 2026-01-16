//! Partial Least Squares (PLS) regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{PlsFitResult, PlsOptions};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Fit a PLS regression model
///
/// PLS regression finds latent components that maximize the covariance between
/// X scores and y, useful for high-dimensional data and multicollinearity.
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
///
/// # Returns
/// * `PlsFitResult` containing coefficients, R-squared, and component info
pub fn fit_pls(y: &[f64], x: &[Vec<f64>], options: &PlsOptions) -> StatsResult<PlsFitResult> {
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

    // Validate n_components
    if options.n_components == 0 {
        return Err(StatsError::InvalidValue {
            field: "n_components",
            message: "n_components must be at least 1".to_string(),
        });
    }
    if options.n_components > n_features {
        return Err(StatsError::InvalidValue {
            field: "n_components",
            message: format!(
                "n_components ({}) cannot exceed number of features ({})",
                options.n_components, n_features
            ),
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

    // Need at least n_components + 1 observations
    let min_obs = options.n_components + 1;
    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    // Build and fit the PLS model
    let fitted = PlsRegressor::new(options.n_components)
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

    Ok(PlsFitResult {
        coefficients,
        intercept,
        r_squared: result.r_squared,
        n_components: options.n_components,
        n_observations: n_valid,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pls_basic() {
        // Simple data with 2 features
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 13.0, 16.0, 18.0, 20.0],
        ];
        let y = vec![3.0, 6.0, 8.0, 12.0, 15.0, 18.0, 20.0, 24.0, 27.0, 30.0];

        let options = PlsOptions {
            n_components: 1,
            fit_intercept: true,
        };

        let result = fit_pls(&y, &x, &options);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.n_components, 1);
        assert!(result.r_squared >= 0.0);
    }

    #[test]
    fn test_pls_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let options = PlsOptions::default();
        let result = fit_pls(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::EmptyInput { .. })));
    }

    #[test]
    fn test_pls_invalid_n_components() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let options = PlsOptions {
            n_components: 5, // More than features
            fit_intercept: true,
        };

        let result = fit_pls(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }
}
