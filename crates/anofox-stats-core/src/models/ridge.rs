//! Ridge Regression (L2 regularization) wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, RidgeOptions};
use faer::{Col, Mat};
use regress_rs::prelude::*;

/// Fit a Ridge regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features)
/// * `options` - Fitting options including alpha (L2 penalty)
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_ridge(y: &[f64], x: &[Vec<f64>], options: &RidgeOptions) -> StatsResult<FitResult> {
    // Validate alpha parameter
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
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
                && x.iter().all(|col| !col[i].is_nan() && !col[i].is_infinite())
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
    let fitted = RidgeRegressor::builder()
        .with_intercept(options.fit_intercept)
        .lambda(options.alpha)
        .confidence_level(options.confidence_level)
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

    // Build inference results if requested
    let inference = if options.compute_inference {
        let std_errors: Vec<f64> = result
            .std_errors
            .as_ref()
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();
        let t_values: Vec<f64> = result
            .t_statistics
            .as_ref()
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();
        let p_values: Vec<f64> = result
            .p_values
            .as_ref()
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();
        let ci_lower: Vec<f64> = result
            .conf_interval_lower
            .as_ref()
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();
        let ci_upper: Vec<f64> = result
            .conf_interval_upper
            .as_ref()
            .map(|c| c.iter().copied().collect())
            .unwrap_or_default();

        Some(FitResultInference {
            std_errors,
            t_values,
            p_values,
            ci_lower,
            ci_upper,
            confidence_level: options.confidence_level,
            f_statistic: Some(result.f_statistic),
            f_pvalue: Some(result.f_pvalue),
        })
    } else {
        None
    };

    Ok(FitResult {
        core,
        inference,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression() {
        // Simple linear relationship with some noise
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = RidgeOptions {
            alpha: 0.1, // Small regularization
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_ridge(&y, &x, &options).unwrap();

        // Ridge should shrink coefficients slightly compared to OLS
        // Coefficient should be close to 2, intercept close to 0
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
        assert!(result.core.r_squared > 0.95);
    }

    #[test]
    fn test_ridge_invalid_alpha() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = RidgeOptions {
            alpha: -1.0, // Invalid negative alpha
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_ridge(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidAlpha(_))));
    }

    #[test]
    fn test_ridge_high_regularization() {
        // High regularization should shrink coefficients more
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let low_reg = RidgeOptions {
            alpha: 0.001,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let high_reg = RidgeOptions {
            alpha: 10.0,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result_low = fit_ridge(&y, &x, &low_reg).unwrap();
        let result_high = fit_ridge(&y, &x, &high_reg).unwrap();

        // Higher regularization = smaller coefficient magnitude
        assert!(result_high.core.coefficients[0].abs() < result_low.core.coefficients[0].abs());
    }
}
