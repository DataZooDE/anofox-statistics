//! Ordinary Least Squares (OLS) regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, OlsOptions};
use faer::{Col, Mat};
use regress_rs::prelude::*;

/// Fit an OLS regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features)
/// * `options` - Fitting options
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_ols(y: &[f64], x: &[Vec<f64>], options: &OlsOptions) -> StatsResult<FitResult> {
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
    let fitted = OlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .confidence_level(options.confidence_level)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract results
    let result = fitted.result();

    // Build core results - use field access, not method calls
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
        residual_std_error: result.rmse, // rmse is the residual standard error
        n_observations: n_valid,
        n_features,
    };

    // Build inference results if requested
    let inference = if options.compute_inference {
        // These fields are Option<Col<f64>>, need to handle None case
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
        diagnostics: None, // TODO: implement diagnostics
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ols() {
        // Simple linear relationship: y = 2*x + 1
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_ols(&y, &x, &options).unwrap();

        // Check coefficient is approximately 2
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.01);
        // Check intercept is approximately 1
        assert!((result.core.intercept.unwrap() - 1.0).abs() < 0.01);
        // R-squared should be very high (perfect fit)
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ols_with_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
        };

        let result = fit_ols(&y, &x, &options).unwrap();

        // Should have inference results
        assert!(result.inference.is_some());
        let inference = result.inference.unwrap();

        // p-value should be significant (< 0.05)
        assert!(inference.p_values[0] < 0.05);
    }

    #[test]
    fn test_ols_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0]; // Wrong length

        let options = OlsOptions::default();
        let result = fit_ols(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ols_insufficient_data() {
        let x = vec![vec![1.0, 2.0]]; // Only 2 observations
        let y = vec![1.0, 2.0];

        let options = OlsOptions {
            fit_intercept: true, // Needs n > p + 1 = 2
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InsufficientData { .. })));
    }
}
