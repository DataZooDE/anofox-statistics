//! Weighted Least Squares (WLS) regression
//!
//! WLS is implemented by transforming the data: multiply X and y by sqrt(weights),
//! then perform OLS on the transformed data.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, WlsOptions};
use faer::{Col, Mat};
use regress_rs::prelude::*;

/// Fit a Weighted Least Squares regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (p features, each with n observations)
/// * `weights` - Observation weights (n observations, must be positive)
/// * `options` - Fitting options
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_wls(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
) -> StatsResult<FitResult> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if weights.is_empty() {
        return Err(StatsError::EmptyInput { field: "weights" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    // Check weights length matches y
    if weights.len() != n_obs {
        return Err(StatsError::DimensionMismatch {
            y_len: n_obs,
            x_rows: weights.len(),
        });
    }

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

    // Filter out rows with NaN/Inf values or non-positive weights
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && weights[i] > 0.0
                && !weights[i].is_nan()
                && !weights[i].is_infinite()
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

    // WLS transformation: multiply X and y by sqrt(weights)
    // This makes OLS on transformed data equivalent to WLS
    let sqrt_weights: Vec<f64> = valid_indices.iter().map(|&i| weights[i].sqrt()).collect();

    // Transform y: y_tilde = sqrt(w) * y
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]] * sqrt_weights[i]);

    // Transform X: X_tilde = diag(sqrt(w)) * X
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| {
        x[j][valid_indices[i]] * sqrt_weights[i]
    });

    // Build and fit the model using OLS on transformed data
    let fitted = OlsRegressor::builder()
        .with_intercept(options.fit_intercept)
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

    // Note: R-squared from the transformed model is not the same as weighted R-squared
    // We compute proper weighted R-squared here
    let y_mean_weighted = {
        let sum_wy: f64 = valid_indices.iter().map(|&i| weights[i] * y[i]).sum();
        let sum_w: f64 = valid_indices.iter().map(|&i| weights[i]).sum();
        sum_wy / sum_w
    };

    // Compute weighted SS_tot and SS_res
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for &i in valid_indices.iter() {
        let y_pred = if let Some(intercept) = intercept {
            intercept
                + coefficients
                    .iter()
                    .zip(x.iter())
                    .map(|(c, col)| c * col[i])
                    .sum::<f64>()
        } else {
            coefficients
                .iter()
                .zip(x.iter())
                .map(|(c, col)| c * col[i])
                .sum::<f64>()
        };
        let w = weights[i];
        ss_tot += w * (y[i] - y_mean_weighted).powi(2);
        ss_res += w * (y[i] - y_pred).powi(2);
    }

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // Adjusted R-squared for weighted regression
    let adj_r_squared = if n_valid > min_obs && ss_tot > 0.0 {
        let p = if options.fit_intercept {
            n_features + 1
        } else {
            n_features
        };
        1.0 - (1.0 - r_squared) * ((n_valid - 1) as f64) / ((n_valid - p) as f64)
    } else {
        r_squared
    };

    // Weighted residual standard error
    let df_residual = n_valid
        - if options.fit_intercept {
            n_features + 1
        } else {
            n_features
        };
    let residual_std_error = if df_residual > 0 {
        (ss_res / df_residual as f64).sqrt()
    } else {
        0.0
    };

    let core = FitResultCore {
        coefficients,
        intercept,
        r_squared,
        adj_r_squared,
        residual_std_error,
        n_observations: n_valid,
        n_features,
    };

    // Build inference results if requested
    // Note: The inference from transformed OLS is approximately correct for WLS
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
    fn test_wls_uniform_weights() {
        // With uniform weights, WLS should give same results as OLS
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0; 10]; // Uniform weights

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should be close to y = 2*x
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_wls_heteroscedastic() {
        // Data with increasing variance - higher weights for more reliable observations
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        // y = 2*x + noise, where noise variance increases with x
        let y = vec![2.1, 4.0, 5.9, 8.3, 9.5, 12.5, 13.0, 17.0, 17.0, 22.0];
        // Higher weights for observations with lower variance (early observations)
        let weights = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Coefficient should still be close to 2
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
    }

    #[test]
    fn test_wls_zero_weights() {
        // Zero weights should be filtered out
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        // Last two observations have zero weight - should be ignored
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should still get good fit on remaining 8 observations
        assert!(result.core.r_squared > 0.99);
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_negative_weights_filtered() {
        // Negative weights should be filtered out (not valid)
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let weights = vec![1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // 8 valid observations (2 with negative weights filtered)
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0]; // Wrong length

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }
}
