//! Weighted Least Squares (WLS) regression
//!
//! WLS is implemented by transforming the data: multiply X and y by sqrt(weights),
//! then perform OLS on the transformed data.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, WlsOptions};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

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

    // Detect zero-variance (constant) columns BEFORE min_obs check
    let is_constant_column: Vec<bool> = x
        .iter()
        .map(|col| {
            if valid_indices.is_empty() {
                return true;
            }
            let first_val = col[valid_indices[0]];
            valid_indices
                .iter()
                .all(|&i| (col[i] - first_val).abs() < 1e-10)
        })
        .collect();

    // Count non-constant features for min_obs calculation
    let n_effective_features = is_constant_column.iter().filter(|&&c| !c).count();

    // Check we have enough observations for the effective (non-constant) features
    let min_obs = if options.fit_intercept {
        n_effective_features + 1
    } else {
        n_effective_features
    };

    // If ALL columns are constant, we can still fit (intercept-only model if fit_intercept=true)
    if n_effective_features == 0 {
        if !options.fit_intercept {
            return Err(StatsError::InsufficientData {
                rows: n_valid,
                cols: n_features,
            });
        }
        // Intercept-only model: compute weighted mean of y as intercept
        let sum_wy: f64 = valid_indices.iter().map(|&i| weights[i] * y[i]).sum();
        let sum_w: f64 = valid_indices.iter().map(|&i| weights[i]).sum();
        let y_mean = sum_wy / sum_w;
        let y_var = valid_indices
            .iter()
            .map(|&i| weights[i] * (y[i] - y_mean).powi(2))
            .sum::<f64>()
            / sum_w;
        let rmse = y_var.sqrt();

        return Ok(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features],
                intercept: Some(y_mean),
                r_squared: 0.0,
                adj_r_squared: 0.0,
                residual_std_error: rmse,
                n_observations: n_valid,
                n_features,
            },
            inference: None,
            diagnostics: None,
        });
    }

    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Build reduced X matrix (only non-constant columns)
    let non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    // WLS transformation: multiply X and y by sqrt(weights)
    // This makes OLS on transformed data equivalent to WLS
    let sqrt_weights: Vec<f64> = valid_indices.iter().map(|&i| weights[i].sqrt()).collect();

    // Transform y: y_tilde = sqrt(w) * y
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]] * sqrt_weights[i]);

    // Transform X: X_tilde = diag(sqrt(w)) * X (only non-constant columns)
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]] * sqrt_weights[i]
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

    // Reconstruct full coefficient vector with NaN for constant columns
    let reduced_coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let mut coefficients = vec![f64::NAN; n_features];
    for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
        coefficients[orig_idx] = reduced_coefficients[reduced_idx];
    }
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
    // Note: NaN coefficients contribute 0 to predictions (skip them)
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for &i in valid_indices.iter() {
        let y_pred = if let Some(intercept) = intercept {
            intercept
                + coefficients
                    .iter()
                    .zip(x.iter())
                    .filter(|(c, _)| !c.is_nan())
                    .map(|(c, col)| c * col[i])
                    .sum::<f64>()
        } else {
            coefficients
                .iter()
                .zip(x.iter())
                .filter(|(c, _)| !c.is_nan())
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

    // Adjusted R-squared for weighted regression (use effective features for df)
    let adj_r_squared = if n_valid > min_obs && ss_tot > 0.0 {
        let p = if options.fit_intercept {
            n_effective_features + 1
        } else {
            n_effective_features
        };
        1.0 - (1.0 - r_squared) * ((n_valid - 1) as f64) / ((n_valid - p) as f64)
    } else {
        r_squared
    };

    // Weighted residual standard error (use effective features for df)
    let df_residual = n_valid
        - if options.fit_intercept {
            n_effective_features + 1
        } else {
            n_effective_features
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
        // Helper to reconstruct reduced vector to full size with NaN for constant columns
        let reconstruct = |reduced: Option<&faer::Col<f64>>| -> Vec<f64> {
            let mut full = vec![f64::NAN; n_features];
            if let Some(col) = reduced {
                for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
                    full[orig_idx] = col[reduced_idx];
                }
            }
            full
        };

        let std_errors = reconstruct(result.std_errors.as_ref());
        let t_values = reconstruct(result.t_statistics.as_ref());
        let p_values = reconstruct(result.p_values.as_ref());
        let ci_lower = reconstruct(result.conf_interval_lower.as_ref());
        let ci_upper = reconstruct(result.conf_interval_upper.as_ref());

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
