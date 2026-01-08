//! Ordinary Least Squares (OLS) regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, OlsOptions};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

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

    // Detect zero-variance (constant) columns BEFORE min_obs check
    // A column is constant if all values in valid rows are the same
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
    // or return error if fit_intercept=false and no features
    if n_effective_features == 0 {
        if !options.fit_intercept {
            return Err(StatsError::InsufficientData {
                rows: n_valid,
                cols: n_features,
            });
        }
        // Intercept-only model: compute mean of y as intercept
        let y_mean = valid_indices.iter().map(|&i| y[i]).sum::<f64>() / n_valid as f64;
        let y_var = valid_indices
            .iter()
            .map(|&i| (y[i] - y_mean).powi(2))
            .sum::<f64>()
            / (n_valid - 1) as f64;
        let rmse = y_var.sqrt();

        return Ok(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features], // All NaN for constant columns
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

    // Allow n == min_obs (fitting with 0 degrees of freedom for residuals)
    // This enables exact fits and models with many zero-variance features
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

    // Convert to faer types (only non-constant columns)
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });

    // Build and fit the model
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
        // With fit_intercept=true and 3 non-constant features, we need n >= 4
        // (1 intercept + 3 coefficients = 4 parameters)
        // With n=2, we can't fit 4 parameters
        let x = vec![
            vec![1.0, 2.0], // 2 observations, varying - not constant
            vec![3.0, 4.0], // varying - not constant
            vec![5.0, 6.0], // varying - not constant
        ]; // 3 features
        let y = vec![1.0, 2.0];

        let options = OlsOptions {
            fit_intercept: true, // Needs n >= p + 1 = 4
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InsufficientData { .. })));
    }

    #[test]
    fn test_ols_exact_fit() {
        // Test that n == p+1 (exact fit, 0 degrees of freedom) is now allowed
        let x = vec![vec![1.0, 2.0]]; // 2 observations
        let y = vec![1.0, 3.0]; // y = 2*x - 1

        let options = OlsOptions {
            fit_intercept: true, // 2 parameters (intercept + 1 coef)
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options);

        // Should succeed (exact fit is now allowed)
        assert!(result.is_ok());
        let result = result.unwrap();
        // With 2 points, coefficient should be (3-1)/(2-1) = 2
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.01);
        // Intercept should be 1 - 2*1 = -1
        assert!((result.core.intercept.unwrap() - (-1.0)).abs() < 0.01);
    }
}
