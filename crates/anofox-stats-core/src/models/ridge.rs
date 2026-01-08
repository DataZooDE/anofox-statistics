//! Ridge Regression (L2 regularization) wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, RidgeOptions};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

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

    // Convert to faer types (only non-constant columns)
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });

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
        residual_std_error: result.rmse,
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
    fn test_ridge_perfect_fit() {
        // Test with perfect fit data (y = 2*x exactly)
        // This is a regression test for the panic in statrs beta function
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let options = RidgeOptions {
            alpha: 0.1,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        let result = fit_ridge(&y, &x, &options).unwrap();

        // Should work without panicking even with perfect fit
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ridge_perfect_fit_minimal() {
        // Test with exact minimal data (same as DuckDB failing query)
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let options = RidgeOptions {
            alpha: 0.1,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        // Should not panic even with minimal data
        let result = fit_ridge(&y, &x, &options).unwrap();
        assert!(result.core.r_squared > 0.99);
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
