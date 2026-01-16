//! Isotonic regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{IsotonicFitResult, IsotonicOptions};
use anofox_regression::prelude::*;
use faer::Col;

/// Fit an Isotonic regression model
///
/// Isotonic regression fits a monotonic (non-decreasing or non-increasing)
/// function to the data. Useful for calibration, dose-response curves, and
/// any situation where monotonicity constraints are required.
///
/// # Arguments
/// * `x` - Feature values (n observations, 1D input)
/// * `y` - Response variable (n observations)
/// * `options` - Fitting options
///
/// # Returns
/// * `IsotonicFitResult` containing fitted values and R-squared
pub fn fit_isotonic(x: &[f64], y: &[f64], options: &IsotonicOptions) -> StatsResult<IsotonicFitResult> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = y.len();
    if x.len() != n_obs {
        return Err(StatsError::DimensionMismatch {
            y_len: n_obs,
            x_rows: x.len(),
        });
    }

    // Filter out rows with NaN values and collect valid data
    let valid_data: Vec<(f64, f64)> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan() && !y[i].is_infinite() && !x[i].is_nan() && !x[i].is_infinite()
        })
        .map(|i| (x[i], y[i]))
        .collect();

    if valid_data.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_data.len();
    if n_valid < 2 {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: 1,
        });
    }

    // Convert to faer types
    let x_col = Col::from_fn(n_valid, |i| valid_data[i].0);
    let y_col = Col::from_fn(n_valid, |i| valid_data[i].1);

    // Build and fit the Isotonic model
    let fitted = IsotonicRegressor::builder()
        .increasing(options.increasing)
        .build()
        .fit_1d(&x_col, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract fitted values
    let fitted_values: Vec<f64> = fitted.fitted_values().iter().copied().collect();

    // Calculate R-squared
    let y_mean = valid_data.iter().map(|(_, y)| y).sum::<f64>() / n_valid as f64;
    let ss_tot: f64 = valid_data.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = valid_data
        .iter()
        .zip(fitted_values.iter())
        .map(|((_, y), yhat)| (y - yhat).powi(2))
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok(IsotonicFitResult {
        fitted_values,
        r_squared,
        n_observations: n_valid,
        increasing: options.increasing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotonic_increasing() {
        // Data with some noise but general increasing trend
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0];

        let options = IsotonicOptions { increasing: true };
        let result = fit_isotonic(&x, &y, &options);

        assert!(result.is_ok());
        let result = result.unwrap();

        // Fitted values should be monotonically increasing
        for i in 1..result.fitted_values.len() {
            assert!(
                result.fitted_values[i] >= result.fitted_values[i - 1],
                "Fitted values should be monotonically increasing"
            );
        }
        assert!(result.increasing);
    }

    #[test]
    fn test_isotonic_decreasing() {
        // Data with decreasing trend
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 9.0, 5.0, 3.0];

        let options = IsotonicOptions { increasing: false };
        let result = fit_isotonic(&x, &y, &options);

        assert!(result.is_ok());
        let result = result.unwrap();

        // Fitted values should be monotonically decreasing
        for i in 1..result.fitted_values.len() {
            assert!(
                result.fitted_values[i] <= result.fitted_values[i - 1],
                "Fitted values should be monotonically decreasing"
            );
        }
        assert!(!result.increasing);
    }

    #[test]
    fn test_isotonic_empty_input() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let options = IsotonicOptions::default();
        let result = fit_isotonic(&x, &y, &options);

        assert!(matches!(result, Err(StatsError::EmptyInput { .. })));
    }

    #[test]
    fn test_isotonic_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0]; // Wrong length

        let options = IsotonicOptions::default();
        let result = fit_isotonic(&x, &y, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }
}
