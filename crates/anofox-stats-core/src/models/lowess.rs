//! LOWESS (Locally Weighted Scatterplot Smoothing)

use crate::errors::{StatsError, StatsResult};
use crate::types::{LowessOptions, LowessResult};
use anofox_regression::solvers::lowess::lowess_smooth;
use faer::Col;

/// Apply LOWESS smoothing to a response variable
///
/// LOWESS uses local weighted regression to produce a smooth curve.
/// Each point is fitted using nearby observations weighted by a tricube kernel.
///
/// # Arguments
/// * `y` - Response variable to smooth
/// * `options` - Smoothing options (span controls bandwidth)
///
/// # Returns
/// * `LowessResult` with smoothed values
pub fn fit_lowess(y: &[f64], options: &LowessOptions) -> StatsResult<LowessResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }

    if y.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "LOWESS requires at least 3 observations".to_string(),
        ));
    }

    if options.span <= 0.0 || options.span > 1.0 {
        return Err(StatsError::InvalidValue {
            field: "span",
            message: "Span must be in (0, 1]".to_string(),
        });
    }

    // Filter NaN/Inf values
    let valid_indices: Vec<usize> = (0..y.len())
        .filter(|&i| !y[i].is_nan() && !y[i].is_infinite())
        .collect();

    if valid_indices.len() < 3 {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);

    let smoothed = lowess_smooth(&y_col, options.span);

    // Map back to full array with NaN for invalid positions
    let mut fitted_values = vec![f64::NAN; y.len()];
    for (smooth_idx, &orig_idx) in valid_indices.iter().enumerate() {
        fitted_values[orig_idx] = smoothed[smooth_idx];
    }

    Ok(LowessResult {
        fitted_values,
        n_observations: n_valid,
        span: options.span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowess_basic() {
        // Noisy sine wave
        let y: Vec<f64> = (0..50)
            .map(|i| {
                let x = i as f64 * 0.1;
                x.sin() + 0.1 * (i as f64 * 0.3).sin()
            })
            .collect();

        let options = LowessOptions { span: 0.3 };
        let result = fit_lowess(&y, &options).unwrap();

        assert_eq!(result.fitted_values.len(), 50);
        assert_eq!(result.n_observations, 50);
        assert!((result.span - 0.3).abs() < 1e-10);

        // Smoothed values should be finite
        for &v in &result.fitted_values {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lowess_constant() {
        let y = vec![5.0; 20];
        let options = LowessOptions::default();
        let result = fit_lowess(&y, &options).unwrap();

        // Smoothing a constant should return the constant
        for &v in &result.fitted_values {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lowess_with_nan() {
        let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        y[3] = f64::NAN;
        y[7] = f64::NAN;

        let options = LowessOptions { span: 0.5 };
        let result = fit_lowess(&y, &options).unwrap();

        // NaN positions should remain NaN
        assert!(result.fitted_values[3].is_nan());
        assert!(result.fitted_values[7].is_nan());
        // Valid positions should be finite
        assert!(result.fitted_values[0].is_finite());
        assert_eq!(result.n_observations, 8);
    }

    #[test]
    fn test_lowess_empty() {
        let y: Vec<f64> = vec![];
        let options = LowessOptions::default();
        assert!(fit_lowess(&y, &options).is_err());
    }

    #[test]
    fn test_lowess_invalid_span() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let options = LowessOptions { span: 0.0 };
        assert!(fit_lowess(&y, &options).is_err());

        let options = LowessOptions { span: 1.5 };
        assert!(fit_lowess(&y, &options).is_err());
    }

    #[test]
    fn test_lowess_reduces_noise() {
        // Linear trend with noise
        let y: Vec<f64> = (0..30)
            .map(|i| {
                let x = i as f64;
                2.0 * x + if i % 2 == 0 { 3.0 } else { -3.0 }
            })
            .collect();

        let options = LowessOptions { span: 0.4 };
        let result = fit_lowess(&y, &options).unwrap();

        // Smoothed values should be closer to the trend (2x) than the noisy data
        // Check that variance of smoothed is less than variance of original
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / y.len() as f64;

        let mean_s: f64 =
            result.fitted_values.iter().sum::<f64>() / result.fitted_values.len() as f64;
        let var_s: f64 = result
            .fitted_values
            .iter()
            .map(|v| (v - mean_s).powi(2))
            .sum::<f64>()
            / result.fitted_values.len() as f64;

        assert!(var_s < var_y, "Smoothed variance should be less than original");
    }
}
