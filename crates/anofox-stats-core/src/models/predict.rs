//! Prediction function for fitted models
//!
//! Computes predictions using fitted coefficients and intercept.

use crate::errors::{StatsError, StatsResult};

/// Make predictions using fitted model coefficients
///
/// # Arguments
/// * `x` - Feature matrix (p features, each with n observations)
/// * `coefficients` - Fitted coefficients (p values)
/// * `intercept` - Optional intercept term
///
/// # Returns
/// * Vector of predicted values (n observations)
#[allow(clippy::needless_range_loop)]
pub fn predict(
    x: &[Vec<f64>],
    coefficients: &[f64],
    intercept: Option<f64>,
) -> StatsResult<Vec<f64>> {
    // Validate inputs
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if coefficients.is_empty() {
        return Err(StatsError::EmptyInput {
            field: "coefficients",
        });
    }

    let n_features = x.len();
    if n_features != coefficients.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: coefficients.len(),
            x_rows: n_features,
        });
    }

    // Get number of observations from first column
    let n_obs = x[0].len();

    // Verify all columns have same length
    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

    // Compute predictions: y_pred = X * beta + intercept
    let mut predictions = Vec::with_capacity(n_obs);

    for i in 0..n_obs {
        let mut y_pred = intercept.unwrap_or(0.0);
        for (j, coef) in coefficients.iter().enumerate() {
            y_pred += coef * x[j][i];
        }
        predictions.push(y_pred);
    }

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_basic() {
        // y = 2*x + 1
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let coefficients = vec![2.0];
        let intercept = Some(1.0);

        let predictions = predict(&x, &coefficients, intercept).unwrap();

        assert_eq!(predictions.len(), 5);
        assert!((predictions[0] - 3.0).abs() < 1e-10); // 2*1 + 1 = 3
        assert!((predictions[1] - 5.0).abs() < 1e-10); // 2*2 + 1 = 5
        assert!((predictions[2] - 7.0).abs() < 1e-10); // 2*3 + 1 = 7
        assert!((predictions[3] - 9.0).abs() < 1e-10); // 2*4 + 1 = 9
        assert!((predictions[4] - 11.0).abs() < 1e-10); // 2*5 + 1 = 11
    }

    #[test]
    fn test_predict_no_intercept() {
        // y = 2*x (no intercept)
        let x = vec![vec![1.0, 2.0, 3.0]];
        let coefficients = vec![2.0];
        let intercept = None;

        let predictions = predict(&x, &coefficients, intercept).unwrap();

        assert_eq!(predictions.len(), 3);
        assert!((predictions[0] - 2.0).abs() < 1e-10); // 2*1 = 2
        assert!((predictions[1] - 4.0).abs() < 1e-10); // 2*2 = 4
        assert!((predictions[2] - 6.0).abs() < 1e-10); // 2*3 = 6
    }

    #[test]
    fn test_predict_multivariate() {
        // y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1.0, 2.0, 3.0], // x1
            vec![1.0, 1.0, 1.0], // x2
        ];
        let coefficients = vec![2.0, 3.0];
        let intercept = Some(1.0);

        let predictions = predict(&x, &coefficients, intercept).unwrap();

        assert_eq!(predictions.len(), 3);
        assert!((predictions[0] - 6.0).abs() < 1e-10); // 2*1 + 3*1 + 1 = 6
        assert!((predictions[1] - 8.0).abs() < 1e-10); // 2*2 + 3*1 + 1 = 8
        assert!((predictions[2] - 10.0).abs() < 1e-10); // 2*3 + 3*1 + 1 = 10
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let coefficients = vec![2.0, 3.0]; // 2 coefficients but only 1 feature

        let result = predict(&x, &coefficients, None);
        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_predict_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let coefficients = vec![2.0];

        let result = predict(&x, &coefficients, None);
        assert!(matches!(result, Err(StatsError::EmptyInput { .. })));
    }
}
