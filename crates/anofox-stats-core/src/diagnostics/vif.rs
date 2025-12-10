//! Variance Inflation Factor (VIF) computation

use crate::errors::{StatsError, StatsResult};
use crate::models::fit_ols;
use crate::types::OlsOptions;

/// Compute VIF (Variance Inflation Factor) for each feature.
///
/// VIF measures multicollinearity. For each feature j, we regress it on
/// all other features and compute VIF_j = 1 / (1 - R²_j).
///
/// VIF interpretation:
/// - VIF = 1: No correlation with other features
/// - VIF < 5: Generally acceptable
/// - VIF 5-10: Moderate multicollinearity
/// - VIF > 10: High multicollinearity, may need to address
///
/// # Arguments
/// * `x` - Feature arrays, each Vec<f64> is one feature column
///
/// # Returns
/// * Vector of VIF values, one per feature
pub fn compute_vif(x: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let n_features = x.len();

    if n_features == 0 {
        return Err(StatsError::InvalidInput("No features provided".into()));
    }

    if n_features == 1 {
        // Single feature has VIF = 1 (no other features to correlate with)
        return Ok(vec![1.0]);
    }

    let n_obs = x[0].len();
    if n_obs == 0 {
        return Err(StatsError::InvalidInput("Empty feature arrays".into()));
    }

    // Check all features have same length
    for (i, col) in x.iter().enumerate() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Feature {} has {} observations, expected {}",
                i,
                col.len(),
                n_obs
            )));
        }
    }

    let mut vif_values = Vec::with_capacity(n_features);

    // For each feature, regress it on all other features
    for j in 0..n_features {
        // y is feature j
        let y = &x[j];

        // X contains all other features
        let other_x: Vec<Vec<f64>> = x
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != j)
            .map(|(_, col)| col.clone())
            .collect();

        // Fit OLS: feature_j ~ other_features
        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        };

        match fit_ols(y, &other_x, &options) {
            Ok(result) => {
                let r_squared = result.core.r_squared;

                // VIF = 1 / (1 - R²)
                // Handle edge case where R² ≈ 1 (perfect multicollinearity)
                let vif = if r_squared >= 0.9999 {
                    f64::INFINITY
                } else if r_squared < 0.0 {
                    // Shouldn't happen, but be defensive
                    1.0
                } else {
                    1.0 / (1.0 - r_squared)
                };

                vif_values.push(vif);
            }
            Err(_) => {
                // If regression fails (e.g., singular matrix), VIF is infinite
                vif_values.push(f64::INFINITY);
            }
        }
    }

    Ok(vif_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vif_single_feature() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_vif_uncorrelated_features() {
        // Two uncorrelated features should have VIF ≈ 1
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![5.0, 3.0, 1.0, 4.0, 2.0]; // Random order
        let x = vec![x1, x2];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 2);
        // VIF should be close to 1 for uncorrelated features
        assert!(result[0] < 2.0);
        assert!(result[1] < 2.0);
    }

    #[test]
    fn test_vif_perfectly_correlated() {
        // x2 = 2 * x1, perfect collinearity
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x = vec![x1, x2];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 2);
        // VIF should be very high (or infinite) for perfectly correlated features
        assert!(result[0] > 1000.0 || result[0].is_infinite());
        assert!(result[1] > 1000.0 || result[1].is_infinite());
    }

    #[test]
    fn test_vif_moderate_correlation() {
        // Features with moderate correlation
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![1.1, 2.2, 2.8, 4.1, 5.0, 5.9, 7.2, 8.0]; // Correlated with noise
        let x3 = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]; // Negatively correlated
        let x = vec![x1, x2, x3];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 3);
        // All VIFs should be > 1 due to correlations
        for vif in &result {
            assert!(*vif >= 1.0);
        }
    }

    #[test]
    fn test_vif_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        assert!(compute_vif(&x).is_err());
    }
}
