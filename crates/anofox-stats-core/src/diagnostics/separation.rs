//! Quasi-separation detection for GLM data

use crate::errors::{StatsError, StatsResult};
use crate::types::{SeparationCheckResult, SeparationType};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Check for binary separation in logistic regression data
///
/// Detects complete or quasi-separation where a predictor perfectly or
/// near-perfectly separates the binary response classes.
///
/// # Arguments
/// * `x` - Feature matrix (column-major: each Vec is one feature)
/// * `y` - Binary response (0 or 1)
pub fn check_binary_separation_diagnostic(
    x: &[Vec<f64>],
    y: &[f64],
) -> StatsResult<SeparationCheckResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| x[j][i]);

    // Use upstream check
    let check = check_binary_separation(&x_mat, &y_col);

    let separation_types: Vec<SeparationType> = check
        .separation_types
        .iter()
        .map(|st| match st {
            anofox_regression::diagnostics::SeparationType::None => SeparationType::None,
            anofox_regression::diagnostics::SeparationType::Complete => SeparationType::Complete,
            anofox_regression::diagnostics::SeparationType::Quasi => SeparationType::Quasi,
            anofox_regression::diagnostics::SeparationType::MonotonicResponse => {
                SeparationType::MonotonicResponse
            }
        })
        .collect();

    Ok(SeparationCheckResult {
        has_separation: check.has_separation,
        separated_predictors: check.separated_predictors,
        separation_types,
        warning: check.warning_message,
    })
}

/// Check for count data sparsity issues in Poisson regression data
///
/// Detects sparse count patterns that may cause convergence issues.
///
/// # Arguments
/// * `x` - Feature matrix (column-major)
/// * `y` - Count response (non-negative)
pub fn check_count_sparsity_diagnostic(
    x: &[Vec<f64>],
    y: &[f64],
) -> StatsResult<SeparationCheckResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| x[j][i]);

    let check = check_count_sparsity(&x_mat, &y_col);

    let separation_types: Vec<SeparationType> = check
        .separation_types
        .iter()
        .map(|st| match st {
            anofox_regression::diagnostics::SeparationType::None => SeparationType::None,
            anofox_regression::diagnostics::SeparationType::Complete => SeparationType::Complete,
            anofox_regression::diagnostics::SeparationType::Quasi => SeparationType::Quasi,
            anofox_regression::diagnostics::SeparationType::MonotonicResponse => {
                SeparationType::MonotonicResponse
            }
        })
        .collect();

    Ok(SeparationCheckResult {
        has_separation: check.has_separation,
        separated_predictors: check.separated_predictors,
        separation_types,
        warning: check.warning_message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_separation() {
        // Overlapping classes - no separation
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let result = check_binary_separation_diagnostic(&x, &y).unwrap();
        assert!(!result.has_separation);
    }

    #[test]
    fn test_complete_separation() {
        // Perfect separation: low x -> 0, high x -> 1
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = check_binary_separation_diagnostic(&x, &y).unwrap();
        assert!(result.has_separation);
    }

    #[test]
    fn test_separation_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let y = vec![0.0, 1.0];
        assert!(check_binary_separation_diagnostic(&x, &y).is_err());
    }

    #[test]
    fn test_count_sparsity_no_issues() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![3.0, 5.0, 2.0, 8.0, 4.0, 7.0, 1.0, 9.0, 6.0, 3.0];

        let result = check_count_sparsity_diagnostic(&x, &y).unwrap();
        // Dense count data should not trigger sparsity warnings
        assert!(!result.has_separation);
    }

    #[test]
    fn test_count_sparsity_sparse() {
        // Sparse count data - check that the function runs without error
        let x = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 7.0, 4.0, 6.0];

        let result = check_count_sparsity_diagnostic(&x, &y).unwrap();
        // Verify the result is well-formed regardless of detection outcome
        assert_eq!(
            result.separated_predictors.len(),
            result.separation_types.len()
        );
    }
}
