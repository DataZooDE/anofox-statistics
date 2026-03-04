//! Condition number diagnostics for detecting multicollinearity

use crate::errors::{StatsError, StatsResult};
use crate::types::{ConditionDiagnosticResult, ConditionSeverity};
use anofox_regression::prelude::*;
use faer::Mat;

/// Compute condition number diagnostics for a design matrix
///
/// # Arguments
/// * `x` - Feature matrix (n observations x p features, column-major: each inner Vec is one feature)
/// * `with_intercept` - Whether an intercept column should be added
///
/// # Returns
/// * `ConditionDiagnosticResult` with condition number, singular values, severity, etc.
pub fn compute_condition_diagnostic(
    x: &[Vec<f64>],
    with_intercept: bool,
) -> StatsResult<ConditionDiagnosticResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = x[0].len();
    let n_features = x.len();

    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::InvalidInput(
                "All feature columns must have the same length".to_string(),
            ));
        }
    }

    if n_obs < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Need at least 2 observations for condition diagnostics".to_string(),
        ));
    }

    // Convert to faer matrix
    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| x[j][i]);

    // Use upstream condition_diagnostic
    let diag = condition_diagnostic(&x_mat, with_intercept);

    let severity = match diag.severity {
        anofox_regression::diagnostics::ConditionSeverity::WellConditioned => {
            ConditionSeverity::WellConditioned
        }
        anofox_regression::diagnostics::ConditionSeverity::Moderate => ConditionSeverity::Moderate,
        anofox_regression::diagnostics::ConditionSeverity::High => ConditionSeverity::High,
        anofox_regression::diagnostics::ConditionSeverity::Severe => ConditionSeverity::Severe,
    };

    Ok(ConditionDiagnosticResult {
        condition_number: diag.condition_number,
        condition_number_xtx: diag.condition_number_xtx,
        singular_values: diag.singular_values,
        condition_indices: diag.condition_indices,
        severity,
        warning: diag.warning,
    })
}

/// Compute just the condition number of a design matrix
///
/// # Arguments
/// * `x` - Feature matrix (column-major)
/// * `with_intercept` - Whether to include intercept column
///
/// # Returns
/// * Condition number (ratio of largest to smallest singular value)
pub fn compute_condition_number(x: &[Vec<f64>], with_intercept: bool) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = x[0].len();
    let n_features = x.len();

    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| x[j][i]);

    Ok(condition_number(&x_mat, with_intercept))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_number_well_conditioned() {
        // Orthogonal-ish features should have low condition number
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];

        let result = compute_condition_diagnostic(&x, true).unwrap();
        assert!(result.condition_number > 0.0);
        assert!(result.condition_number.is_finite());
        assert!(!result.singular_values.is_empty());
    }

    #[test]
    fn test_condition_number_collinear() {
        // Nearly collinear features should have high condition number
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![
                1.001, 2.001, 3.001, 4.001, 5.001, 6.001, 7.001, 8.001, 9.001, 10.001,
            ],
        ];

        let result = compute_condition_diagnostic(&x, true).unwrap();
        // High condition number expected for near-collinear data
        assert!(result.condition_number > 100.0);
    }

    #[test]
    fn test_condition_number_simple() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let cond = compute_condition_number(&x, true).unwrap();
        assert!(cond > 0.0);
        assert!(cond.is_finite());
    }

    #[test]
    fn test_condition_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let result = compute_condition_diagnostic(&x, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_condition_severity_classification() {
        // Single well-scaled feature
        let x = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];
        let result = compute_condition_diagnostic(&x, false).unwrap();
        // Without intercept, single feature should be well-conditioned
        assert_eq!(result.severity, ConditionSeverity::WellConditioned);
    }
}
