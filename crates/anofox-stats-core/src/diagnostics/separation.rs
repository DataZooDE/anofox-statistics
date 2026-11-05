//! Quasi-separation detection for GLM data

use crate::errors::{StatsError, StatsResult};
use crate::types::{LinearSeparationResult, SeparationCheckResult, SeparationType};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Validate the shape of `(x, y)` and that every `y` passes `valid_y`.
fn validate_xy(
    x: &[Vec<f64>],
    y: &[f64],
    valid_y: fn(f64) -> bool,
    y_msg: &str,
) -> StatsResult<()> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    for col in x.iter() {
        if col.len() != y.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: y.len(),
                x_rows: col.len(),
            });
        }
    }
    if !y.iter().all(|&v| valid_y(v)) {
        return Err(StatsError::InvalidInput(y_msg.to_string()));
    }
    Ok(())
}

fn is_binary(v: f64) -> bool {
    v == 0.0 || v == 1.0
}

fn is_count(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}

const BINARY_Y: &str = "y must be 0 or 1";

fn map_separation_type(st: &anofox_regression::diagnostics::SeparationType) -> SeparationType {
    match st {
        anofox_regression::diagnostics::SeparationType::None => SeparationType::None,
        anofox_regression::diagnostics::SeparationType::Complete => SeparationType::Complete,
        anofox_regression::diagnostics::SeparationType::Quasi => SeparationType::Quasi,
        anofox_regression::diagnostics::SeparationType::MonotonicResponse => {
            SeparationType::MonotonicResponse
        }
        // `#[non_exhaustive]` upstream; a newer kind is still a separation.
        _ => SeparationType::Quasi,
    }
}

/// Exact detection of (quasi-)complete separation for a binary response by
/// upstream's linear program (Konis 2007), as
/// `detectseparation::detect_separation`. Unlike
/// [`check_binary_separation_diagnostic`] it sees separation by a
/// combination of predictors.
///
/// # Arguments
/// * `x` - Feature matrix (column-major: each Vec is one feature)
/// * `y` - Binary response (0 or 1)
/// * `with_intercept` - Whether an intercept column is part of the design
pub fn detect_linear_separation(
    x: &[Vec<f64>],
    y: &[f64],
    with_intercept: bool,
) -> StatsResult<LinearSeparationResult> {
    validate_xy(x, y, is_binary, BINARY_Y)?;
    let n_obs = y.len();
    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let x_mat = Mat::from_fn(n_obs, x.len(), |i, j| x[j][i]);
    let sep = detect_separation(&x_mat, &y_col, with_intercept);
    Ok(LinearSeparationResult {
        separated: sep.separated,
        infinite: sep.infinite,
        direction: sep.direction,
    })
}

/// Check for binary separation in logistic regression data
///
/// Detects complete or quasi-separation where a predictor perfectly or
/// near-perfectly separates the binary response classes (a per-column
/// heuristic; see [`detect_linear_separation`] for the exact check).
///
/// # Arguments
/// * `x` - Feature matrix (column-major: each Vec is one feature)
/// * `y` - Binary response (0 or 1)
///
/// `separation_types` is aligned with `separated_predictors`.
pub fn check_binary_separation_diagnostic(
    x: &[Vec<f64>],
    y: &[f64],
) -> StatsResult<SeparationCheckResult> {
    validate_xy(x, y, is_binary, BINARY_Y)?;
    let n_obs = y.len();

    // Convert to faer types
    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let x_mat = Mat::from_fn(n_obs, x.len(), |i, j| x[j][i]);

    // Use upstream check; it reports a type for every predictor.
    let check = check_binary_separation(&x_mat, &y_col);
    let separation_types = check
        .separated_predictors
        .iter()
        .map(|&j| map_separation_type(&check.separation_types[j]))
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
///
/// `separation_types` is aligned with `separated_predictors`.
pub fn check_count_sparsity_diagnostic(
    x: &[Vec<f64>],
    y: &[f64],
) -> StatsResult<SeparationCheckResult> {
    validate_xy(x, y, is_count, "y must be a non-negative count")?;
    let n_obs = y.len();

    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let x_mat = Mat::from_fn(n_obs, x.len(), |i, j| x[j][i]);

    // Upstream reports a type only for the flagged predictors.
    let check = check_count_sparsity(&x_mat, &y_col);
    let separation_types = check
        .separation_types
        .iter()
        .map(map_separation_type)
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
    fn test_binary_types_aligned_with_predictors() {
        // Feature 0 overlaps the classes, feature 1 separates them.
        let x = vec![
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = check_binary_separation_diagnostic(&x, &y).unwrap();
        assert_eq!(result.separated_predictors, vec![1]);
        assert_eq!(result.separation_types, vec![SeparationType::Complete]);
    }

    #[test]
    fn test_binary_rejects_non_binary_y() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        assert!(check_binary_separation_diagnostic(&x, &[0.0, 2.0, 1.0]).is_err());
        assert!(detect_linear_separation(&x, &[0.0, 0.5, 1.0], true).is_err());
        assert!(check_count_sparsity_diagnostic(&x, &[0.0, -1.0, 1.0]).is_err());
    }

    #[test]
    fn test_detect_linear_separation() {
        // Separated by x1 + x2 jointly, not by either column alone.
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let lin = detect_linear_separation(&x, &y, true).unwrap();
        assert!(lin.separated);
        assert_eq!(lin.infinite.len(), 3);
        let y2 = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let lin2 = detect_linear_separation(&x, &y2, true).unwrap();
        assert!(!lin2.separated);
        assert!(lin2.infinite.iter().all(|&v| v == 0.0));
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
