//! Shared input-validation helpers for the model fit entry points.

use crate::errors::{StatsError, StatsResult};

/// Validate that every feature column has exactly `y_len` observations.
///
/// Every fit entry point must call this (or an equivalent check) before
/// indexing `x[j][i]` for `i < y_len`; otherwise a short column panics.
pub(crate) fn validate_x_columns(y_len: usize, x: &[Vec<f64>]) -> StatsResult<()> {
    for col in x.iter() {
        if col.len() != y_len {
            return Err(StatsError::DimensionMismatch {
                y_len,
                x_rows: col.len(),
            });
        }
    }
    Ok(())
}

/// Relative tolerance for treating a column as constant; the same default
/// `rank_tolerance` the upstream linear solvers use.
const CONSTANT_RANGE_RTOL: f64 = 1e-10;

/// Decide, per column, whether it must be left out of the design (restricted
/// to `rows`), using upstream's detectors so the extension agrees with the
/// solvers:
///
/// * With an intercept, a constant column (scale-relative test,
///   `utils::detect_constant_columns_relative`) is perfectly collinear with
///   the intercept and is dropped (its coefficient is reported as NaN).
/// * WITHOUT an intercept a constant non-zero column (e.g. all ones) *is* the
///   intercept and is kept; only identically-zero columns
///   (`utils::detect_zero_columns`) are dropped.
pub(crate) fn droppable_columns(x: &[Vec<f64>], rows: &[usize], fit_intercept: bool) -> Vec<bool> {
    detect(
        &faer::Mat::from_fn(rows.len(), x.len(), |i, j| x[j][rows[i]]),
        fit_intercept,
    )
}

fn detect(m: &faer::Mat<f64>, fit_intercept: bool) -> Vec<bool> {
    if fit_intercept {
        anofox_regression::utils::detect_constant_columns_relative(m, CONSTANT_RANGE_RTOL)
    } else {
        anofox_regression::utils::detect_zero_columns(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_droppable_column(col: &[f64], rows: &[usize], fit_intercept: bool) -> bool {
        droppable_columns(&[col.to_vec()], rows, fit_intercept)[0]
    }

    #[test]
    fn tiny_unit_column_is_not_constant() {
        let col = vec![1e-8, 2e-8, 3e-8, 4e-8];
        assert!(!is_droppable_column(&col, &[0, 1, 2, 3], true));
    }

    #[test]
    fn exactly_constant_column_is_constant() {
        assert!(is_droppable_column(&[5.0; 4], &[0, 1, 2, 3], true));
        assert!(is_droppable_column(&[0.0; 4], &[0, 1, 2, 3], true));
    }

    #[test]
    fn no_intercept_keeps_constant_nonzero_columns() {
        let x = vec![vec![1.0; 4], vec![0.0; 4], vec![1.0, 2.0, 3.0, 4.0]];
        let rows = [0, 1, 2, 3];
        assert_eq!(
            droppable_columns(&x, &rows, false),
            vec![false, true, false]
        );
        assert_eq!(droppable_columns(&x, &rows, true), vec![true, true, false]);
    }

    #[test]
    fn mismatched_columns_error() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert!(matches!(
            validate_x_columns(3, &x),
            Err(StatsError::DimensionMismatch {
                y_len: 3,
                x_rows: 2
            })
        ));
    }
}
