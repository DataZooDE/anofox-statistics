//! Residual computation for regression models

use crate::errors::{StatsError, StatsResult};

/// Type of residuals to compute
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualType {
    /// Raw residuals: e = y - y_hat
    Raw,
    /// Standardized residuals: e / s (where s = residual std error)
    Standardized,
    /// Studentized residuals: e / (s * sqrt(1 - h_ii)) where h_ii is leverage
    Studentized,
}

/// Result containing different types of residuals
#[derive(Debug)]
pub struct ResidualsResult {
    /// Raw residuals
    pub raw: Vec<f64>,
    /// Standardized residuals (if computed)
    pub standardized: Option<Vec<f64>>,
    /// Studentized residuals (if computed)
    pub studentized: Option<Vec<f64>>,
    /// Leverage values (hat diagonal)
    pub leverage: Option<Vec<f64>>,
}

/// Compute residuals from y and predicted values
pub fn compute_residuals(
    y: &[f64],
    y_hat: &[f64],
    x: Option<&[Vec<f64>]>,
    residual_std_error: Option<f64>,
    include_studentized: bool,
) -> StatsResult<ResidualsResult> {
    let n = y.len();

    if n == 0 {
        return Err(StatsError::InvalidInput("Empty y array".into()));
    }

    if y_hat.len() != n {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "y has {} elements, y_hat has {}",
            n,
            y_hat.len()
        )));
    }

    // Compute raw residuals
    let raw: Vec<f64> = y.iter().zip(y_hat).map(|(yi, yhi)| yi - yhi).collect();

    // Compute standardized residuals if we have residual_std_error
    let standardized = residual_std_error.map(|s| {
        if s > 0.0 {
            raw.iter().map(|e| e / s).collect()
        } else {
            raw.clone()
        }
    });

    // Compute studentized residuals and leverage if requested and we have X
    let (studentized, leverage) = if let (true, Some(x_data)) = (include_studentized, x) {
        compute_studentized_residuals(&raw, x_data, residual_std_error)?
    } else {
        (None, None)
    };

    Ok(ResidualsResult {
        raw,
        standardized,
        studentized,
        leverage,
    })
}

/// Compute studentized residuals and leverage values using simple matrix math
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)]
fn compute_studentized_residuals(
    raw_residuals: &[f64],
    x: &[Vec<f64>],
    residual_std_error: Option<f64>,
) -> StatsResult<(Option<Vec<f64>>, Option<Vec<f64>>)> {
    let n = raw_residuals.len();
    let k = x.len();

    if k == 0 {
        return Ok((None, None));
    }

    // Build design matrix with intercept: n x (k+1)
    let n_cols = k + 1;

    // X'X matrix (k+1) x (k+1)
    let mut xtx = vec![vec![0.0; n_cols]; n_cols];
    for i in 0..n {
        for j in 0..n_cols {
            for l in 0..n_cols {
                let x_ij = if j == 0 { 1.0 } else { x[j - 1][i] };
                let x_il = if l == 0 { 1.0 } else { x[l - 1][i] };
                xtx[j][l] += x_ij * x_il;
            }
        }
    }

    // Invert X'X using Gauss-Jordan elimination
    let xtx_inv = match invert_matrix(&xtx) {
        Some(inv) => inv,
        None => return Ok((None, None)), // Singular matrix
    };

    // Compute leverage for each observation: h_ii = x_i' (X'X)^(-1) x_i
    let mut leverage = Vec::with_capacity(n);
    for i in 0..n {
        let mut h_ii = 0.0;
        for j in 0..n_cols {
            for l in 0..n_cols {
                let x_ij = if j == 0 { 1.0 } else { x[j - 1][i] };
                let x_il = if l == 0 { 1.0 } else { x[l - 1][i] };
                h_ii += x_ij * xtx_inv[j][l] * x_il;
            }
        }
        leverage.push(h_ii);
    }

    // Compute studentized residuals if we have residual_std_error
    let studentized = residual_std_error.map(|s| {
        raw_residuals
            .iter()
            .zip(&leverage)
            .map(|(e, h)| {
                let denom = s * (1.0 - h).max(1e-10).sqrt();
                e / denom
            })
            .collect()
    });

    Ok((studentized, Some(leverage)))
}

/// Simple matrix inversion using Gauss-Jordan elimination
#[allow(clippy::needless_range_loop)]
fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 {
        return None;
    }

    // Create augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular matrix
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse (right half of augmented matrix)
    let inv: Vec<Vec<f64>> = aug.iter().map(|row| row[n..].to_vec()).collect();

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_residuals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_hat = vec![1.1, 1.9, 3.0, 4.1, 4.9];

        let result = compute_residuals(&y, &y_hat, None, None, false).unwrap();

        assert_eq!(result.raw.len(), 5);
        assert!((result.raw[0] - (-0.1)).abs() < 1e-10);
        assert!((result.raw[1] - 0.1).abs() < 1e-10);
        assert!((result.raw[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardized_residuals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_hat = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual_std_error = 0.1;

        let result = compute_residuals(&y, &y_hat, None, Some(residual_std_error), false).unwrap();

        assert!(result.standardized.is_some());
        let std_resid = result.standardized.unwrap();
        for r in std_resid {
            assert_eq!(r, 0.0);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let y = vec![1.0, 2.0, 3.0];
        let y_hat = vec![1.0, 2.0];

        let result = compute_residuals(&y, &y_hat, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_studentized_residuals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_hat = vec![1.1, 1.9, 3.0, 4.1, 4.9];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let residual_std_error = 0.1;

        let result =
            compute_residuals(&y, &y_hat, Some(&x), Some(residual_std_error), true).unwrap();

        assert!(result.studentized.is_some());
        assert!(result.leverage.is_some());

        let leverage = result.leverage.unwrap();
        assert_eq!(leverage.len(), 5);
        // Leverage should be between 0 and 1
        for h in leverage {
            assert!((0.0..=1.0).contains(&h));
        }
    }
}
