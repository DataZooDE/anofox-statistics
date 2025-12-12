//! Bounded Least Squares (BLS) regression implementation
//!
//! BLS extends ordinary least squares to handle coefficient constraints:
//! - Non-Negative Least Squares (NNLS): all coefficients >= 0
//! - Box-constrained LS: lower <= coefficients <= upper

use anofox_regression::solvers::{BlsRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

use crate::errors::{StatsError, StatsResult};
use crate::types::{BlsFitResult, BlsOptions};

/// Validate input arrays for BLS
fn validate_inputs(y: &[f64], x: &[Vec<f64>]) -> StatsResult<()> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n = y.len();
    for (i, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Feature column {} has {} rows, expected {}",
                i,
                col.len(),
                n
            )));
        }
    }

    Ok(())
}

/// Fit a Bounded Least Squares model
///
/// Solves: minimize ||Xβ - y||² subject to lower ≤ β ≤ upper
///
/// # Arguments
/// * `y` - Response variable
/// * `x` - Feature matrix (column-major: each inner Vec is one feature)
/// * `options` - Fitting options including bounds
///
/// # Returns
/// * `BlsFitResult` containing coefficients and constraint info
///
/// # Example
/// ```ignore
/// use anofox_stats_core::models::fit_bls;
/// use anofox_stats_core::types::BlsOptions;
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let options = BlsOptions::nnls(); // Non-negative LS
/// let result = fit_bls(&y, &x, &options).unwrap();
/// ```
pub fn fit_bls(y: &[f64], x: &[Vec<f64>], options: &BlsOptions) -> StatsResult<BlsFitResult> {
    validate_inputs(y, x)?;

    let n_features = x.len();
    let n_obs = y.len();

    // Filter out rows with NaN/NULL values
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && y[i].is_finite()
                && x.iter().all(|col| !col[i].is_nan() && col[i].is_finite())
        })
        .collect();

    let n_valid = valid_indices.len();
    if n_valid == 0 {
        return Err(StatsError::NoValidData);
    }

    // Need at least n_features + 1 observations (for intercept)
    let min_obs = if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    };
    if n_valid <= min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    // Build regressor based on options
    let fitted = if options.lower_bounds.is_none() && options.upper_bounds.is_none() {
        // NNLS: default to lower bound of 0 for all coefficients
        BlsRegressor::nnls()
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations as usize)
            .tolerance(options.tolerance)
            .build()
            .fit(&x_mat, &y_col)
            .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?
    } else {
        // Custom bounds
        let mut builder = BlsRegressor::builder()
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations as usize)
            .tolerance(options.tolerance);

        if let Some(ref lower) = options.lower_bounds {
            if lower.len() == 1 {
                // Single value = apply to all
                builder = builder.lower_bound_all(lower[0]);
            } else {
                builder = builder.lower_bounds(lower.clone());
            }
        }

        if let Some(ref upper) = options.upper_bounds {
            if upper.len() == 1 {
                // Single value = apply to all
                builder = builder.upper_bound_all(upper[0]);
            } else {
                builder = builder.upper_bounds(upper.clone());
            }
        }

        builder
            .build()
            .fit(&x_mat, &y_col)
            .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?
    };

    // Extract results
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = result.intercept;

    // Determine which coefficients are at bounds
    let (at_lower_bound, at_upper_bound, n_active) =
        compute_active_constraints(&coefficients, options, options.tolerance);

    Ok(BlsFitResult {
        coefficients,
        intercept,
        ssr: result.mse * n_valid as f64, // SSR = MSE * n
        r_squared: result.r_squared,
        n_observations: n_valid,
        n_features,
        n_active_constraints: n_active,
        at_lower_bound,
        at_upper_bound,
    })
}

/// Fit Non-Negative Least Squares (NNLS) - convenience function
///
/// All coefficients are constrained to be >= 0.
///
/// # Arguments
/// * `y` - Response variable
/// * `x` - Feature matrix (column-major)
///
/// # Returns
/// * `BlsFitResult` containing non-negative coefficients
pub fn fit_nnls(y: &[f64], x: &[Vec<f64>]) -> StatsResult<BlsFitResult> {
    fit_bls(y, x, &BlsOptions::nnls())
}

/// Compute which constraints are active (coefficients at bounds)
fn compute_active_constraints(
    coefficients: &[f64],
    options: &BlsOptions,
    tolerance: f64,
) -> (Vec<bool>, Vec<bool>, usize) {
    let n = coefficients.len();
    let mut at_lower = vec![false; n];
    let mut at_upper = vec![false; n];
    let mut n_active = 0;

    for (i, &coef) in coefficients.iter().enumerate() {
        // Check lower bound
        let lower = match &options.lower_bounds {
            Some(bounds) if bounds.len() == 1 => bounds[0],
            Some(bounds) if i < bounds.len() => bounds[i],
            None => 0.0, // Default for NNLS
            _ => f64::NEG_INFINITY,
        };

        // Check upper bound
        let upper = match &options.upper_bounds {
            Some(bounds) if bounds.len() == 1 => bounds[0],
            Some(bounds) if i < bounds.len() => bounds[i],
            _ => f64::INFINITY,
        };

        if lower.is_finite() && (coef - lower).abs() < tolerance {
            at_lower[i] = true;
            n_active += 1;
        } else if upper.is_finite() && (coef - upper).abs() < tolerance {
            at_upper[i] = true;
            n_active += 1;
        }
    }

    (at_lower, at_upper, n_active)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bls_nnls_basic() {
        // Simple data where true coefficients are positive
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 2.0 * xi).collect();

        let result = fit_nnls(&y, &x).unwrap();

        // Coefficient should be positive and close to 2.0
        assert!(result.coefficients[0] >= 0.0);
        assert!(result.coefficients[0] > 1.5);
    }

    #[test]
    fn test_bls_nnls_constrained() {
        // Data where OLS would give negative coefficient but NNLS should give 0
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| -2.0 * xi + 20.0).collect();

        let options = BlsOptions::nnls();
        let result = fit_bls(&y, &x, &options).unwrap();

        // Coefficient should be non-negative (constrained)
        assert!(result.coefficients[0] >= 0.0);
    }

    #[test]
    fn test_bls_with_bounds() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 5.0 * xi).collect();

        let options = BlsOptions {
            fit_intercept: false,
            lower_bounds: Some(vec![0.0]),
            upper_bounds: Some(vec![3.0]), // Upper bound below true coefficient
            max_iterations: 1000,
            tolerance: 1e-10,
        };

        let result = fit_bls(&y, &x, &options).unwrap();

        // Coefficient should be at upper bound
        assert!((result.coefficients[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_bls_active_constraints() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| -xi + 10.0).collect();

        let result = fit_nnls(&y, &x).unwrap();

        // With negative true coefficient, NNLS should hit the lower bound
        assert!(result.at_lower_bound[0] || result.coefficients[0] < 0.1);
        // n_active_constraints should be valid (usize is always >= 0)
        assert!(result.n_active_constraints <= result.coefficients.len());
    }

    #[test]
    fn test_bls_empty_input() {
        let y: Vec<f64> = vec![];
        let x: Vec<Vec<f64>> = vec![vec![]];

        let result = fit_nnls(&y, &x);
        assert!(result.is_err());
    }
}
