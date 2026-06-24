//! Least Angle Regression (LARS) and its Lasso variant (LassoLars) wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, LarsOptions};
use anofox_regression::prelude::*;
use anofox_regression::solvers::{LarsMethod, LarsRegressor};
use faer::{Col, Mat};

/// Fit a Least Angle Regression model (LARS or LassoLars).
///
/// LARS produces a piecewise-linear coefficient path; with `method_lasso` it
/// reproduces the exact Lasso solution (LassoLars). Coefficients are returned
/// on the original predictor scale.
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (p features, each a column of n observations)
/// * `options` - Fitting options
pub fn fit_lars(y: &[f64], x: &[Vec<f64>], options: &LarsOptions) -> StatsResult<FitResult> {
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
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

    // Drop rows with NaN/inf in y or any feature.
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect();

    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n_valid = valid_indices.len();

    // Detect zero-variance (constant) columns; they are dropped from the fit
    // and reported as NaN coefficients.
    let is_constant_column: Vec<bool> = x
        .iter()
        .map(|col| {
            let first_val = col[valid_indices[0]];
            valid_indices
                .iter()
                .all(|&i| (col[i] - first_val).abs() < 1e-10)
        })
        .collect();

    let n_effective_features = is_constant_column.iter().filter(|&&c| !c).count();
    let min_obs = if options.fit_intercept {
        n_effective_features + 1
    } else {
        n_effective_features
    };

    // All columns constant: intercept-only model (mean of y) when fitting an intercept.
    if n_effective_features == 0 {
        if !options.fit_intercept {
            return Err(StatsError::InsufficientData {
                rows: n_valid,
                cols: n_features,
            });
        }
        let y_mean = valid_indices.iter().map(|&i| y[i]).sum::<f64>() / n_valid as f64;
        let y_var = valid_indices
            .iter()
            .map(|&i| (y[i] - y_mean).powi(2))
            .sum::<f64>()
            / (n_valid.max(2) - 1) as f64;
        return Ok(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features],
                intercept: Some(y_mean),
                r_squared: 0.0,
                adj_r_squared: 0.0,
                residual_std_error: y_var.sqrt(),
                n_observations: n_valid,
                n_features,
            },
            inference: None,
            diagnostics: None,
        });
    }

    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    let non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    // Build reduced design (non-constant columns, valid rows only).
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });

    let mut builder = LarsRegressor::builder()
        .fit_intercept(options.fit_intercept)
        .method(if options.method_lasso {
            LarsMethod::Lasso
        } else {
            LarsMethod::Lar
        })
        .alpha(options.alpha)
        .standardize(options.standardize);
    if options.n_nonzero_coefs > 0 {
        builder = builder.n_nonzero_coefs(options.n_nonzero_coefs as usize);
    }

    let fitted = builder
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    let result = fitted.result();

    // Reconstruct full coefficient vector with NaN for the dropped constant columns.
    let reduced: Vec<f64> = result.coefficients.iter().copied().collect();
    let mut coefficients = vec![f64::NAN; n_features];
    for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
        coefficients[orig_idx] = reduced[reduced_idx];
    }
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    Ok(FitResult {
        core: FitResultCore {
            coefficients,
            intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            residual_std_error: result.rmse,
            n_observations: n_valid,
            n_features,
        },
        inference: None,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lars_recovers_linear() {
        // y = 3 + 2*x  -> slope ~2, strong fit
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![5.1, 7.0, 8.9, 11.1, 13.0, 14.9, 17.1, 19.0, 20.9, 23.1];
        let opts = LarsOptions::default();
        let r = fit_lars(&y, &x, &opts).unwrap();
        assert!(r.core.coefficients[0] > 1.5 && r.core.coefficients[0] < 2.5);
        assert!(r.core.r_squared > 0.95);
    }

    #[test]
    fn test_lars_invalid_alpha() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let opts = LarsOptions {
            alpha: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            fit_lars(&y, &x, &opts),
            Err(StatsError::InvalidAlpha(_))
        ));
    }
}
