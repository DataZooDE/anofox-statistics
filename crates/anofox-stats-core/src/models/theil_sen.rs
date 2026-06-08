//! Theil-Sen robust regression wrapper.
//!
//! Wraps `anofox_regression::solvers::TheilSenRegressor` and reshapes its
//! result into the workspace-local `FitResult`. Theil-Sen has no
//! consensus-set diagnostics to surface (no inlier mask, no trial count) —
//! the upstream `FittedTheilSen` exposes only the standard `RegressionResult`
//! and a `with_intercept` flag.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, TheilSenOptions};
use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenRegressor};
use faer::{Col, Mat};

/// Theil-Sen regression fit. Identical fields to `FitResult` — kept as a
/// distinct type so the FFI / downstream wrappers can grow estimator-specific
/// extras later without touching the OLS / Huber / RANSAC contracts.
#[derive(Debug, Clone)]
pub struct TheilSenResult {
    pub fit: FitResult,
}

/// Fit a Theil-Sen robust regression model.
pub fn fit_theilsen(
    y: &[f64],
    x: &[Vec<f64>],
    options: &TheilSenOptions,
) -> StatsResult<TheilSenResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if options.max_subpopulation == 0 {
        return Err(StatsError::InvalidInput(
            "max_subpopulation must be > 0".to_string(),
        ));
    }
    if options.max_iterations == 0 {
        return Err(StatsError::InvalidInput(
            "max_iterations must be > 0".to_string(),
        ));
    }
    if !(options.tolerance > 0.0 && options.tolerance.is_finite()) {
        return Err(StatsError::InvalidInput(format!(
            "tolerance must be finite and positive, got {}",
            options.tolerance
        )));
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
    let p_eff = if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    };
    let effective_n_subsamples = options.n_subsamples.unwrap_or(p_eff).max(p_eff);
    if n_valid < effective_n_subsamples {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    let mut builder = TheilSenRegressor::builder()
        .with_intercept(options.fit_intercept)
        .max_subpopulation(options.max_subpopulation as usize)
        .max_iter(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .random_state(options.random_state);
    if let Some(n) = options.n_subsamples {
        builder = builder.n_subsamples(n);
    }

    let fitted = builder
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    let result = fitted.result();

    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    let core = FitResultCore {
        coefficients,
        intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: n_valid,
        n_features,
    };

    // Same inference projection as RANSAC: only emit fields the upstream
    // RegressionResult actually populates.
    let inference = if options.compute_inference {
        result.std_errors.as_ref().map(|se| FitResultInference {
            std_errors: se.iter().copied().collect(),
            t_values: result
                .t_statistics
                .as_ref()
                .map(|c| c.iter().copied().collect())
                .unwrap_or_else(|| vec![f64::NAN; n_features]),
            p_values: result
                .p_values
                .as_ref()
                .map(|c| c.iter().copied().collect())
                .unwrap_or_else(|| vec![f64::NAN; n_features]),
            ci_lower: result
                .conf_interval_lower
                .as_ref()
                .map(|c| c.iter().copied().collect())
                .unwrap_or_else(|| vec![f64::NAN; n_features]),
            ci_upper: result
                .conf_interval_upper
                .as_ref()
                .map(|c| c.iter().copied().collect())
                .unwrap_or_else(|| vec![f64::NAN; n_features]),
            confidence_level: options.confidence_level,
            f_statistic: Some(result.f_statistic),
            f_pvalue: Some(result.f_pvalue),
        })
    } else {
        None
    };

    Ok(TheilSenResult {
        fit: FitResult {
            core,
            inference,
            diagnostics: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn fits_clean_linear_data() {
        // y = 1 + 2x on clean data.
        let x = vec![(0..30).map(|i| i as f64 * 0.2).collect::<Vec<_>>()];
        let y: Vec<f64> = x[0].iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        let opts = TheilSenOptions {
            random_state: 42,
            ..TheilSenOptions::default()
        };
        let r = fit_theilsen(&y, &x, &opts).unwrap();

        assert!(
            approx(r.fit.core.coefficients[0], 2.0, 0.05),
            "expected slope ≈ 2, got {}",
            r.fit.core.coefficients[0]
        );
        assert!(approx(r.fit.core.intercept.unwrap(), 1.0, 0.1));
    }

    #[test]
    fn robust_against_outliers() {
        // 50 inliers on y = 1 + 2x plus 20 wild outliers.
        let n_in = 50usize;
        let n_out = 20usize;
        let mut xs: Vec<f64> = (0..n_in).map(|i| i as f64 * 0.2).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
        for i in 0..n_out {
            xs.push(i as f64 * 0.1);
            ys.push(50.0 + i as f64);
        }

        let x = vec![xs];
        let opts = TheilSenOptions {
            random_state: 42,
            ..TheilSenOptions::default()
        };
        let r = fit_theilsen(&ys, &x, &opts).unwrap();

        // Theil-Sen's high breakdown point should keep slope close to 2 despite
        // outliers; tolerance is wider than RANSAC because Theil-Sen blends
        // rather than excluding.
        assert!(
            approx(r.fit.core.coefficients[0], 2.0, 0.6),
            "expected slope ≈ 2 despite outliers, got {}",
            r.fit.core.coefficients[0]
        );
    }

    #[test]
    fn rejects_zero_max_subpopulation() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let opts = TheilSenOptions {
            max_subpopulation: 0,
            ..TheilSenOptions::default()
        };
        let err = fit_theilsen(&y, &x, &opts).unwrap_err();
        match err {
            StatsError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let err = fit_theilsen(&y, &x, &TheilSenOptions::default()).unwrap_err();
        match err {
            StatsError::DimensionMismatch { .. } => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }
}
