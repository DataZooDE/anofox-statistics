//! Huber M-estimator robust regression wrapper.
//!
//! Wraps `anofox_regression::solvers::HuberRegressor` and reshapes its result
//! into the workspace-local `FitResult` plus a small Huber-specific extension
//! carrying the MAD-based scale estimate and outlier mask.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, HuberOptions, ModelSummary};
use anofox_regression::core::HasModelInfo;
use anofox_regression::solvers::{FittedRegressor, HuberRegressor, Regressor};
use faer::{Col, Mat};

/// Huber regression fit, bundling the standard `FitResult` with the
/// Huber-specific scale and outlier diagnostics produced by IRLS.
#[derive(Debug, Clone)]
pub struct HuberResult {
    /// Standard fit result (coefficients, R², optional inference).
    pub fit: FitResult,
    /// MAD-based scale estimate (sigma).
    pub scale: f64,
    /// Epsilon parameter that was used (echoed back for reproducibility).
    pub epsilon: f64,
    /// Per-observation outlier mask (true where |r_i| > epsilon * scale).
    /// Length matches the number of *non-NaN* observations actually used in
    /// the fit, not the original input length.
    pub outliers: Vec<bool>,
    /// Number of observations flagged as outliers.
    pub n_outliers: usize,
}

/// Fit a Huber M-estimator regression model.
pub fn fit_huber(y: &[f64], x: &[Vec<f64>], options: &HuberOptions) -> StatsResult<HuberResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if options.epsilon <= 1.0 {
        return Err(StatsError::InvalidInput(format!(
            "epsilon must be > 1.0, got {}",
            options.epsilon
        )));
    }
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }

    let n_obs = y.len();
    let n_features = x.len();

    crate::validation::validate_x_columns(n_obs, x)?;

    // NaN / infinite filtering — identical policy to OLS so per-group
    // call sites can swap estimators transparently.
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
    let min_obs = if options.fit_intercept {
        n_features + 2
    } else {
        n_features + 1
    };
    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    let fitted = HuberRegressor::builder()
        .epsilon(options.epsilon)
        .alpha(options.alpha)
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .compute_inference(options.compute_inference)
        .confidence_level(options.confidence_level)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;

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

    // MASS `summary.rlm(method = "XtX")` covariance, computed upstream; NaN
    // (NULL in SQL) where it is unavailable.
    let inference = options
        .compute_inference
        .then(|| super::inference_from_result(result, n_features, options.confidence_level));

    let outliers = fitted.outliers().to_vec();
    let n_outliers = fitted.n_outliers();
    let scale = fitted.scale();
    let epsilon = fitted.epsilon();

    Ok(HuberResult {
        fit: FitResult {
            core,
            inference,
            diagnostics: None,
            summary: ModelSummary::from_result(
                fitted.model_info(),
                result,
                options.compute_inference,
            ),
        },
        scale,
        epsilon,
        outliers,
        n_outliers,
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
        // y = 1 + 2*x with no outliers — Huber should match OLS closely.
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];
        let r = fit_huber(&y, &x, &HuberOptions::default()).unwrap();

        assert!(approx(r.fit.core.coefficients[0], 2.0, 0.05));
        assert!(approx(r.fit.core.intercept.unwrap(), 1.0, 0.1));
        assert_eq!(r.n_outliers, 0);
    }

    #[test]
    fn downweights_outliers() {
        // y = 2*x exactly, except for two extreme outliers. OLS would be
        // pulled toward the outliers; Huber should recover slope ≈ 2.
        let x = vec![vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ]];
        let mut y: Vec<f64> = x[0].iter().map(|&v| 2.0 * v).collect();
        // Inject two outliers that an OLS fit would chase.
        y[5] = 200.0;
        y[10] = -150.0;

        let r = fit_huber(&y, &x, &HuberOptions::default()).unwrap();

        // Slope should still be near 2 despite the outliers.
        assert!(
            approx(r.fit.core.coefficients[0], 2.0, 0.5),
            "expected slope ≈ 2, got {}",
            r.fit.core.coefficients[0]
        );
        assert!(
            r.n_outliers >= 2,
            "expected at least 2 outliers detected, got {}",
            r.n_outliers
        );
        assert!(r.outliers[5]);
        assert!(r.outliers[10]);
    }

    #[test]
    fn rejects_epsilon_below_one() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&v| 2.0 * v).collect();
        let opts = HuberOptions {
            epsilon: 0.5,
            ..HuberOptions::default()
        };
        let err = fit_huber(&y, &x, &opts).unwrap_err();
        match err {
            StatsError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let err = fit_huber(&y, &x, &HuberOptions::default()).unwrap_err();
        match err {
            StatsError::DimensionMismatch { .. } => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// Coefficients, scale and standard errors against R:
    /// `MASS::rlm(y ~ x, psi = psi.huber, k = 1.35, scale.est = "MAD")` and
    /// `summary(fit, method = "XtX")`.
    #[test]
    fn inference_matches_mass_rlm() {
        let xs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut y: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + (((x as i64 * 7) % 5) as f64 - 2.0) * 0.4)
            .collect();
        y[5] = 40.0;
        y[14] = 5.0;
        let opts = HuberOptions {
            alpha: 0.0,
            max_iterations: 500,
            tolerance: 1e-12,
            compute_inference: true,
            ..HuberOptions::default()
        };
        let r = fit_huber(&y, &[xs], &opts).unwrap();
        let inf = r.fit.inference.as_ref().unwrap();
        eprintln!(
            "coef {:?} icpt {:?} scale {} se {:?}",
            r.fit.core.coefficients, r.fit.core.intercept, r.scale, inf.std_errors
        );
        assert!(approx(r.fit.core.coefficients[0], 1.97692968311, 1e-6));
        assert!(approx(r.fit.core.intercept.unwrap(), 1.28668277180, 1e-5));
        assert!(approx(inf.std_errors[0], 0.0296479346015, 1e-6));
        assert!(approx(inf.t_values[0], 66.68018226841, 1e-2));
    }
}
