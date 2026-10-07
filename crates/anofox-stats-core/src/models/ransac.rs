//! RANSAC (RAndom SAmple Consensus) robust regression wrapper.
//!
//! Wraps `anofox_regression::solvers::RansacRegressor` and reshapes its
//! result into the workspace-local `FitResult` plus the RANSAC-specific
//! diagnostics (inlier mask, trial count, residual threshold actually used).

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, RansacOptions};
use anofox_regression::solvers::{FittedRegressor, RansacRegressor, Regressor};
use faer::{Col, Mat};

/// RANSAC regression fit, bundling the standard `FitResult` with the
/// RANSAC-specific consensus-set diagnostics.
#[derive(Debug, Clone)]
pub struct RansacResult {
    /// Standard fit result (final inlier-only OLS coefficients + statistics).
    pub fit: FitResult,
    /// Per-observation inlier mask (true if the observation was part of the
    /// final consensus set). Length matches the number of *non-NaN* rows
    /// actually used in the fit, not the original input length.
    pub inliers: Vec<bool>,
    /// Number of observations classified as inliers in the final consensus.
    pub n_inliers: usize,
    /// Actual number of RANSAC trials run before early termination.
    pub n_trials: usize,
    /// Residual threshold the algorithm used (either the user-supplied value
    /// or the MAD(y) default).
    pub residual_threshold: f64,
}

/// Fit a RANSAC robust regression model.
/// Inlier count above which the OLS refit avoids the n x n SVD/QR factors.
const LARGE_REFIT_ROWS: usize = 10_000;

pub fn fit_ransac(y: &[f64], x: &[Vec<f64>], options: &RansacOptions) -> StatsResult<RansacResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if options.max_trials == 0 {
        return Err(StatsError::InvalidInput(
            "max_trials must be > 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&options.stop_probability) {
        return Err(StatsError::InvalidInput(format!(
            "stop_probability must be in [0, 1], got {}",
            options.stop_probability
        )));
    }
    if let Some(thr) = options.residual_threshold {
        if !thr.is_finite() || thr <= 0.0 {
            return Err(StatsError::InvalidInput(format!(
                "residual_threshold must be finite and positive, got {}",
                thr
            )));
        }
    }

    let n_obs = y.len();
    let n_features = x.len();

    crate::validation::validate_x_columns(n_obs, x)?;

    // Same NaN/infinite filtering policy as OLS/Huber so per-group call
    // sites can swap estimators transparently.
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

    // RANSAC needs at least min_samples observations; if the user did not
    // override min_samples, the upstream solver picks n_features + 1
    // (or +0 without intercept). Surface a clear error in either case
    // rather than letting the upstream return InsufficientObservations.
    let effective_min_samples = options.min_samples.unwrap_or(if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    });
    if n_valid < effective_min_samples {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    let mut builder = RansacRegressor::builder()
        .with_intercept(options.fit_intercept)
        .max_trials(options.max_trials as usize)
        .stop_probability(options.stop_probability)
        .random_state(options.random_state);
    if let Some(s) = options.min_samples {
        builder = builder.min_samples(s);
    }
    if let Some(thr) = options.residual_threshold {
        builder = builder.residual_threshold(thr);
    }
    if let Some(n) = options.stop_n_inliers {
        builder = builder.stop_n_inliers(n);
    }

    let fitted = builder
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

    let inliers = fitted.inlier_mask().to_vec();

    // The upstream result leaves adj_r_squared and rmse at 0 and the inference
    // lists empty. RANSAC's final model is the OLS fit on the inliers, so the
    // summary statistics and inference come from refitting OLS on those rows.
    // The inference ignores that the inliers were selected from the data, so it
    // is optimistic (see CHANGELOG).
    let y_in: Vec<f64> = (0..n_valid)
        .filter(|&i| inliers.get(i).copied().unwrap_or(false))
        .map(|i| y[valid_indices[i]])
        .collect();
    let x_in: Vec<Vec<f64>> = (0..n_features)
        .map(|j| {
            (0..n_valid)
                .filter(|&i| inliers.get(i).copied().unwrap_or(false))
                .map(|i| x[j][valid_indices[i]])
                .collect()
        })
        .collect();
    let refit = crate::models::ols::fit_ols(
        &y_in,
        &x_in,
        &crate::types::OlsOptions {
            fit_intercept: options.fit_intercept,
            compute_inference: options.compute_inference,
            confidence_level: options.confidence_level,
            // anofox-regression's SVD and QR solvers materialise a full n x n
            // factor (U / Q), which cannot be allocated for large inlier sets
            // (~300k rows -> 700 GB). Above a modest size use the normal
            // equations (Cholesky), which are O(n p^2) in time and O(p^2) in memory.
            solver: if y_in.len() > LARGE_REFIT_ROWS {
                crate::types::SolverType::Cholesky
            } else {
                crate::types::SolverType::default()
            },
            ..Default::default()
        },
    )
    .ok();

    let core = FitResultCore {
        coefficients,
        intercept,
        r_squared: refit
            .as_ref()
            .map(|r| r.core.r_squared)
            .unwrap_or(result.r_squared),
        adj_r_squared: refit
            .as_ref()
            .map(|r| r.core.adj_r_squared)
            .unwrap_or(f64::NAN),
        residual_std_error: refit
            .as_ref()
            .map(|r| r.core.residual_std_error)
            .unwrap_or(f64::NAN),
        n_observations: n_valid,
        n_features,
    };

    let inference = if options.compute_inference {
        refit.and_then(|r| r.inference)
    } else {
        None
    };

    let n_inliers = fitted.n_inliers();
    let n_trials = fitted.n_trials();
    let residual_threshold = fitted.residual_threshold();

    Ok(RansacResult {
        fit: FitResult {
            core,
            inference,
            diagnostics: None,
        },
        inliers,
        n_inliers,
        n_trials,
        residual_threshold,
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
        // y = 1 + 2x on clean data — RANSAC should match OLS closely.
        let x = vec![(0..30).map(|i| i as f64 * 0.2).collect::<Vec<_>>()];
        let y: Vec<f64> = x[0].iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        let opts = RansacOptions {
            random_state: 42,
            residual_threshold: Some(0.5),
            ..RansacOptions::default()
        };
        let r = fit_ransac(&y, &x, &opts).unwrap();

        assert!(
            approx(r.fit.core.coefficients[0], 2.0, 0.05),
            "expected slope ≈ 2, got {}",
            r.fit.core.coefficients[0]
        );
        assert!(approx(r.fit.core.intercept.unwrap(), 1.0, 0.1));
        assert_eq!(r.n_inliers, 30, "all clean points should be inliers");
    }

    #[test]
    fn ignores_extreme_outliers() {
        // 50 inliers on y = 1 + 2x, plus 20 wild outliers around y ≈ 50.
        let n_in = 50usize;
        let n_out = 20usize;
        let mut xs: Vec<f64> = (0..n_in).map(|i| i as f64 * 0.2).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
        for i in 0..n_out {
            xs.push(i as f64 * 0.1);
            ys.push(50.0 + i as f64);
        }

        let x = vec![xs];
        let opts = RansacOptions {
            random_state: 42,
            residual_threshold: Some(0.5),
            max_trials: 200,
            ..RansacOptions::default()
        };
        let r = fit_ransac(&ys, &x, &opts).unwrap();

        assert!(
            approx(r.fit.core.coefficients[0], 2.0, 0.2),
            "expected slope ≈ 2 despite outliers, got {}",
            r.fit.core.coefficients[0]
        );
        // Inliers should be roughly the 50 clean points, not the 20 outliers.
        assert!(
            r.n_inliers >= n_in - 2 && r.n_inliers <= n_in + 2,
            "expected ≈ {} inliers, got {}",
            n_in,
            r.n_inliers
        );
        assert!(r.n_trials >= 1);
        assert!(r.residual_threshold > 0.0);
    }

    #[test]
    fn rejects_zero_max_trials() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let opts = RansacOptions {
            max_trials: 0,
            ..RansacOptions::default()
        };
        let err = fit_ransac(&y, &x, &opts).unwrap_err();
        match err {
            StatsError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let err = fit_ransac(&y, &x, &RansacOptions::default()).unwrap_err();
        match err {
            StatsError::DimensionMismatch { .. } => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }
}
