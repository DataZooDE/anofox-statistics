//! Ridge Regression (L2 regularization) wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    FitResult, FitResultCore, FitResultInference, LambdaScaling, RidgeOptions, SolverType,
};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Convert our SolverType to anofox_regression's SolverType
fn convert_solver(solver: SolverType) -> anofox_regression::SolverType {
    match solver {
        SolverType::Qr => anofox_regression::SolverType::Qr,
        SolverType::Svd => anofox_regression::SolverType::Svd,
        SolverType::Cholesky => anofox_regression::SolverType::Cholesky,
    }
}

/// Convert our LambdaScaling to anofox_regression's LambdaScaling
fn convert_lambda_scaling(scaling: LambdaScaling) -> anofox_regression::LambdaScaling {
    match scaling {
        LambdaScaling::Raw => anofox_regression::LambdaScaling::Raw,
        LambdaScaling::Glmnet => anofox_regression::LambdaScaling::Glmnet,
    }
}

/// Fit a Ridge regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features)
/// * `options` - Fitting options including alpha (L2 penalty)
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_ridge(y: &[f64], x: &[Vec<f64>], options: &RidgeOptions) -> StatsResult<FitResult> {
    // Validate alpha parameter
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }

    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    // Check all feature vectors have same length as y
    crate::validation::validate_x_columns(n_obs, x)?;

    // Filter out rows with NaN values
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

    // Non-estimable columns (constant with an intercept, all-zero without) are
    // left out of the design and reported as NaN; intercept-only fits and
    // (for alpha == 0) constant columns without an intercept are upstream's.
    let dropped = crate::validation::droppable_columns(x, &valid_indices, options.fit_intercept);
    let kept: Vec<usize> = (0..n_features).filter(|&j| !dropped[j]).collect();
    if (kept.is_empty() && !options.fit_intercept)
        || n_valid < kept.len() + usize::from(options.fit_intercept)
    {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }
    if kept.is_empty() && n_valid == 1 {
        // A single observation: the intercept is that value, its residual
        // variance is undefined (upstream needs two rows).
        return Ok(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features],
                intercept: Some(y[valid_indices[0]]),
                r_squared: 0.0,
                adj_r_squared: 0.0,
                residual_std_error: f64::NAN,
                n_observations: 1,
                n_features,
            },
            inference: None,
            diagnostics: None,
        });
    }

    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, kept.len(), |i, j| x[kept[j]][valid_indices[i]]);

    let fitted = RidgeRegressor::builder()
        .with_intercept(options.fit_intercept)
        .lambda(options.alpha)
        .lambda_scaling(convert_lambda_scaling(options.lambda_scaling))
        .compute_inference(options.compute_inference)
        .confidence_level(options.confidence_level)
        .solve_method(convert_solver(options.solver))
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let result = fitted.result();

    // Scatter a reduced (kept-columns) vector back to full width, NaN elsewhere.
    let expand = |reduced: Option<&Col<f64>>| -> Vec<f64> {
        let mut full = vec![f64::NAN; n_features];
        if let Some(col) = reduced {
            for (r, &j) in kept.iter().enumerate() {
                full[j] = col[r];
            }
        }
        full
    };

    let core = FitResultCore {
        coefficients: expand(Some(&result.coefficients)),
        intercept: result.intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: n_valid,
        n_features,
    };

    // Upstream reports, for alpha > 0, the ridge sandwich standard errors
    // sigma * sqrt(diag(A X'X A)), A = (X'X + lambda P)^-1, and leaves t, p and
    // the confidence interval unset: a t-test centred on the shrunken, biased
    // coefficient is not a valid test (NaN, i.e. NULL in SQL). The overall F
    // test is likewise not reported for alpha > 0. alpha == 0 is plain OLS.
    let inference = options.compute_inference.then(|| FitResultInference {
        std_errors: expand(result.std_errors.as_ref()),
        t_values: expand(result.t_statistics.as_ref()),
        p_values: expand(result.p_values.as_ref()),
        ci_lower: expand(result.conf_interval_lower.as_ref()),
        ci_upper: expand(result.conf_interval_upper.as_ref()),
        confidence_level: options.confidence_level,
        f_statistic: (options.alpha == 0.0).then_some(result.f_statistic),
        f_pvalue: (options.alpha == 0.0).then_some(result.f_pvalue),
    });

    Ok(FitResult {
        core,
        inference,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression() {
        // Simple linear relationship with some noise
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = RidgeOptions {
            alpha: 0.1, // Small regularization
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_ridge(&y, &x, &options).unwrap();

        // Ridge should shrink coefficients slightly compared to OLS
        // Coefficient should be close to 2, intercept close to 0
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
        assert!(result.core.r_squared > 0.95);
    }

    #[test]
    fn test_ridge_invalid_alpha() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = RidgeOptions {
            alpha: -1.0, // Invalid negative alpha
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_ridge(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidAlpha(_))));
    }

    #[test]
    fn test_ridge_perfect_fit() {
        // Test with perfect fit data (y = 2*x exactly)
        // This is a regression test for the panic in statrs beta function
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let options = RidgeOptions {
            alpha: 0.1,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_ridge(&y, &x, &options).unwrap();

        // Should work without panicking even with perfect fit
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ridge_perfect_fit_minimal() {
        // Test with exact minimal data (same as DuckDB failing query)
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let options = RidgeOptions {
            alpha: 0.1,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        // Should not panic even with minimal data
        let result = fit_ridge(&y, &x, &options).unwrap();
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ridge_high_regularization() {
        // High regularization should shrink coefficients more
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let low_reg = RidgeOptions {
            alpha: 0.001,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let high_reg = RidgeOptions {
            alpha: 10.0,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result_low = fit_ridge(&y, &x, &low_reg).unwrap();
        let result_high = fit_ridge(&y, &x, &high_reg).unwrap();

        // Higher regularization = smaller coefficient magnitude
        assert!(result_high.core.coefficients[0].abs() < result_low.core.coefficients[0].abs());
    }

    #[test]
    fn test_ridge_svd_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = RidgeOptions {
            alpha: 0.1,
            solver: SolverType::Svd,
            ..Default::default()
        };

        let result = fit_ridge(&y, &x, &options).unwrap();
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
        assert!(result.core.r_squared > 0.95);
    }

    #[test]
    fn test_ridge_cholesky_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        // Cholesky is ideal for Ridge since L2 guarantees positive definite X'X + λI
        let options = RidgeOptions {
            alpha: 0.1,
            solver: SolverType::Cholesky,
            ..Default::default()
        };

        let result = fit_ridge(&y, &x, &options).unwrap();
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
        assert!(result.core.r_squared > 0.95);
    }

    #[test]
    fn test_ridge_glmnet_scaling() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let raw = RidgeOptions {
            alpha: 1.0,
            lambda_scaling: LambdaScaling::Raw,
            ..Default::default()
        };

        let glmnet = RidgeOptions {
            alpha: 1.0,
            lambda_scaling: LambdaScaling::Glmnet,
            ..Default::default()
        };

        let result_raw = fit_ridge(&y, &x, &raw).unwrap();
        let result_glmnet = fit_ridge(&y, &x, &glmnet).unwrap();

        // Verify that different lambda scaling modes produce different coefficients
        assert!(
            (result_raw.core.coefficients[0] - result_glmnet.core.coefficients[0]).abs() > 1e-10,
            "Raw and Glmnet scaling should produce different coefficients"
        );
    }
}
