//! Ordinary Least Squares (OLS) regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, HcType, OlsOptions, SolverType};
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

/// Convert our HcType to anofox_regression's HcType
fn convert_hc_type(hc: HcType) -> anofox_regression::HcType {
    match hc {
        HcType::HC0 => anofox_regression::HcType::HC0,
        HcType::HC1 => anofox_regression::HcType::HC1,
        HcType::HC2 => anofox_regression::HcType::HC2,
        HcType::HC3 => anofox_regression::HcType::HC3,
    }
}

/// Fit an OLS regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features)
/// * `options` - Fitting options
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_ols(y: &[f64], x: &[Vec<f64>], options: &OlsOptions) -> StatsResult<FitResult> {
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

    // Columns that cannot be estimated (constant with an intercept, all-zero
    // without) are left out of the design so they do not count against the
    // observation requirement; they are reported as NaN. Everything else —
    // aliasing, the no-intercept (uncentered) R²/F, intercept-only fits — is
    // upstream's.
    let dropped = crate::validation::droppable_columns(x, &valid_indices, options.fit_intercept);
    let kept: Vec<usize> = (0..n_features).filter(|&j| !dropped[j]).collect();
    if kept.is_empty() && !options.fit_intercept {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }
    if n_valid < kept.len() + usize::from(options.fit_intercept) {
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

    let fitted = OlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .compute_inference(options.compute_inference && options.hc_type.is_none())
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

    let inference = if !options.compute_inference {
        None
    } else if let Some(hc_type) = options.hc_type {
        match anofox_regression::inference::compute_hc_inference(
            &x_mat,
            &result.coefficients,
            result.intercept,
            &result.residuals,
            &result.aliased,
            options.fit_intercept,
            convert_hc_type(hc_type),
            options.confidence_level,
        ) {
            Ok(hc) => Some(FitResultInference {
                std_errors: expand(Some(&hc.std_errors)),
                t_values: expand(Some(&hc.t_statistics)),
                p_values: expand(Some(&hc.p_values)),
                ci_lower: expand(Some(&hc.conf_interval_lower)),
                ci_upper: expand(Some(&hc.conf_interval_upper)),
                confidence_level: hc.confidence_level,
                f_statistic: Some(result.f_statistic),
                f_pvalue: Some(result.f_pvalue),
            }),
            // The requested HC estimator is not available for this fit (e.g. a
            // leverage-1 observation for HC2/HC3). Substituting classical
            // standard errors would hand back numbers the caller did not ask
            // for, so the coefficient inference is NaN (NULL in SQL); the F
            // test does not depend on the HC estimator and is kept.
            Err(_) => Some(FitResultInference {
                std_errors: vec![f64::NAN; n_features],
                t_values: vec![f64::NAN; n_features],
                p_values: vec![f64::NAN; n_features],
                ci_lower: vec![f64::NAN; n_features],
                ci_upper: vec![f64::NAN; n_features],
                confidence_level: options.confidence_level,
                f_statistic: Some(result.f_statistic),
                f_pvalue: Some(result.f_pvalue),
            }),
        }
    } else {
        Some(FitResultInference {
            std_errors: expand(result.std_errors.as_ref()),
            t_values: expand(result.t_statistics.as_ref()),
            p_values: expand(result.p_values.as_ref()),
            ci_lower: expand(result.conf_interval_lower.as_ref()),
            ci_upper: expand(result.conf_interval_upper.as_ref()),
            confidence_level: options.confidence_level,
            f_statistic: Some(result.f_statistic),
            f_pvalue: Some(result.f_pvalue),
        })
    };

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
    fn test_simple_ols() {
        // Simple linear relationship: y = 2*x + 1
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_ols(&y, &x, &options).unwrap();

        // Check coefficient is approximately 2
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.01);
        // Check intercept is approximately 1
        assert!((result.core.intercept.unwrap() - 1.0).abs() < 0.01);
        // R-squared should be very high (perfect fit)
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ols_with_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_ols(&y, &x, &options).unwrap();

        // Should have inference results
        assert!(result.inference.is_some());
        let inference = result.inference.unwrap();

        // p-value should be significant (< 0.05)
        assert!(inference.p_values[0] < 0.05);
    }

    #[test]
    fn test_ols_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0]; // Wrong length

        let options = OlsOptions::default();
        let result = fit_ols(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ols_insufficient_data() {
        // With fit_intercept=true and 3 non-constant features, we need n >= 4
        // (1 intercept + 3 coefficients = 4 parameters)
        // With n=2, we can't fit 4 parameters
        let x = vec![
            vec![1.0, 2.0], // 2 observations, varying - not constant
            vec![3.0, 4.0], // varying - not constant
            vec![5.0, 6.0], // varying - not constant
        ]; // 3 features
        let y = vec![1.0, 2.0];

        let options = OlsOptions {
            fit_intercept: true, // Needs n >= p + 1 = 4
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InsufficientData { .. })));
    }

    #[test]
    fn test_ols_exact_fit() {
        // Test that n == p+1 (exact fit, 0 degrees of freedom) is now allowed
        let x = vec![vec![1.0, 2.0]]; // 2 observations
        let y = vec![1.0, 3.0]; // y = 2*x - 1

        let options = OlsOptions {
            fit_intercept: true, // 2 parameters (intercept + 1 coef)
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options);

        // Should succeed (exact fit is now allowed)
        assert!(result.is_ok());
        let result = result.unwrap();
        // With 2 points, coefficient should be (3-1)/(2-1) = 2
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.01);
        // Intercept should be 1 - 2*1 = -1
        assert!((result.core.intercept.unwrap() - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_ols_svd_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = OlsOptions {
            solver: SolverType::Svd,
            ..Default::default()
        };

        let result = fit_ols(&y, &x, &options).unwrap();
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ols_cholesky_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = OlsOptions {
            solver: SolverType::Cholesky,
            ..Default::default()
        };

        let result = fit_ols(&y, &x, &options).unwrap();
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_ols_hc1_inference() {
        // HC1 (default) heteroscedasticity-consistent standard errors
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            hc_type: Some(HcType::HC1),
            ..Default::default()
        };

        let result = fit_ols(&y, &x, &options).unwrap();
        assert!(result.inference.is_some());
        let inf = result.inference.unwrap();
        // HC standard errors should be finite and positive
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        // p-value should be significant
        assert!(inf.p_values[0] < 0.05);
    }

    /// A saturated fit (n == p) has no HC covariance. The requested estimator
    /// must not be silently replaced by classical standard errors: coefficient
    /// inference is NaN instead.
    #[test]
    fn test_ols_hc_failure_reports_nan_not_classical() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![3.0, 5.5];
        let options = OlsOptions {
            fit_intercept: true,
            compute_inference: true,
            hc_type: Some(HcType::HC1),
            ..Default::default()
        };
        let result = fit_ols(&y, &x, &options).unwrap();
        let inf = result.inference.unwrap();
        assert!(inf.std_errors.iter().all(|v| v.is_nan()));
        assert!(inf.t_values.iter().all(|v| v.is_nan()));
        assert!(inf.p_values.iter().all(|v| v.is_nan()));
        assert!(inf.ci_lower.iter().all(|v| v.is_nan()));
        assert!(inf.ci_upper.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_ols_hc3_inference() {
        // HC3 (jackknife-like) should produce larger SE than classical
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];

        let classical = OlsOptions {
            compute_inference: true,
            hc_type: None,
            ..Default::default()
        };
        let hc3 = OlsOptions {
            compute_inference: true,
            hc_type: Some(HcType::HC3),
            ..Default::default()
        };

        let result_classical = fit_ols(&y, &x, &classical).unwrap();
        let result_hc3 = fit_ols(&y, &x, &hc3).unwrap();

        let se_classical = result_classical.inference.unwrap().std_errors[0];
        let se_hc3 = result_hc3.inference.unwrap().std_errors[0];

        // Both should be finite and positive
        assert!(se_classical.is_finite() && se_classical > 0.0);
        assert!(se_hc3.is_finite() && se_hc3 > 0.0);
        // HC3 and classical should differ (HC3 is typically larger but not guaranteed)
        assert!((se_hc3 - se_classical).abs() > 1e-15);
    }

    #[test]
    fn test_no_intercept_ones_column_reproduces_intercept_model() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];

        let with_int = fit_ols(&y, std::slice::from_ref(&x1), &OlsOptions::default()).unwrap();

        // No intercept, but an explicit all-ones column: it must NOT be dropped.
        let opts = OlsOptions {
            fit_intercept: false,
            ..Default::default()
        };
        let no_int = fit_ols(&y, &[vec![1.0; 8], x1], &opts).unwrap();

        assert!(no_int.core.intercept.is_none());
        let b0 = no_int.core.coefficients[0];
        let b1 = no_int.core.coefficients[1];
        assert!(b0.is_finite(), "ones column was dropped");
        assert!((b0 - with_int.core.intercept.unwrap()).abs() < 1e-10);
        assert!((b1 - with_int.core.coefficients[0]).abs() < 1e-10);
        assert!((no_int.core.residual_std_error - with_int.core.residual_std_error).abs() < 1e-10);
    }

    #[test]
    fn test_no_intercept_all_constant_columns_fit_normally() {
        // Only a constant column and no intercept: it is the intercept → mean(y).
        let y = vec![1.0, 2.0, 3.0, 6.0];
        let opts = OlsOptions {
            fit_intercept: false,
            ..Default::default()
        };
        let r = fit_ols(&y, &[vec![2.0; 4]], &opts).unwrap();
        assert!((r.core.coefficients[0] - 1.5).abs() < 1e-10); // 2 * 1.5 = mean 3
    }

    #[test]
    fn test_tiny_unit_column_is_not_dropped() {
        // Values on the order of 1e-8 vary in relative terms; must be kept.
        let x = vec![vec![1e-8, 2e-8, 3e-8, 4e-8, 5e-8]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2e8 * x + 1
        let r = fit_ols(&y, &x, &OlsOptions::default()).unwrap();
        assert!(
            r.core.coefficients[0].is_finite(),
            "tiny-unit column dropped"
        );
        assert!((r.core.coefficients[0] / 2e8 - 1.0).abs() < 1e-8);
        assert!((r.core.intercept.unwrap() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_intercept_only_single_observation() {
        let r = fit_ols(&[4.0], &[vec![1.0]], &OlsOptions::default()).unwrap();
        assert_eq!(r.core.intercept, Some(4.0));
        assert!(r.core.residual_std_error.is_nan());
    }

    #[test]
    fn test_ols_mismatched_second_column() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let r = fit_ols(&y, &x, &OlsOptions::default());
        assert!(matches!(r, Err(StatsError::DimensionMismatch { .. })));
    }
}
