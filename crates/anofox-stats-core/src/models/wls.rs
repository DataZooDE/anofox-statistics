//! Weighted Least Squares (WLS) regression using native WlsRegressor

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResult, FitResultCore, FitResultInference, HcType, SolverType, WlsOptions};
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

/// Residual standard error of a weighted intercept-only fit, as R's
/// `summary(lm(y ~ 1, weights = w))$sigma`: `sqrt(sum(w (y - ybar_w)^2) / (n - 1))`.
fn intercept_only_weighted_sigma(rows: &[usize], y: &[f64], weights: &[f64], y_mean: f64) -> f64 {
    let n = rows.len();
    if n < 2 {
        return f64::NAN;
    }
    let rss: f64 = rows
        .iter()
        .map(|&i| weights[i] * (y[i] - y_mean).powi(2))
        .sum();
    (rss / (n - 1) as f64).sqrt()
}

/// Fit a Weighted Least Squares regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (p features, each with n observations)
/// * `weights` - Observation weights (n observations, must be positive)
/// * `options` - Fitting options
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_wls(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
) -> StatsResult<FitResult> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if weights.is_empty() {
        return Err(StatsError::EmptyInput { field: "weights" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    // Check weights length matches y
    if weights.len() != n_obs {
        return Err(StatsError::DimensionMismatch {
            y_len: n_obs,
            x_rows: weights.len(),
        });
    }

    // Check all feature vectors have same length as y
    crate::validation::validate_x_columns(n_obs, x)?;

    // Filter out rows with NaN/Inf values or non-positive weights
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && weights[i] > 0.0
                && !weights[i].is_nan()
                && !weights[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect();

    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();

    // Detect zero-variance (constant) columns BEFORE min_obs check
    // Constant columns are only dropped when an intercept is fitted (see
    // `validation::droppable_columns`); without one, a constant column IS the intercept.
    let is_constant_column: Vec<bool> =
        crate::validation::droppable_columns(x, &valid_indices, options.fit_intercept);

    // Count non-constant features for min_obs calculation
    let n_effective_features = is_constant_column.iter().filter(|&&c| !c).count();

    // Check we have enough observations for the effective (non-constant) features
    let min_obs = if options.fit_intercept {
        n_effective_features + 1
    } else {
        n_effective_features
    };

    // If ALL columns are constant, we can still fit (intercept-only model if fit_intercept=true)
    if n_effective_features == 0 {
        if !options.fit_intercept {
            return Err(StatsError::InsufficientData {
                rows: n_valid,
                cols: n_features,
            });
        }
        // Intercept-only model: compute weighted mean of y as intercept
        let sum_wy: f64 = valid_indices.iter().map(|&i| weights[i] * y[i]).sum();
        let sum_w: f64 = valid_indices.iter().map(|&i| weights[i]).sum();
        let y_mean = sum_wy / sum_w;
        // R lm(y ~ 1, weights = w): sigma^2 = sum(w (y - ybar_w)^2) / (n - 1)
        // (the residual degrees of freedom), NaN with a single observation.
        let rmse = intercept_only_weighted_sigma(&valid_indices, y, weights, y_mean);

        return Ok(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features],
                intercept: Some(y_mean),
                r_squared: 0.0,
                adj_r_squared: 0.0,
                residual_std_error: rmse,
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

    // Build reduced X matrix (only non-constant columns)
    let mut non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    // No intercept but a constant column (e.g. all ones) is present: that column
    // IS the intercept. The upstream solver would drop it, so fit the equivalent
    // intercept model without the constant column(s) and fold the intercept back
    // into its coefficient afterwards (see `validation::find_pseudo_intercept`).
    let pseudo_intercept = if options.fit_intercept {
        None
    } else {
        crate::validation::find_pseudo_intercept(x, &valid_indices, &non_constant_indices)
    };
    if pseudo_intercept.is_some() {
        non_constant_indices
            .retain(|&j| !crate::validation::is_constant_over(&x[j], &valid_indices));
    }
    let use_intercept = options.fit_intercept || pseudo_intercept.is_some();

    if let (Some((pcol, pval)), true) = (pseudo_intercept, non_constant_indices.is_empty()) {
        // Only constant columns: the model is y = beta * c, i.e. an intercept-only fit.
        let sum_wy: f64 = valid_indices.iter().map(|&i| weights[i] * y[i]).sum();
        let sum_w: f64 = valid_indices.iter().map(|&i| weights[i]).sum();
        let y_mean = sum_wy / sum_w;
        let rmse = intercept_only_weighted_sigma(&valid_indices, y, weights, y_mean);
        let mut coefficients = vec![f64::NAN; n_features];
        coefficients[pcol] = y_mean / pval;
        let mut core = FitResultCore {
            coefficients,
            intercept: None,
            r_squared: 0.0,
            adj_r_squared: 0.0,
            residual_std_error: rmse,
            n_observations: n_valid,
            n_features,
        };
        crate::validation::apply_no_intercept_fit_stats(
            &mut core,
            None,
            y,
            x,
            Some(weights),
            &valid_indices,
        );
        return Ok(FitResult {
            core,
            inference: None,
            diagnostics: None,
        });
    }

    // Convert to faer types (only non-constant columns, only valid rows)
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, non_constant_indices.len(), |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });
    let w_col = Col::from_fn(n_valid, |i| weights[valid_indices[i]]);

    // Build and fit the model using native WlsRegressor
    let fitted = WlsRegressor::builder()
        .with_intercept(use_intercept)
        .weights(w_col)
        .compute_inference(options.compute_inference || options.hc_type.is_some())
        .confidence_level(options.confidence_level)
        .solve_method(convert_solver(options.solver))
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract results
    let result = fitted.result();

    // Reconstruct full coefficient vector with NaN for constant columns
    let reduced_coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let mut coefficients = vec![f64::NAN; n_features];
    for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
        coefficients[orig_idx] = reduced_coefficients[reduced_idx];
    }
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    let mut core = FitResultCore {
        coefficients,
        intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: n_valid,
        n_features,
    };

    // Intercept statistics, used only to fold a pseudo-intercept back into its column.
    let mut pseudo_stats = crate::validation::InterceptStats {
        estimate: result.intercept.unwrap_or(f64::NAN),
        std_error: result.intercept_std_error.unwrap_or(f64::NAN),
        t_value: result.intercept_t_statistic.unwrap_or(f64::NAN),
        p_value: result.intercept_p_value.unwrap_or(f64::NAN),
        ci: result
            .intercept_conf_interval
            .unwrap_or((f64::NAN, f64::NAN)),
    };

    // Build inference results if requested
    let mut inference = if options.compute_inference {
        // Helper to reconstruct reduced vector to full size with NaN for constant columns
        let reconstruct = |reduced: Option<&faer::Col<f64>>| -> Vec<f64> {
            let mut full = vec![f64::NAN; n_features];
            if let Some(col) = reduced {
                for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
                    full[orig_idx] = col[reduced_idx];
                }
            }
            full
        };
        let reconstruct_col = |col: &faer::Col<f64>| -> Vec<f64> {
            let mut full = vec![f64::NAN; n_features];
            for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
                full[orig_idx] = col[reduced_idx];
            }
            full
        };

        // If HC inference is requested, use heteroscedasticity-consistent standard errors
        if let Some(hc_type) = options.hc_type {
            let hc_result = anofox_regression::inference::compute_hc_inference(
                &x_mat,
                &result.coefficients,
                result.intercept,
                &result.residuals,
                &result.aliased,
                use_intercept,
                convert_hc_type(hc_type),
                options.confidence_level,
            );

            match hc_result {
                Ok(hc) => Some({
                    if let Some(ic) = &hc.intercept {
                        pseudo_stats.std_error = ic.std_error;
                        pseudo_stats.t_value = ic.t_statistic;
                        pseudo_stats.p_value = ic.p_value;
                        pseudo_stats.ci = ic.conf_interval;
                    }
                    FitResultInference {
                        std_errors: reconstruct_col(&hc.std_errors),
                        t_values: reconstruct_col(&hc.t_statistics),
                        p_values: reconstruct_col(&hc.p_values),
                        ci_lower: reconstruct_col(&hc.conf_interval_lower),
                        ci_upper: reconstruct_col(&hc.conf_interval_upper),
                        confidence_level: hc.confidence_level,
                        f_statistic: Some(result.f_statistic),
                        f_pvalue: Some(result.f_pvalue),
                    }
                }),
                Err(_) => {
                    // Fall back to classical inference if HC fails
                    Some(FitResultInference {
                        std_errors: reconstruct(result.std_errors.as_ref()),
                        t_values: reconstruct(result.t_statistics.as_ref()),
                        p_values: reconstruct(result.p_values.as_ref()),
                        ci_lower: reconstruct(result.conf_interval_lower.as_ref()),
                        ci_upper: reconstruct(result.conf_interval_upper.as_ref()),
                        confidence_level: options.confidence_level,
                        f_statistic: Some(result.f_statistic),
                        f_pvalue: Some(result.f_pvalue),
                    })
                }
            }
        } else {
            // Classical inference from the native WLS fit
            Some(FitResultInference {
                std_errors: reconstruct(result.std_errors.as_ref()),
                t_values: reconstruct(result.t_statistics.as_ref()),
                p_values: reconstruct(result.p_values.as_ref()),
                ci_lower: reconstruct(result.conf_interval_lower.as_ref()),
                ci_upper: reconstruct(result.conf_interval_upper.as_ref()),
                confidence_level: options.confidence_level,
                f_statistic: Some(result.f_statistic),
                f_pvalue: Some(result.f_pvalue),
            })
        }
    } else {
        None
    };

    if let Some((pcol, pval)) = pseudo_intercept {
        crate::validation::fold_pseudo_intercept(
            &mut core,
            inference.as_mut(),
            pcol,
            pval,
            pseudo_stats,
        );
    }

    if !options.fit_intercept {
        crate::validation::apply_no_intercept_fit_stats(
            &mut core,
            inference.as_mut(),
            y,
            x,
            Some(weights),
            &valid_indices,
        );
    }

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
    fn test_wls_uniform_weights() {
        // With uniform weights, WLS should give same results as OLS
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0; 10]; // Uniform weights

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should be close to y = 2*x
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_wls_heteroscedastic() {
        // Data with increasing variance - higher weights for more reliable observations
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        // y = 2*x + noise, where noise variance increases with x
        let y = vec![2.1, 4.0, 5.9, 8.3, 9.5, 12.5, 13.0, 17.0, 17.0, 22.0];
        // Higher weights for observations with lower variance (early observations)
        let weights = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Coefficient should still be close to 2
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
    }

    #[test]
    fn test_wls_zero_weights() {
        // Zero weights should be filtered out
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        // Last two observations have zero weight - should be ignored
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should still get good fit on remaining 8 observations
        assert!(result.core.r_squared > 0.99);
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_negative_weights_filtered() {
        // Negative weights should be filtered out (not valid)
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let weights = vec![1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // 8 valid observations (2 with negative weights filtered)
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0]; // Wrong length

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_wls_with_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let options = WlsOptions {
            compute_inference: true,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!(result.inference.is_some());
        let inf = result.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        assert!(inf.p_values[0] < 0.05);
    }

    #[test]
    fn test_wls_hc1_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let options = WlsOptions {
            compute_inference: true,
            hc_type: Some(HcType::HC1),
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!(result.inference.is_some());
        let inf = result.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
    }

    #[test]
    fn test_wls_svd_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0; 10];

        let options = WlsOptions {
            solver: SolverType::Svd,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_wls_no_intercept_ones_column_matches_intercept_model() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];
        let w = vec![1.0; 8];
        let opts_int = WlsOptions {
            compute_inference: true,
            ..Default::default()
        };
        let with_int = fit_wls(&y, std::slice::from_ref(&x1), &w, &opts_int).unwrap();
        let opts = WlsOptions {
            fit_intercept: false,
            compute_inference: true,
            ..Default::default()
        };
        // Constant column of 2s: its coefficient is intercept / 2.
        let no_int = fit_wls(&y, &[vec![2.0; 8], x1], &w, &opts).unwrap();
        let b0 = no_int.core.coefficients[0];
        assert!((b0 * 2.0 - with_int.core.intercept.unwrap()).abs() < 1e-10);
        assert!((no_int.core.coefficients[1] - with_int.core.coefficients[0]).abs() < 1e-10);
        let inf = no_int.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        assert!(inf.ci_lower[0] < b0 && b0 < inf.ci_upper[0]);
    }
}
