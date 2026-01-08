//! Generalized Linear Models (GLM) - Poisson, Binomial, Negative Binomial, Tweedie

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    BinomialLink, BinomialOptions, GlmFitResult, GlmInferenceResult, NegBinomialOptions,
    PoissonLink, PoissonOptions, TweedieOptions,
};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Combined GLM result with optional inference
#[derive(Debug, Clone)]
pub struct GlmResult {
    pub core: GlmFitResult,
    pub inference: Option<GlmInferenceResult>,
}

/// Fit a Poisson regression model (for count data)
///
/// # Arguments
/// * `y` - Response variable (counts, must be non-negative integers)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_poisson(y: &[f64], x: &[Vec<f64>], options: &PoissonOptions) -> StatsResult<GlmResult> {
    validate_inputs(y, x)?;

    let n_features = x.len();

    // Check for non-negative y values
    for &val in y.iter() {
        if val < 0.0 {
            return Err(StatsError::InvalidValue {
                field: "y",
                message: "Poisson regression requires non-negative response values".to_string(),
            });
        }
    }

    // Filter out rows with NaN/Inf values
    let valid_indices = get_valid_indices(y, x);
    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();

    // Detect zero-variance (constant) columns BEFORE min_obs check
    let is_constant_column: Vec<bool> = x
        .iter()
        .map(|col| {
            if valid_indices.is_empty() {
                return true;
            }
            let first_val = col[valid_indices[0]];
            valid_indices
                .iter()
                .all(|&i| (col[i] - first_val).abs() < 1e-10)
        })
        .collect();

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
        // Intercept-only model: compute log of mean of y as intercept (Poisson log link)
        let y_mean = valid_indices.iter().map(|&i| y[i]).sum::<f64>() / n_valid as f64;

        let core = GlmFitResult {
            coefficients: vec![f64::NAN; n_features],
            intercept: Some(y_mean.ln()),
            null_deviance: f64::NAN,
            residual_deviance: f64::NAN,
            pseudo_r_squared: 0.0,
            aic: f64::NAN,
            n_observations: n_valid,
            n_features,
            iterations: 0,
            converged: true,
            dispersion: Some(1.0),
        };

        return Ok(GlmResult {
            core,
            inference: None,
        });
    }

    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Build reduced X matrix (only non-constant columns)
    let non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    // Convert to faer types (only non-constant columns)
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });

    // Build regressor based on link function
    let fitted = match options.link {
        PoissonLink::Log => PoissonRegressor::log()
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations as usize)
            .tolerance(options.tolerance)
            .confidence_level(options.confidence_level)
            .build()
            .fit(&x_mat, &y_col),
        PoissonLink::Identity => PoissonRegressor::identity()
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations as usize)
            .tolerance(options.tolerance)
            .confidence_level(options.confidence_level)
            .build()
            .fit(&x_mat, &y_col),
        PoissonLink::Sqrt => PoissonRegressor::sqrt()
            .with_intercept(options.fit_intercept)
            .max_iterations(options.max_iterations as usize)
            .tolerance(options.tolerance)
            .confidence_level(options.confidence_level)
            .build()
            .fit(&x_mat, &y_col),
    }
    .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract GLM-specific results from FittedPoisson
    let result = fitted.result();

    // Reconstruct full coefficient vector with NaN for constant columns
    let reduced_coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let mut coefficients = vec![f64::NAN; n_features];
    for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
        coefficients[orig_idx] = reduced_coefficients[reduced_idx];
    }
    let intercept = result.intercept;

    // Calculate pseudo R-squared
    let pseudo_r_squared = if fitted.null_deviance > 0.0 {
        1.0 - fitted.deviance / fitted.null_deviance
    } else {
        0.0
    };

    let core = GlmFitResult {
        coefficients,
        intercept,
        null_deviance: fitted.null_deviance,
        residual_deviance: fitted.deviance,
        pseudo_r_squared,
        aic: result.aic,
        n_observations: n_valid,
        n_features,
        iterations: fitted.iterations as u32,
        converged: true, // IRLS converged if we got here
        dispersion: Some(fitted.dispersion),
    };

    let inference = if options.compute_inference {
        extract_inference_with_nan(
            result,
            &non_constant_indices,
            n_features,
            options.confidence_level,
        )
    } else {
        None
    };

    Ok(GlmResult { core, inference })
}

/// Fit a Binomial (Logistic) regression model (for binary outcomes)
///
/// # Arguments
/// * `y` - Response variable (0 or 1 for binary, or proportion in [0,1])
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &BinomialOptions,
) -> StatsResult<GlmResult> {
    validate_inputs(y, x)?;

    let n_features = x.len();

    // Check y values are in [0, 1]
    for &val in y.iter() {
        if !(0.0..=1.0).contains(&val) {
            return Err(StatsError::InvalidValue {
                field: "y",
                message: "Binomial regression requires y values in [0, 1]".to_string(),
            });
        }
    }

    // Filter out rows with NaN/Inf values
    let valid_indices = get_valid_indices(y, x);
    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();
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

    // Map our link function to the library's - only 3 supported
    let link = match options.link {
        BinomialLink::Logit => anofox_regression::core::BinomialLink::Logit,
        BinomialLink::Probit => anofox_regression::core::BinomialLink::Probit,
        BinomialLink::Cloglog => anofox_regression::core::BinomialLink::Cloglog,
        // Cauchit and Log not supported by upstream, fall back to Logit
        BinomialLink::Cauchit | BinomialLink::Log => {
            return Err(StatsError::InvalidValue {
                field: "link",
                message: "Cauchit and Log links are not supported. Use Logit, Probit, or Cloglog."
                    .to_string(),
            });
        }
    };

    let fitted = BinomialRegressor::builder()
        .link(link)
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .confidence_level(options.confidence_level)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract GLM-specific results from FittedBinomial
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = result.intercept;

    // Calculate pseudo R-squared
    let pseudo_r_squared = if fitted.null_deviance > 0.0 {
        1.0 - fitted.deviance / fitted.null_deviance
    } else {
        0.0
    };

    let core = GlmFitResult {
        coefficients,
        intercept,
        null_deviance: fitted.null_deviance,
        residual_deviance: fitted.deviance,
        pseudo_r_squared,
        aic: result.aic,
        n_observations: n_valid,
        n_features,
        iterations: fitted.iterations as u32,
        converged: true,
        dispersion: Some(fitted.dispersion),
    };

    let inference = if options.compute_inference {
        extract_inference(result, options.confidence_level)
    } else {
        None
    };

    Ok(GlmResult { core, inference })
}

/// Fit a Negative Binomial regression model (for overdispersed count data)
///
/// # Arguments
/// * `y` - Response variable (counts, must be non-negative)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &NegBinomialOptions,
) -> StatsResult<GlmResult> {
    validate_inputs(y, x)?;

    let n_features = x.len();

    // Check for non-negative y values
    for &val in y.iter() {
        if val < 0.0 {
            return Err(StatsError::InvalidValue {
                field: "y",
                message: "Negative Binomial regression requires non-negative response values"
                    .to_string(),
            });
        }
    }

    // Filter out rows with NaN/Inf values
    let valid_indices = get_valid_indices(y, x);
    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();
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

    // Build regressor
    // Note: NegativeBinomialRegressor estimates alpha (dispersion) from data
    let fitted = NegativeBinomialRegressor::builder()
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .confidence_level(options.confidence_level)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract GLM-specific results
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = result.intercept;

    // Calculate pseudo R-squared
    let pseudo_r_squared = if fitted.null_deviance > 0.0 {
        1.0 - fitted.deviance / fitted.null_deviance
    } else {
        0.0
    };

    let core = GlmFitResult {
        coefficients,
        intercept,
        null_deviance: fitted.null_deviance,
        residual_deviance: fitted.deviance,
        pseudo_r_squared,
        aic: result.aic,
        n_observations: n_valid,
        n_features,
        iterations: fitted.iterations as u32,
        converged: true,
        dispersion: Some(fitted.dispersion), // NegBinomial dispersion (alpha/theta)
    };

    let inference = if options.compute_inference {
        extract_inference(result, options.confidence_level)
    } else {
        None
    };

    Ok(GlmResult { core, inference })
}

/// Fit a Tweedie regression model (for zero-inflated continuous data)
///
/// # Arguments
/// * `y` - Response variable (non-negative, can include zeros)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_tweedie(y: &[f64], x: &[Vec<f64>], options: &TweedieOptions) -> StatsResult<GlmResult> {
    validate_inputs(y, x)?;

    let n_features = x.len();

    // Check for non-negative y values
    for &val in y.iter() {
        if val < 0.0 {
            return Err(StatsError::InvalidValue {
                field: "y",
                message: "Tweedie regression requires non-negative response values".to_string(),
            });
        }
    }

    // Validate power parameter
    if !(1.0..=2.0).contains(&options.power) {
        return Err(StatsError::InvalidValue {
            field: "power",
            message: "Tweedie power parameter must be in [1, 2]".to_string(),
        });
    }

    // Filter out rows with NaN/Inf values
    let valid_indices = get_valid_indices(y, x);
    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();
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

    // Build regressor using var_power (not power)
    let fitted = TweedieRegressor::builder()
        .var_power(options.power)
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .confidence_level(options.confidence_level)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(|e| StatsError::RegressError(format!("{:?}", e)))?;

    // Extract GLM-specific results
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = result.intercept;

    // Calculate pseudo R-squared
    let pseudo_r_squared = if fitted.null_deviance > 0.0 {
        1.0 - fitted.deviance / fitted.null_deviance
    } else {
        0.0
    };

    let core = GlmFitResult {
        coefficients,
        intercept,
        null_deviance: fitted.null_deviance,
        residual_deviance: fitted.deviance,
        pseudo_r_squared,
        aic: result.aic,
        n_observations: n_valid,
        n_features,
        iterations: fitted.iterations as u32,
        converged: true,
        dispersion: Some(fitted.dispersion),
    };

    let inference = if options.compute_inference {
        extract_inference(result, options.confidence_level)
    } else {
        None
    };

    Ok(GlmResult { core, inference })
}

// Helper functions

fn validate_inputs(y: &[f64], x: &[Vec<f64>]) -> StatsResult<()> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = y.len();
    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }
    Ok(())
}

fn get_valid_indices(y: &[f64], x: &[Vec<f64>]) -> Vec<usize> {
    let n_obs = y.len();
    (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect()
}

fn extract_inference_with_nan(
    result: &RegressionResult,
    non_constant_indices: &[usize],
    n_features: usize,
    confidence_level: f64,
) -> Option<GlmInferenceResult> {
    // Helper to reconstruct reduced vector to full size with NaN for constant columns
    let reconstruct = |reduced: Option<&faer::Col<f64>>| -> Vec<f64> {
        let mut full = vec![f64::NAN; n_features];
        if let Some(col) = reduced {
            for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
                if reduced_idx < col.nrows() {
                    full[orig_idx] = col[reduced_idx];
                }
            }
        }
        full
    };

    let std_errors = reconstruct(result.std_errors.as_ref());
    let z_values = reconstruct(result.t_statistics.as_ref());
    let p_values = reconstruct(result.p_values.as_ref());
    let ci_lower = reconstruct(result.conf_interval_lower.as_ref());
    let ci_upper = reconstruct(result.conf_interval_upper.as_ref());

    Some(GlmInferenceResult {
        std_errors,
        z_values,
        p_values,
        ci_lower,
        ci_upper,
        confidence_level,
    })
}

fn extract_inference(
    result: &RegressionResult,
    confidence_level: f64,
) -> Option<GlmInferenceResult> {
    let std_errors: Vec<f64> = result
        .std_errors
        .as_ref()
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default();

    // For GLMs, we use z-statistics (same as t-statistics for large samples)
    let z_values: Vec<f64> = result
        .t_statistics
        .as_ref()
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default();

    let p_values: Vec<f64> = result
        .p_values
        .as_ref()
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default();

    let ci_lower: Vec<f64> = result
        .conf_interval_lower
        .as_ref()
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default();

    let ci_upper: Vec<f64> = result
        .conf_interval_upper
        .as_ref()
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default();

    Some(GlmInferenceResult {
        std_errors,
        z_values,
        p_values,
        ci_lower,
        ci_upper,
        confidence_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_basic() {
        // Simple count data
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0];

        let options = PoissonOptions::default();
        let result = fit_poisson(&y, &x, &options);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.core.pseudo_r_squared > 0.0);
    }

    #[test]
    fn test_binomial_basic() {
        // Binary outcome data
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let options = BinomialOptions::default();
        let result = fit_binomial(&y, &x, &options);

        assert!(result.is_ok());
    }

    #[test]
    fn test_poisson_negative_y_error() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![-1.0, 2.0, 3.0, 4.0, 5.0]; // Invalid negative count

        let options = PoissonOptions::default();
        let result = fit_poisson(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn test_binomial_invalid_y_error() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![0.5, 1.5, 0.0, 0.5, 0.5]; // 1.5 is invalid

        let options = BinomialOptions::default();
        let result = fit_binomial(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }
}
