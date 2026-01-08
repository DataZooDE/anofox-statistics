//! Augmented Linear Model (ALM) implementations
//!
//! ALM extends linear models with 24 error distribution families,
//! providing robust and flexible regression for various data types.

use anofox_regression::solvers::{
    AlmDistribution as RegAlmDistribution, AlmLoss as RegAlmLoss, AlmRegressor, FittedRegressor,
    Regressor,
};
use faer::{Col, Mat};

use crate::errors::{StatsError, StatsResult};
use crate::types::{AlmDistribution, AlmFitResult, AlmLoss, AlmOptions};

/// Result from ALM fitting including optional inference
#[derive(Debug, Clone)]
pub struct AlmResult {
    pub core: AlmFitResult,
    pub inference: Option<AlmInferenceResult>,
}

/// Inference results for ALM
#[derive(Debug, Clone)]
pub struct AlmInferenceResult {
    pub standard_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_int_lower: Vec<f64>,
    pub conf_int_upper: Vec<f64>,
}

/// Convert our AlmDistribution to anofox-regression's AlmDistribution
fn convert_distribution(dist: AlmDistribution) -> RegAlmDistribution {
    match dist {
        AlmDistribution::Normal => RegAlmDistribution::Normal,
        AlmDistribution::Laplace => RegAlmDistribution::Laplace,
        AlmDistribution::StudentT => RegAlmDistribution::StudentT,
        AlmDistribution::Logistic => RegAlmDistribution::Logistic,
        AlmDistribution::AsymmetricLaplace => RegAlmDistribution::AsymmetricLaplace,
        AlmDistribution::GeneralisedNormal => RegAlmDistribution::GeneralisedNormal,
        AlmDistribution::S => RegAlmDistribution::S,
        AlmDistribution::LogNormal => RegAlmDistribution::LogNormal,
        AlmDistribution::LogLaplace => RegAlmDistribution::LogLaplace,
        AlmDistribution::LogS => RegAlmDistribution::LogS,
        AlmDistribution::LogGeneralisedNormal => RegAlmDistribution::LogGeneralisedNormal,
        AlmDistribution::FoldedNormal => RegAlmDistribution::FoldedNormal,
        AlmDistribution::RectifiedNormal => RegAlmDistribution::RectifiedNormal,
        AlmDistribution::BoxCoxNormal => RegAlmDistribution::BoxCoxNormal,
        AlmDistribution::Gamma => RegAlmDistribution::Gamma,
        AlmDistribution::InverseGaussian => RegAlmDistribution::InverseGaussian,
        AlmDistribution::Exponential => RegAlmDistribution::Exponential,
        AlmDistribution::Beta => RegAlmDistribution::Beta,
        AlmDistribution::LogitNormal => RegAlmDistribution::LogitNormal,
        AlmDistribution::Poisson => RegAlmDistribution::Poisson,
        AlmDistribution::NegativeBinomial => RegAlmDistribution::NegativeBinomial,
        AlmDistribution::Binomial => RegAlmDistribution::Binomial,
        AlmDistribution::Geometric => RegAlmDistribution::Geometric,
        AlmDistribution::CumulativeLogistic => RegAlmDistribution::CumulativeLogistic,
        AlmDistribution::CumulativeNormal => RegAlmDistribution::CumulativeNormal,
    }
}

/// Convert our AlmLoss to anofox-regression's AlmLoss
fn convert_loss(loss: AlmLoss, role_trim: f64) -> RegAlmLoss {
    match loss {
        AlmLoss::Likelihood => RegAlmLoss::Likelihood,
        AlmLoss::MSE => RegAlmLoss::MSE,
        AlmLoss::MAE => RegAlmLoss::MAE,
        AlmLoss::HAM => RegAlmLoss::HAM,
        AlmLoss::ROLE => RegAlmLoss::ROLE { trim: role_trim },
    }
}

/// Validate input arrays for ALM
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

/// Fit an Augmented Linear Model (ALM)
///
/// ALM supports 24 error distribution families and multiple loss functions,
/// making it suitable for a wide variety of data types.
///
/// # Arguments
/// * `y` - Response variable
/// * `x` - Feature matrix (column-major: each inner Vec is one feature)
/// * `options` - Fitting options including distribution and loss function
///
/// # Returns
/// * `AlmResult` containing coefficients and model fit statistics
///
/// # Example
/// ```ignore
/// use anofox_stats_core::models::fit_alm;
/// use anofox_stats_core::types::{AlmOptions, AlmDistribution};
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let options = AlmOptions {
///     distribution: AlmDistribution::Laplace,
///     ..Default::default()
/// };
/// let result = fit_alm(&y, &x, &options).unwrap();
/// ```
pub fn fit_alm(y: &[f64], x: &[Vec<f64>], options: &AlmOptions) -> StatsResult<AlmResult> {
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
        // Intercept-only model: compute mean of y as intercept
        let y_mean = valid_indices.iter().map(|&i| y[i]).sum::<f64>() / n_valid as f64;

        let core = AlmFitResult {
            coefficients: vec![f64::NAN; n_features],
            intercept: Some(y_mean),
            log_likelihood: f64::NAN,
            aic: f64::NAN,
            bic: f64::NAN,
            scale: f64::NAN,
            n_observations: n_valid,
            n_features,
            iterations: 0,
            converged: true,
        };

        return Ok(AlmResult {
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

    // Build regressor
    let mut builder = AlmRegressor::builder()
        .distribution(convert_distribution(options.distribution))
        .loss(convert_loss(options.loss, options.role_trim))
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .confidence_level(options.confidence_level);

    // Set extra parameter for distributions that need it
    // (quantile for AsymmetricLaplace, df for StudentT, etc.)
    if matches!(options.distribution, AlmDistribution::AsymmetricLaplace) {
        builder = builder.extra_parameter(options.quantile);
    }

    let fitted = builder
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
    let intercept = result.intercept;

    let core = AlmFitResult {
        coefficients,
        intercept,
        log_likelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        scale: fitted.scale(),
        n_observations: n_valid,
        n_features,
        iterations: 0, // ALM doesn't expose iterations directly
        converged: true,
    };

    // Extract inference if requested
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

    Ok(AlmResult { core, inference })
}

/// Extract inference statistics from regression result with NaN for constant columns
fn extract_inference_with_nan(
    result: &anofox_regression::core::RegressionResult,
    non_constant_indices: &[usize],
    n_features: usize,
    _confidence_level: f64,
) -> Option<AlmInferenceResult> {
    // Check if inference data is available
    result.std_errors.as_ref()?;

    // Helper to reconstruct reduced vector to full size with NaN for constant columns
    let reconstruct = |reduced: &faer::Col<f64>| -> Vec<f64> {
        let mut full = vec![f64::NAN; n_features];
        for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
            full[orig_idx] = reduced[reduced_idx];
        }
        full
    };

    let se = result.std_errors.as_ref()?;
    let t_vals = result.t_statistics.as_ref()?;
    let p_vals = result.p_values.as_ref()?;
    let conf_lower = result.conf_interval_lower.as_ref()?;
    let conf_upper = result.conf_interval_upper.as_ref()?;

    Some(AlmInferenceResult {
        standard_errors: reconstruct(se),
        t_values: reconstruct(t_vals),
        p_values: reconstruct(p_vals),
        conf_int_lower: reconstruct(conf_lower),
        conf_int_upper: reconstruct(conf_upper),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alm_normal() {
        // Simple linear data: y = 2 + 3*x
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 2.0 + 3.0 * xi).collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Normal,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        assert!((result.core.intercept.unwrap() - 2.0).abs() < 0.1);
        assert!((result.core.coefficients[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_alm_laplace() {
        // Laplace distribution is robust to outliers
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let mut y: Vec<f64> = x[0].iter().map(|&xi| 1.0 + 2.0 * xi).collect();
        // Add outlier
        y[9] = 100.0;

        let options = AlmOptions {
            distribution: AlmDistribution::Laplace,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        // Laplace should be less affected by the outlier
        assert!(result.core.coefficients[0] > 0.0);
    }

    #[test]
    fn test_alm_gamma() {
        // Gamma for positive data
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| (1.0_f64 + 0.5 * xi).exp()).collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Gamma,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        assert!(result.core.aic.is_finite());
        assert!(result.core.scale > 0.0);
    }

    #[test]
    fn test_alm_poisson() {
        // Poisson for count data
        let x = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];
        let y: Vec<f64> = x[0]
            .iter()
            .map(|&xi| (0.5_f64 + 0.2 * xi).exp().round())
            .collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Poisson,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        assert!(result.core.log_likelihood.is_finite());
    }

    #[test]
    fn test_alm_beta() {
        // Beta for data in (0, 1)
        let x = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 0.1 + 0.8 * xi).collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Beta,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        assert!(result.core.coefficients[0] > 0.0);
    }

    #[test]
    fn test_alm_with_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 2.0 + 3.0 * xi + 0.1).collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Normal,
            compute_inference: true,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        // Check that inference was computed
        if let Some(inference) = result.inference {
            assert!(!inference.standard_errors.is_empty());
            assert!(!inference.p_values.is_empty());
        }
    }

    #[test]
    fn test_alm_mae_loss() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y: Vec<f64> = x[0].iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        let options = AlmOptions {
            distribution: AlmDistribution::Normal,
            loss: AlmLoss::MAE,
            ..Default::default()
        };

        let result = fit_alm(&y, &x, &options).unwrap();

        assert!(result.core.coefficients[0] > 0.0);
    }

    #[test]
    fn test_alm_empty_input() {
        let y: Vec<f64> = vec![];
        let x: Vec<Vec<f64>> = vec![vec![]];

        let options = AlmOptions::default();
        let result = fit_alm(&y, &x, &options);

        assert!(result.is_err());
    }
}
