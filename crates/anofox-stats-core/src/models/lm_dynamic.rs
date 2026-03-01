//! LmDynamic (Time-Varying Coefficient) regression model
//!
//! Uses information-theoretic model averaging with optional LOWESS smoothing
//! to produce time-varying (observation-varying) coefficient estimates.

use crate::errors::{StatsError, StatsResult};
use crate::types::{AlmDistribution, InformationCriterion, LmDynamicOptions, LmDynamicResult};
use anofox_regression::prelude::AlmDistribution as UpstreamAlmDist;
use anofox_regression::solvers::lm_dynamic::{
    InformationCriterion as UpstreamIC, LmDynamicRegressor,
};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Convert our InformationCriterion to upstream
fn convert_ic(ic: InformationCriterion) -> UpstreamIC {
    match ic {
        InformationCriterion::AIC => UpstreamIC::AIC,
        InformationCriterion::AICc => UpstreamIC::AICc,
        InformationCriterion::BIC => UpstreamIC::BIC,
    }
}

/// Convert our AlmDistribution to upstream
fn convert_distribution(dist: AlmDistribution) -> UpstreamAlmDist {
    match dist {
        AlmDistribution::Normal => UpstreamAlmDist::Normal,
        AlmDistribution::Laplace => UpstreamAlmDist::Laplace,
        AlmDistribution::StudentT => UpstreamAlmDist::StudentT,
        AlmDistribution::Logistic => UpstreamAlmDist::Logistic,
        AlmDistribution::AsymmetricLaplace => {
            UpstreamAlmDist::AsymmetricLaplace
        }
        AlmDistribution::GeneralisedNormal => {
            UpstreamAlmDist::GeneralisedNormal
        }
        AlmDistribution::S => UpstreamAlmDist::S,
        AlmDistribution::LogNormal => UpstreamAlmDist::LogNormal,
        AlmDistribution::LogLaplace => UpstreamAlmDist::LogLaplace,
        AlmDistribution::LogS => UpstreamAlmDist::LogS,
        AlmDistribution::LogGeneralisedNormal => {
            UpstreamAlmDist::LogGeneralisedNormal
        }
        AlmDistribution::FoldedNormal => UpstreamAlmDist::FoldedNormal,
        AlmDistribution::RectifiedNormal => UpstreamAlmDist::RectifiedNormal,
        AlmDistribution::BoxCoxNormal => UpstreamAlmDist::BoxCoxNormal,
        AlmDistribution::Gamma => UpstreamAlmDist::Gamma,
        AlmDistribution::InverseGaussian => UpstreamAlmDist::InverseGaussian,
        AlmDistribution::Exponential => UpstreamAlmDist::Exponential,
        AlmDistribution::Beta => UpstreamAlmDist::Beta,
        AlmDistribution::LogitNormal => UpstreamAlmDist::LogitNormal,
        AlmDistribution::Poisson => UpstreamAlmDist::Poisson,
        AlmDistribution::NegativeBinomial => UpstreamAlmDist::NegativeBinomial,
        AlmDistribution::Binomial => UpstreamAlmDist::Binomial,
        AlmDistribution::Geometric => UpstreamAlmDist::Geometric,
        AlmDistribution::CumulativeLogistic => {
            UpstreamAlmDist::CumulativeLogistic
        }
        AlmDistribution::CumulativeNormal => UpstreamAlmDist::CumulativeNormal,
    }
}

/// Fit a time-varying coefficient model using information-theoretic model averaging
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (p features, each with n observations)
/// * `options` - Fitting options
///
/// # Returns
/// * `LmDynamicResult` containing averaged coefficients and time-varying coefficients
pub fn fit_lm_dynamic(
    y: &[f64],
    x: &[Vec<f64>],
    options: &LmDynamicOptions,
) -> StatsResult<LmDynamicResult> {
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
    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

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

    // LmDynamic requires at least 3 observations
    if n_valid < 3 {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Detect zero-variance (constant) columns
    let is_constant_column: Vec<bool> = x
        .iter()
        .map(|col| {
            let first_val = col[valid_indices[0]];
            valid_indices
                .iter()
                .all(|&i| (col[i] - first_val).abs() < 1e-10)
        })
        .collect();

    let non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    let n_effective_features = non_constant_indices.len();

    if n_effective_features == 0 && !options.fit_intercept {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types (only non-constant columns)
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_effective_features, |i, j| {
        x[non_constant_indices[j]][valid_indices[i]]
    });

    // Build and fit the model
    let mut builder = LmDynamicRegressor::builder()
        .ic(convert_ic(options.ic))
        .distribution(convert_distribution(options.distribution))
        .with_intercept(options.fit_intercept)
        .confidence_level(options.confidence_level);

    if let Some(span) = options.lowess_span {
        builder = builder.lowess_span(span);
    } else {
        builder = builder.no_smoothing();
    }

    if let Some(max_models) = options.max_models {
        builder = builder.max_models(max_models);
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

    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    // Extract dynamic coefficients
    let dyn_coefs = fitted.dynamic_coefficients();
    let n_dyn_cols = dyn_coefs.ncols(); // n_effective_features (+ intercept if present)
    let mut dynamic_coefficients = Vec::with_capacity(n_valid);
    for i in 0..n_valid {
        let mut row = Vec::with_capacity(n_dyn_cols);
        for j in 0..n_dyn_cols {
            row.push(dyn_coefs[(i, j)]);
        }
        dynamic_coefficients.push(row);
    }

    Ok(LmDynamicResult {
        coefficients,
        intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        rmse: result.rmse,
        n_observations: n_valid,
        n_features,
        dynamic_coefficients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lm_dynamic_basic() {
        // Simple linear relationship: y = 2*x + 1
        let x = vec![vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]];
        let y = vec![3.1, 5.0, 6.9, 9.1, 11.0, 12.9, 15.1, 17.0, 18.9, 21.1];

        let options = LmDynamicOptions::default();
        let result = fit_lm_dynamic(&y, &x, &options).unwrap();

        // Averaged coefficient should be close to 2
        assert!(
            result.coefficients[0] > 1.5 && result.coefficients[0] < 2.5,
            "coefficient = {}",
            result.coefficients[0]
        );
        assert!(result.r_squared > 0.95);
    }

    #[test]
    fn test_lm_dynamic_has_dynamic_coefficients() {
        let x = vec![vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]];
        let y = vec![3.1, 5.0, 6.9, 9.1, 11.0, 12.9, 15.1, 17.0, 18.9, 21.1];

        let options = LmDynamicOptions::default();
        let result = fit_lm_dynamic(&y, &x, &options).unwrap();

        // Should have dynamic coefficients for each observation
        assert_eq!(result.dynamic_coefficients.len(), 10);
        // Each row should have coefficients (features + possibly intercept)
        assert!(!result.dynamic_coefficients[0].is_empty());
    }

    #[test]
    fn test_lm_dynamic_no_smoothing() {
        let x = vec![vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]];
        let y = vec![3.1, 5.0, 6.9, 9.1, 11.0, 12.9, 15.1, 17.0, 18.9, 21.1];

        let options = LmDynamicOptions {
            lowess_span: None,
            ..Default::default()
        };
        let result = fit_lm_dynamic(&y, &x, &options).unwrap();

        assert!(result.r_squared > 0.90);
    }

    #[test]
    fn test_lm_dynamic_bic() {
        let x = vec![vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]];
        let y = vec![3.1, 5.0, 6.9, 9.1, 11.0, 12.9, 15.1, 17.0, 18.9, 21.1];

        let options = LmDynamicOptions {
            ic: InformationCriterion::BIC,
            ..Default::default()
        };
        let result = fit_lm_dynamic(&y, &x, &options).unwrap();

        assert!(result.r_squared > 0.90);
    }

    #[test]
    fn test_lm_dynamic_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        let options = LmDynamicOptions::default();
        assert!(fit_lm_dynamic(&y, &x, &options).is_err());
    }

    #[test]
    fn test_lm_dynamic_insufficient_data() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0, 2.0];

        let options = LmDynamicOptions::default();
        assert!(fit_lm_dynamic(&y, &x, &options).is_err());
    }
}
