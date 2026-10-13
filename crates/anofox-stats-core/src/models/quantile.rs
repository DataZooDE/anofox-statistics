//! Quantile regression wrapper

use crate::errors::{StatsError, StatsResult};
use crate::types::{ModelSummary, QuantileFitResult, QuantileOptions};
use anofox_regression::core::HasModelInfo;
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Fit a Quantile regression model
///
/// Quantile regression estimates conditional quantiles of the response variable,
/// rather than the conditional mean. It is robust to outliers and useful for
/// understanding the full distribution of the response.
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
///
/// # Returns
/// * `QuantileFitResult` containing coefficients for the specified quantile
pub fn fit_quantile(
    y: &[f64],
    x: &[Vec<f64>],
    options: &QuantileOptions,
) -> StatsResult<QuantileFitResult> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    // Validate tau
    if options.tau <= 0.0 || options.tau >= 1.0 {
        return Err(StatsError::InvalidValue {
            field: "tau",
            message: "tau must be in (0, 1)".to_string(),
        });
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

    // Check minimum observations
    let min_obs = if options.fit_intercept {
        n_features + 1
    } else {
        n_features
    };

    if n_valid < min_obs {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }

    // Convert to faer types
    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, n_features, |i, j| x[j][valid_indices[i]]);

    // Build and fit the Quantile model
    let fitted = QuantileRegressor::builder()
        .tau(options.tau)
        .with_intercept(options.fit_intercept)
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;

    // Extract results
    let result = fitted.result();
    let coefficients: Vec<f64> = result.coefficients.iter().copied().collect();
    let intercept = if options.fit_intercept {
        result.intercept
    } else {
        None
    };

    Ok(QuantileFitResult {
        coefficients,
        intercept,
        tau: options.tau,
        n_observations: n_valid,
        n_features,
        summary: ModelSummary::from_result(fitted.model_info(), result, false),
    })
}

/// Standard-error type of [`quantile_process`] (`quantreg::summary.rq`'s `se`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileSeType {
    /// `se = "iid"`.
    Iid,
    /// `se = "nid"` (Hendricks-Koenker sandwich, Hall-Sheather bandwidth).
    Nid,
    /// `se = "boot"`, `bsmethod = "xy"`: `n_boot` seeded pair resamples.
    Boot { seed: u64, n_boot: usize },
}

/// Coefficient table of one quantile; vectors start with the intercept when
/// the model has one.
#[derive(Debug, Clone)]
pub struct QuantileProcessRow {
    pub tau: f64,
    pub estimate: Vec<f64>,
    pub std_error: Vec<f64>,
    pub statistic: Vec<f64>,
    pub p_value: Vec<f64>,
    pub conf_low: Vec<f64>,
    pub conf_high: Vec<f64>,
    pub conf_level: f64,
    pub df_residual: f64,
}

/// Regression-quantile process with inference (`summary(rq(y ~ x, tau = taus),
/// se = ...)`), computed by `anofox_regression::solvers::quantile_inference`.
/// Rows with a non-finite value are dropped; `se = None` gives point
/// estimates with NaN inference. Also returns the number of rows used.
pub fn quantile_process(
    y: &[f64],
    x: &[Vec<f64>],
    taus: &[f64],
    fit_intercept: bool,
    se: Option<QuantileSeType>,
    level: f64,
) -> StatsResult<(Vec<QuantileProcessRow>, usize)> {
    use anofox_regression::solvers::quantile_inference::{
        quantile_process as up_process, QuantileSe,
    };
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if taus.is_empty() {
        return Err(StatsError::EmptyInput { field: "taus" });
    }
    if let Some(&t) = taus.iter().find(|t| !(**t > 0.0 && **t < 1.0)) {
        return Err(StatsError::InvalidValue {
            field: "tau",
            message: format!("tau must be in (0, 1), got {t}"),
        });
    }
    crate::validation::validate_x_columns(y.len(), x)?;
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let k = x.len() + usize::from(fit_intercept);
    if rows.len() <= k {
        return Err(StatsError::InsufficientData {
            rows: rows.len(),
            cols: x.len(),
        });
    }
    let y_col = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let x_mat = Mat::from_fn(rows.len(), x.len(), |i, j| x[j][rows[i]]);
    let se = se.map(|s| match s {
        QuantileSeType::Iid => QuantileSe::Iid,
        QuantileSeType::Nid => QuantileSe::Nid,
        QuantileSeType::Boot { seed, n_boot } => QuantileSe::Boot { seed, r: n_boot },
    });
    let table =
        up_process(&x_mat, &y_col, taus, fit_intercept, se, level).map_err(StatsError::from)?;
    Ok((
        table
            .into_iter()
            .map(|r| QuantileProcessRow {
                tau: r.tau,
                estimate: r.estimate,
                std_error: r.std_error,
                statistic: r.statistic,
                p_value: r.p_value,
                conf_low: r.conf_low,
                conf_high: r.conf_high,
                conf_level: r.conf_level,
                df_residual: r.df_residual,
            })
            .collect(),
        rows.len(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_has_one_row_per_tau() {
        let x = vec![(0..30).map(|i| i as f64).collect::<Vec<_>>()];
        let y: Vec<f64> = (0..30)
            .map(|i| 1.0 + 0.5 * i as f64 + ((i * 7) % 5) as f64)
            .collect();
        let (rows, n) =
            quantile_process(&y, &x, &[0.25, 0.5], true, Some(QuantileSeType::Nid), 0.95).unwrap();
        assert_eq!(n, 30);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].estimate.len(), 2);
        assert!(rows[1].std_error.iter().all(|s| s.is_finite() && *s > 0.0));
        let (none, _) = quantile_process(&y, &x, &[0.5], true, None, 0.95).unwrap();
        assert!(none[0].std_error[0].is_nan());
        assert!(quantile_process(&y, &x, &[1.0], true, None, 0.95).is_err());
    }

    #[test]
    fn test_quantile_median() {
        // Simple linear data
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let options = QuantileOptions {
            tau: 0.5, // Median
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_quantile(&y, &x, &options);
        assert!(result.is_ok());
        let result = result.unwrap();

        // Coefficient should be approximately 2 (y = 2*x)
        assert!((result.coefficients[0] - 2.0).abs() < 0.5);
        assert!((result.tau - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_75th_percentile() {
        // Data with some spread
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        let options = QuantileOptions {
            tau: 0.75, // 75th percentile
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        };

        let result = fit_quantile(&y, &x, &options);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!((result.tau - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let options = QuantileOptions::default();
        let result = fit_quantile(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::EmptyInput { .. })));
    }

    #[test]
    fn test_quantile_invalid_tau() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // tau = 0 is invalid
        let options = QuantileOptions {
            tau: 0.0,
            ..Default::default()
        };
        let result = fit_quantile(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));

        // tau = 1 is invalid
        let options = QuantileOptions {
            tau: 1.0,
            ..Default::default()
        };
        let result = fit_quantile(&y, &x, &options);
        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn test_quantile_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0]; // Wrong length

        let options = QuantileOptions::default();
        let result = fit_quantile(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }
}
