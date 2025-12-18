//! Forecast evaluation tests
//!
//! - Diebold-Mariano test
//! - Clark-West test
//! - Superior Predictive Ability (SPA) test
//! - Model Confidence Set

use super::{convert_error, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    clark_west as lib_clark_west, diebold_mariano as lib_diebold_mariano,
    model_confidence_set as lib_model_confidence_set, spa_test as lib_spa_test, Alternative,
    LossFunction, MCSStatistic, VarEstimator,
};

/// Loss functions for forecast comparison
#[derive(Debug, Clone, Copy)]
pub enum ForecastLoss {
    /// Squared error loss
    SquaredError,
    /// Absolute error loss
    AbsoluteError,
}

impl From<ForecastLoss> for LossFunction {
    fn from(loss: ForecastLoss) -> Self {
        match loss {
            ForecastLoss::SquaredError => LossFunction::SquaredError,
            ForecastLoss::AbsoluteError => LossFunction::AbsoluteError,
        }
    }
}

/// Variance estimators for forecast tests
#[derive(Debug, Clone, Copy)]
pub enum ForecastVarEstimator {
    /// ACF-based (Newey-West)
    Acf,
    /// Bartlett kernel
    Bartlett,
}

impl From<ForecastVarEstimator> for VarEstimator {
    fn from(est: ForecastVarEstimator) -> Self {
        match est {
            ForecastVarEstimator::Acf => VarEstimator::Acf,
            ForecastVarEstimator::Bartlett => VarEstimator::Bartlett,
        }
    }
}

/// Options for Diebold-Mariano test
#[derive(Debug, Clone)]
pub struct DieboldMarianoOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Loss function
    pub loss: ForecastLoss,
    /// Variance estimator
    pub var_estimator: ForecastVarEstimator,
    /// Forecast horizon (h-step ahead)
    pub horizon: usize,
}

impl Default for DieboldMarianoOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            loss: ForecastLoss::SquaredError,
            var_estimator: ForecastVarEstimator::Acf,
            horizon: 1,
        }
    }
}

/// Diebold-Mariano test for forecast accuracy comparison
///
/// Tests whether two forecasts have equal predictive accuracy.
/// The test is performed on forecast errors.
///
/// # Arguments
/// * `actual` - Actual values
/// * `forecast1` - First forecast
/// * `forecast2` - Second forecast
/// * `options` - Test options
pub fn diebold_mariano(
    actual: &[f64],
    forecast1: &[f64],
    forecast2: &[f64],
    options: &DieboldMarianoOptions,
) -> StatsResult<TestResult> {
    if actual.len() != forecast1.len() || actual.len() != forecast2.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "All arrays must have the same length".into(),
        ));
    }

    // Filter rows where any value is NaN
    let valid: Vec<(f64, f64, f64)> = actual
        .iter()
        .zip(forecast1.iter())
        .zip(forecast2.iter())
        .filter(|((a, f1), f2)| !a.is_nan() && !f1.is_nan() && !f2.is_nan())
        .map(|((a, f1), f2)| (*a, *f1, *f2))
        .collect();

    if valid.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Diebold-Mariano test requires at least 3 valid observations".into(),
        ));
    }

    // Compute forecast errors
    let e1: Vec<f64> = valid.iter().map(|(a, f1, _)| a - f1).collect();
    let e2: Vec<f64> = valid.iter().map(|(a, _, f2)| a - f2).collect();

    let result = lib_diebold_mariano(
        &e1,
        &e2,
        options.loss.into(),
        options.horizon,
        options.alternative,
        options.var_estimator.into(),
    )
    .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: valid.len(),
        n1: 0,
        n2: 0,
        alternative: options.alternative,
        method: "Diebold-Mariano test".into(),
    })
}

/// Clark-West test for nested model comparison
///
/// Tests whether a restricted model forecasts as well as an unrestricted model.
/// Adjusts for the bias when comparing nested models.
///
/// # Arguments
/// * `actual` - Actual values
/// * `forecast_restricted` - Forecast from restricted (parsimonious) model
/// * `forecast_unrestricted` - Forecast from unrestricted model
/// * `horizon` - Forecast horizon
pub fn clark_west(
    actual: &[f64],
    forecast_restricted: &[f64],
    forecast_unrestricted: &[f64],
    horizon: usize,
) -> StatsResult<TestResult> {
    if actual.len() != forecast_restricted.len() || actual.len() != forecast_unrestricted.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "All arrays must have the same length".into(),
        ));
    }

    let valid: Vec<(f64, f64, f64)> = actual
        .iter()
        .zip(forecast_restricted.iter())
        .zip(forecast_unrestricted.iter())
        .filter(|((a, f1), f2)| !a.is_nan() && !f1.is_nan() && !f2.is_nan())
        .map(|((a, f1), f2)| (*a, *f1, *f2))
        .collect();

    if valid.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Clark-West test requires at least 3 valid observations".into(),
        ));
    }

    // Compute forecast errors
    let e1: Vec<f64> = valid.iter().map(|(a, f1, _)| a - f1).collect();
    let e2: Vec<f64> = valid.iter().map(|(a, _, f2)| a - f2).collect();

    let result = lib_clark_west(&e1, &e2, horizon).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: valid.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::Greater,
        method: "Clark-West test".into(),
    })
}

/// Options for SPA test
#[derive(Debug, Clone)]
pub struct SpaTestOptions {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Block length for stationary bootstrap (0 for automatic)
    pub block_length: f64,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for SpaTestOptions {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            block_length: 0.0, // Automatic
            seed: None,
        }
    }
}

/// SPA test result
#[derive(Debug, Clone)]
pub struct SpaResult {
    /// Test statistic
    pub statistic: f64,
    /// p-value (consistent)
    pub p_value_consistent: f64,
    /// p-value (upper, more conservative)
    pub p_value_upper: f64,
    /// Index of best model (if any outperforms benchmark)
    pub best_model_idx: Option<usize>,
}

/// Superior Predictive Ability (SPA) test
///
/// Tests whether a benchmark model is outperformed by any of the alternatives.
/// Takes pre-computed losses for all models.
///
/// # Arguments
/// * `benchmark_losses` - Loss values from the benchmark model
/// * `model_losses` - Loss values for each alternative model (each inner Vec is one model)
/// * `options` - Test options
pub fn spa_test(
    benchmark_losses: &[f64],
    model_losses: &[Vec<f64>],
    options: &SpaTestOptions,
) -> StatsResult<SpaResult> {
    if model_losses.is_empty() {
        return Err(StatsError::InvalidInput(
            "At least one alternative model required".into(),
        ));
    }

    for (i, model) in model_losses.iter().enumerate() {
        if model.len() != benchmark_losses.len() {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Model {} has different length than benchmark",
                i
            )));
        }
    }

    let result = lib_spa_test(
        benchmark_losses,
        model_losses,
        options.n_bootstrap,
        options.block_length,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(SpaResult {
        statistic: result.statistic,
        p_value_consistent: result.p_value_consistent,
        p_value_upper: result.p_value_upper,
        best_model_idx: result.best_model_idx,
    })
}

/// MCS statistic type
#[derive(Debug, Clone, Copy)]
pub enum McsStatistic {
    /// Range statistic
    Range,
    /// Max statistic
    Max,
}

impl From<McsStatistic> for MCSStatistic {
    fn from(s: McsStatistic) -> Self {
        match s {
            McsStatistic::Range => MCSStatistic::Range,
            McsStatistic::Max => MCSStatistic::Max,
        }
    }
}

/// Options for Model Confidence Set
#[derive(Debug, Clone)]
pub struct McsOptions {
    /// Significance level
    pub alpha: f64,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Test statistic type
    pub statistic: McsStatistic,
    /// Block length for stationary bootstrap (0 for automatic)
    pub block_length: f64,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for McsOptions {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            n_bootstrap: 1000,
            statistic: McsStatistic::Range,
            block_length: 0.0,
            seed: None,
        }
    }
}

/// MCS result
#[derive(Debug, Clone)]
pub struct McsResult {
    /// Indices of models in the confidence set
    pub included_models: Vec<usize>,
    /// Indices of eliminated models
    pub eliminated_models: Vec<usize>,
    /// p-value for the MCS
    pub mcs_p_value: f64,
    /// Elimination sequence
    pub elimination_sequence: Vec<usize>,
}

/// Model Confidence Set
///
/// Identifies the set of best models at a given confidence level.
/// Takes pre-computed losses for all models.
///
/// # Arguments
/// * `losses` - Loss values for each model (each inner Vec is one model)
/// * `options` - MCS options
pub fn model_confidence_set(losses: &[Vec<f64>], options: &McsOptions) -> StatsResult<McsResult> {
    if losses.len() < 2 {
        return Err(StatsError::InvalidInput(
            "MCS requires at least 2 models".into(),
        ));
    }

    let t = losses[0].len();
    for (i, model) in losses.iter().enumerate() {
        if model.len() != t {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Model {} has different length",
                i
            )));
        }
    }

    let result = lib_model_confidence_set(
        losses,
        options.alpha,
        options.statistic.into(),
        options.n_bootstrap,
        options.block_length,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(McsResult {
        included_models: result.included_models,
        eliminated_models: result.eliminated_models,
        mcs_p_value: result.mcs_p_value,
        elimination_sequence: result
            .elimination_sequence
            .iter()
            .map(|s| s.model_idx)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diebold_mariano() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let forecast1 = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]; // Good forecast
        let forecast2 = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]; // Worse forecast
        let opts = DieboldMarianoOptions::default();

        let result = diebold_mariano(&actual, &forecast1, &forecast2, &opts).unwrap();
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_clark_west() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let restricted = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let unrestricted = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];

        let result = clark_west(&actual, &restricted, &unrestricted, 1).unwrap();
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }
}
