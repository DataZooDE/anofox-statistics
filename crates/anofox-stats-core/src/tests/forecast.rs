//! Forecast evaluation tests
//!
//! - Diebold-Mariano test
//! - Clark-West test
//! - Superior Predictive Ability (SPA) test
//! - MSPE-adjusted SPA test
//! - Model Confidence Set

use super::{convert_error, TestResult, UnifiedTestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    clark_west as lib_clark_west, diebold_mariano as lib_diebold_mariano,
    model_confidence_set_with_method as lib_model_confidence_set_with_method,
    mspe_adjusted_spa as lib_mspe_adjusted_spa, spa_test as lib_spa_test, Alternative,
    BootstrapMethod, LossFunction, MCSStatistic, VarEstimator,
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
        unified: super::UnifiedTestResult::from(&result).with_n(valid.len()),
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
        unified: super::UnifiedTestResult::from(&result).with_n(valid.len()),
    })
}

/// Options for the SPA-type tests (`spa_test`, `mspe_adjusted_spa`)
#[derive(Debug, Clone)]
pub struct SpaTestOptions {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Expected block length of the stationary bootstrap; `<= 0` (or NaN) selects
    /// the rule of thumb `T^(1/3)` documented upstream
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

/// Result of an SPA-type test
#[derive(Debug, Clone)]
pub struct SpaResult {
    /// Hansen's (2005) statistic `T_SPA = max(0, max_k sqrt(n) d̄_k / ω̂_k)`
    pub statistic: f64,
    /// Lower p-value (re-centring `max(d̄_k, 0)`; liberal)
    pub p_value_lower: f64,
    /// Consistent p-value (Hansen's threshold re-centring)
    pub p_value_consistent: f64,
    /// Upper p-value (every model re-centred at the null boundary, White's
    /// Reality Check; conservative). `p_value_lower <= p_value_consistent <= p_value_upper`
    pub p_value_upper: f64,
    /// 0-based index of the best competing model
    pub best_model_idx: Option<usize>,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Block length used by the stationary bootstrap
    pub block_length: f64,
    /// Number of observations (time points)
    pub n: usize,
    /// Number of competing models
    pub n_models: usize,
    /// Unified result (contract `test` row), from the upstream result
    pub unified: UnifiedTestResult,
}

/// Resolve the stationary-bootstrap block length: a non-positive (or NaN) value
/// selects the `T^(1/3)` rule of thumb named in the upstream documentation.
pub fn resolve_block_length(block_length: f64, t: usize) -> f64 {
    if block_length.is_nan() || block_length <= 0.0 {
        (t as f64).cbrt()
    } else {
        block_length
    }
}

fn check_spa_inputs(benchmark: &[f64], models: &[Vec<f64>], what: &str) -> StatsResult<()> {
    if models.is_empty() {
        return Err(StatsError::InvalidInput(format!(
            "At least one {} required",
            what
        )));
    }
    for (i, model) in models.iter().enumerate() {
        if model.len() != benchmark.len() {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Model {} has different length than benchmark",
                i
            )));
        }
    }
    if benchmark.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "SPA test requires at least 3 observations".into(),
        ));
    }
    Ok(())
}

/// Superior Predictive Ability (SPA) test (Hansen, 2005)
///
/// Tests whether any competing model has a lower expected loss than the
/// benchmark. Takes pre-computed losses for all models.
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
    check_spa_inputs(benchmark_losses, model_losses, "competing model")?;
    let t = benchmark_losses.len();
    let block_length = resolve_block_length(options.block_length, t);

    let result = lib_spa_test(
        benchmark_losses,
        model_losses,
        options.n_bootstrap,
        block_length,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(SpaResult {
        statistic: result.statistic,
        p_value_lower: result.p_value_lower,
        p_value_consistent: result.p_value_consistent,
        p_value_upper: result.p_value_upper,
        best_model_idx: result.best_model_idx,
        n_bootstrap: result.n_bootstrap,
        block_length,
        n: t,
        n_models: model_losses.len(),
        unified: UnifiedTestResult::from(&result)
            .with_alternative(Alternative::Greater)
            .with_n(t),
    })
}

/// MSPE-adjusted SPA test (Clark and McCracken, 2012)
///
/// SPA bootstrap on Clark-West adjusted loss differentials, for a benchmark
/// nested in each alternative. Takes forecast errors, not losses.
///
/// # Arguments
/// * `benchmark_errors` - Forecast errors of the benchmark (restricted) model
/// * `model_errors` - Forecast errors of each alternative (unrestricted) model
/// * `options` - Test options
pub fn mspe_adjusted_spa(
    benchmark_errors: &[f64],
    model_errors: &[Vec<f64>],
    options: &SpaTestOptions,
) -> StatsResult<SpaResult> {
    check_spa_inputs(benchmark_errors, model_errors, "alternative model")?;
    let t = benchmark_errors.len();
    let block_length = resolve_block_length(options.block_length, t);

    let result = lib_mspe_adjusted_spa(
        benchmark_errors,
        model_errors,
        options.n_bootstrap,
        block_length,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(SpaResult {
        statistic: result.statistic,
        p_value_lower: result.p_value_lower,
        p_value_consistent: result.p_value_consistent,
        p_value_upper: result.p_value_upper,
        best_model_idx: result.best_model_idx,
        n_bootstrap: result.n_bootstrap,
        block_length,
        n: t,
        n_models: model_errors.len(),
        unified: UnifiedTestResult::from(&result)
            .with_alternative(Alternative::Greater)
            .with_n(t),
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

/// Bootstrap scheme of the Model Confidence Set
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McsBootstrap {
    /// Stationary bootstrap (geometric blocks with mean `block_length`)
    Stationary,
    /// Moving (non-circular) block bootstrap with blocks of `block_length`;
    /// `block_length = k` is R `MCS::MCSprocedure(k = k)`
    MovingBlock,
    /// Circular block bootstrap with blocks of `block_length`
    CircularBlock,
    /// IID bootstrap (block length 1)
    Iid,
}

impl McsBootstrap {
    /// Name used in the SQL result.
    pub fn name(&self) -> &'static str {
        match self {
            McsBootstrap::Stationary => "stationary",
            McsBootstrap::MovingBlock => "moving_block",
            McsBootstrap::CircularBlock => "circular_block",
            McsBootstrap::Iid => "iid",
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
    /// Bootstrap scheme
    pub bootstrap: McsBootstrap,
    /// (Expected) block length; `<= 0` (or NaN) selects `T^(1/3)` (rounded to
    /// a whole number >= 1 for the fixed-block schemes). Ignored for `Iid`.
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
            bootstrap: McsBootstrap::Stationary,
            block_length: 0.0,
            seed: None,
        }
    }
}

/// The upstream bootstrap scheme and the block length actually used.
fn resolve_mcs_bootstrap(
    bootstrap: McsBootstrap,
    block_length: f64,
    t: usize,
) -> StatsResult<(BootstrapMethod, f64)> {
    let auto = block_length.is_nan() || block_length <= 0.0;
    let fixed = || -> StatsResult<usize> {
        if auto {
            return Ok(((t as f64).cbrt().round() as usize).max(1));
        }
        if !block_length.is_finite() || block_length.fract() != 0.0 {
            return Err(StatsError::InvalidInput(format!(
                "block_length must be a whole number for the {} bootstrap, got {}",
                bootstrap.name(),
                block_length
            )));
        }
        Ok(block_length as usize)
    };
    Ok(match bootstrap {
        McsBootstrap::Stationary => {
            let l = resolve_block_length(block_length, t);
            if !l.is_finite() || l < 1.0 {
                return Err(StatsError::InvalidInput(format!(
                    "block_length must be >= 1 for the stationary bootstrap, got {}",
                    l
                )));
            }
            (
                BootstrapMethod::Stationary {
                    expected_block_length: l,
                },
                l,
            )
        }
        McsBootstrap::MovingBlock => {
            let l = fixed()?;
            (BootstrapMethod::MovingBlock { block_length: l }, l as f64)
        }
        McsBootstrap::CircularBlock => {
            let l = fixed()?;
            (BootstrapMethod::CircularBlock { block_length: l }, l as f64)
        }
        McsBootstrap::Iid => (BootstrapMethod::Iid, 1.0),
    })
}

/// One step of the MCS elimination sequence
#[derive(Debug, Clone, Copy)]
pub struct McsStep {
    /// 0-based index of the model tested at this step
    pub model_idx: usize,
    /// p-value of the equal-predictive-ability test at this step
    pub p_value: f64,
    /// MCS p-value of the model (running maximum of the step p-values)
    pub mcs_p_value: f64,
    /// Whether the model was eliminated (`mcs_p_value < alpha`)
    pub eliminated: bool,
}

/// MCS result
#[derive(Debug, Clone)]
pub struct McsResult {
    /// 0-based indices of the models in the confidence set
    pub included_models: Vec<usize>,
    /// 0-based indices of the eliminated models, in elimination order
    pub eliminated_models: Vec<usize>,
    /// p-value of the MCS (the p-value at which elimination stopped)
    pub mcs_p_value: f64,
    /// Elimination steps up to and including the first retained model
    pub elimination_sequence: Vec<McsStep>,
    /// Full elimination order, worst model first and the best model last
    /// (R `elimination.order`), 0-based
    pub elimination_order: Vec<usize>,
    /// MCS p-value of every model, indexed by model (the best model has 1)
    pub mcs_p_values: Vec<f64>,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Bootstrap scheme used
    pub bootstrap: McsBootstrap,
    /// (Expected) block length used (1 for the IID bootstrap)
    pub block_length: f64,
    /// Number of observations (time points)
    pub n: usize,
    /// Number of models
    pub n_models: usize,
}

/// Model Confidence Set (Hansen, Lunde and Nason, 2011)
///
/// Identifies the set of best models at a given confidence level.
/// Takes pre-computed losses for all models.
///
/// # Arguments
/// * `losses` - Loss values for each model (each inner Vec is one model)
/// * `options` - MCS options
pub fn model_confidence_set(losses: &[Vec<f64>], options: &McsOptions) -> StatsResult<McsResult> {
    if losses.is_empty() {
        return Err(StatsError::InvalidInput(
            "MCS requires at least 1 model".into(),
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
    if t < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "MCS requires at least 3 observations".into(),
        ));
    }
    let (method, block_length) = resolve_mcs_bootstrap(options.bootstrap, options.block_length, t)?;

    let result = lib_model_confidence_set_with_method(
        losses,
        options.alpha,
        options.statistic.into(),
        options.n_bootstrap,
        method,
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
            .map(|s| McsStep {
                model_idx: s.model_idx,
                p_value: s.p_value,
                mcs_p_value: s.mcs_p_value,
                eliminated: s.eliminated,
            })
            .collect(),
        elimination_order: result.elimination_order,
        mcs_p_values: result.mcs_p_values,
        n_bootstrap: result.n_bootstrap,
        bootstrap: options.bootstrap,
        block_length,
        n: t,
        n_models: losses.len(),
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

    fn losses(mean: f64, amp: f64, phase: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| mean + amp * ((i as f64) * 0.7 + phase).sin().abs())
            .collect()
    }

    #[test]
    fn test_resolve_block_length() {
        assert!((resolve_block_length(0.0, 125) - 5.0).abs() < 1e-12);
        assert!((resolve_block_length(f64::NAN, 8) - 2.0).abs() < 1e-12);
        assert_eq!(resolve_block_length(3.0, 125), 3.0);
    }

    #[test]
    fn test_spa_seeded_is_reproducible() {
        let bench = losses(2.0, 0.5, 0.0, 80);
        let models = vec![losses(1.0, 0.3, 1.0, 80), losses(2.2, 0.4, 2.0, 80)];
        let opts = SpaTestOptions {
            n_bootstrap: 199,
            block_length: 4.0,
            seed: Some(7),
        };
        let a = spa_test(&bench, &models, &opts).unwrap();
        let b = spa_test(&bench, &models, &opts).unwrap();
        assert_eq!(a.p_value_consistent, b.p_value_consistent);
        assert_eq!(a.best_model_idx, Some(0));
        assert_eq!(a.n_models, 2);
        assert_eq!(a.n, 80);
        assert!(a.p_value_consistent < 0.05);
    }

    #[test]
    fn test_spa_errors() {
        let bench = losses(2.0, 0.5, 0.0, 10);
        assert!(matches!(
            spa_test(&bench, &[], &SpaTestOptions::default()),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            spa_test(&bench, &[vec![1.0; 9]], &SpaTestOptions::default()),
            Err(StatsError::DimensionMismatchMsg(_))
        ));
        assert!(matches!(
            spa_test(&[1.0, 2.0], &[vec![1.0, 1.5]], &SpaTestOptions::default()),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }

    #[test]
    fn test_mspe_adjusted_spa() {
        let bench = losses(2.0, 0.2, 0.0, 60);
        let models = vec![losses(0.5, 0.2, 1.0, 60)];
        let opts = SpaTestOptions {
            n_bootstrap: 199,
            block_length: 3.0,
            seed: Some(42),
        };
        let r = mspe_adjusted_spa(&bench, &models, &opts).unwrap();
        assert!(r.p_value_consistent < 0.05);
        assert_eq!(r.best_model_idx, Some(0));
    }

    #[test]
    fn test_mcs() {
        let l = vec![
            losses(1.0, 0.3, 0.0, 100),
            losses(5.0, 0.3, 1.0, 100),
            losses(1.0, 0.3, 0.0, 100),
        ];
        let opts = McsOptions {
            n_bootstrap: 199,
            block_length: 3.0,
            seed: Some(42),
            ..McsOptions::default()
        };
        let r = model_confidence_set(&l, &opts).unwrap();
        assert_eq!(r.eliminated_models, vec![1]);
        assert_eq!(r.n_models, 3);
        assert!(!r.elimination_sequence.is_empty());
        assert!(r.elimination_sequence[0].eliminated);

        assert_eq!(r.elimination_order.len(), 3);
        assert_eq!(r.elimination_order[0], 1);
        assert_eq!(r.mcs_p_values.len(), 3);
        assert_eq!(r.mcs_p_values[*r.elimination_order.last().unwrap()], 1.0);

        let single = model_confidence_set(&l[..1], &opts).unwrap();
        assert_eq!(single.included_models, vec![0]);
        assert!(model_confidence_set(&[], &opts).is_err());
    }

    #[test]
    fn test_mcs_reproducible_with_many_models() {
        let l: Vec<Vec<f64>> = (0..5)
            .map(|k| losses(1.0 + 0.05 * k as f64, 0.4, k as f64, 120))
            .collect();
        for bootstrap in [
            McsBootstrap::Stationary,
            McsBootstrap::MovingBlock,
            McsBootstrap::CircularBlock,
            McsBootstrap::Iid,
        ] {
            let opts = McsOptions {
                n_bootstrap: 300,
                bootstrap,
                block_length: 3.0,
                seed: Some(9),
                ..McsOptions::default()
            };
            let a = model_confidence_set(&l, &opts).unwrap();
            for _ in 0..3 {
                let b = model_confidence_set(&l, &opts).unwrap();
                assert_eq!(a.mcs_p_values, b.mcs_p_values);
                assert_eq!(a.elimination_order, b.elimination_order);
            }
            assert_eq!(a.bootstrap, bootstrap);
        }
    }

    #[test]
    fn test_mcs_block_length_resolution() {
        let (_, l) = resolve_mcs_bootstrap(McsBootstrap::MovingBlock, 0.0, 125).unwrap();
        assert_eq!(l, 5.0);
        let (_, l) = resolve_mcs_bootstrap(McsBootstrap::Stationary, 0.0, 100).unwrap();
        assert!((l - 100f64.cbrt()).abs() < 1e-12);
        let (_, l) = resolve_mcs_bootstrap(McsBootstrap::Iid, 7.0, 100).unwrap();
        assert_eq!(l, 1.0);
        assert!(resolve_mcs_bootstrap(McsBootstrap::MovingBlock, 2.5, 100).is_err());
        assert!(resolve_mcs_bootstrap(McsBootstrap::Stationary, 0.5, 100).is_err());
    }
}
