//! Equivalence testing (TOST)
//!
//! Two One-Sided Tests (TOST) for equivalence testing

use crate::{StatsError, StatsResult};
use super::{convert_error, filter_nan};
use anofox_tests::{
    tost_t_test_one_sample as lib_tost_one_sample,
    tost_t_test_two_sample as lib_tost_two_sample,
    tost_t_test_paired as lib_tost_paired,
    tost_correlation as lib_tost_correlation,
    tost_prop_one as lib_tost_prop_one,
    tost_prop_two as lib_tost_prop_two,
    tost_wilcoxon_paired as lib_tost_wilcoxon_paired,
    tost_wilcoxon_two_sample as lib_tost_wilcoxon_two_sample,
    tost_bootstrap as lib_tost_bootstrap,
    tost_yuen as lib_tost_yuen,
    EquivalenceBounds, CorrelationTostMethod,
};

/// TOST result
#[derive(Debug, Clone)]
pub struct TostResult {
    /// Point estimate
    pub estimate: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Equivalence bounds lower
    pub bounds_lower: f64,
    /// Equivalence bounds upper
    pub bounds_upper: f64,
    /// Lower bound test statistic
    pub statistic_lower: f64,
    /// Upper bound test statistic
    pub statistic_upper: f64,
    /// p-value for lower bound test
    pub p_value_lower: f64,
    /// p-value for upper bound test
    pub p_value_upper: f64,
    /// Overall TOST p-value (max of lower and upper)
    pub p_value: f64,
    /// Degrees of freedom
    pub df: f64,
    /// Whether equivalence is established
    pub equivalent: bool,
    /// Sample size
    pub n: usize,
    /// Test method
    pub method: String,
}

/// Helper function to convert library TostResult to our TostResult
fn convert_tost_result(result: anofox_tests::TostResult) -> TostResult {
    TostResult {
        estimate: result.estimate,
        ci_lower: result.ci.0,
        ci_upper: result.ci.1,
        bounds_lower: result.bounds.0,
        bounds_upper: result.bounds.1,
        statistic_lower: result.lower_test.statistic,
        statistic_upper: result.upper_test.statistic,
        p_value_lower: result.lower_test.p_value,
        p_value_upper: result.upper_test.p_value,
        p_value: result.tost_p_value,
        df: result.df.unwrap_or(f64::NAN),
        equivalent: result.equivalent,
        n: result.n,
        method: result.method,
    }
}

/// Equivalence bounds specification
#[derive(Debug, Clone)]
pub enum TostBounds {
    /// Raw bounds (lower, upper)
    Raw { lower: f64, upper: f64 },
    /// Symmetric bounds (-delta, +delta)
    Symmetric { delta: f64 },
    /// Cohen's d effect size bounds
    CohenD { d: f64 },
}

impl TostBounds {
    fn to_lib_bounds(&self) -> StatsResult<EquivalenceBounds> {
        match self {
            TostBounds::Raw { lower, upper } => EquivalenceBounds::raw(*lower, *upper)
                .map_err(|e| StatsError::InvalidInput(e.to_string())),
            TostBounds::Symmetric { delta } => EquivalenceBounds::symmetric(*delta)
                .map_err(|e| StatsError::InvalidInput(e.to_string())),
            TostBounds::CohenD { d } => EquivalenceBounds::cohen_d(*d)
                .map_err(|e| StatsError::InvalidInput(e.to_string())),
        }
    }
}

/// Options for TOST t-test
#[derive(Debug, Clone)]
pub struct TostTTestOptions {
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Significance level (default: 0.05)
    pub alpha: f64,
    /// Use pooled variance (Student's t) vs Welch's t (for two-sample)
    pub pooled: bool,
}

impl Default for TostTTestOptions {
    fn default() -> Self {
        Self {
            bounds: TostBounds::Symmetric { delta: 0.5 },
            alpha: 0.05,
            pooled: false, // Welch's t by default
        }
    }
}

/// One-sample TOST t-test
///
/// Tests whether a sample mean is equivalent to a hypothesized value.
///
/// # Arguments
/// * `data` - Sample data
/// * `mu` - Hypothesized mean
/// * `options` - Test options
pub fn tost_t_test_one_sample(
    data: &[f64],
    mu: f64,
    options: &TostTTestOptions,
) -> StatsResult<TostResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST one-sample t-test requires at least 2 observations".into(),
        ));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_one_sample(&filtered, mu, &bounds, options.alpha)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Two-sample TOST t-test
///
/// Tests whether two group means are equivalent.
///
/// # Arguments
/// * `group1` - First sample
/// * `group2` - Second sample
/// * `options` - Test options
pub fn tost_t_test_two_sample(
    group1: &[f64],
    group2: &[f64],
    options: &TostTTestOptions,
) -> StatsResult<TostResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 || g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST two-sample t-test requires at least 2 observations per group".into(),
        ));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_two_sample(&g1, &g2, &bounds, options.alpha, options.pooled)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Paired TOST t-test
///
/// Tests whether paired differences are equivalent to zero.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (paired with x)
/// * `options` - Test options
pub fn tost_t_test_paired(
    x: &[f64],
    y: &[f64],
    options: &TostTTestOptions,
) -> StatsResult<TostResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Paired samples must have same length".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST paired t-test requires at least 2 valid pairs".into(),
        ));
    }

    let (x_f, y_f): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_paired(&x_f, &y_f, &bounds, options.alpha)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Correlation method for TOST
#[derive(Debug, Clone, Copy)]
pub enum TostCorrelationMethod {
    Pearson,
    Spearman,
}

impl From<TostCorrelationMethod> for CorrelationTostMethod {
    fn from(m: TostCorrelationMethod) -> Self {
        match m {
            TostCorrelationMethod::Pearson => CorrelationTostMethod::Pearson,
            TostCorrelationMethod::Spearman => CorrelationTostMethod::Spearman,
        }
    }
}

/// Options for TOST correlation
#[derive(Debug, Clone)]
pub struct TostCorrelationOptions {
    /// Null value for correlation (usually 0)
    pub rho_null: f64,
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Correlation method
    pub method: TostCorrelationMethod,
    /// Significance level
    pub alpha: f64,
}

impl Default for TostCorrelationOptions {
    fn default() -> Self {
        Self {
            rho_null: 0.0,
            bounds: TostBounds::Symmetric { delta: 0.3 },
            method: TostCorrelationMethod::Pearson,
            alpha: 0.05,
        }
    }
}

/// TOST correlation test
///
/// Tests whether a correlation is negligibly small (equivalent to zero).
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
/// * `options` - Test options
pub fn tost_correlation(
    x: &[f64],
    y: &[f64],
    options: &TostCorrelationOptions,
) -> StatsResult<TostResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Variables must have same length".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST correlation requires at least 4 valid pairs".into(),
        ));
    }

    let (x_f, y_f): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_correlation(
        &x_f,
        &y_f,
        options.rho_null,
        &bounds,
        options.alpha,
        options.method.into(),
    ).map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Options for TOST proportion
#[derive(Debug, Clone)]
pub struct TostPropOptions {
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Significance level
    pub alpha: f64,
}

impl Default for TostPropOptions {
    fn default() -> Self {
        Self {
            bounds: TostBounds::Symmetric { delta: 0.1 },
            alpha: 0.05,
        }
    }
}

/// One-sample TOST proportion test
///
/// Tests whether a proportion is equivalent to a hypothesized value.
///
/// # Arguments
/// * `successes` - Number of successes
/// * `trials` - Number of trials
/// * `p0` - Hypothesized proportion
/// * `options` - Test options
pub fn tost_prop_one(
    successes: usize,
    trials: usize,
    p0: f64,
    options: &TostPropOptions,
) -> StatsResult<TostResult> {
    if trials == 0 {
        return Err(StatsError::InvalidInput("Number of trials must be > 0".into()));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_prop_one(
        successes,
        trials,
        p0,
        &bounds,
        options.alpha,
    ).map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Two-sample TOST proportion test
///
/// Tests whether two proportions are equivalent.
pub fn tost_prop_two(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
    options: &TostPropOptions,
) -> StatsResult<TostResult> {
    if trials1 == 0 || trials2 == 0 {
        return Err(StatsError::InvalidInput("Number of trials must be > 0".into()));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_prop_two(
        successes1, trials1, successes2, trials2,
        &bounds,
        options.alpha,
    ).map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Options for TOST Wilcoxon
#[derive(Debug, Clone)]
pub struct TostWilcoxonOptions {
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Significance level
    pub alpha: f64,
}

impl Default for TostWilcoxonOptions {
    fn default() -> Self {
        Self {
            bounds: TostBounds::Symmetric { delta: 0.5 },
            alpha: 0.05,
        }
    }
}

/// Paired TOST Wilcoxon test
///
/// Nonparametric equivalence test for paired samples.
pub fn tost_wilcoxon_paired(
    x: &[f64],
    y: &[f64],
    options: &TostWilcoxonOptions,
) -> StatsResult<TostResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Paired samples must have same length".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST Wilcoxon requires at least 3 valid pairs".into(),
        ));
    }

    let (x_f, y_f): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_wilcoxon_paired(&x_f, &y_f, &bounds, options.alpha)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Two-sample TOST Wilcoxon test
///
/// Nonparametric equivalence test for two independent samples.
pub fn tost_wilcoxon_two_sample(
    group1: &[f64],
    group2: &[f64],
    options: &TostWilcoxonOptions,
) -> StatsResult<TostResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 3 || g2.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "TOST Wilcoxon requires at least 3 observations per group".into(),
        ));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_wilcoxon_two_sample(&g1, &g2, &bounds, options.alpha)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Options for bootstrap TOST
#[derive(Debug, Clone)]
pub struct TostBootstrapOptions {
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Significance level
    pub alpha: f64,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TostBootstrapOptions {
    fn default() -> Self {
        Self {
            bounds: TostBounds::Symmetric { delta: 0.5 },
            n_bootstrap: 10000,
            alpha: 0.05,
            seed: None,
        }
    }
}

/// Bootstrap TOST
///
/// Distribution-free equivalence test using bootstrap.
pub fn tost_bootstrap(
    group1: &[f64],
    group2: &[f64],
    options: &TostBootstrapOptions,
) -> StatsResult<TostResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 3 || g2.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Bootstrap TOST requires at least 3 observations per group".into(),
        ));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_bootstrap(
        &g1,
        &g2,
        &bounds,
        options.alpha,
        options.n_bootstrap,
        options.seed,
    ).map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

/// Options for Yuen TOST
#[derive(Debug, Clone)]
pub struct TostYuenOptions {
    /// Equivalence bounds
    pub bounds: TostBounds,
    /// Trim proportion
    pub trim: f64,
    /// Significance level
    pub alpha: f64,
}

impl Default for TostYuenOptions {
    fn default() -> Self {
        Self {
            bounds: TostBounds::Symmetric { delta: 0.5 },
            trim: 0.2,
            alpha: 0.05,
        }
    }
}

/// Yuen TOST (robust with trimmed means)
///
/// Robust equivalence test using trimmed means.
pub fn tost_yuen(
    group1: &[f64],
    group2: &[f64],
    options: &TostYuenOptions,
) -> StatsResult<TostResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 4 || g2.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "Yuen TOST requires at least 4 observations per group".into(),
        ));
    }

    let bounds = options.bounds.to_lib_bounds()?;
    let result = lib_tost_yuen(&g1, &g2, &bounds, options.trim, options.alpha)
        .map_err(convert_error)?;

    Ok(convert_tost_result(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tost_two_sample() {
        // Similar groups should be equivalent with wide bounds
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let opts = TostTTestOptions {
            bounds: TostBounds::Symmetric { delta: 2.0 }, // Wide bounds for small sample
            ..Default::default()
        };
        let result = tost_t_test_two_sample(&g1, &g2, &opts).unwrap();

        // Verify test ran successfully
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // With wide bounds and similar groups, should be equivalent
        assert!(result.equivalent);
    }

    #[test]
    fn test_tost_correlation() {
        // Near-zero correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.8, 8.1];
        let opts = TostCorrelationOptions::default();
        let result = tost_correlation(&x, &y, &opts).unwrap();

        // Strong correlation, so not equivalent to zero
        assert!(!result.equivalent);
    }
}
