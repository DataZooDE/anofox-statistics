//! Resampling methods
//!
//! - Permutation t-test
//! - Bootstrap methods (stationary, circular block)

use super::{convert_error, filter_nan, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    bootstrap_mean_ci as lib_bootstrap_mean_ci, permutation_t_test as lib_permutation_t_test,
    Alternative,
};

/// Options for permutation t-test
#[derive(Debug, Clone)]
pub struct PermutationTTestOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Number of permutations
    pub n_permutations: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PermutationTTestOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            n_permutations: 10000,
            seed: None,
        }
    }
}

/// Permutation t-test
///
/// Distribution-free alternative to the t-test using permutation.
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
pub fn permutation_t_test(
    group1: &[f64],
    group2: &[f64],
    options: &PermutationTTestOptions,
) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Permutation t-test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Permutation t-test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_permutation_t_test(
        &g1,
        &g2,
        options.alternative,
        options.n_permutations,
        options.seed,
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
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!(
            "Permutation t-test ({} permutations)",
            options.n_permutations
        ),
    })
}

/// Bootstrap result
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Original statistic
    pub statistic: f64,
    /// Bootstrap standard error
    pub se: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
}

/// Options for bootstrap
#[derive(Debug, Clone)]
pub struct BootstrapOptions {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Confidence level
    pub confidence_level: f64,
    /// Block length for block bootstrap (0 for iid bootstrap)
    pub block_length: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            n_bootstrap: 10000,
            confidence_level: 0.95,
            block_length: 0,
            seed: None,
        }
    }
}

/// Bootstrap mean confidence interval
///
/// Percentile bootstrap (delegates to `anofox_statistics::bootstrap_mean_ci`):
/// IID resampling for `block_length == 0`, stationary bootstrap for 1,
/// circular block bootstrap for larger block lengths.
///
/// # Arguments
/// * `data` - Sample data
/// * `options` - Bootstrap options
pub fn bootstrap_mean(data: &[f64], options: &BootstrapOptions) -> StatsResult<BootstrapResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Bootstrap requires at least 2 observations".into(),
        ));
    }

    if options.n_bootstrap < 2 {
        return Err(StatsError::InvalidInput(format!(
            "n_bootstrap must be >= 2, got {}",
            options.n_bootstrap
        )));
    }
    if !(options.confidence_level > 0.0 && options.confidence_level < 1.0) {
        return Err(StatsError::InvalidInput(format!(
            "confidence_level must be in (0, 1), got {}",
            options.confidence_level
        )));
    }

    let result = lib_bootstrap_mean_ci(
        &filtered,
        options.n_bootstrap,
        options.confidence_level,
        options.block_length,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(BootstrapResult {
        statistic: result.estimate,
        se: result.se,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        n_bootstrap: result.n_bootstrap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_t_test() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let opts = PermutationTTestOptions {
            n_permutations: 1000,
            seed: Some(42), // For reproducibility
            ..Default::default()
        };
        let result = permutation_t_test(&g1, &g2, &opts).unwrap();

        assert!(result.p_value < 0.05); // Should be significant
    }

    #[test]
    fn test_bootstrap_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let opts = BootstrapOptions {
            n_bootstrap: 1000,
            seed: Some(42),
            ..Default::default()
        };
        let result = bootstrap_mean(&data, &opts).unwrap();

        assert!((result.statistic - 5.5).abs() < 0.01); // Mean should be 5.5
        assert!(result.ci_lower < 5.5 && result.ci_upper > 5.5);
    }

    #[test]
    fn test_bootstrap_rejects_degenerate_options() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        for n_bootstrap in [0, 1] {
            let opts = BootstrapOptions {
                n_bootstrap,
                seed: Some(1),
                ..Default::default()
            };
            assert!(bootstrap_mean(&data, &opts).is_err());
        }
        for cl in [0.0, 1.0, f64::NAN, 1.5] {
            let opts = BootstrapOptions {
                n_bootstrap: 100,
                confidence_level: cl,
                seed: Some(1),
                ..Default::default()
            };
            assert!(bootstrap_mean(&data, &opts).is_err());
        }
    }

    #[test]
    fn test_bootstrap_with_infinities_does_not_panic() {
        let data = vec![f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0];
        let opts = BootstrapOptions {
            n_bootstrap: 200,
            seed: Some(7),
            ..Default::default()
        };
        assert!(bootstrap_mean(&data, &opts).is_ok());
    }
}
