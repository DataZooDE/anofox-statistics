//! Resampling methods
//!
//! - Permutation t-test
//! - Bootstrap methods (stationary, circular block)

use super::{convert_error, filter_nan, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{permutation_t_test as lib_permutation_t_test, Alternative};

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
/// Uses percentile method for CI estimation.
///
/// # Arguments
/// * `data` - Sample data
/// * `options` - Bootstrap options
pub fn bootstrap_mean(data: &[f64], options: &BootstrapOptions) -> StatsResult<BootstrapResult> {
    use anofox_tests::{CircularBlockBootstrap, StationaryBootstrap};

    let filtered = filter_nan(data);

    if filtered.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Bootstrap requires at least 2 observations".into(),
        ));
    }

    // Compute original mean
    let original_mean: f64 = filtered.iter().sum::<f64>() / filtered.len() as f64;
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(options.n_bootstrap);

    if options.block_length > 0 {
        // Block bootstrap for dependent data
        if options.block_length > 1 {
            // Circular block bootstrap
            let mut cb = CircularBlockBootstrap::new(options.block_length, options.seed);
            let samples = cb.samples(&filtered, filtered.len(), options.n_bootstrap);
            for sample in samples {
                let sample_mean = sample.iter().sum::<f64>() / sample.len() as f64;
                bootstrap_means.push(sample_mean);
            }
        } else {
            // Stationary bootstrap with expected block length = 1
            let mut sb = StationaryBootstrap::new(1.0, options.seed);
            let samples = sb.samples(&filtered, filtered.len(), options.n_bootstrap);
            for sample in samples {
                let sample_mean = sample.iter().sum::<f64>() / sample.len() as f64;
                bootstrap_means.push(sample_mean);
            }
        }
    } else {
        // IID bootstrap
        use rand::prelude::*;
        let mut rng = match options.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        for _ in 0..options.n_bootstrap {
            let sample: Vec<f64> = (0..filtered.len())
                .map(|_| filtered[rng.gen_range(0..filtered.len())])
                .collect();
            let sample_mean = sample.iter().sum::<f64>() / sample.len() as f64;
            bootstrap_means.push(sample_mean);
        }
    }

    // Sort for percentile method
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - options.confidence_level;
    let lower_idx = ((alpha / 2.0) * options.n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * options.n_bootstrap as f64).ceil() as usize;

    let se = {
        let mean_of_means = bootstrap_means.iter().sum::<f64>() / options.n_bootstrap as f64;
        let variance = bootstrap_means
            .iter()
            .map(|x| (x - mean_of_means).powi(2))
            .sum::<f64>()
            / (options.n_bootstrap - 1) as f64;
        variance.sqrt()
    };

    Ok(BootstrapResult {
        statistic: original_mean,
        se,
        ci_lower: bootstrap_means[lower_idx.min(bootstrap_means.len() - 1)],
        ci_upper: bootstrap_means[upper_idx.min(bootstrap_means.len() - 1)],
        n_bootstrap: options.n_bootstrap,
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
}
