//! Modern distribution tests
//!
//! - Energy distance test
//! - Maximum Mean Discrepancy (MMD) test

use crate::{StatsError, StatsResult};
use super::{convert_error, filter_nan, TestResult};
use anofox_tests::{
    energy_distance_test_1d as lib_energy_distance_test,
    mmd_test_1d as lib_mmd_test,
    Alternative,
};

/// Options for energy distance test
#[derive(Debug, Clone)]
pub struct EnergyDistanceOptions {
    /// Number of permutations for p-value
    pub n_permutations: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for EnergyDistanceOptions {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: None,
        }
    }
}

/// Energy distance two-sample test
///
/// Multivariate two-sample test based on energy distance.
/// Can detect differences in both location and scale.
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
pub fn energy_distance_test(
    group1: &[f64],
    group2: &[f64],
    options: &EnergyDistanceOptions,
) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Energy distance test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Energy distance test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_energy_distance_test(&g1, &g2, options.n_permutations, options.seed)
        .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: result.statistic, // Energy distance as effect size
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: Alternative::TwoSided,
        method: format!("Energy distance test ({} permutations)", options.n_permutations),
    })
}

/// Options for MMD test
#[derive(Debug, Clone)]
pub struct MmdOptions {
    /// Number of permutations
    pub n_permutations: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for MmdOptions {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: None,
        }
    }
}

/// Maximum Mean Discrepancy (MMD) two-sample test
///
/// Kernel-based two-sample test that can detect complex distributional differences.
/// Uses Gaussian kernel with automatic bandwidth selection (median heuristic).
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
pub fn mmd_test(
    group1: &[f64],
    group2: &[f64],
    options: &MmdOptions,
) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "MMD test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "MMD test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_mmd_test(&g1, &g2, options.n_permutations, options.seed)
        .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: result.statistic,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: Alternative::TwoSided,
        method: format!("MMD test (Gaussian kernel, {} permutations)", options.n_permutations),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_distance() {
        // Same distribution should give high p-value
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let opts = EnergyDistanceOptions {
            n_permutations: 500,
            seed: Some(42),
        };
        let result = energy_distance_test(&g1, &g2, &opts).unwrap();

        // Statistic should be valid (not NaN)
        assert!(!result.statistic.is_nan());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mmd() {
        // Different distributions should have low p-value
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let opts = MmdOptions {
            n_permutations: 500,
            seed: Some(42),
        };
        let result = mmd_test(&g1, &g2, &opts).unwrap();

        assert!(result.statistic >= 0.0);
        // Should detect the difference
        assert!(result.p_value < 0.1);
    }
}
