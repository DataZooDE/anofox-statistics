//! Nonparametric statistical tests
//!
//! - Mann-Whitney U test
//! - Wilcoxon signed-rank test
//! - Kruskal-Wallis test
//! - Brunner-Munzel test

use crate::{StatsError, StatsResult};
use super::{convert_error, filter_nan, TestResult};
use anofox_tests::{
    mann_whitney_u as lib_mann_whitney_u,
    wilcoxon_signed_rank as lib_wilcoxon_signed_rank,
    kruskal_wallis as lib_kruskal_wallis,
    brunner_munzel as lib_brunner_munzel,
    Alternative,
};

/// Options for Mann-Whitney U test
#[derive(Debug, Clone)]
pub struct MannWhitneyOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Use exact distribution (for small samples)
    pub exact: bool,
    /// Apply continuity correction
    pub continuity_correction: bool,
    /// Confidence level for CI
    pub confidence_level: Option<f64>,
    /// Hypothesized location shift
    pub mu: Option<f64>,
}

impl Default for MannWhitneyOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            exact: false,
            continuity_correction: true,
            confidence_level: Some(0.95),
            mu: None,
        }
    }
}

/// Mann-Whitney U test (Wilcoxon rank-sum test)
///
/// Nonparametric test for comparing two independent samples.
pub fn mann_whitney_u(
    group1: &[f64],
    group2: &[f64],
    options: &MannWhitneyOptions,
) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.is_empty() || g2.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Mann-Whitney U test requires at least 1 observation per group".into(),
        ));
    }

    let result = lib_mann_whitney_u(
        &g1,
        &g2,
        options.alternative,
        options.continuity_correction,
        options.exact,
        options.confidence_level,
        options.mu,
    ).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: f64::NAN, // Not provided by library
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: "Mann-Whitney U test".into(),
    })
}

/// Options for Wilcoxon signed-rank test
#[derive(Debug, Clone)]
pub struct WilcoxonOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Use exact distribution (for small samples)
    pub exact: bool,
    /// Apply continuity correction
    pub continuity_correction: bool,
    /// Confidence level for CI
    pub confidence_level: Option<f64>,
    /// Hypothesized median
    pub mu: Option<f64>,
}

impl Default for WilcoxonOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            exact: false,
            continuity_correction: true,
            confidence_level: Some(0.95),
            mu: None,
        }
    }
}

/// Wilcoxon signed-rank test
///
/// Nonparametric test for paired samples.
pub fn wilcoxon_signed_rank(
    x: &[f64],
    y: &[f64],
    options: &WilcoxonOptions,
) -> StatsResult<TestResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Wilcoxon signed-rank test requires equal length samples".into(),
        ));
    }

    // Filter paired values where neither is NaN
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Wilcoxon signed-rank test requires at least 2 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result = lib_wilcoxon_signed_rank(
        &x_filtered,
        &y_filtered,
        options.alternative,
        options.continuity_correction,
        options.exact,
        options.confidence_level,
        options.mu,
    ).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: f64::NAN, // Not provided by library
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: x_filtered.len(),
        n1: x_filtered.len(),
        n2: y_filtered.len(),
        alternative: options.alternative,
        method: "Wilcoxon signed-rank test".into(),
    })
}

/// Kruskal-Wallis H test
///
/// Nonparametric test for comparing multiple independent groups.
pub fn kruskal_wallis(groups: &[Vec<f64>]) -> StatsResult<TestResult> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Kruskal-Wallis test requires at least 2 groups".into(),
        ));
    }

    let filtered: Vec<Vec<f64>> = groups.iter().map(|g| filter_nan(g)).collect();

    for (i, g) in filtered.iter().enumerate() {
        if g.is_empty() {
            return Err(StatsError::InsufficientDataMsg(
                format!("Kruskal-Wallis test requires at least 1 observation per group (group {} is empty)", i),
            ));
        }
    }

    let refs: Vec<&[f64]> = filtered.iter().map(|v| v.as_slice()).collect();
    let result = lib_kruskal_wallis(&refs).map_err(convert_error)?;

    let n: usize = filtered.iter().map(|g| g.len()).sum();

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: (filtered.len() - 1) as f64,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n,
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "Kruskal-Wallis H test".into(),
    })
}

/// Options for Brunner-Munzel test
#[derive(Debug, Clone)]
pub struct BrunnerMunzelOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Confidence level
    pub confidence_level: f64,
}

impl Default for BrunnerMunzelOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            confidence_level: 0.95,
        }
    }
}

/// Brunner-Munzel test
///
/// Tests for stochastic equality between two groups.
pub fn brunner_munzel(
    group1: &[f64],
    group2: &[f64],
    options: &BrunnerMunzelOptions,
) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Brunner-Munzel test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Brunner-Munzel test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_brunner_munzel(&g1, &g2, options.alternative, Some(options.confidence_level))
        .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: result.estimate, // Probability estimate
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level,
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: "Brunner-Munzel test".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mann_whitney_u() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let opts = MannWhitneyOptions::default();
        let result = mann_whitney_u(&g1, &g2, &opts).unwrap();

        assert!(result.p_value < 0.05); // Should be significant
    }

    #[test]
    fn test_wilcoxon_signed_rank() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let opts = WilcoxonOptions::default();
        let result = wilcoxon_signed_rank(&x, &y, &opts).unwrap();

        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_kruskal_wallis() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = kruskal_wallis(&groups).unwrap();

        assert!(result.p_value < 0.05); // Should be significant
    }
}
