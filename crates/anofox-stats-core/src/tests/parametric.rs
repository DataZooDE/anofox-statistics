//! Parametric statistical tests
//!
//! - t-test (Welch, Student, Paired)
//! - Yuen test (robust with trimmed means)
//! - One-way ANOVA (Fisher, Welch)
//! - Brown-Forsythe test

use crate::{StatsError, StatsResult};
use super::{convert_error, filter_nan, AnovaResult, TestResult};
use anofox_tests::{
    t_test as lib_t_test,
    yuen_test as lib_yuen_test,
    brown_forsythe as lib_brown_forsythe,
    one_way_anova as lib_one_way_anova,
    Alternative, TTestKind, AnovaKind,
};

/// Options for t-test
#[derive(Debug, Clone)]
pub struct TTestOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Test kind: Welch (default), Student, or Paired
    pub kind: TTestKind,
    /// Confidence level for CI (default: 0.95)
    pub confidence_level: Option<f64>,
    /// Hypothesized mean difference (default: 0.0)
    pub mu: f64,
}

impl Default for TTestOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            kind: TTestKind::Welch,
            confidence_level: Some(0.95),
            mu: 0.0,
        }
    }
}

/// Two-sample t-test
///
/// Performs Welch's t-test (default), Student's t-test, or paired t-test.
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
///
/// # Returns
/// Test result with t-statistic, p-value, df, and CI
pub fn t_test(group1: &[f64], group2: &[f64], options: &TTestOptions) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "t-test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "t-test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_t_test(
        &g1,
        &g2,
        options.kind,
        options.alternative,
        options.mu,
        options.confidence_level,
    ).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: f64::NAN, // TTestResult doesn't include effect size
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!("{:?} t-test", options.kind),
    })
}

/// Options for Yuen test
#[derive(Debug, Clone)]
pub struct YuenOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Trim proportion (default: 0.2)
    pub trim: f64,
    /// Confidence level for CI (default: 0.95)
    pub confidence_level: Option<f64>,
}

impl Default for YuenOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            trim: 0.2,
            confidence_level: Some(0.95),
        }
    }
}

/// Yuen test for trimmed means
///
/// A robust alternative to the t-test using trimmed means.
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
pub fn yuen_test(group1: &[f64], group2: &[f64], options: &YuenOptions) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "Yuen test requires at least 4 observations in group 1".into(),
        ));
    }
    if g2.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "Yuen test requires at least 4 observations in group 2".into(),
        ));
    }

    let result = lib_yuen_test(&g1, &g2, options.trim, options.alternative, options.confidence_level)
        .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: f64::NAN,
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!("Yuen test (trim={:.2})", options.trim),
    })
}

/// Options for one-way ANOVA
#[derive(Debug, Clone)]
pub struct AnovaOptions {
    /// ANOVA kind: Fisher (default) or Welch
    pub kind: AnovaKind,
}

impl Default for AnovaOptions {
    fn default() -> Self {
        Self {
            kind: AnovaKind::Fisher,
        }
    }
}

/// One-way ANOVA
///
/// Compares means across multiple groups.
///
/// # Arguments
/// * `groups` - Vector of group data
/// * `options` - Test options
pub fn one_way_anova(groups: &[Vec<f64>], options: &AnovaOptions) -> StatsResult<AnovaResult> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "ANOVA requires at least 2 groups".into(),
        ));
    }

    let filtered: Vec<Vec<f64>> = groups.iter().map(|g| filter_nan(g)).collect();

    for (i, g) in filtered.iter().enumerate() {
        if g.len() < 2 {
            return Err(StatsError::InsufficientDataMsg(
                format!("ANOVA requires at least 2 observations per group (group {} has {})", i, g.len()),
            ));
        }
    }

    let refs: Vec<&[f64]> = filtered.iter().map(|v| v.as_slice()).collect();
    let result = lib_one_way_anova(&refs, options.kind).map_err(convert_error)?;

    let n: usize = filtered.iter().map(|g| g.len()).sum();

    Ok(AnovaResult {
        f_statistic: result.statistic,
        p_value: result.p_value,
        df_between: result.df_between as usize,
        df_within: result.df_within as usize,
        ss_between: result.ss_between.unwrap_or(f64::NAN),
        ss_within: result.ss_within.unwrap_or(f64::NAN),
        n_groups: filtered.len(),
        n,
        method: format!("{:?} One-way ANOVA", options.kind),
    })
}

/// Brown-Forsythe test for homogeneity of variances
///
/// Tests whether groups have equal variances.
///
/// # Arguments
/// * `groups` - Vector of group data
pub fn brown_forsythe(groups: &[Vec<f64>]) -> StatsResult<TestResult> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Brown-Forsythe test requires at least 2 groups".into(),
        ));
    }

    let filtered: Vec<Vec<f64>> = groups.iter().map(|g| filter_nan(g)).collect();

    for (i, g) in filtered.iter().enumerate() {
        if g.len() < 2 {
            return Err(StatsError::InsufficientDataMsg(
                format!("Brown-Forsythe test requires at least 2 observations per group (group {} has {})", i, g.len()),
            ));
        }
    }

    let refs: Vec<&[f64]> = filtered.iter().map(|v| v.as_slice()).collect();
    let result = lib_brown_forsythe(&refs).map_err(convert_error)?;

    let n: usize = filtered.iter().map(|g| g.len()).sum();

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df1,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n,
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "Brown-Forsythe test".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test_welch() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let opts = TTestOptions::default();
        let result = t_test(&g1, &g2, &opts).unwrap();

        assert!(result.statistic < 0.0); // g1 mean < g2 mean
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
        assert!(result.df > 0.0);
    }

    #[test]
    fn test_one_way_anova() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let opts = AnovaOptions::default();
        let result = one_way_anova(&groups, &opts).unwrap();

        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }
}
