//! Parametric statistical tests
//!
//! - t-test (Welch, Student, Paired)
//! - Yuen test (robust with trimmed means)
//! - One-way ANOVA (Fisher, Welch)
//! - Brown-Forsythe test

use super::{convert_error, filter_nan, AnovaResult, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    brown_forsythe as lib_brown_forsythe, cohens_d as lib_cohens_d,
    cohens_d_one_sample as lib_cohens_d_one_sample, one_way_anova as lib_one_way_anova,
    t_test as lib_t_test, t_test_one_sample as lib_t_test_one_sample, yuen_test as lib_yuen_test,
    Alternative, AnovaKind, TTestKind,
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
    )
    .map_err(convert_error)?;

    // Cohen's d (R effectsize::cohens_d): pooled SD for Student, average
    // variance for Welch, d_z for paired samples.
    let cohens_d = lib_cohens_d(&g1, &g2, options.kind, options.mu).unwrap_or(f64::NAN);
    let mut unified = super::UnifiedTestResult::from(&result)
        .with_alternative(options.alternative)
        .with_n(g1.len() + g2.len());
    if !cohens_d.is_nan() {
        unified = unified.with_effect_size("cohens_d", cohens_d);
    }

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: cohens_d,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!("{:?} t-test", options.kind),
        unified,
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

    let result = lib_yuen_test(
        &g1,
        &g2,
        options.trim,
        options.alternative,
        options.confidence_level,
    )
    .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: f64::NAN,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!("Yuen test (trim={:.2})", options.trim),
        unified: super::UnifiedTestResult::from(&result)
            .with_alternative(options.alternative)
            .with_n(g1.len() + g2.len()),
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
            return Err(StatsError::InsufficientDataMsg(format!(
                "ANOVA requires at least 2 observations per group (group {} has {})",
                i,
                g.len()
            )));
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
        unified: super::UnifiedTestResult::from(&result).with_n(n),
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
            return Err(StatsError::InsufficientDataMsg(format!(
                "Brown-Forsythe test requires at least 2 observations per group (group {} has {})",
                i,
                g.len()
            )));
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
        unified: super::UnifiedTestResult::from(&result).with_n(n),
    })
}

/// Build a [`TestResult`] from an upstream paired / one-sample t-test result.
fn paired_like_result(
    result: &anofox_tests::TTestResult,
    cohens_d: f64,
    n: usize,
    n2: usize,
    options: &TTestOptions,
    method: &str,
) -> TestResult {
    let mut unified = super::UnifiedTestResult::from(result)
        .with_method(method)
        .with_alternative(options.alternative)
        .with_n(n);
    if !cohens_d.is_nan() {
        unified = unified.with_effect_size("cohens_d", cohens_d);
    }
    TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: cohens_d,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n,
        n1: n,
        n2,
        alternative: options.alternative,
        method: method.into(),
        unified,
    }
}

/// One-sample t-test of H0: mean(x) = `options.mu` (R `t.test(x, mu = mu)`),
/// upstream `t_test_one_sample`.
///
/// `options.kind` is ignored. Cohen's d is upstream `cohens_d_one_sample`
/// ((mean - mu) / sd). A one-sided alternative gives R's one-sided interval
/// (one infinite bound).
pub fn t_test_one_sample(x: &[f64], options: &TTestOptions) -> StatsResult<TestResult> {
    let x = filter_nan(x);
    if x.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "one-sample t-test requires at least 2 observations".into(),
        ));
    }
    let result = lib_t_test_one_sample(
        &x,
        options.alternative,
        options.mu,
        options.confidence_level,
    )
    .map_err(convert_error)?;
    let cohens_d = lib_cohens_d_one_sample(&x, options.mu).unwrap_or(f64::NAN);
    Ok(paired_like_result(
        &result,
        cohens_d,
        x.len(),
        0,
        options,
        "One Sample t-test",
    ))
}

/// Paired t-test of H0: mean(x - y) = `options.mu` (R `t.test(x, y, paired =
/// TRUE)`). Pairs with a NaN in either sample are dropped; `options.kind` is
/// ignored. Cohen's d is d_z = (mean(d) - mu) / sd(d) (upstream `cohens_d`).
pub fn t_test_paired(x: &[f64], y: &[f64], options: &TTestOptions) -> StatsResult<TestResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "paired t-test requires equal length samples".into(),
        ));
    }
    let (xs, ys): (Vec<f64>, Vec<f64>) = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .unzip();
    if xs.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "paired t-test requires at least 2 valid pairs".into(),
        ));
    }
    let result = lib_t_test(
        &xs,
        &ys,
        TTestKind::Paired,
        options.alternative,
        options.mu,
        options.confidence_level,
    )
    .map_err(convert_error)?;
    let cohens_d = lib_cohens_d(&xs, &ys, TTestKind::Paired, options.mu).unwrap_or(f64::NAN);
    Ok(paired_like_result(
        &result,
        cohens_d,
        xs.len(),
        ys.len(),
        options,
        "Paired t-test",
    ))
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

    // R: t.test(c(5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1), mu = 5)
    #[test]
    fn test_t_test_one_sample_matches_r() {
        let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let opts = TTestOptions {
            mu: 5.0,
            ..Default::default()
        };
        let r = t_test_one_sample(&x, &opts).unwrap();
        assert!((r.statistic - 3.562099364).abs() < 1e-8, "{}", r.statistic);
        assert_eq!(r.df, 7.0);
        assert_eq!(r.n, 8);
        assert!(r.effect_size > 0.0);
        assert!(t_test_one_sample(&[1.0], &opts).is_err());
    }

    // R: t.test(x, mu = 5, alternative = "greater" / "less")$conf.int
    #[test]
    fn test_t_test_one_sample_one_sided_ci_matches_r() {
        let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let greater = TTestOptions {
            mu: 5.0,
            alternative: Alternative::Greater,
            ..Default::default()
        };
        let r = t_test_one_sample(&x, &greater).unwrap();
        assert!((r.ci_lower - 5.28965460085684).abs() < 1e-10);
        assert_eq!(r.ci_upper, f64::INFINITY);
        assert!((r.p_value - 0.00459678543466782).abs() < 1e-12);
        let less = TTestOptions {
            alternative: Alternative::Less,
            ..greater
        };
        let r = t_test_one_sample(&x, &less).unwrap();
        assert_eq!(r.ci_lower, f64::NEG_INFINITY);
        assert!((r.ci_upper - 5.94784539914316).abs() < 1e-10);
    }

    #[test]
    fn test_t_test_paired_drops_nan_pairs() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 5.5];
        let y = [1.5, 2.1, 3.0, 4.6, 5.0];
        let r = t_test_paired(&x, &y, &TTestOptions::default()).unwrap();
        assert_eq!(r.n, 4);
        assert_eq!(r.df, 3.0);
        assert!(t_test_paired(&x, &y[..4], &TTestOptions::default()).is_err());
    }
}
