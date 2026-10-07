//! Nonparametric statistical tests
//!
//! - Mann-Whitney U test
//! - Wilcoxon signed-rank test
//! - Kruskal-Wallis test
//! - Brunner-Munzel test

use super::{convert_error, filter_nan, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    brunner_munzel as lib_brunner_munzel, kruskal_wallis as lib_kruskal_wallis,
    mann_whitney_u as lib_mann_whitney_u, wilcoxon_signed_rank as lib_wilcoxon_signed_rank,
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
    )
    .map_err(convert_error)?;

    // Workarounds for anofox-statistics <= 0.4.2 (normal approximation):
    // * every observation tied -> Var(U) = 0; the library divides by zero and
    //   reports p = 0. U equals its null expectation, so p = 1 (scipy; R gives
    //   NaN two-sided and 1 one-sided).
    // * U == n1*n2/2 (two-sided): R applies no continuity correction
    //   (sign(z) * 0.5 = 0), so z = 0 and p = 1.
    let shift = options.mu.unwrap_or(0.0);
    let first = g1[0];
    let all_tied = g1.iter().all(|&v| v == first) && g2.iter().all(|&v| v + shift == first);
    let at_center = options.alternative == Alternative::TwoSided
        && result.statistic == g1.len() as f64 * g2.len() as f64 / 2.0;
    let p_value = if all_tied || at_center {
        1.0
    } else {
        result.p_value
    };

    Ok(TestResult {
        statistic: result.statistic,
        p_value,
        df: f64::NAN,
        // Rank-biserial correlation r = 1 - 2*U1 / (n1*n2), where U1 (the
        // reported statistic, R's W) counts pairs with group1 > group2 (ties
        // 0.5), after any `mu` shift. Sign convention: r = P(g2 > g1) - P(g1 > g2),
        // so r > 0 when group 2 tends to be LARGER than group 1, r < 0 when
        // group 1 tends to be larger; r is in [-1, 1].
        effect_size: 1.0 - 2.0 * result.statistic / (g1.len() as f64 * g2.len() as f64),
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
    )
    .map_err(convert_error)?;

    // anofox-statistics <= 0.4.2 applies the continuity correction even when
    // V equals its null expectation; R uses sign(z) * 0.5 = 0 there, so p = 1.
    let shift = options.mu.unwrap_or(0.0);
    let n_nonzero = x_filtered
        .iter()
        .zip(y_filtered.iter())
        .filter(|(a, b)| *a - *b - shift != 0.0)
        .count() as f64;
    let p_value = if options.alternative == Alternative::TwoSided
        && result.statistic == n_nonzero * (n_nonzero + 1.0) / 4.0
    {
        1.0
    } else {
        result.p_value
    };

    Ok(TestResult {
        statistic: result.statistic,
        p_value,
        df: f64::NAN,
        effect_size: f64::NAN, // Not provided by library
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
            return Err(StatsError::InsufficientDataMsg(format!(
                "Kruskal-Wallis test requires at least 1 observation per group (group {} is empty)",
                i
            )));
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

    let result = lib_brunner_munzel(
        &g1,
        &g2,
        options.alternative,
        // The library takes alpha (CI level = 1 - alpha), not the confidence level.
        Some(1.0 - options.confidence_level),
    )
    .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: result.estimate, // Probability estimate
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
                                        // Complete separation with group 2 larger: U1 = 0 -> r = +1.
        assert!((result.effect_size - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mann_whitney_rank_biserial() {
        // Pairs (g1 > g2): only 3 > 2 -> U1 = 1; r = 1 - 2*1/(2*2) = 0.5.
        let r = mann_whitney_u(&[1.0, 3.0], &[2.0, 4.0], &MannWhitneyOptions::default()).unwrap();
        assert!((r.statistic - 1.0).abs() < 1e-12);
        assert!((r.effect_size - 0.5).abs() < 1e-12);
        // Swapping the groups flips the sign.
        let r = mann_whitney_u(&[2.0, 4.0], &[1.0, 3.0], &MannWhitneyOptions::default()).unwrap();
        assert!((r.effect_size + 0.5).abs() < 1e-12);
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

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_brunner_munzel_ci_level() {
        // lawstat::brunner.munzel.test: p_hat -/+ qt(0.975, df) * se
        let g1 = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let g2 = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];
        let r = brunner_munzel(&g1, &g2, &BrunnerMunzelOptions::default()).unwrap();
        assert!((r.ci_lower - 0.8722553373834635).abs() < 1e-9);
        assert!((r.ci_upper - 1.0527446626165362).abs() < 1e-9);
    }

    #[test]
    fn test_mann_whitney_all_tied_and_center() {
        let opts = MannWhitneyOptions::default();
        let r = mann_whitney_u(&[5.0; 6], &[5.0; 7], &opts).unwrap();
        assert_eq!(r.p_value, 1.0);
        // R: wilcox.test(c(1, 4), c(2, 3), exact = FALSE)$p.value == 1
        let r = mann_whitney_u(&[1.0, 4.0], &[2.0, 3.0], &opts).unwrap();
        assert_eq!(r.p_value, 1.0);
        // R: wilcox.test(1:4, 4:1, paired = TRUE, exact = FALSE)$p.value == 1
        let r = wilcoxon_signed_rank(
            &[1.0, 2.0, 3.0, 4.0],
            &[4.0, 3.0, 2.0, 1.0],
            &WilcoxonOptions::default(),
        )
        .unwrap();
        assert_eq!(r.p_value, 1.0);
    }
}
