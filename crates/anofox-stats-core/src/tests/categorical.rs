//! Categorical tests
//!
//! - Chi-square test (independence)
//! - Chi-square goodness-of-fit
//! - G-test (log-likelihood ratio)
//! - Fisher's exact test
//! - McNemar's test
//! - Effect sizes (Cramer's V, phi, contingency coefficient)
//! - Cohen's kappa
//! - Proportion tests
//! - Binomial test

use super::{convert_error, ChiSquareResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    binom_test_with_conf_level as lib_binom_test, chisq_goodness_of_fit as lib_chisq_gof,
    chisq_test as lib_chisq_test, cohen_kappa as lib_cohen_kappa,
    contingency_coef as lib_contingency_coef, cramers_v as lib_cramers_v,
    fisher_exact_conditional as lib_fisher_exact_conditional,
    fisher_exact_with_conf_level as lib_fisher_exact_with_conf_level, g_test as lib_g_test,
    mcnemar_exact as lib_mcnemar_exact, mcnemar_test as lib_mcnemar_test,
    phi_coefficient as lib_phi_coefficient, prop_test_one_with_conf_level as lib_prop_test_one,
    prop_test_two_with_conf_level as lib_prop_test_two, Alternative,
};

/// Options for chi-square test
#[derive(Debug, Clone)]
pub struct ChiSquareOptions {
    /// Apply Yates' continuity correction (for 2x2 tables)
    pub correction: bool,
}

impl Default for ChiSquareOptions {
    fn default() -> Self {
        Self { correction: true }
    }
}

/// Chi-square test for independence
///
/// Tests whether two categorical variables are independent.
///
/// # Arguments
/// * `table` - Contingency table (2D array of counts)
/// * `options` - Test options
pub fn chisq_test(
    table: &[Vec<usize>],
    options: &ChiSquareOptions,
) -> StatsResult<ChiSquareResult> {
    if table.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Empty contingency table".into(),
        ));
    }

    let n_cols = table[0].len();
    for (i, row) in table.iter().enumerate() {
        if row.len() != n_cols {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Row {} has different number of columns",
                i
            )));
        }
    }

    let result = lib_chisq_test(table, options.correction).map_err(convert_error)?;

    Ok(ChiSquareResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df as usize,
        method: "Chi-square test for independence".into(),
    })
}

/// Chi-square goodness-of-fit test
///
/// Tests whether observed frequencies match expected frequencies.
///
/// # Arguments
/// * `observed` - Observed counts
/// * `expected` - Expected proportions (will be normalized)
pub fn chisq_goodness_of_fit(observed: &[usize], expected: &[f64]) -> StatsResult<ChiSquareResult> {
    if observed.len() != expected.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Observed and expected must have same length".into(),
        ));
    }

    if observed.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Goodness-of-fit test requires at least 2 categories".into(),
        ));
    }

    let result = lib_chisq_gof(observed, Some(expected)).map_err(convert_error)?;

    Ok(ChiSquareResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df as usize,
        method: "Chi-square goodness-of-fit test".into(),
    })
}

/// Chi-square goodness-of-fit test against uniform distribution
pub fn chisq_goodness_of_fit_uniform(observed: &[usize]) -> StatsResult<ChiSquareResult> {
    if observed.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Goodness-of-fit test requires at least 2 categories".into(),
        ));
    }

    let result = lib_chisq_gof(observed, None).map_err(convert_error)?;

    Ok(ChiSquareResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df as usize,
        method: "Chi-square goodness-of-fit test (uniform)".into(),
    })
}

/// G-test (log-likelihood ratio test)
///
/// Alternative to chi-square test using log-likelihood ratio.
///
/// # Arguments
/// * `table` - Contingency table
pub fn g_test(table: &[Vec<usize>]) -> StatsResult<ChiSquareResult> {
    if table.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Empty contingency table".into(),
        ));
    }

    let n_cols = table[0].len();
    for (i, row) in table.iter().enumerate() {
        if row.len() != n_cols {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Row {} has different number of columns",
                i
            )));
        }
    }

    let result = lib_g_test(table).map_err(convert_error)?;

    Ok(ChiSquareResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df as usize,
        method: "G-test (log-likelihood ratio)".into(),
    })
}

/// Fisher's exact test result
#[derive(Debug, Clone)]
pub struct FisherExactResult {
    /// p-value
    pub p_value: f64,
    /// Odds ratio
    pub odds_ratio: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
}

/// Options for Fisher's exact test
#[derive(Debug, Clone)]
pub struct FisherExactOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Confidence level for the odds-ratio confidence interval, in (0, 1).
    pub confidence_level: f64,
}

impl Default for FisherExactOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            confidence_level: 0.95,
        }
    }
}

fn validate_fisher_confidence(confidence_level: f64) -> StatsResult<()> {
    if confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidInput(format!(
            "confidence_level must be in (0, 1), got {confidence_level}"
        )))
    }
}

/// Fisher's exact test for 2x2 tables
///
/// p-value: exact hypergeometric. Odds ratio: the *sample* odds ratio `ad/bc`.
/// Confidence interval: Woolf (log-odds Wald) interval at
/// `options.confidence_level` (two-sided), with the Haldane-Anscombe +0.5
/// correction when any cell is zero. For R `fisher.test` parity (conditional
/// MLE and exact conditional CI) use [`fisher_exact_conditional`].
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `options` - Test options
pub fn fisher_exact(
    table: &[[usize; 2]; 2],
    options: &FisherExactOptions,
) -> StatsResult<FisherExactResult> {
    validate_fisher_confidence(options.confidence_level)?;
    let result =
        lib_fisher_exact_with_conf_level(table, options.alternative, options.confidence_level)
            .map_err(convert_error)?;
    Ok(fisher_result(result, options.alternative))
}

/// Fisher's exact test for 2x2 tables with R `fisher.test` semantics
/// (delegates to `anofox_statistics::fisher_exact_conditional`).
///
/// * `odds_ratio` is the **conditional maximum-likelihood estimate** (as R
///   reports), not the sample odds ratio.
/// * The confidence interval is the exact conditional interval at
///   `options.confidence_level`, one-sided for `Less` (`[0, U]`) / `Greater`
///   (`[L, inf)`), two-sided otherwise.
pub fn fisher_exact_conditional(
    table: &[[usize; 2]; 2],
    options: &FisherExactOptions,
) -> StatsResult<FisherExactResult> {
    validate_fisher_confidence(options.confidence_level)?;
    let result = lib_fisher_exact_conditional(table, options.alternative, options.confidence_level)
        .map_err(convert_error)?;
    Ok(fisher_result(result, options.alternative))
}

fn fisher_result(
    result: anofox_tests::FisherResult,
    alternative: Alternative,
) -> FisherExactResult {
    FisherExactResult {
        p_value: result.p_value,
        odds_ratio: result.odds_ratio,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative,
    }
}

/// Options for McNemar's test
#[derive(Debug, Clone)]
pub struct McNemarOptions {
    /// Apply continuity correction
    pub correction: bool,
    /// Use exact test
    pub exact: bool,
}

impl Default for McNemarOptions {
    fn default() -> Self {
        Self {
            correction: true,
            exact: false,
        }
    }
}

/// McNemar's test for paired nominal data
///
/// Tests whether marginal frequencies are equal in a 2x2 table
/// from paired observations.
///
/// # Arguments
/// * `table` - 2x2 contingency table
/// * `options` - Test options
pub fn mcnemar_test(
    table: &[[usize; 2]; 2],
    options: &McNemarOptions,
) -> StatsResult<ChiSquareResult> {
    if options.exact {
        let result = lib_mcnemar_exact(table).map_err(convert_error)?;
        Ok(ChiSquareResult {
            statistic: f64::NAN, // Exact test doesn't have a chi-square statistic
            p_value: result.p_value,
            df: 0, // Exact test doesn't have df
            method: "McNemar's exact test".into(),
        })
    } else {
        let result = lib_mcnemar_test(table, options.correction).map_err(convert_error)?;
        Ok(ChiSquareResult {
            statistic: result.statistic,
            p_value: result.p_value,
            df: 1,
            method: "McNemar's test".into(),
        })
    }
}

/// Cramer's V effect size
///
/// Measures association strength for contingency tables (0 to 1).
pub fn cramers_v(table: &[Vec<usize>]) -> StatsResult<f64> {
    if table.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Empty contingency table".into(),
        ));
    }

    let result = lib_cramers_v(table).map_err(convert_error)?;
    Ok(result.estimate)
}

/// Phi coefficient for 2x2 tables
///
/// Measures association strength for 2x2 tables (-1 to 1).
pub fn phi_coefficient(table: &[[usize; 2]; 2]) -> StatsResult<f64> {
    let result = lib_phi_coefficient(table).map_err(convert_error)?;
    Ok(result.estimate)
}

/// Pearson's contingency coefficient
///
/// Measures association strength (0 to < 1).
pub fn contingency_coef(table: &[Vec<usize>]) -> StatsResult<f64> {
    if table.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Empty contingency table".into(),
        ));
    }

    let result = lib_contingency_coef(table).map_err(convert_error)?;
    Ok(result.estimate)
}

/// Cohen's kappa result
#[derive(Debug, Clone)]
pub struct KappaResult {
    /// Kappa coefficient
    pub kappa: f64,
    /// Standard error
    pub se: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// z-statistic
    pub z: f64,
    /// p-value
    pub p_value: f64,
}

/// Cohen's kappa for inter-rater agreement
///
/// # Arguments
/// * `table` - Confusion matrix (square matrix of agreement counts)
/// * `weighted` - Use weighted kappa
pub fn cohen_kappa(table: &[Vec<usize>], weighted: bool) -> StatsResult<KappaResult> {
    if table.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Cohen's kappa requires a non-empty table".into(),
        ));
    }

    let result = lib_cohen_kappa(table, weighted).map_err(convert_error)?;

    Ok(KappaResult {
        kappa: result.kappa,
        se: result.se,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        z: result.z,
        p_value: result.p_value,
    })
}

/// Proportion test result
#[derive(Debug, Clone)]
pub struct PropTestResult {
    /// Test statistic (z)
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Estimated proportion(s)
    pub estimate: Vec<f64>,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
}

/// Options for proportion tests
#[derive(Debug, Clone)]
pub struct PropTestOptions {
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Apply continuity correction (for two-sample test)
    pub correction: bool,
    /// Confidence level of the reported interval, in (0, 1)
    pub confidence_level: f64,
}

impl Default for PropTestOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            correction: true,
            confidence_level: 0.95,
        }
    }
}

/// One-sample proportion z-test
///
/// # Arguments
/// * `successes` - Number of successes
/// * `trials` - Number of trials
/// * `p0` - Null hypothesis proportion
/// * `options` - Test options
pub fn prop_test_one(
    successes: usize,
    trials: usize,
    p0: f64,
    options: &PropTestOptions,
) -> StatsResult<PropTestResult> {
    if trials == 0 {
        return Err(StatsError::InsufficientDataMsg(
            "Number of trials must be > 0".into(),
        ));
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatsError::InvalidInput(
            "p0 must be between 0 and 1".into(),
        ));
    }

    validate_prop_confidence(options.confidence_level)?;

    let result = lib_prop_test_one(
        successes,
        trials,
        p0,
        options.alternative,
        options.confidence_level,
    )
    .map_err(convert_error)?;

    Ok(PropTestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative: options.alternative,
    })
}

/// Two-sample proportion z-test
///
/// # Arguments
/// * `successes1` - Number of successes in group 1
/// * `trials1` - Number of trials in group 1
/// * `successes2` - Number of successes in group 2
/// * `trials2` - Number of trials in group 2
/// * `options` - Test options
pub fn prop_test_two(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
    options: &PropTestOptions,
) -> StatsResult<PropTestResult> {
    if trials1 == 0 || trials2 == 0 {
        return Err(StatsError::InsufficientDataMsg(
            "Number of trials must be > 0".into(),
        ));
    }

    validate_prop_confidence(options.confidence_level)?;

    let result = lib_prop_test_two(
        [successes1, successes2],
        [trials1, trials2],
        options.alternative,
        options.correction,
        options.confidence_level,
    )
    .map_err(convert_error)?;

    Ok(PropTestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative: options.alternative,
    })
}

/// Exact binomial test
///
/// # Arguments
/// * `successes` - Number of successes
/// * `trials` - Number of trials
/// * `p0` - Null hypothesis proportion
/// * `options` - Test options
pub fn binom_test(
    successes: usize,
    trials: usize,
    p0: f64,
    options: &PropTestOptions,
) -> StatsResult<PropTestResult> {
    if trials == 0 {
        return Err(StatsError::InsufficientDataMsg(
            "Number of trials must be > 0".into(),
        ));
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatsError::InvalidInput(
            "p0 must be between 0 and 1".into(),
        ));
    }

    if successes > trials {
        return Err(StatsError::InvalidInput(
            "successes cannot exceed trials".into(),
        ));
    }
    validate_prop_confidence(options.confidence_level)?;

    let result = lib_binom_test(
        successes,
        trials,
        p0,
        options.alternative,
        options.confidence_level,
    )
    .map_err(convert_error)?;

    Ok(PropTestResult {
        statistic: f64::NAN, // Binomial test doesn't have a test statistic
        p_value: result.p_value,
        estimate: vec![result.estimate],
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative: options.alternative,
    })
}

fn validate_prop_confidence(confidence_level: f64) -> StatsResult<()> {
    if confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidInput(format!(
            "confidence_level must be in (0, 1), got {confidence_level}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chisq_test() {
        // Example: Testing independence of gender and preference
        let table = vec![vec![10, 20], vec![15, 25]];
        let opts = ChiSquareOptions::default();
        let result = chisq_test(&table, &opts).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_fisher_exact() {
        let table = [[10, 2], [1, 10]];
        let opts = FisherExactOptions::default();
        let result = fisher_exact(&table, &opts).unwrap();

        assert!(result.p_value < 0.05); // Should be significant
        assert!(result.odds_ratio > 1.0);
    }

    /// Reference values from R 4.x:
    /// `fisher.test(matrix(c(a, c, b, d), nrow = 2), alternative, conf.level)`.
    /// R's `uniroot` tolerance (~1e-4 in t, amplified by the 1/t substitution)
    /// limits agreement of the CI to ~2e-2 relative (e.g. R's upper bound 621.93
    /// for the tea table has tail probability 0.02517, not 0.025).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_fisher_exact_conditional_matches_r() {
        type Case = (
            usize,
            usize,
            usize,
            usize,
            Alternative,
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        #[rustfmt::skip]
        let cases: &[Case] = &[
            (3, 1, 1, 3, Alternative::TwoSided, 0.95, 0.4857142857142856, 6.4083088670057906, 0.21173291530550312, 621.93375054541684),
            (5, 2, 2, 8, Alternative::TwoSided, 0.95, 0.058412176059234912, 8.4320042515238818, 0.73024723722077733, 162.80487560806503),
            (5, 2, 2, 8, Alternative::Less, 0.95, 0.99634923899629779, 8.4320042515238818, 0.0, 105.93002550968224),
            (5, 2, 2, 8, Alternative::Greater, 0.95, 0.052241875771287548, 8.4320042515238818, 0.98024496141817996, f64::INFINITY),
            (5, 2, 2, 8, Alternative::TwoSided, 0.90, 0.058412176059234912, 8.4320042515238818, 0.98024496141817941, 105.93002550968208),
            (0, 5, 3, 2, Alternative::TwoSided, 0.95, 0.16666666666666657, 0.0, 0.0, 2.0268713096128086),
            (1, 0, 0, 1, Alternative::TwoSided, 0.95, 1.0, f64::INFINITY, 0.025640664062500023, f64::INFINITY),
            (10, 3, 2, 15, Alternative::TwoSided, 0.99, 0.00053672411914343582, 21.305331275016723, 1.7275583637669687, 721.32748240000512),
            (2, 7, 8, 2, Alternative::Greater, 0.80, 0.9990149169715733, 0.085862351357362074, 0.019373127779967497, f64::INFINITY),
        ];
        let close = |got: f64, want: f64, rtol: f64| {
            if want.is_infinite() || want == 0.0 {
                got == want
            } else {
                ((got - want) / want).abs() < rtol
            }
        };
        for &(a, b, c, d, alt, cl, p, est, lo, hi) in cases {
            let opts = FisherExactOptions {
                alternative: alt,
                confidence_level: cl,
            };
            let r = fisher_exact_conditional(&[[a, b], [c, d]], &opts).unwrap();
            let tag = format!("{a},{b},{c},{d} {alt:?} {cl}");
            assert!(close(r.p_value, p, 1e-9), "p {tag}: {} vs {p}", r.p_value);
            assert!(
                close(r.odds_ratio, est, 1e-3),
                "est {tag}: {} vs {est}",
                r.odds_ratio
            );
            assert!(
                close(r.ci_lower, lo, 2e-2),
                "lo {tag}: {} vs {lo}",
                r.ci_lower
            );
            assert!(
                close(r.ci_upper, hi, 2e-2),
                "hi {tag}: {} vs {hi}",
                r.ci_upper
            );
        }
    }

    #[test]
    fn test_fisher_confidence_level_validation() {
        for cl in [0.0, 1.0, -0.5, f64::NAN] {
            let opts = FisherExactOptions {
                confidence_level: cl,
                ..Default::default()
            };
            assert!(fisher_exact(&[[1, 2], [3, 4]], &opts).is_err());
            assert!(fisher_exact_conditional(&[[1, 2], [3, 4]], &opts).is_err());
        }
        assert!(
            fisher_exact_conditional(&[[0, 0], [0, 0]], &FisherExactOptions::default()).is_err()
        );
    }

    #[test]
    fn test_fisher_wald_ci_honours_level() {
        let t = [[10, 2], [1, 10]];
        let r95 = fisher_exact(&t, &FisherExactOptions::default()).unwrap();
        let r80 = fisher_exact(
            &t,
            &FisherExactOptions {
                confidence_level: 0.8,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r80.ci_lower > r95.ci_lower && r80.ci_upper < r95.ci_upper);
    }

    #[test]
    fn test_cohen_kappa() {
        // Simple 2x2 agreement table
        let table = vec![vec![10, 2], vec![1, 7]];
        let result = cohen_kappa(&table, false).unwrap();

        assert!(result.kappa > 0.5); // High agreement
    }

    /// R binom.test / prop.test reference values.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_binom_and_prop_ci_match_r() {
        let close = |a: f64, b: f64, t: f64| assert!((a - b).abs() <= t, "{a} vs {b}");
        let o = PropTestOptions::default();
        let r = binom_test(13, 20, 0.5, &o).unwrap();
        close(r.p_value, 0.26317596435546875, 1e-14);
        close(r.ci_lower, 0.4078114654671719, 1e-12);
        close(r.ci_upper, 0.84609079521545882, 1e-12);
        let r = binom_test(4500, 10000, 0.5, &o).unwrap();
        assert!((r.p_value / 1.5510640568246068e-23 - 1.0).abs() < 1e-6);
        let less = PropTestOptions {
            alternative: Alternative::Less,
            confidence_level: 0.9,
            ..PropTestOptions::default()
        };
        let r = binom_test(13, 20, 0.5, &less).unwrap();
        close(r.p_value, 0.94234085083007812, 1e-14);
        close(r.ci_upper, 0.79333596671715334, 1e-12);

        let r = prop_test_two(18, 30, 11, 28, &o).unwrap();
        close(r.ci_lower, -0.079284640161607245, 1e-12);
        close(r.ci_upper, 0.4935703544473215, 1e-12);
        let r = prop_test_one(
            13,
            20,
            0.5,
            &PropTestOptions {
                confidence_level: 0.9,
                ..PropTestOptions::default()
            },
        )
        .unwrap();
        close(r.ci_lower, 0.46651268884847175, 1e-12);
        close(r.ci_upper, 0.79773995994323899, 1e-12);
    }
}
