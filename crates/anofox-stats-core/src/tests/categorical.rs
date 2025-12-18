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

use crate::{StatsError, StatsResult};
use super::{convert_error, ChiSquareResult};
use anofox_tests::{
    chisq_test as lib_chisq_test,
    chisq_goodness_of_fit as lib_chisq_gof,
    g_test as lib_g_test,
    fisher_exact as lib_fisher_exact,
    mcnemar_test as lib_mcnemar_test,
    mcnemar_exact as lib_mcnemar_exact,
    cramers_v as lib_cramers_v,
    phi_coefficient as lib_phi_coefficient,
    contingency_coef as lib_contingency_coef,
    cohen_kappa as lib_cohen_kappa,
    prop_test_one as lib_prop_test_one,
    prop_test_two as lib_prop_test_two,
    binom_test as lib_binom_test,
    Alternative,
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
pub fn chisq_test(table: &[Vec<usize>], options: &ChiSquareOptions) -> StatsResult<ChiSquareResult> {
    if table.is_empty() {
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
    }

    let n_cols = table[0].len();
    for (i, row) in table.iter().enumerate() {
        if row.len() != n_cols {
            return Err(StatsError::DimensionMismatchMsg(
                format!("Row {} has different number of columns", i),
            ));
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
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
    }

    let n_cols = table[0].len();
    for (i, row) in table.iter().enumerate() {
        if row.len() != n_cols {
            return Err(StatsError::DimensionMismatchMsg(
                format!("Row {} has different number of columns", i),
            ));
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
}

impl Default for FisherExactOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
        }
    }
}

/// Fisher's exact test for 2x2 tables
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `options` - Test options
pub fn fisher_exact(table: &[[usize; 2]; 2], options: &FisherExactOptions) -> StatsResult<FisherExactResult> {
    let result = lib_fisher_exact(table, options.alternative)
        .map_err(convert_error)?;

    Ok(FisherExactResult {
        p_value: result.p_value,
        odds_ratio: result.odds_ratio,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative: options.alternative,
    })
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
pub fn mcnemar_test(table: &[[usize; 2]; 2], options: &McNemarOptions) -> StatsResult<ChiSquareResult> {
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
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
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
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
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
}

impl Default for PropTestOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            correction: true,
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
        return Err(StatsError::InvalidInput("Number of trials must be > 0".into()));
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatsError::InvalidInput("p0 must be between 0 and 1".into()));
    }

    let result = lib_prop_test_one(successes, trials, p0, options.alternative)
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
        return Err(StatsError::InvalidInput("Number of trials must be > 0".into()));
    }

    let result = lib_prop_test_two(
        [successes1, successes2],
        [trials1, trials2],
        options.alternative,
        options.correction,
    ).map_err(convert_error)?;

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
        return Err(StatsError::InvalidInput("Number of trials must be > 0".into()));
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatsError::InvalidInput("p0 must be between 0 and 1".into()));
    }

    let result = lib_binom_test(successes, trials, p0, options.alternative)
        .map_err(convert_error)?;

    Ok(PropTestResult {
        statistic: f64::NAN, // Binomial test doesn't have a test statistic
        p_value: result.p_value,
        estimate: vec![result.estimate], // Wrap single estimate in vec for consistency
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        alternative: options.alternative,
    })
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

    #[test]
    fn test_cohen_kappa() {
        // Simple 2x2 agreement table
        let table = vec![vec![10, 2], vec![1, 7]];
        let result = cohen_kappa(&table, false).unwrap();

        assert!(result.kappa > 0.5); // High agreement
    }
}
