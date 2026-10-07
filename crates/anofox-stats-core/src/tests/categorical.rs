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
    chisq_goodness_of_fit as lib_chisq_gof, chisq_test as lib_chisq_test,
    cohen_kappa as lib_cohen_kappa, contingency_coef as lib_contingency_coef,
    cramers_v as lib_cramers_v, fisher_exact as lib_fisher_exact, g_test as lib_g_test,
    mcnemar_exact as lib_mcnemar_exact, mcnemar_test as lib_mcnemar_test,
    phi_coefficient as lib_phi_coefficient, prop_test_one as lib_prop_test_one,
    prop_test_two as lib_prop_test_two, Alternative,
};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::factorial::ln_binomial;

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
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
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
        return Err(StatsError::InvalidInput("Empty contingency table".into()));
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
/// p-value: exact hypergeometric (upstream `anofox-statistics`). Odds ratio:
/// the *sample* odds ratio `ad/bc`. Confidence interval: Woolf (log-odds Wald)
/// interval at `options.confidence_level` (two-sided), with the Haldane-Anscombe
/// +0.5 correction when any cell is zero. For R `fisher.test` parity
/// (conditional MLE and exact conditional CI) use [`fisher_exact_conditional`].
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `options` - Test options
pub fn fisher_exact(
    table: &[[usize; 2]; 2],
    options: &FisherExactOptions,
) -> StatsResult<FisherExactResult> {
    validate_fisher_confidence(options.confidence_level)?;
    let result = lib_fisher_exact(table, options.alternative).map_err(convert_error)?;

    // Upstream hardcodes z = 1.96; recompute the same Woolf interval at the
    // requested level.
    let [[a, b], [c, d]] = *table;
    let corr = if a == 0 || b == 0 || c == 0 || d == 0 {
        0.5
    } else {
        0.0
    };
    let (a, b, c, d) = (
        a as f64 + corr,
        b as f64 + corr,
        c as f64 + corr,
        d as f64 + corr,
    );
    let log_or = (a * d).ln() - (b * c).ln();
    let se = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
    let z = Normal::new(0.0, 1.0)
        .map(|n| n.inverse_cdf((1.0 + options.confidence_level) / 2.0))
        .unwrap_or(f64::NAN);

    Ok(FisherExactResult {
        p_value: result.p_value,
        odds_ratio: result.odds_ratio,
        ci_lower: (log_or - z * se).exp(),
        ci_upper: (log_or + z * se).exp(),
        alternative: options.alternative,
    })
}

/// Noncentral hypergeometric helper for [`fisher_exact_conditional`]
/// (a direct port of the closures inside R's `fisher.test` for 2x2 tables).
struct NcHyper {
    lo: usize,
    hi: usize,
    support: Vec<f64>,
    logdc: Vec<f64>,
}

impl NcHyper {
    /// `m` = first column total, `n` = second column total, `k` = first row total.
    fn new(m: usize, n: usize, k: usize) -> Self {
        let lo = k.saturating_sub(n);
        let hi = k.min(m);
        let support: Vec<f64> = (lo..=hi).map(|s| s as f64).collect();
        let ln_total = ln_binomial((m + n) as u64, k as u64);
        let logdc = (lo..=hi)
            .map(|s| {
                ln_binomial(m as u64, s as u64) + ln_binomial(n as u64, (k - s) as u64) - ln_total
            })
            .collect();
        Self {
            lo,
            hi,
            support,
            logdc,
        }
    }

    /// Density over the support for odds ratio `ncp` (0 < ncp < inf).
    fn density(&self, ncp: f64) -> Vec<f64> {
        let ln_ncp = ncp.ln();
        let d: Vec<f64> = self
            .logdc
            .iter()
            .zip(&self.support)
            .map(|(l, s)| l + ln_ncp * s)
            .collect();
        let max = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = d.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        e.into_iter().map(|v| v / sum).collect()
    }

    /// Mean of the distribution for odds ratio `ncp`.
    fn mean(&self, ncp: f64) -> f64 {
        if ncp == 0.0 {
            return self.lo as f64;
        }
        if ncp.is_infinite() {
            return self.hi as f64;
        }
        self.density(ncp)
            .iter()
            .zip(&self.support)
            .map(|(d, s)| d * s)
            .sum()
    }

    /// P(X <= q) (or P(X >= q) when `upper`) for odds ratio `ncp`.
    fn cdf(&self, q: f64, ncp: f64, upper: bool) -> f64 {
        let edge = |e: usize| {
            let e = e as f64;
            if (upper && q <= e) || (!upper && q >= e) {
                1.0
            } else {
                0.0
            }
        };
        if ncp == 0.0 {
            return edge(self.lo);
        }
        if ncp.is_infinite() {
            return edge(self.hi);
        }
        self.density(ncp)
            .iter()
            .zip(&self.support)
            .filter(|(_, &s)| if upper { s >= q } else { s <= q })
            .map(|(d, _)| d)
            .sum()
    }
}

/// Root of a monotone function on (a, b) by bisection (to ~machine precision).
fn bisect<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64) -> f64 {
    let fa_neg = f(a) < 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if mid <= a || mid >= b {
            break;
        }
        if (f(mid) < 0.0) == fa_neg {
            a = mid;
        } else {
            b = mid;
        }
    }
    0.5 * (a + b)
}

/// Fisher's exact test for 2x2 tables with R `fisher.test` semantics.
///
/// * `odds_ratio` is the **conditional maximum-likelihood estimate** (as R
///   reports), not the sample odds ratio.
/// * The confidence interval is the exact conditional interval at
///   `options.confidence_level`, one-sided for `Less` (`[0, U]`) / `Greater`
///   (`[L, inf)`), two-sided otherwise.
/// * The two-sided p-value sums the probabilities of all tables no more
///   likely than the observed one (relative tolerance 1e-7, as R).
///
/// Works for any table with at least one observation (including n < 4).
/// R solves the root-finding problems with `uniroot` (tolerance ~1e-4); this
/// implementation bisects to machine precision, so the CI / MLE agree with R to
/// about 1e-4 relative.
pub fn fisher_exact_conditional(
    table: &[[usize; 2]; 2],
    options: &FisherExactOptions,
) -> StatsResult<FisherExactResult> {
    validate_fisher_confidence(options.confidence_level)?;
    let [[a, b], [c, d]] = *table;
    if a + b + c + d == 0 {
        return Err(StatsError::InsufficientDataMsg(
            "Fisher's exact test requires at least one observation".into(),
        ));
    }
    let x = a as f64;
    let h = NcHyper::new(a + c, b + d, a + b);
    let eps = f64::EPSILON;

    let p_value = match options.alternative {
        Alternative::Less => h.cdf(x, 1.0, false),
        Alternative::Greater => h.cdf(x, 1.0, true),
        Alternative::TwoSided => {
            let dens = h.density(1.0);
            let obs = dens[a - h.lo] * (1.0 + 1e-7);
            dens.iter().filter(|&&v| v <= obs).sum::<f64>()
        }
    }
    .min(1.0);

    let mle = if a == h.lo {
        0.0
    } else if a == h.hi {
        f64::INFINITY
    } else {
        let mu = h.mean(1.0);
        if mu > x {
            bisect(|t| h.mean(t) - x, 0.0, 1.0)
        } else if mu < x {
            1.0 / bisect(|t| h.mean(1.0 / t) - x, eps, 1.0)
        } else {
            1.0
        }
    };

    let ncp_upper = |alpha: f64| -> f64 {
        if a == h.hi {
            return f64::INFINITY;
        }
        let p = h.cdf(x, 1.0, false);
        if p < alpha {
            bisect(|t| h.cdf(x, t, false) - alpha, 0.0, 1.0)
        } else if p > alpha {
            1.0 / bisect(|t| h.cdf(x, 1.0 / t, false) - alpha, eps, 1.0)
        } else {
            1.0
        }
    };
    let ncp_lower = |alpha: f64| -> f64 {
        if a == h.lo {
            return 0.0;
        }
        let p = h.cdf(x, 1.0, true);
        if p > alpha {
            bisect(|t| h.cdf(x, t, true) - alpha, 0.0, 1.0)
        } else if p < alpha {
            1.0 / bisect(|t| h.cdf(x, 1.0 / t, true) - alpha, eps, 1.0)
        } else {
            1.0
        }
    };

    let cl = options.confidence_level;
    let (ci_lower, ci_upper) = match options.alternative {
        Alternative::Less => (0.0, ncp_upper(1.0 - cl)),
        Alternative::Greater => (ncp_lower(1.0 - cl), f64::INFINITY),
        Alternative::TwoSided => {
            let alpha = (1.0 - cl) / 2.0;
            (ncp_lower(alpha), ncp_upper(alpha))
        }
    };

    Ok(FisherExactResult {
        p_value,
        odds_ratio: mle,
        ci_lower,
        ci_upper,
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
        return Err(StatsError::InvalidInput(
            "Number of trials must be > 0".into(),
        ));
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatsError::InvalidInput(
            "p0 must be between 0 and 1".into(),
        ));
    }

    validate_prop_confidence(options.confidence_level)?;

    let result =
        lib_prop_test_one(successes, trials, p0, options.alternative).map_err(convert_error)?;

    // anofox-statistics <= 0.4.2 hard-codes a two-sided 95% interval; compute
    // R's prop.test(correct = FALSE) Wilson interval at the requested level.
    let (ci_lower, ci_upper) = wilson_ci(
        successes,
        trials,
        options.confidence_level,
        options.alternative,
    );

    Ok(PropTestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        ci_lower,
        ci_upper,
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
        return Err(StatsError::InvalidInput(
            "Number of trials must be > 0".into(),
        ));
    }

    validate_prop_confidence(options.confidence_level)?;

    let result = lib_prop_test_two(
        [successes1, successes2],
        [trials1, trials2],
        options.alternative,
        options.correction,
    )
    .map_err(convert_error)?;

    // anofox-statistics <= 0.4.2 uses z = 1.96 and ignores the continuity
    // correction; compute R's prop.test interval here.
    let (ci_lower, ci_upper) = two_prop_ci(
        successes1,
        trials1,
        successes2,
        trials2,
        options.correction,
        options.confidence_level,
        options.alternative,
    );

    Ok(PropTestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        ci_lower,
        ci_upper,
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
        return Err(StatsError::InvalidInput(
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

    // Computed here rather than via anofox-statistics <= 0.4.2, whose
    // Clopper-Pearson interval used an inaccurate beta quantile and whose
    // two-sided p-value used an absolute tolerance (floored for large n).
    let p_value = binom_p_value(successes, trials, p0, options.alternative);
    let (ci_lower, ci_upper) = clopper_pearson_ci(
        successes,
        trials,
        options.confidence_level,
        options.alternative,
    );

    Ok(PropTestResult {
        statistic: f64::NAN, // Binomial test doesn't have a test statistic
        p_value,
        estimate: vec![successes as f64 / trials as f64],
        ci_lower,
        ci_upper,
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

/// `qnorm((1 + cl) / 2)` for two-sided, `qnorm(cl)` for one-sided intervals (R prop.test).
fn prop_z(confidence_level: f64, alternative: Alternative) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match alternative {
        Alternative::TwoSided => normal.inverse_cdf((1.0 + confidence_level) / 2.0),
        _ => normal.inverse_cdf(confidence_level),
    }
}

/// Wilson score interval, R `prop.test(x, n, correct = FALSE)$conf.int`.
fn wilson_ci(x: usize, n: usize, confidence_level: f64, alternative: Alternative) -> (f64, f64) {
    let n_f = n as f64;
    let p_hat = x as f64 / n_f;
    let z = prop_z(confidence_level, alternative);
    let z22n = z * z / (2.0 * n_f);
    let half = z * (p_hat * (1.0 - p_hat) / n_f + z22n / (2.0 * n_f)).sqrt();
    let upper = if p_hat >= 1.0 {
        1.0
    } else {
        ((p_hat + z22n + half) / (1.0 + 2.0 * z22n)).min(1.0)
    };
    let lower = if p_hat <= 0.0 {
        0.0
    } else {
        ((p_hat + z22n - half) / (1.0 + 2.0 * z22n)).max(0.0)
    };
    match alternative {
        Alternative::TwoSided => (lower, upper),
        Alternative::Less => (0.0, upper),
        Alternative::Greater => (lower, 1.0),
    }
}

/// Interval for p1 - p2, R `prop.test(c(x1, x2), c(n1, n2), correct = ...)$conf.int`.
fn two_prop_ci(
    x1: usize,
    n1: usize,
    x2: usize,
    n2: usize,
    correction: bool,
    confidence_level: f64,
    alternative: Alternative,
) -> (f64, f64) {
    let (n1, n2) = (n1 as f64, n2 as f64);
    let (p1, p2) = (x1 as f64 / n1, x2 as f64 / n2);
    let delta = p1 - p2;
    let inv_sum = 1.0 / n1 + 1.0 / n2;
    let yates = if correction {
        0.5_f64.min(delta.abs() / inv_sum)
    } else {
        0.0
    };
    let z = prop_z(confidence_level, alternative);
    let width = z * (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt() + yates * inv_sum;
    match alternative {
        Alternative::TwoSided => ((delta - width).max(-1.0), (delta + width).min(1.0)),
        Alternative::Less => (-1.0, (delta + width).min(1.0)),
        Alternative::Greater => ((delta - width).max(-1.0), 1.0),
    }
}

/// Exact binomial p-value, a port of R's `binom.test` (relative tolerance 1 + 1e-7).
fn binom_p_value(x: usize, n: usize, p0: f64, alternative: Alternative) -> f64 {
    use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
    let binom = Binomial::new(p0, n as u64).unwrap();
    // P(X <= k), P(X > k)
    let cdf = |k: i64| if k < 0 { 0.0 } else { binom.cdf(k as u64) };
    let sf = |k: i64| if k < 0 { 1.0 } else { binom.sf(k as u64) };
    let (xi, ni) = (x as i64, n as i64);
    let p = match alternative {
        Alternative::Less => cdf(xi),
        Alternative::Greater => sf(xi - 1),
        Alternative::TwoSided => {
            if p0 == 0.0 {
                f64::from(u8::from(x == 0))
            } else if p0 == 1.0 {
                f64::from(u8::from(x == n))
            } else {
                let d = binom.pmf(x as u64) * (1.0 + 1e-7);
                let m = n as f64 * p0;
                let xf = x as f64;
                if xf == m {
                    1.0
                } else if xf < m {
                    let y = (m.ceil() as i64..=ni)
                        .filter(|&i| binom.pmf(i as u64) <= d)
                        .count() as i64;
                    cdf(xi) + sf(ni - y)
                } else {
                    let y = (0..=m.floor() as i64)
                        .filter(|&i| binom.pmf(i as u64) <= d)
                        .count() as i64;
                    cdf(y - 1) + sf(xi - 1)
                }
            }
        }
    };
    p.clamp(0.0, 1.0)
}

/// Clopper-Pearson interval, R `binom.test(x, n)$conf.int` (one-sided for one-sided alternatives).
fn clopper_pearson_ci(
    x: usize,
    n: usize,
    confidence_level: f64,
    alternative: Alternative,
) -> (f64, f64) {
    use statrs::distribution::Beta;
    let p_lower = |alpha: f64| {
        if x == 0 {
            0.0
        } else {
            Beta::new(x as f64, (n - x + 1) as f64)
                .unwrap()
                .inverse_cdf(alpha)
        }
    };
    let p_upper = |alpha: f64| {
        if x == n {
            1.0
        } else {
            Beta::new((x + 1) as f64, (n - x) as f64)
                .unwrap()
                .inverse_cdf(1.0 - alpha)
        }
    };
    match alternative {
        Alternative::TwoSided => {
            let alpha = (1.0 - confidence_level) / 2.0;
            (p_lower(alpha), p_upper(alpha))
        }
        Alternative::Less => (0.0, p_upper(1.0 - confidence_level)),
        Alternative::Greater => (p_lower(1.0 - confidence_level), 1.0),
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

            // Our bounds solve the defining equations to machine precision
            // (R's do not, see above): P(X <= a | U) = alpha, P(X >= a | L) = alpha.
            let h = NcHyper::new(a + c, b + d, a + b);
            let alpha = match alt {
                Alternative::TwoSided => (1.0 - cl) / 2.0,
                _ => 1.0 - cl,
            };
            if r.ci_upper.is_finite() && alt != Alternative::Greater {
                assert!(
                    (h.cdf(a as f64, r.ci_upper, false) - alpha).abs() < 1e-9,
                    "U {tag}"
                );
            }
            if r.ci_lower > 0.0 && alt != Alternative::Less {
                assert!(
                    (h.cdf(a as f64, r.ci_lower, true) - alpha).abs() < 1e-9,
                    "L {tag}"
                );
            }
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
