//! Proportion tests and binomial tests.

use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use statrs::distribution::{Binomial, ContinuousCDF, Discrete, Normal};

/// Result of a proportion test
#[derive(Debug, Clone)]
pub struct PropTestResult {
    /// Estimated proportion(s)
    pub estimate: Vec<f64>,
    /// Test statistic (chi-square or z)
    pub statistic: f64,
    /// Degrees of freedom (for chi-square test)
    pub df: Option<f64>,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound
    pub conf_int_upper: f64,
    /// Null hypothesis proportion
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Name of the method
    pub method: String,
}

/// Result of an exact binomial test
#[derive(Debug, Clone)]
pub struct BinomTestResult {
    /// Estimated proportion
    pub estimate: f64,
    /// Number of successes
    pub successes: usize,
    /// Number of trials
    pub n: usize,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound (Clopper-Pearson)
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound (Clopper-Pearson)
    pub conf_int_upper: f64,
    /// Null hypothesis proportion
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Name of the method
    pub method: String,
}

/// One-sample proportion test (z-test approximation).
///
/// Tests the null hypothesis that the true proportion equals p0.
///
/// # Arguments
/// * `successes` - Number of successes
/// * `n` - Total number of trials
/// * `p0` - Null hypothesis proportion
/// * `alternative` - Alternative hypothesis
///
/// # Returns
/// * `PropTestResult` containing test statistic and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{prop_test_one, Alternative};
///
/// // Test if coin is fair: 60 heads out of 100 flips
/// let result = prop_test_one(60, 100, 0.5, Alternative::TwoSided).unwrap();
/// println!("z = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `prop.test(x, n, p = p0, alternative = "...", correct = FALSE)`
pub fn prop_test_one(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<PropTestResult> {
    if n == 0 {
        return Err(StatError::EmptyData);
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatError::InvalidParameter(format!(
            "Null proportion must be between 0 and 1, got {}",
            p0
        )));
    }
    if successes > n {
        return Err(StatError::InvalidParameter(format!(
            "Successes ({}) cannot exceed n ({})",
            successes, n
        )));
    }

    let p_hat = successes as f64 / n as f64;
    let n_f = n as f64;

    // Standard error under null hypothesis
    let se = (p0 * (1.0 - p0) / n_f).sqrt();

    // Z-statistic
    let z = if se > 0.0 { (p_hat - p0) / se } else { 0.0 };

    // p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z.abs())),
        Alternative::Greater => 1.0 - normal.cdf(z),
        Alternative::Less => normal.cdf(z),
    };

    // Wilson score confidence interval
    let (conf_int_lower, conf_int_upper) = wilson_ci(successes, n, 0.95);

    Ok(PropTestResult {
        estimate: vec![p_hat],
        statistic: z,
        df: None,
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: p0,
        alternative,
        method: "1-sample proportions test without continuity correction".to_string(),
    })
}

/// Two-sample proportion test.
///
/// Tests the null hypothesis that two population proportions are equal.
///
/// # Arguments
/// * `successes` - [x1, x2] number of successes in each group
/// * `totals` - [n1, n2] total number of trials in each group
/// * `alternative` - Alternative hypothesis
/// * `correction` - Apply Yates' continuity correction
///
/// # Returns
/// * `PropTestResult` containing test statistic and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{prop_test_two, Alternative};
///
/// // Compare conversion rates: 30/100 vs 45/120
/// let result = prop_test_two([30, 45], [100, 120], Alternative::TwoSided, false).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `prop.test(c(x1, x2), c(n1, n2), alternative = "...")`
pub fn prop_test_two(
    successes: [usize; 2],
    totals: [usize; 2],
    alternative: Alternative,
    correction: bool,
) -> Result<PropTestResult> {
    if totals[0] == 0 || totals[1] == 0 {
        return Err(StatError::EmptyData);
    }
    if successes[0] > totals[0] || successes[1] > totals[1] {
        return Err(StatError::InvalidParameter(
            "Successes cannot exceed total trials".to_string(),
        ));
    }

    let p1 = successes[0] as f64 / totals[0] as f64;
    let p2 = successes[1] as f64 / totals[1] as f64;

    let n1 = totals[0] as f64;
    let n2 = totals[1] as f64;

    // Pooled proportion
    let p_pooled = (successes[0] + successes[1]) as f64 / (totals[0] + totals[1]) as f64;

    // Standard error under null (equal proportions)
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();

    // Z-statistic (or chi-square)
    let diff = p1 - p2;
    let z = if se > 0.0 {
        if correction {
            // Yates' correction
            let correction_term = 0.5 * (1.0 / n1 + 1.0 / n2);
            let adj_diff = if diff.abs() > correction_term {
                diff.abs() - correction_term
            } else {
                0.0
            };
            adj_diff / se * diff.signum()
        } else {
            diff / se
        }
    } else {
        0.0
    };

    // Chi-square statistic (z^2)
    let chi_sq = z * z;

    // p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z.abs())),
        Alternative::Greater => 1.0 - normal.cdf(z),
        Alternative::Less => normal.cdf(z),
    };

    // Confidence interval for difference in proportions
    let se_diff = (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt();
    let z_crit = 1.96;
    let conf_int_lower = diff - z_crit * se_diff;
    let conf_int_upper = diff + z_crit * se_diff;

    Ok(PropTestResult {
        estimate: vec![p1, p2],
        statistic: chi_sq,
        df: Some(1.0),
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: 0.0, // Difference = 0 under null
        alternative,
        method: if correction {
            "2-sample test for equality of proportions with continuity correction"
        } else {
            "2-sample test for equality of proportions without continuity correction"
        }
        .to_string(),
    })
}

/// Exact binomial test.
///
/// Tests the null hypothesis that the probability of success equals p0.
/// Uses the exact binomial distribution rather than normal approximation.
///
/// # Arguments
/// * `successes` - Number of successes
/// * `n` - Total number of trials
/// * `p0` - Null hypothesis probability
/// * `alternative` - Alternative hypothesis
///
/// # Returns
/// * `BinomTestResult` containing the exact p-value and Clopper-Pearson CI
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{binom_test, Alternative};
///
/// // Test if proportion of heads is 0.5: 7 heads out of 10 flips
/// let result = binom_test(7, 10, 0.5, Alternative::TwoSided).unwrap();
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `binom.test(x, n, p = p0, alternative = "...")`
pub fn binom_test(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<BinomTestResult> {
    if n == 0 {
        return Err(StatError::EmptyData);
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatError::InvalidParameter(format!(
            "Null probability must be between 0 and 1, got {}",
            p0
        )));
    }
    if successes > n {
        return Err(StatError::InvalidParameter(format!(
            "Successes ({}) cannot exceed n ({})",
            successes, n
        )));
    }

    let p_hat = successes as f64 / n as f64;

    // Compute exact p-value using binomial distribution
    let binom = Binomial::new(p0, n as u64).unwrap();
    let observed_prob = binom.pmf(successes as u64);

    let p_value = match alternative {
        Alternative::TwoSided => {
            // Sum probabilities of all outcomes as or less likely than observed
            let mut p = 0.0;
            for k in 0..=n {
                let prob_k = binom.pmf(k as u64);
                if prob_k <= observed_prob + 1e-10 {
                    p += prob_k;
                }
            }
            p.min(1.0)
        }
        Alternative::Greater => {
            // P(X >= successes)
            let mut p = 0.0;
            for k in successes..=n {
                p += binom.pmf(k as u64);
            }
            p
        }
        Alternative::Less => {
            // P(X <= successes)
            let mut p = 0.0;
            for k in 0..=successes {
                p += binom.pmf(k as u64);
            }
            p
        }
    };

    // Clopper-Pearson exact confidence interval
    let (conf_int_lower, conf_int_upper) = clopper_pearson_ci(successes, n, 0.95);

    Ok(BinomTestResult {
        estimate: p_hat,
        successes,
        n,
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: p0,
        alternative,
        method: "Exact binomial test".to_string(),
    })
}

/// Wilson score confidence interval for a proportion.
fn wilson_ci(successes: usize, n: usize, conf_level: f64) -> (f64, f64) {
    let p_hat = successes as f64 / n as f64;
    let n_f = n as f64;

    let alpha = 1.0 - conf_level;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = normal.inverse_cdf(1.0 - alpha / 2.0);
    let z2 = z * z;

    let denom = 1.0 + z2 / n_f;
    let center = (p_hat + z2 / (2.0 * n_f)) / denom;
    let margin = z * (p_hat * (1.0 - p_hat) / n_f + z2 / (4.0 * n_f * n_f)).sqrt() / denom;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Clopper-Pearson exact confidence interval for a proportion.
fn clopper_pearson_ci(successes: usize, n: usize, conf_level: f64) -> (f64, f64) {
    let alpha = 1.0 - conf_level;

    // Lower bound: find p such that P(X >= successes | p) = alpha/2
    // This is the alpha/2 quantile of Beta(successes, n - successes + 1)
    let lower = if successes == 0 {
        0.0
    } else {
        beta_quantile(alpha / 2.0, successes as f64, (n - successes + 1) as f64)
    };

    // Upper bound: find p such that P(X <= successes | p) = alpha/2
    // This is the 1 - alpha/2 quantile of Beta(successes + 1, n - successes)
    let upper = if successes == n {
        1.0
    } else {
        beta_quantile(
            1.0 - alpha / 2.0,
            (successes + 1) as f64,
            (n - successes) as f64,
        )
    };

    (lower, upper)
}

/// Approximate beta quantile using Newton-Raphson iteration.
fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    // Simple approximation using the normal approximation to beta
    // Mean of Beta(a,b) = a/(a+b), Var = ab/((a+b)^2*(a+b+1))
    let mean = a / (a + b);
    let var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
    let sd = var.sqrt();

    // Normal approximation
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = normal.inverse_cdf(p);

    // Initial guess
    let mut x = mean + z * sd;
    x = x.clamp(0.001, 0.999);

    // Newton-Raphson refinement (a few iterations)
    for _ in 0..10 {
        let cdf = beta_cdf(x, a, b);
        let pdf = beta_pdf(x, a, b);

        if pdf.abs() < 1e-12 {
            break;
        }

        let delta = (cdf - p) / pdf;
        x -= delta;
        x = x.clamp(0.001, 0.999);

        if delta.abs() < 1e-10 {
            break;
        }
    }

    x
}

/// Beta CDF using incomplete beta function approximation.
fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use regularized incomplete beta function
    // I_x(a,b) = B(x;a,b) / B(a,b)
    incomplete_beta(x, a, b)
}

/// Beta PDF
fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }

    // f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
    let log_b = log_beta(a, b);
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_b;

    log_pdf.exp()
}

/// Log of beta function: log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Log-gamma function using Lanczos approximation.
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation coefficients
    #[allow(clippy::excessive_precision)]
    let g = 7.0;
    #[allow(clippy::excessive_precision)]
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut y = c[0];
    for (i, coef) in c.iter().enumerate().skip(1) {
        y += coef / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + y.ln()
}

/// Regularized incomplete beta function using continued fraction.
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation if needed for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(1.0 - x, b, a);
    }

    // Continued fraction representation
    let bt = (a * x.ln() + b * (1.0 - x).ln() - log_beta(a, b)).exp();

    // Lentz's algorithm for continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;

    for m in 0..200 {
        let m_f = m as f64;

        // Even term
        let an = if m == 0 {
            1.0
        } else {
            let num = (a + m_f - 1.0) * (a + b + m_f - 1.0) * m_f * (b - m_f) * x * x;
            let den = (a + 2.0 * m_f - 1.0).powi(2);
            num / den
        };

        d = 1.0 + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;

        c = 1.0 + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }

        f *= d * c;

        // Odd term
        let an = {
            let num = -(a + m_f) * (a + b + m_f) * (m_f + 1.0) * (b - m_f - 1.0) * x * x;
            let den = (a + 2.0 * m_f + 1.0).powi(2);
            num / den
        };

        d = 1.0 + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;

        c = 1.0 + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }

        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    bt * f / a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_test_one() {
        let result = prop_test_one(60, 100, 0.5, Alternative::TwoSided).unwrap();

        // z = (0.6 - 0.5) / sqrt(0.5*0.5/100) = 0.1 / 0.05 = 2.0
        assert!((result.statistic - 2.0).abs() < 0.01);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_prop_test_one_fair_coin() {
        let result = prop_test_one(50, 100, 0.5, Alternative::TwoSided).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prop_test_two() {
        let result = prop_test_two([30, 50], [100, 100], Alternative::TwoSided, false).unwrap();

        // Pooled proportion = 80/200 = 0.4
        // SE = sqrt(0.4*0.6*(1/100 + 1/100)) = sqrt(0.0048) ≈ 0.069
        // z = (0.3-0.5)/0.069 ≈ -2.9
        assert!(result.statistic > 0.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_prop_test_two_equal() {
        let result = prop_test_two([50, 50], [100, 100], Alternative::TwoSided, false).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binom_test() {
        let result = binom_test(7, 10, 0.5, Alternative::TwoSided).unwrap();

        // 7 heads out of 10 with fair coin
        // p-value should be significant but not extremely small
        assert!(result.p_value > 0.1);
        assert!(result.p_value < 1.0);
    }

    #[test]
    fn test_binom_test_extreme() {
        let result = binom_test(10, 10, 0.5, Alternative::Greater).unwrap();

        // 10 out of 10 successes, one-sided test
        // P(X >= 10) = 0.5^10 ≈ 0.001
        assert!(result.p_value < 0.01);
    }

    #[test]
    fn test_binom_test_ci() {
        let result = binom_test(30, 100, 0.5, Alternative::TwoSided).unwrap();

        // CI should contain the estimate
        assert!(result.conf_int_lower < result.estimate);
        assert!(result.conf_int_upper > result.estimate);

        // CI should be reasonable (0.2-0.4ish for 30/100)
        assert!(result.conf_int_lower > 0.15);
        assert!(result.conf_int_upper < 0.45);
    }
}
