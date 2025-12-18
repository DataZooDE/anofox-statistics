//! McNemar's test for paired nominal data.

use crate::categorical::validate_2x2_table;
use crate::error::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of McNemar's test
#[derive(Debug, Clone)]
pub struct McNemarkResult {
    /// Chi-square test statistic
    pub statistic: f64,
    /// Degrees of freedom (always 1)
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Whether continuity correction was applied
    pub corrected: bool,
    /// Name of the method
    pub method: String,
}

/// McNemar's test for paired nominal data.
///
/// Tests the null hypothesis that marginal homogeneity holds for
/// a 2x2 contingency table of paired observations.
///
/// # Table structure
/// ```text
///              | Time 2: +  | Time 2: -  |
/// -------------|------------|------------|
/// Time 1: +    |    a       |    b       |
/// Time 1: -    |    c       |    d       |
/// ```
///
/// The test examines whether b and c differ significantly.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `correction` - Apply Edwards' continuity correction
///
/// # Returns
/// * `McNemarkResult` containing test statistic, df, and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::mcnemar_test;
///
/// // Before/after treatment: 10 stayed positive, 20 went from + to -,
/// // 5 went from - to +, 65 stayed negative
/// let table = [[10, 20], [5, 65]];
///
/// let result = mcnemar_test(&table, false).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `mcnemar.test(matrix, correct = FALSE)`
pub fn mcnemar_test(table: &[[usize; 2]; 2], correction: bool) -> Result<McNemarkResult> {
    validate_2x2_table(table)?;

    let b = table[0][1] as f64;
    let c = table[1][0] as f64;

    let bc_sum = b + c;

    let statistic = if bc_sum == 0.0 {
        0.0
    } else if correction {
        // Edwards' continuity correction
        let diff = (b - c).abs() - 1.0;
        if diff <= 0.0 {
            0.0
        } else {
            diff * diff / bc_sum
        }
    } else {
        (b - c).powi(2) / bc_sum
    };

    // p-value from chi-square distribution with 1 df
    let df = 1.0;
    let p_value = if statistic > 0.0 {
        let chi_dist = ChiSquared::new(df).unwrap();
        1.0 - chi_dist.cdf(statistic)
    } else {
        1.0
    };

    let method = if correction {
        "McNemar's Chi-squared test with continuity correction"
    } else {
        "McNemar's Chi-squared test"
    };

    Ok(McNemarkResult {
        statistic,
        df,
        p_value,
        corrected: correction,
        method: method.to_string(),
    })
}

/// Result of an exact binomial test (McNemar's exact test)
#[derive(Debug, Clone)]
pub struct McNemarkExactResult {
    /// p-value
    pub p_value: f64,
    /// Number of discordant pairs with b > c
    pub b: usize,
    /// Number of discordant pairs with c > b
    pub c: usize,
    /// Name of the method
    pub method: String,
}

/// McNemar's exact test using binomial distribution.
///
/// More accurate than the chi-square approximation for small samples.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
///
/// # Returns
/// * `McNemarkExactResult` containing the exact p-value
///
/// # R equivalent
/// `mcnemar.test(matrix, correct = FALSE)` with small n uses exact test
pub fn mcnemar_exact(table: &[[usize; 2]; 2]) -> Result<McNemarkExactResult> {
    validate_2x2_table(table)?;

    let b = table[0][1];
    let c = table[1][0];
    let n = b + c;

    // Two-sided exact p-value using binomial distribution
    // P(X <= min(b,c)) + P(X >= max(b,c)) where X ~ Binomial(n, 0.5)
    let p_value = if n == 0 {
        1.0
    } else {
        let k = b.min(c);
        // Sum P(X = 0) + P(X = 1) + ... + P(X = k) + P(X = n-k) + ... + P(X = n)
        // = 2 * sum(P(X = i) for i in 0..=k) if k < n/2
        // For binomial(n, 0.5): P(X = k) = C(n,k) / 2^n
        let mut p = 0.0;
        for i in 0..=k {
            p += binomial_pmf(n, i, 0.5);
        }
        // Two-sided
        (2.0 * p).min(1.0)
    };

    Ok(McNemarkExactResult {
        p_value,
        b,
        c,
        method: "McNemar's Chi-squared test (exact)".to_string(),
    })
}

/// Binomial PMF: P(X = k) where X ~ Binomial(n, p)
fn binomial_pmf(n: usize, k: usize, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    log_binomial_coeff(n, k).exp() * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

/// Log of binomial coefficient
fn log_binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Log factorial
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n <= 20 {
        let mut result = 0.0;
        for i in 2..=n {
            result += (i as f64).ln();
        }
        return result;
    }
    // Stirling's approximation
    let n_f = n as f64;
    n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcnemar_basic() {
        // Example from many textbooks
        let table = [[59, 6], [16, 80]];

        let result = mcnemar_test(&table, false).unwrap();

        // Chi-square = (6-16)^2 / (6+16) = 100/22 ≈ 4.545
        assert!((result.statistic - 4.545454545).abs() < 0.001);
        assert!((result.df - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcnemar_with_correction() {
        let table = [[59, 6], [16, 80]];

        let without = mcnemar_test(&table, false).unwrap();
        let with_correction = mcnemar_test(&table, true).unwrap();

        // Correction should reduce the statistic
        assert!(with_correction.statistic < without.statistic);
    }

    #[test]
    fn test_mcnemar_symmetric() {
        // When b = c, no significant difference
        let table = [[50, 10], [10, 30]];

        let result = mcnemar_test(&table, false).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcnemar_exact() {
        let table = [[10, 3], [7, 80]];

        let result = mcnemar_exact(&table).unwrap();

        // With b=3, c=7, n=10, exact test
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
    }

    #[test]
    fn test_mcnemar_exact_equal() {
        let table = [[50, 5], [5, 40]];

        let result = mcnemar_exact(&table).unwrap();

        // b = c = 5, should give p-value = 1.0
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }
}
