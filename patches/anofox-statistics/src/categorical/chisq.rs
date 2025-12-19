//! Chi-square tests for categorical data.

use crate::categorical::{expected_frequencies, validate_contingency_table};
use crate::error::{Result, StatError};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of a chi-square test
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square test statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Expected frequencies under the null hypothesis
    pub expected: Vec<Vec<f64>>,
    /// Standardized residuals: (O - E) / sqrt(E)
    pub residuals: Option<Vec<Vec<f64>>>,
    /// Name of the method used
    pub method: String,
}

/// Pearson's chi-square test of independence for contingency tables.
///
/// Tests the null hypothesis that the row and column variables are independent.
///
/// # Arguments
/// * `observed` - Contingency table as a 2D vector of observed counts
/// * `correction` - Apply Yates' continuity correction (only for 2x2 tables)
///
/// # Returns
/// * `ChiSquareResult` containing the test statistic, df, p-value, and expected frequencies
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::chisq_test;
///
/// // 2x2 contingency table
/// let observed = vec![
///     vec![10, 20],
///     vec![15, 25],
/// ];
///
/// let result = chisq_test(&observed, false).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `chisq.test(matrix, correct = FALSE)`
pub fn chisq_test(observed: &[Vec<usize>], correction: bool) -> Result<ChiSquareResult> {
    let (n_rows, n_cols) = validate_contingency_table(observed)?;

    // Need at least 2x2 table
    if n_rows < 2 || n_cols < 2 {
        return Err(StatError::InvalidParameter(
            "Contingency table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    let expected = expected_frequencies(observed);

    // Check for expected frequencies < 5 (warning in R)
    // We'll still compute but this affects validity

    // Degrees of freedom: (r-1)(c-1)
    let df = ((n_rows - 1) * (n_cols - 1)) as f64;

    // Compute chi-square statistic
    let mut chi_sq = 0.0;
    let apply_yates = correction && n_rows == 2 && n_cols == 2;

    for i in 0..n_rows {
        for j in 0..n_cols {
            let o = observed[i][j] as f64;
            let e = expected[i][j];
            if e > 0.0 {
                let diff = if apply_yates {
                    // Yates' continuity correction: |O - E| - 0.5
                    (o - e).abs() - 0.5
                } else {
                    o - e
                };
                chi_sq += diff * diff / e;
            }
        }
    }

    // Ensure non-negative after correction
    chi_sq = chi_sq.max(0.0);

    // Compute p-value
    let p_value = if df > 0.0 && chi_sq.is_finite() {
        let chi_dist = ChiSquared::new(df).unwrap();
        1.0 - chi_dist.cdf(chi_sq)
    } else {
        1.0
    };

    // Compute standardized residuals
    let residuals: Vec<Vec<f64>> = (0..n_rows)
        .map(|i| {
            (0..n_cols)
                .map(|j| {
                    let o = observed[i][j] as f64;
                    let e = expected[i][j];
                    if e > 0.0 {
                        (o - e) / e.sqrt()
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let method = if apply_yates {
        "Pearson's Chi-squared test with Yates' continuity correction".to_string()
    } else {
        "Pearson's Chi-squared test".to_string()
    };

    Ok(ChiSquareResult {
        statistic: chi_sq,
        df,
        p_value,
        expected,
        residuals: Some(residuals),
        method,
    })
}

/// Chi-square goodness-of-fit test.
///
/// Tests whether observed frequencies match expected proportions.
///
/// # Arguments
/// * `observed` - Vector of observed counts
/// * `expected_props` - Expected proportions (must sum to 1.0). If None, assumes uniform.
///
/// # Returns
/// * `ChiSquareResult` containing the test statistic, df, and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::chisq_goodness_of_fit;
///
/// // Test if die is fair
/// let observed = vec![16, 18, 14, 17, 15, 20];  // 100 rolls
///
/// // Test against uniform distribution (fair die)
/// let result = chisq_goodness_of_fit(&observed, None).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `chisq.test(x, p = expected_props)`
pub fn chisq_goodness_of_fit(
    observed: &[usize],
    expected_props: Option<&[f64]>,
) -> Result<ChiSquareResult> {
    if observed.is_empty() {
        return Err(StatError::EmptyData);
    }

    let k = observed.len();
    if k < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: k });
    }

    let total: usize = observed.iter().sum();
    if total == 0 {
        return Err(StatError::EmptyData);
    }

    // Compute expected frequencies
    let expected: Vec<f64> = if let Some(props) = expected_props {
        if props.len() != k {
            return Err(StatError::InvalidParameter(format!(
                "Expected proportions length {} does not match observed length {}",
                props.len(),
                k
            )));
        }

        // Validate proportions sum to 1
        let sum: f64 = props.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(StatError::InvalidParameter(format!(
                "Expected proportions must sum to 1.0, got {}",
                sum
            )));
        }

        props.iter().map(|&p| p * total as f64).collect()
    } else {
        // Uniform distribution
        let uniform = total as f64 / k as f64;
        vec![uniform; k]
    };

    // Degrees of freedom: k - 1
    let df = (k - 1) as f64;

    // Compute chi-square statistic
    let mut chi_sq = 0.0;
    for i in 0..k {
        let o = observed[i] as f64;
        let e = expected[i];
        if e > 0.0 {
            chi_sq += (o - e).powi(2) / e;
        }
    }

    // Compute p-value
    let p_value = if df > 0.0 && chi_sq.is_finite() {
        let chi_dist = ChiSquared::new(df).unwrap();
        1.0 - chi_dist.cdf(chi_sq)
    } else {
        1.0
    };

    // Convert to 2D format for consistency
    let expected_2d = vec![expected.clone()];
    let residuals: Vec<Vec<f64>> = vec![observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| {
            if e > 0.0 {
                (o as f64 - e) / e.sqrt()
            } else {
                0.0
            }
        })
        .collect()];

    Ok(ChiSquareResult {
        statistic: chi_sq,
        df,
        p_value,
        expected: expected_2d,
        residuals: Some(residuals),
        method: "Chi-squared test for given probabilities".to_string(),
    })
}

/// G-test (likelihood ratio test) for contingency tables.
///
/// Uses log-likelihood ratio instead of squared differences.
/// Asymptotically equivalent to chi-square but may perform better
/// with sparse data.
///
/// # Arguments
/// * `observed` - Contingency table as a 2D vector of observed counts
///
/// # Returns
/// * `ChiSquareResult` containing the G statistic (in place of chi-square), df, and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::g_test;
///
/// let observed = vec![
///     vec![10, 20],
///     vec![15, 25],
/// ];
///
/// let result = g_test(&observed).unwrap();
/// println!("G = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `DescTools::GTest(matrix)`
pub fn g_test(observed: &[Vec<usize>]) -> Result<ChiSquareResult> {
    let (n_rows, n_cols) = validate_contingency_table(observed)?;

    // Need at least 2x2 table
    if n_rows < 2 || n_cols < 2 {
        return Err(StatError::InvalidParameter(
            "Contingency table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    let expected = expected_frequencies(observed);

    // Degrees of freedom: (r-1)(c-1)
    let df = ((n_rows - 1) * (n_cols - 1)) as f64;

    // Compute G statistic: 2 * sum(O * ln(O/E))
    let mut g_stat = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let o = observed[i][j] as f64;
            let e = expected[i][j];
            if o > 0.0 && e > 0.0 {
                g_stat += o * (o / e).ln();
            }
        }
    }
    g_stat *= 2.0;

    // Ensure non-negative
    g_stat = g_stat.max(0.0);

    // Compute p-value (G is approximately chi-square distributed)
    let p_value = if df > 0.0 && g_stat.is_finite() {
        let chi_dist = ChiSquared::new(df).unwrap();
        1.0 - chi_dist.cdf(g_stat)
    } else {
        1.0
    };

    Ok(ChiSquareResult {
        statistic: g_stat,
        df,
        p_value,
        expected,
        residuals: None,
        method: "Log likelihood ratio (G-test) test of independence".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chisq_2x2() {
        let observed = vec![vec![10, 20], vec![30, 40]];

        let result = chisq_test(&observed, false).unwrap();

        assert!(result.statistic > 0.0);
        assert!((result.df - 1.0).abs() < 1e-10);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_chisq_with_yates() {
        let observed = vec![vec![10, 20], vec![30, 40]];

        let without = chisq_test(&observed, false).unwrap();
        let with_yates = chisq_test(&observed, true).unwrap();

        // Yates' correction should reduce the chi-square statistic
        assert!(with_yates.statistic < without.statistic);
    }

    #[test]
    fn test_chisq_goodness_of_fit_uniform() {
        let observed = vec![20, 20, 20, 20, 20]; // Perfect uniform

        let result = chisq_goodness_of_fit(&observed, None).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!((result.df - 4.0).abs() < 1e-10);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chisq_goodness_of_fit_custom_props() {
        let observed = vec![50, 30, 20]; // 100 total

        let props = vec![0.5, 0.3, 0.2]; // Match observed proportions
        let result = chisq_goodness_of_fit(&observed, Some(&props)).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_g_test_2x2() {
        let observed = vec![vec![10, 20], vec![30, 40]];

        let result = g_test(&observed).unwrap();

        assert!(result.statistic > 0.0);
        assert!((result.df - 1.0).abs() < 1e-10);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_chisq_larger_table() {
        let observed = vec![vec![10, 20, 30], vec![20, 30, 40], vec![30, 40, 50]];

        let result = chisq_test(&observed, false).unwrap();

        // df = (3-1)(3-1) = 4
        assert!((result.df - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_chisq_empty_error() {
        let observed: Vec<Vec<usize>> = vec![];
        assert!(chisq_test(&observed, false).is_err());
    }

    #[test]
    fn test_chisq_1x2_error() {
        let observed = vec![vec![10, 20]];
        assert!(chisq_test(&observed, false).is_err());
    }
}
