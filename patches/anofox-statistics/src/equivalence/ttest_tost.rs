//! TOST equivalence tests for means using t-tests.
//!
//! Provides one-sample, two-sample (independent), and paired t-test based TOST.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use crate::utils::math::{mean, variance};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Perform one-sample TOST to test if a mean is equivalent to a specified value.
///
/// Tests whether the population mean is practically equivalent to `mu` within
/// the specified equivalence bounds.
///
/// # Arguments
/// * `x` - Sample data
/// * `mu` - Value to test equivalence against (usually 0)
/// * `bounds` - Equivalence bounds specification
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_t_test_one_sample, EquivalenceBounds};
///
/// let x = vec![0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.15, 0.05];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();
/// println!("Equivalent: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTone(m = mean(x), mu = mu, sd = sd(x), n = length(x), low_eqbound = -delta, high_eqbound = delta)`
pub fn tost_t_test_one_sample(
    x: &[f64],
    mu: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_alpha(alpha)?;

    let n = x.len();
    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    let mean_x = mean(x)?;
    let var_x = variance(x)?;
    let sd_x = var_x.sqrt();
    let se = sd_x / (n as f64).sqrt();
    let df = (n - 1) as f64;

    // Convert bounds to raw units
    let (lower_bound, upper_bound) = bounds.to_raw(Some(sd_x))?;

    // Estimate: difference from mu
    let estimate = mean_x - mu;

    // t-distribution for inference
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create t-distribution: {}", e))
    })?;

    // Lower test: H0: estimate <= lower_bound (effect too negative)
    // Reject if estimate significantly greater than lower_bound
    let t_lower = (estimate - lower_bound) / se;
    let p_lower = 1.0 - t_dist.cdf(t_lower);

    // Upper test: H0: estimate >= upper_bound (effect too positive)
    // Reject if estimate significantly less than upper_bound
    let t_upper = (estimate - upper_bound) / se;
    let p_upper = t_dist.cdf(t_upper);

    // TOST p-value is the maximum of the two one-sided p-values
    let tost_p = p_lower.max(p_upper);

    // Confidence interval at (1 - 2*alpha) level for TOST
    let t_crit = t_dist.inverse_cdf(1.0 - alpha);
    let margin = t_crit * se;
    let ci = (estimate - margin, estimate + margin);

    // Equivalence is established if CI falls entirely within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: effect <= {:.4}", lower_bound),
            statistic: t_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: effect >= {:.4}", upper_bound),
            statistic: t_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n,
        df: Some(df),
        method: "One-sample TOST".to_string(),
    })
}

/// Perform two-sample TOST to test if two means are equivalent.
///
/// Uses Welch's t-test (unequal variances) to test whether the difference
/// in population means is practically equivalent to zero.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `bounds` - Equivalence bounds specification
/// * `alpha` - Significance level (default: 0.05)
/// * `pooled` - If true, use pooled variance (Student's t); if false, use Welch's t (default)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_t_test_two_sample, EquivalenceBounds};
///
/// let x = vec![10.1, 9.8, 10.2, 10.0, 9.9];
/// let y = vec![10.0, 10.1, 9.9, 10.2, 10.0];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();
/// println!("Groups equivalent: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTtwo(m1, m2, sd1, sd2, n1, n2, low_eqbound_d, high_eqbound_d)`
pub fn tost_t_test_two_sample(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    pooled: bool,
) -> Result<TostResult> {
    validate_alpha(alpha)?;

    let nx = x.len();
    let ny = y.len();

    if nx < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: nx });
    }
    if ny < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: ny });
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let var_x = variance(x)?;
    let var_y = variance(y)?;

    let nx_f = nx as f64;
    let ny_f = ny as f64;

    // Compute pooled standard deviation for Cohen's d conversion
    let pooled_var = ((nx_f - 1.0) * var_x + (ny_f - 1.0) * var_y) / (nx_f + ny_f - 2.0);
    let pooled_sd = pooled_var.sqrt();

    // Convert bounds to raw units using pooled SD
    let (lower_bound, upper_bound) = bounds.to_raw(Some(pooled_sd))?;

    // Estimate: difference in means
    let estimate = mean_x - mean_y;

    // Compute SE and df based on whether we use pooled or Welch
    let (se, df) = if pooled {
        // Student's t-test (pooled variance)
        let se = (pooled_var * (1.0 / nx_f + 1.0 / ny_f)).sqrt();
        let df = nx_f + ny_f - 2.0;
        (se, df)
    } else {
        // Welch's t-test (unequal variances)
        let se_x = var_x / nx_f;
        let se_y = var_y / ny_f;
        let se = (se_x + se_y).sqrt();

        // Welch-Satterthwaite degrees of freedom
        let num = (se_x + se_y).powi(2);
        let denom = (se_x.powi(2) / (nx_f - 1.0)) + (se_y.powi(2) / (ny_f - 1.0));
        let df = num / denom;
        (se, df)
    };

    // t-distribution for inference
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create t-distribution: {}", e))
    })?;

    // Lower test: H0: estimate <= lower_bound
    let t_lower = (estimate - lower_bound) / se;
    let p_lower = 1.0 - t_dist.cdf(t_lower);

    // Upper test: H0: estimate >= upper_bound
    let t_upper = (estimate - upper_bound) / se;
    let p_upper = t_dist.cdf(t_upper);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // (1 - 2*alpha) confidence interval
    let t_crit = t_dist.inverse_cdf(1.0 - alpha);
    let margin = t_crit * se;
    let ci = (estimate - margin, estimate + margin);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    let method = if pooled {
        "Two-sample TOST (Student's t)"
    } else {
        "Two-sample TOST (Welch's t)"
    };

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: effect <= {:.4}", lower_bound),
            statistic: t_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: effect >= {:.4}", upper_bound),
            statistic: t_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n: nx + ny,
        df: Some(df),
        method: method.to_string(),
    })
}

/// Perform paired-samples TOST to test if paired differences are equivalent to zero.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (must be same length as x)
/// * `bounds` - Equivalence bounds specification
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_t_test_paired, EquivalenceBounds};
///
/// let before = vec![10.0, 12.0, 11.0, 13.0, 10.5];
/// let after = vec![10.2, 11.8, 11.1, 13.1, 10.4];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_t_test_paired(&before, &after, &bounds, 0.05).unwrap();
/// println!("Treatment effect equivalent to zero: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTpaired(m_diff, sd_diff, n, low_eqbound, high_eqbound)`
pub fn tost_t_test_paired(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_alpha(alpha)?;

    let n = x.len();
    if n != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "Paired samples must have equal length: {} vs {}",
            n,
            y.len()
        )));
    }

    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    // Compute differences
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi - yi).collect();

    let mean_diff = mean(&diffs)?;
    let var_diff = variance(&diffs)?;
    let sd_diff = var_diff.sqrt();
    let se = sd_diff / (n as f64).sqrt();
    let df = (n - 1) as f64;

    // Convert bounds to raw units using SD of differences
    let (lower_bound, upper_bound) = bounds.to_raw(Some(sd_diff))?;

    // Estimate: mean difference
    let estimate = mean_diff;

    // t-distribution for inference
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create t-distribution: {}", e))
    })?;

    // Lower test: H0: estimate <= lower_bound
    let t_lower = (estimate - lower_bound) / se;
    let p_lower = 1.0 - t_dist.cdf(t_lower);

    // Upper test: H0: estimate >= upper_bound
    let t_upper = (estimate - upper_bound) / se;
    let p_upper = t_dist.cdf(t_upper);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // (1 - 2*alpha) confidence interval
    let t_crit = t_dist.inverse_cdf(1.0 - alpha);
    let margin = t_crit * se;
    let ci = (estimate - margin, estimate + margin);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: effect <= {:.4}", lower_bound),
            statistic: t_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: effect >= {:.4}", upper_bound),
            statistic: t_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n,
        df: Some(df),
        method: "Paired-samples TOST".to_string(),
    })
}

/// Validate alpha parameter.
fn validate_alpha(alpha: f64) -> Result<()> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_sample_tost_equivalent() {
        // Data with small mean close to 0
        let x = vec![0.1, -0.1, 0.05, -0.05, 0.0, 0.08, -0.03, 0.02];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();

        // Should be equivalent (mean is very close to 0, well within ±0.5)
        assert!(result.equivalent);
        assert!(result.tost_p_value < 0.05);
    }

    #[test]
    fn test_one_sample_tost_not_equivalent() {
        // Data with mean clearly outside bounds
        let x = vec![2.0, 2.1, 1.9, 2.2, 1.8, 2.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();

        // Should NOT be equivalent (mean ~2.0 is outside ±0.5)
        assert!(!result.equivalent);
    }

    #[test]
    fn test_two_sample_tost_equivalent() {
        // Two groups with very similar means
        let x = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0];
        let y = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

        // Should be equivalent
        assert!(result.equivalent);
    }

    #[test]
    fn test_two_sample_tost_not_equivalent() {
        // Two groups with different means
        let x = vec![10.0, 10.1, 9.9, 10.2, 10.0];
        let y = vec![12.0, 12.1, 11.9, 12.2, 12.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

        // Should NOT be equivalent (difference ~2)
        assert!(!result.equivalent);
    }

    #[test]
    fn test_paired_tost_equivalent() {
        // Paired data with small differences
        let before = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0];
        let after = vec![10.1, 11.9, 11.05, 13.1, 10.45, 11.55, 12.45, 10.05];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_t_test_paired(&before, &after, &bounds, 0.05).unwrap();

        // Should be equivalent (differences are tiny)
        assert!(result.equivalent);
    }

    #[test]
    fn test_cohen_d_bounds() {
        // Test with Cohen's d bounds
        let x = vec![0.1, 0.2, 0.0, -0.1, 0.15, -0.05, 0.1, 0.0];
        let bounds = EquivalenceBounds::CohenD { d: 0.5 };

        let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();

        // Bounds should be converted based on sample SD
        assert!(result.bounds.0 < 0.0);
        assert!(result.bounds.1 > 0.0);
    }

    #[test]
    fn test_invalid_alpha() {
        let x = vec![1.0, 2.0, 3.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 0.0).is_err());
        assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 1.0).is_err());
        assert!(tost_t_test_one_sample(&x, 0.0, &bounds, -0.1).is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).is_err());
    }
}
