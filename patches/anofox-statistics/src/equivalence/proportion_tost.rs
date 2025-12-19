//! TOST equivalence tests for proportions.
//!
//! Tests whether one or two proportions are practically equivalent to a
//! specified value using normal approximation.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Perform TOST for a single proportion.
///
/// Tests whether an observed proportion is practically equivalent to a
/// specified value (usually 0.5 or some other benchmark).
///
/// # Arguments
/// * `x` - Number of successes
/// * `n` - Total number of trials
/// * `p0` - Null proportion to test equivalence against
/// * `bounds` - Equivalence bounds for the difference (p - p0)
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_prop_one, EquivalenceBounds};
///
/// // Test if observed proportion (48/100 = 0.48) is equivalent to 0.5
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
/// let result = tost_prop_one(48, 100, 0.5, &bounds, 0.05).unwrap();
/// println!("Proportion equivalent to 0.5: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTone.prop(prop1, n, prop_null, low_eqbound, high_eqbound)`
pub fn tost_prop_one(
    x: usize,
    n: usize,
    p0: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_alpha(alpha)?;

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    if x > n {
        return Err(StatError::InvalidParameter(format!(
            "x ({}) cannot exceed n ({})",
            x, n
        )));
    }

    if !(0.0..=1.0).contains(&p0) {
        return Err(StatError::InvalidParameter(format!(
            "p0 must be between 0 and 1, got {}",
            p0
        )));
    }

    // Observed proportion
    let p_hat = x as f64 / n as f64;

    // Get bounds (must be in proportion difference scale)
    let (lower_bound, upper_bound) = bounds_for_proportion(bounds)?;

    // Standard error using observed proportion
    let se = (p_hat * (1.0 - p_hat) / n as f64).sqrt();

    // Handle edge cases where SE is 0
    if se == 0.0 {
        // All successes or all failures - equivalence depends on whether
        // the estimate falls within bounds
        let estimate = p_hat - p0;
        let equivalent = estimate >= lower_bound && estimate <= upper_bound;

        return Ok(TostResult {
            estimate,
            ci: (estimate, estimate),
            bounds: (lower_bound, upper_bound),
            lower_test: OneSidedTestResult {
                hypothesis: format!("H0: p - p0 <= {:.4}", lower_bound),
                statistic: f64::INFINITY,
                p_value: if estimate > lower_bound { 0.0 } else { 1.0 },
                rejected: estimate > lower_bound,
            },
            upper_test: OneSidedTestResult {
                hypothesis: format!("H0: p - p0 >= {:.4}", upper_bound),
                statistic: f64::NEG_INFINITY,
                p_value: if estimate < upper_bound { 0.0 } else { 1.0 },
                rejected: estimate < upper_bound,
            },
            tost_p_value: if equivalent { 0.0 } else { 1.0 },
            equivalent,
            alpha,
            n,
            df: None,
            method: "One-proportion TOST".to_string(),
        });
    }

    // Estimate: difference from null
    let estimate = p_hat - p0;

    // Standard normal for inference
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    // Lower test: H0: (p - p0) <= lower_bound
    let z_lower = (estimate - lower_bound) / se;
    let p_lower = 1.0 - normal.cdf(z_lower);

    // Upper test: H0: (p - p0) >= upper_bound
    let z_upper = (estimate - upper_bound) / se;
    let p_upper = normal.cdf(z_upper);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // (1 - 2*alpha) confidence interval
    let z_crit = normal.inverse_cdf(1.0 - alpha);
    let margin = z_crit * se;
    let ci = (estimate - margin, estimate + margin);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: p - p0 <= {:.4}", lower_bound),
            statistic: z_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: p - p0 >= {:.4}", upper_bound),
            statistic: z_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n,
        df: None,
        method: "One-proportion TOST".to_string(),
    })
}

/// Perform TOST for two independent proportions.
///
/// Tests whether the difference between two proportions is practically
/// equivalent to zero.
///
/// # Arguments
/// * `x1` - Number of successes in group 1
/// * `n1` - Total trials in group 1
/// * `x2` - Number of successes in group 2
/// * `n2` - Total trials in group 2
/// * `bounds` - Equivalence bounds for the difference (p1 - p2)
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_prop_two, EquivalenceBounds};
///
/// // Test if two proportions (45/100 vs 48/100) are equivalent
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
/// let result = tost_prop_two(45, 100, 48, 100, &bounds, 0.05).unwrap();
/// println!("Proportions equivalent: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTtwo.prop(prop1, prop2, n1, n2, low_eqbound, high_eqbound)`
pub fn tost_prop_two(
    x1: usize,
    n1: usize,
    x2: usize,
    n2: usize,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_alpha(alpha)?;

    if n1 == 0 || n2 == 0 {
        return Err(StatError::EmptyData);
    }

    if x1 > n1 {
        return Err(StatError::InvalidParameter(format!(
            "x1 ({}) cannot exceed n1 ({})",
            x1, n1
        )));
    }

    if x2 > n2 {
        return Err(StatError::InvalidParameter(format!(
            "x2 ({}) cannot exceed n2 ({})",
            x2, n2
        )));
    }

    // Observed proportions
    let p1 = x1 as f64 / n1 as f64;
    let p2 = x2 as f64 / n2 as f64;

    // Get bounds
    let (lower_bound, upper_bound) = bounds_for_proportion(bounds)?;

    // Estimate: difference in proportions
    let estimate = p1 - p2;

    // Standard error for difference (unpooled, since we're not testing H0: p1 = p2)
    let var1 = p1 * (1.0 - p1) / n1 as f64;
    let var2 = p2 * (1.0 - p2) / n2 as f64;
    let se = (var1 + var2).sqrt();

    // Handle edge case where SE is very small
    if se < 1e-15 {
        let equivalent = estimate >= lower_bound && estimate <= upper_bound;
        return Ok(TostResult {
            estimate,
            ci: (estimate, estimate),
            bounds: (lower_bound, upper_bound),
            lower_test: OneSidedTestResult {
                hypothesis: format!("H0: p1 - p2 <= {:.4}", lower_bound),
                statistic: f64::INFINITY,
                p_value: if estimate > lower_bound { 0.0 } else { 1.0 },
                rejected: estimate > lower_bound,
            },
            upper_test: OneSidedTestResult {
                hypothesis: format!("H0: p1 - p2 >= {:.4}", upper_bound),
                statistic: f64::NEG_INFINITY,
                p_value: if estimate < upper_bound { 0.0 } else { 1.0 },
                rejected: estimate < upper_bound,
            },
            tost_p_value: if equivalent { 0.0 } else { 1.0 },
            equivalent,
            alpha,
            n: n1 + n2,
            df: None,
            method: "Two-proportion TOST".to_string(),
        });
    }

    // Standard normal for inference
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    // Lower test: H0: (p1 - p2) <= lower_bound
    let z_lower = (estimate - lower_bound) / se;
    let p_lower = 1.0 - normal.cdf(z_lower);

    // Upper test: H0: (p1 - p2) >= upper_bound
    let z_upper = (estimate - upper_bound) / se;
    let p_upper = normal.cdf(z_upper);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // (1 - 2*alpha) confidence interval
    let z_crit = normal.inverse_cdf(1.0 - alpha);
    let margin = z_crit * se;
    let ci = (estimate - margin, estimate + margin);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: p1 - p2 <= {:.4}", lower_bound),
            statistic: z_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: p1 - p2 >= {:.4}", upper_bound),
            statistic: z_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n: n1 + n2,
        df: None,
        method: "Two-proportion TOST".to_string(),
    })
}

/// Convert bounds for proportion tests (must be Raw or Symmetric, not CohenD).
fn bounds_for_proportion(bounds: &EquivalenceBounds) -> Result<(f64, f64)> {
    match bounds {
        EquivalenceBounds::Raw { lower, upper } => {
            if *lower < -1.0 || *upper > 1.0 {
                return Err(StatError::InvalidParameter(
                    "Proportion bounds must be between -1 and 1".to_string(),
                ));
            }
            Ok((*lower, *upper))
        }
        EquivalenceBounds::Symmetric { delta } => {
            if *delta > 1.0 {
                return Err(StatError::InvalidParameter(
                    "Symmetric delta for proportions must not exceed 1".to_string(),
                ));
            }
            Ok((-*delta, *delta))
        }
        EquivalenceBounds::CohenD { .. } => Err(StatError::InvalidParameter(
            "Cohen's d bounds not applicable for proportion TOST; use Raw or Symmetric bounds"
                .to_string(),
        )),
    }
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
    fn test_one_prop_equivalent() {
        // Proportion close to 0.5
        let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
        let result = tost_prop_one(52, 100, 0.5, &bounds, 0.05).unwrap();

        // 52% vs 50% with ±10% bounds should likely be equivalent
        assert!(result.estimate.abs() < 0.1);
    }

    #[test]
    fn test_one_prop_not_equivalent() {
        // Proportion far from null
        let bounds = EquivalenceBounds::Symmetric { delta: 0.05 };
        let result = tost_prop_one(70, 100, 0.5, &bounds, 0.05).unwrap();

        // 70% vs 50% with ±5% bounds should NOT be equivalent
        assert!(!result.equivalent);
    }

    #[test]
    fn test_two_prop_equivalent() {
        // Very similar proportions
        let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
        let result = tost_prop_two(51, 100, 49, 100, &bounds, 0.05).unwrap();

        // 51% vs 49% with ±10% bounds
        assert!(result.estimate.abs() < 0.1);
    }

    #[test]
    fn test_two_prop_not_equivalent() {
        // Different proportions
        let bounds = EquivalenceBounds::Symmetric { delta: 0.05 };
        let result = tost_prop_two(70, 100, 50, 100, &bounds, 0.05).unwrap();

        // 70% vs 50% with ±5% bounds should NOT be equivalent
        assert!(!result.equivalent);
    }

    #[test]
    fn test_invalid_x_greater_than_n() {
        let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
        assert!(tost_prop_one(110, 100, 0.5, &bounds, 0.05).is_err());
    }

    #[test]
    fn test_invalid_p0() {
        let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
        assert!(tost_prop_one(50, 100, 1.5, &bounds, 0.05).is_err());
        assert!(tost_prop_one(50, 100, -0.1, &bounds, 0.05).is_err());
    }

    #[test]
    fn test_cohen_d_not_allowed() {
        let bounds = EquivalenceBounds::CohenD { d: 0.5 };
        assert!(tost_prop_one(50, 100, 0.5, &bounds, 0.05).is_err());
        assert!(tost_prop_two(50, 100, 50, 100, &bounds, 0.05).is_err());
    }

    #[test]
    fn test_edge_case_all_successes() {
        let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
        let result = tost_prop_one(100, 100, 0.95, &bounds, 0.05).unwrap();

        // 100% vs 95% - estimate is 0.05
        assert!((result.estimate - 0.05).abs() < 1e-10);
    }
}
