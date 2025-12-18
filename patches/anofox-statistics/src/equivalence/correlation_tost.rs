//! TOST equivalence tests for correlations.
//!
//! Tests whether a correlation coefficient is practically equivalent to zero
//! (or another specified value) using Fisher's z-transformation.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Method for computing the correlation coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationTostMethod {
    /// Pearson product-moment correlation
    Pearson,
    /// Spearman rank correlation
    Spearman,
}

/// Perform TOST for a correlation coefficient.
///
/// Tests whether a correlation is practically equivalent to a specified value
/// (typically 0) using Fisher's z-transformation for inference.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
/// * `rho_null` - Null value for correlation (usually 0)
/// * `bounds` - Equivalence bounds in correlation scale (must be Raw or Symmetric)
/// * `alpha` - Significance level (default: 0.05)
/// * `method` - Correlation method (Pearson or Spearman)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_correlation, EquivalenceBounds, CorrelationTostMethod};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![1.1, 2.3, 2.8, 4.2, 5.0, 5.8, 7.2, 8.1, 8.9, 10.2];
///
/// // Test if correlation is equivalent to zero within ±0.3
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.3 };
/// let result = tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson).unwrap();
/// ```
///
/// # R equivalent
/// `TOSTER::TOSTr(r, n, low_eqbound_r, high_eqbound_r)`
pub fn tost_correlation(
    x: &[f64],
    y: &[f64],
    rho_null: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
    method: CorrelationTostMethod,
) -> Result<TostResult> {
    validate_inputs(x, y, alpha)?;

    let n = x.len();

    // Check we have enough data for Fisher's z (requires n >= 4)
    if n < 4 {
        return Err(StatError::InsufficientData { needed: 4, got: n });
    }

    // Compute correlation
    let r = match method {
        CorrelationTostMethod::Pearson => pearson_cor(x, y)?,
        CorrelationTostMethod::Spearman => spearman_cor(x, y)?,
    };

    // Get bounds in correlation scale
    let (lower_bound, upper_bound) = bounds_for_correlation(bounds)?;

    // Validate bounds are valid correlations
    if lower_bound < -1.0 || upper_bound > 1.0 {
        return Err(StatError::InvalidParameter(
            "Correlation bounds must be between -1 and 1".to_string(),
        ));
    }

    // Fisher's z-transformation for inference
    // z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
    // SE(z) = 1 / sqrt(n - 3)

    let z_r = fisher_z(r);
    let z_lower = fisher_z(lower_bound + rho_null);
    let z_upper = fisher_z(upper_bound + rho_null);
    let se_z = 1.0 / ((n - 3) as f64).sqrt();

    // Standard normal for inference
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    // Lower test: H0: rho <= lower_bound (in z-scale)
    // Test statistic: (z_r - z_lower) / se_z
    let stat_lower = (z_r - z_lower) / se_z;
    let p_lower = 1.0 - normal.cdf(stat_lower);

    // Upper test: H0: rho >= upper_bound (in z-scale)
    // Test statistic: (z_r - z_upper) / se_z
    let stat_upper = (z_r - z_upper) / se_z;
    let p_upper = normal.cdf(stat_upper);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // (1 - 2*alpha) confidence interval for r
    let z_crit = normal.inverse_cdf(1.0 - alpha);
    let ci_z_lower = z_r - z_crit * se_z;
    let ci_z_upper = z_r + z_crit * se_z;

    // Transform CI back to r scale
    let ci_r_lower = fisher_z_inv(ci_z_lower);
    let ci_r_upper = fisher_z_inv(ci_z_upper);

    // The estimate relative to null
    let estimate = r - rho_null;

    // CI for (r - rho_null)
    let ci = (ci_r_lower - rho_null, ci_r_upper - rho_null);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    let method_name = match method {
        CorrelationTostMethod::Pearson => "Correlation TOST (Pearson)",
        CorrelationTostMethod::Spearman => "Correlation TOST (Spearman)",
    };

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: r <= {:.4}", lower_bound + rho_null),
            statistic: stat_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: r >= {:.4}", upper_bound + rho_null),
            statistic: stat_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n,
        df: Some((n - 3) as f64), // Effective df for Fisher's z
        method: method_name.to_string(),
    })
}

/// Fisher's z-transformation: z = arctanh(r)
fn fisher_z(r: f64) -> f64 {
    // Clamp r to avoid infinity at ±1
    let r = r.clamp(-0.9999999, 0.9999999);
    0.5 * ((1.0 + r) / (1.0 - r)).ln()
}

/// Inverse Fisher's z-transformation: r = tanh(z)
fn fisher_z_inv(z: f64) -> f64 {
    z.tanh()
}

/// Compute Pearson correlation coefficient.
fn pearson_cor(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len() as f64;

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = (sum_x2 * sum_y2).sqrt();
    if denom == 0.0 {
        return Err(StatError::InvalidParameter(
            "Cannot compute correlation: zero variance".to_string(),
        ));
    }

    Ok(sum_xy / denom)
}

/// Compute Spearman rank correlation coefficient.
fn spearman_cor(x: &[f64], y: &[f64]) -> Result<f64> {
    let ranks_x = compute_ranks(x);
    let ranks_y = compute_ranks(y);

    pearson_cor(&ranks_x, &ranks_y)
}

/// Compute ranks with average tie handling.
fn compute_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // Find all ties
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// Convert bounds for correlation (must be Raw or Symmetric, not CohenD).
fn bounds_for_correlation(bounds: &EquivalenceBounds) -> Result<(f64, f64)> {
    match bounds {
        EquivalenceBounds::Raw { lower, upper } => Ok((*lower, *upper)),
        EquivalenceBounds::Symmetric { delta } => Ok((-*delta, *delta)),
        EquivalenceBounds::CohenD { .. } => Err(StatError::InvalidParameter(
            "Cohen's d bounds not applicable for correlation TOST; use Raw or Symmetric bounds"
                .to_string(),
        )),
    }
}

/// Validate inputs for correlation TOST.
fn validate_inputs(x: &[f64], y: &[f64], alpha: f64) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "x and y must have same length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    // Check for non-finite values
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatError::InvalidParameter(format!(
                "Non-finite value at index {}: x={}, y={}",
                i, xi, yi
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_tost_weak_correlation() {
        // Data with weak correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 1.8, 3.2, 4.1, 4.8, 6.3, 7.0, 7.9, 9.2, 10.1];

        let bounds = EquivalenceBounds::Symmetric { delta: 0.3 };
        let result =
            tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson).unwrap();

        // Strong positive correlation (~0.99), should NOT be equivalent to zero within ±0.3
        assert!(!result.equivalent);
    }

    #[test]
    fn test_correlation_tost_near_zero() {
        // Data with correlation close to zero
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 4.9, 5.0, 5.1, 4.9];

        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
        let result =
            tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson).unwrap();

        // Correlation should be near zero
        assert!(result.estimate.abs() < 0.3);
    }

    #[test]
    fn test_spearman_correlation_tost() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let bounds = EquivalenceBounds::Symmetric { delta: 0.3 };
        let result =
            tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Spearman).unwrap();

        // Perfect monotonic relationship, r_s = 1.0
        assert!((result.estimate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fisher_z_transformation() {
        // Test basic properties
        assert!((fisher_z(0.0)).abs() < 1e-10);
        assert!(fisher_z(0.5) > 0.0);
        assert!(fisher_z(-0.5) < 0.0);

        // Test round-trip
        for r in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let z = fisher_z(r);
            let r_back = fisher_z_inv(z);
            assert!((r - r_back).abs() < 1e-10);
        }
    }

    #[test]
    fn test_invalid_bounds_for_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Cohen's d bounds should not work for correlation
        let bounds = EquivalenceBounds::CohenD { d: 0.5 };
        let result = tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson);
        assert!(result.is_err());
    }

    #[test]
    fn test_correlation_bounds_outside_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Bounds outside [-1, 1] should fail
        let bounds = EquivalenceBounds::Raw {
            lower: -1.5,
            upper: 1.5,
        };
        let result = tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson);
        assert!(result.is_err());
    }
}
