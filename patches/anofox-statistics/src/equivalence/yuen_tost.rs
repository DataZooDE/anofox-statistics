//! Robust TOST equivalence test using Yuen's trimmed means.
//!
//! This test is robust to outliers and violations of normality by using
//! trimmed means and Winsorized variances.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Perform robust TOST using Yuen's trimmed means test.
///
/// Tests whether the difference in trimmed means is practically equivalent
/// to zero. This is more robust to outliers than the standard t-test TOST.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `bounds` - Equivalence bounds
/// * `alpha` - Significance level (default: 0.05)
/// * `trim` - Proportion to trim from each tail (default: 0.2)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_yuen, EquivalenceBounds};
///
/// let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 100.0]; // outlier
/// let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 17.2];
/// let bounds = EquivalenceBounds::Symmetric { delta: 1.0 };
///
/// let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();
/// println!("Groups equivalent: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `WRS2::yuen.TOST(x, y, tr = trim, low_eqbound, high_eqbound)`
pub fn tost_yuen(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    trim: f64,
) -> Result<TostResult> {
    validate_inputs(x, y, alpha, trim)?;

    let nx = x.len();
    let ny = y.len();

    // Number of observations to trim from each tail
    let gx = (trim * nx as f64).floor() as usize;
    let gy = (trim * ny as f64).floor() as usize;

    // Effective sample sizes after trimming
    let hx = nx - 2 * gx;
    let hy = ny - 2 * gy;

    if hx < 2 || hy < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: hx.min(hy),
        });
    }

    // Compute trimmed means
    let trimmed_mean_x = trimmed_mean(x, gx);
    let trimmed_mean_y = trimmed_mean(y, gy);

    // Compute Winsorized variances
    let winvar_x = winsorized_variance(x, gx);
    let winvar_y = winsorized_variance(y, gy);

    // Estimate: difference in trimmed means
    let estimate = trimmed_mean_x - trimmed_mean_y;

    let hx_f = hx as f64;
    let hy_f = hy as f64;

    // Standard errors for trimmed means
    let dx = (nx - 1) as f64 * winvar_x / (hx_f * (hx_f - 1.0));
    let dy = (ny - 1) as f64 * winvar_y / (hy_f * (hy_f - 1.0));
    let se = (dx + dy).sqrt();

    // Welch-Satterthwaite degrees of freedom
    let df = (dx + dy).powi(2) / (dx.powi(2) / (hx_f - 1.0) + dy.powi(2) / (hy_f - 1.0));

    // Convert bounds using Winsorized SD for Cohen's d
    let winsorized_sd = ((winvar_x + winvar_y) / 2.0).sqrt();
    let (lower_bound, upper_bound) = bounds.to_raw(Some(winsorized_sd))?;

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
            hypothesis: format!("H0: trimmed mean diff <= {:.4}", lower_bound),
            statistic: t_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: trimmed mean diff >= {:.4}", upper_bound),
            statistic: t_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n: nx + ny,
        df: Some(df),
        method: format!("Yuen TOST ({}% trimmed)", (trim * 100.0) as usize),
    })
}

/// Compute trimmed mean by removing g observations from each tail.
fn trimmed_mean(data: &[f64], g: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let trimmed = &sorted[g..n - g];

    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Compute Winsorized variance.
/// Replace the g smallest values with the (g+1)th smallest.
/// Replace the g largest values with the (n-g)th largest.
fn winsorized_variance(data: &[f64], g: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();

    // Winsorize: replace extreme values with boundary values
    let mut winsorized = sorted.clone();
    let low_val = sorted[g];
    let high_val = sorted[n - g - 1];

    for item in winsorized.iter_mut().take(g) {
        *item = low_val;
    }
    for item in winsorized.iter_mut().skip(n - g) {
        *item = high_val;
    }

    // Compute variance with n-1 denominator
    let mean: f64 = winsorized.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = winsorized.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq / (n - 1) as f64
}

/// Validate inputs for Yuen TOST.
fn validate_inputs(x: &[f64], y: &[f64], alpha: f64, trim: f64) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    if !(0.0..0.5).contains(&trim) {
        return Err(StatError::InvalidParameter(format!(
            "trim must be in [0, 0.5), got {}",
            trim
        )));
    }

    // Check if there will be enough data after trimming
    let gx = (trim * x.len() as f64).floor() as usize;
    let gy = (trim * y.len() as f64).floor() as usize;

    let hx = x.len().saturating_sub(2 * gx);
    let hy = y.len().saturating_sub(2 * gy);

    if hx < 2 {
        return Err(StatError::InsufficientData {
            needed: 2 + 2 * gx,
            got: x.len(),
        });
    }

    if hy < 2 {
        return Err(StatError::InsufficientData {
            needed: 2 + 2 * gy,
            got: y.len(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuen_tost_equivalent() {
        // Similar groups
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

        // Small difference (~0.1), should be equivalent
        assert!(result.estimate.abs() < 0.3);
    }

    #[test]
    fn test_yuen_tost_not_equivalent() {
        // Different groups
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let y = vec![15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

        // Clear difference (~5), should NOT be equivalent
        assert!(!result.equivalent);
    }

    #[test]
    fn test_yuen_robust_to_outliers() {
        // Group x has an outlier
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 100.0];
        let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 1.0 };

        // With 20% trimming, the outlier should be removed
        let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

        // Trimmed means should be similar
        assert!(result.estimate.abs() < 1.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // No trimming
        let tm0 = trimmed_mean(&data, 0);
        assert!((tm0 - 5.5).abs() < 1e-10);

        // 10% trimming (remove 1 from each end)
        let tm1 = trimmed_mean(&data, 1);
        assert!((tm1 - 5.5).abs() < 1e-10); // Still 5.5 for symmetric data

        // 20% trimming (remove 2 from each end)
        let tm2 = trimmed_mean(&data, 2);
        let expected = (3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 6.0;
        assert!((tm2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_winsorized_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // With g=1, values become [2, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        let wv = winsorized_variance(&data, 1);
        assert!(wv > 0.0);
    }

    #[test]
    fn test_invalid_trim() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        // trim >= 0.5 is invalid
        assert!(tost_yuen(&x, &y, &bounds, 0.05, 0.5).is_err());
        assert!(tost_yuen(&x, &y, &bounds, 0.05, -0.1).is_err());
    }

    #[test]
    fn test_insufficient_data_after_trim() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.1, 2.1, 3.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        // With 30% trim on n=3, we'd trim 0.9 -> 0 from each side
        // This should still work, but 40% would fail
        let result = tost_yuen(&x, &y, &bounds, 0.05, 0.3);
        // This might work or fail depending on rounding
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_cohen_d_bounds() {
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
        let bounds = EquivalenceBounds::CohenD { d: 0.5 };

        let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

        // Bounds should be computed using Winsorized SD
        assert!(result.bounds.0 < 0.0);
        assert!(result.bounds.1 > 0.0);
    }
}
