//! Pearson product-moment correlation coefficient.

use crate::correlation::{
    mean, std_dev, validate_correlation_input, CorrelationConfInt, CorrelationMethod,
    CorrelationResult,
};
use crate::error::Result;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Compute Pearson's product-moment correlation coefficient with significance test.
///
/// Tests the null hypothesis that the true correlation is zero using a t-test.
/// Confidence intervals are computed using Fisher's z-transformation.
///
/// # Arguments
/// * `x` - First variable (must have at least 3 observations)
/// * `y` - Second variable (same length as x)
/// * `conf_level` - Optional confidence level for CI (e.g., 0.95 for 95%)
///
/// # Returns
/// * `CorrelationResult` containing the correlation coefficient, t-statistic,
///   degrees of freedom, p-value, and optional confidence interval
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::pearson;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
///
/// let result = pearson(&x, &y, Some(0.95)).unwrap();
/// println!("Pearson r = {:.4}", result.estimate);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `cor.test(x, y, method = "pearson")`
pub fn pearson(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<CorrelationResult> {
    let n = validate_correlation_input(x, y)?;

    // Compute means
    let mean_x = mean(x);
    let mean_y = mean(y);

    // Compute standard deviations
    let sd_x = std_dev(x, mean_x);
    let sd_y = std_dev(y, mean_y);

    // Compute Pearson correlation coefficient
    let mut sum_xy = 0.0;
    for i in 0..n {
        sum_xy += (x[i] - mean_x) * (y[i] - mean_y);
    }
    let r = sum_xy / ((n - 1) as f64 * sd_x * sd_y);

    // Clamp r to [-1, 1] to handle numerical precision issues
    let r = r.clamp(-1.0, 1.0);

    // Compute t-statistic: t = r * sqrt((n-2) / (1-r²))
    let df = (n - 2) as f64;
    let t_stat = if (1.0 - r * r).abs() < 1e-15 {
        // Perfect correlation: t approaches infinity
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };

    // Compute two-sided p-value using t-distribution
    let p_value = if t_stat.is_infinite() {
        0.0
    } else {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
    };

    // Compute confidence interval using Fisher z-transformation
    let conf_int = conf_level.map(|level| fisher_z_confidence_interval(r, n, level));

    Ok(CorrelationResult {
        estimate: r,
        statistic: t_stat,
        df: Some(df),
        p_value,
        conf_int,
        method: CorrelationMethod::Pearson,
        n,
    })
}

/// Compute confidence interval for correlation using Fisher's z-transformation.
///
/// Fisher's z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
/// SE(z) = 1 / sqrt(n-3)
/// CI for z: z ± z_alpha/2 * SE(z)
/// Transform back: r = tanh(z)
fn fisher_z_confidence_interval(r: f64, n: usize, conf_level: f64) -> CorrelationConfInt {
    // Handle perfect correlations
    if r.abs() >= 1.0 - 1e-10 {
        return CorrelationConfInt {
            lower: r.signum(),
            upper: r.signum(),
            conf_level,
        };
    }

    // Fisher z-transformation: z = arctanh(r)
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();

    // Standard error of z
    let se_z = 1.0 / ((n - 3) as f64).sqrt();

    // Get z critical value
    let alpha = 1.0 - conf_level;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);

    // Confidence interval in z-space
    let z_lower = z - z_crit * se_z;
    let z_upper = z + z_crit * se_z;

    // Transform back to r-space: r = tanh(z)
    let r_lower = z_lower.tanh().clamp(-1.0, 1.0);
    let r_upper = z_upper.tanh().clamp(-1.0, 1.0);

    CorrelationConfInt {
        lower: r_lower,
        upper: r_upper,
        conf_level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = pearson(&x, &y, None).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert_eq!(result.method, CorrelationMethod::Pearson);
        assert_eq!(result.n, 5);
    }

    #[test]
    fn test_pearson_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];

        let result = pearson(&x, &y, None).unwrap();

        assert!((result.estimate - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_with_ci() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 2.8, 3.2, 4.1, 5.5, 5.8, 7.2, 8.1, 8.9, 10.2];

        let result = pearson(&x, &y, Some(0.95)).unwrap();

        assert!(result.conf_int.is_some());
        let ci = result.conf_int.unwrap();
        assert!(ci.lower < result.estimate);
        assert!(ci.upper > result.estimate);
        assert!((ci.conf_level - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_mismatched_length() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = pearson(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pearson_too_few_observations() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];

        let result = pearson(&x, &y, None);
        assert!(result.is_err());
    }
}
