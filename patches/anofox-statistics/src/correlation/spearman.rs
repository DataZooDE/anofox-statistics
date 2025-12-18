//! Spearman's rank correlation coefficient.

use crate::correlation::{
    validate_correlation_input, CorrelationConfInt, CorrelationMethod, CorrelationResult,
};
use crate::error::Result;
use crate::nonparametric::rank;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Compute Spearman's rank correlation coefficient with significance test.
///
/// Spearman's rho is the Pearson correlation of the ranks. It measures
/// monotonic relationships and is robust to outliers.
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
/// use anofox_statistics::correlation::spearman;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0, 6.0, 7.0, 8.0, 7.0];
///
/// let result = spearman(&x, &y, Some(0.95)).unwrap();
/// println!("Spearman rho = {:.4}", result.estimate);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `cor.test(x, y, method = "spearman")`
pub fn spearman(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<CorrelationResult> {
    let n = validate_correlation_input(x, y)?;

    // Compute ranks
    let ranks_x = rank(x)?;
    let ranks_y = rank(y)?;

    // Compute Pearson correlation of ranks
    let mean_rx: f64 = ranks_x.iter().sum::<f64>() / n as f64;
    let mean_ry: f64 = ranks_y.iter().sum::<f64>() / n as f64;

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;

    for i in 0..n {
        let dx = ranks_x[i] - mean_rx;
        let dy = ranks_y[i] - mean_ry;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let rho = if sum_xx == 0.0 || sum_yy == 0.0 {
        // All values in one variable are the same (all tied)
        0.0
    } else {
        sum_xy / (sum_xx * sum_yy).sqrt()
    };

    // Clamp rho to [-1, 1] to handle numerical precision issues
    let rho = rho.clamp(-1.0, 1.0);

    // Compute t-statistic (same as Pearson)
    let df = (n - 2) as f64;
    let t_stat = if (1.0 - rho * rho).abs() < 1e-15 {
        if rho > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        rho * (df / (1.0 - rho * rho)).sqrt()
    };

    // Compute two-sided p-value using t-distribution
    // R uses t-distribution approximation for Spearman as well
    let p_value = if t_stat.is_infinite() {
        0.0
    } else {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
    };

    // Compute confidence interval using Fisher z-transformation
    let conf_int = conf_level.map(|level| fisher_z_confidence_interval(rho, n, level));

    Ok(CorrelationResult {
        estimate: rho,
        statistic: t_stat,
        df: Some(df),
        p_value,
        conf_int,
        method: CorrelationMethod::Spearman,
        n,
    })
}

/// Compute confidence interval for correlation using Fisher's z-transformation.
fn fisher_z_confidence_interval(r: f64, n: usize, conf_level: f64) -> CorrelationConfInt {
    // Handle perfect correlations
    if r.abs() >= 1.0 - 1e-10 {
        return CorrelationConfInt {
            lower: r.signum(),
            upper: r.signum(),
            conf_level,
        };
    }

    // Fisher z-transformation
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();

    // Standard error of z (same formula as Pearson)
    let se_z = 1.0 / ((n - 3) as f64).sqrt();

    // Get z critical value
    let alpha = 1.0 - conf_level;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);

    // Confidence interval in z-space
    let z_lower = z - z_crit * se_z;
    let z_upper = z + z_crit * se_z;

    // Transform back to r-space
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
    fn test_spearman_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = spearman(&x, &y, None).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert_eq!(result.method, CorrelationMethod::Spearman);
        assert_eq!(result.n, 5);
    }

    #[test]
    fn test_spearman_monotonic_nonlinear() {
        // Perfect monotonic but non-linear relationship
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2

        let result = spearman(&x, &y, None).unwrap();

        // Spearman should be 1 for perfect monotonic relationship
        assert!((result.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 4.0, 5.0];
        let y = vec![1.0, 3.0, 3.0, 4.0, 5.0];

        let result = spearman(&x, &y, None).unwrap();

        // Should still work with ties using average ranks
        assert!(result.estimate > 0.9);
    }

    #[test]
    fn test_spearman_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result = spearman(&x, &y, None).unwrap();

        assert!((result.estimate - (-1.0)).abs() < 1e-10);
    }
}
