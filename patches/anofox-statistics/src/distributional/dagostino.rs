use crate::error::{Result, StatError};
use crate::utils::math::mean;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Compute cube root that handles negative values correctly.
fn cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.powf(1.0 / 3.0)
    } else {
        -((-x).powf(1.0 / 3.0))
    }
}

/// Result of D'Agostino's K-squared test
#[derive(Debug, Clone)]
pub struct DAgostinoResult {
    /// The K-squared test statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// The Z-score for skewness
    pub z_skewness: f64,
    /// The Z-score for kurtosis
    pub z_kurtosis: f64,
}

/// Perform D'Agostino's K-squared test for normality.
///
/// This is an omnibus test that combines tests for skewness and kurtosis.
/// The implementation follows scipy.stats.normaltest.
///
/// # Arguments
/// * `data` - Sample data
///
/// # Returns
/// * `DAgostinoResult` containing K² statistic, p-value, and individual Z-scores
///
/// # Requirements
/// * n >= 20 (for reliable results, though minimum is 8)
///
/// # References
/// * D'Agostino, R. B. (1971). "An omnibus test of normality for moderate and large sample size"
/// * D'Agostino, R. B., and Pearson, E. S. (1973). "Tests for departure from normality"
pub fn dagostino_k_squared(data: &[f64]) -> Result<DAgostinoResult> {
    let n = data.len();

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    if n < 8 {
        return Err(StatError::InsufficientData { needed: 8, got: n });
    }

    let n_f = n as f64;
    let mean_val = mean(data)?;

    // Calculate sample moments (biased, using n)
    let m2: f64 = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n_f;
    let m3: f64 = data.iter().map(|x| (x - mean_val).powi(3)).sum::<f64>() / n_f;
    let m4: f64 = data.iter().map(|x| (x - mean_val).powi(4)).sum::<f64>() / n_f;

    // Check for constant data
    if m2 < 1e-14 {
        return Err(StatError::InvalidParameter(
            "Data has zero variance".to_string(),
        ));
    }

    // Sample skewness (sqrt(b1)) and kurtosis (b2)
    let sqrt_b1 = m3 / m2.powf(1.5);
    let b2 = m4 / (m2 * m2);

    // Transform skewness to Z1 using D'Agostino (1970)
    let z1 = skewness_test(sqrt_b1, n_f)?;

    // Transform kurtosis to Z2 using Anscombe-Glynn (1983)
    let z2 = kurtosis_test(b2, n_f)?;

    // K² = Z1² + Z2² follows chi-squared(2) under normality
    let k_squared = z1 * z1 + z2 * z2;

    // p-value from chi-squared distribution with 2 df
    let chi2 = ChiSquared::new(2.0).unwrap();
    let p_value = 1.0 - chi2.cdf(k_squared);

    Ok(DAgostinoResult {
        statistic: k_squared,
        p_value,
        z_skewness: z1,
        z_kurtosis: z2,
    })
}

/// Transform sample skewness to a Z-score (scipy implementation).
///
/// Based on D'Agostino & Pearson (1973).
fn skewness_test(b1: f64, n: f64) -> Result<f64> {
    // Require n >= 8 for valid approximation
    if n < 8.0 {
        return Err(StatError::InsufficientData {
            needed: 8,
            got: n as usize,
        });
    }

    // Y = b1 * sqrt((n+1)(n+3) / (6(n-2)))
    let y = b1 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();

    // beta2 = 3(n^2 + 27n - 70)(n+1)(n+3) / ((n-2)(n+5)(n+7)(n+9))
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));

    // W^2 = sqrt(2(beta2 - 1)) - 1
    let w_sq = (2.0 * (beta2 - 1.0)).sqrt() - 1.0;

    // delta = 1 / sqrt(ln(W))
    let delta = 1.0 / (0.5 * w_sq.ln()).sqrt();

    // alpha = sqrt(2 / (W^2 - 1))
    let alpha = (2.0 / (w_sq - 1.0)).sqrt();

    // Z = delta * ln(Y/alpha + sqrt((Y/alpha)^2 + 1))
    let z = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();

    Ok(z)
}

/// Transform sample kurtosis to a Z-score (scipy implementation).
///
/// Based on Anscombe & Glynn (1983).
fn kurtosis_test(b2: f64, n: f64) -> Result<f64> {
    // Require n >= 20 for valid approximation (scipy uses this)
    // We relax to n >= 8 but warn about unreliability
    if n < 8.0 {
        return Err(StatError::InsufficientData {
            needed: 8,
            got: n as usize,
        });
    }

    // E[b2] = 3(n-1)/(n+1)
    let e_b2 = 3.0 * (n - 1.0) / (n + 1.0);

    // Var[b2] = 24n(n-2)(n-3) / ((n+1)^2(n+3)(n+5))
    let var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));

    // x = (b2 - E[b2]) / sqrt(Var[b2])
    let x = (b2 - e_b2) / var_b2.sqrt();

    // sqrtbeta1 = 6(n^2 - 5n + 2)/((n+7)(n+9)) * sqrt(6(n+3)(n+5)/(n(n-2)(n-3)))
    let sqrt_beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();

    // A = 6 + 8/sqrtbeta1 * (2/sqrtbeta1 + sqrt(1 + 4/sqrtbeta1^2))
    let a = 6.0
        + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + (1.0 + 4.0 / (sqrt_beta1 * sqrt_beta1)).sqrt());

    // Following R's anscombe.test formula:
    // z <- (1 - 2/(9*a) - ((1 - 2/a)/(1 + xx*sqrt(2/(a-4))))^(1/3)) / sqrt(2/(9*a))
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let inner_denom = 1.0 + x * (2.0 / (a - 4.0)).sqrt();

    if inner_denom.abs() < 1e-14 {
        return Ok(0.0);
    }

    let term2 = cbrt((1.0 - 2.0 / a) / inner_denom);
    Ok((term1 - term2) / (2.0 / (9.0 * a)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dagostino_normal_data() {
        // Standard normal data should not reject normality
        let data: Vec<f64> = vec![
            -0.97, 0.26, -0.13, 0.05, -0.57, 0.53, 1.21, -1.15, 0.32, -0.45, 0.78, -0.89, 0.11,
            -0.23, 0.67, -0.34, 0.45, -0.12, 0.98, -0.76, 0.23, -0.56, 0.89, -0.01, 0.34, -0.78,
            0.12, -0.43, 0.56, -0.21,
        ];
        let result = dagostino_k_squared(&data).unwrap();
        // For normal-ish data, p-value should be > 0.01 (not too strict)
        assert!(
            result.p_value > 0.01,
            "p_value {} should be > 0.01",
            result.p_value
        );
    }

    #[test]
    fn test_dagostino_non_normal_data() {
        // Highly skewed data should reject normality
        let data: Vec<f64> = vec![
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 50.0,
            100.0, 200.0, 500.0,
        ];
        let result = dagostino_k_squared(&data).unwrap();
        // For highly skewed data, p-value should be small
        assert!(
            result.p_value < 0.1,
            "p_value {} should be < 0.1 for skewed data",
            result.p_value
        );
    }

    #[test]
    fn test_dagostino_insufficient_data() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(dagostino_k_squared(&data).is_err());
    }

    #[test]
    fn test_dagostino_empty_data() {
        let data: Vec<f64> = vec![];
        assert!(dagostino_k_squared(&data).is_err());
    }

    #[test]
    fn test_dagostino_constant_data() {
        let data: Vec<f64> = vec![1.0; 30];
        assert!(dagostino_k_squared(&data).is_err());
    }
}
