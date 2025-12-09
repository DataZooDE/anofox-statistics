//! Jarque-Bera test for normality
//!
//! The Jarque-Bera test is used to check if sample data has skewness and kurtosis
//! matching a normal distribution.

use crate::errors::{StatsError, StatsResult};

/// Result of Jarque-Bera test
#[derive(Debug, Clone)]
pub struct JarqueBeraResult {
    /// JB test statistic
    pub statistic: f64,
    /// p-value for the test
    pub p_value: f64,
    /// Sample skewness
    pub skewness: f64,
    /// Sample kurtosis (excess)
    pub kurtosis: f64,
    /// Number of observations
    pub n: usize,
}

/// Compute the Jarque-Bera test statistic for normality
///
/// # Arguments
/// * `data` - Sample data (typically residuals)
///
/// # Returns
/// JarqueBeraResult with test statistic, p-value, skewness, and kurtosis
pub fn jarque_bera(data: &[f64]) -> StatsResult<JarqueBeraResult> {
    // Filter NaN values
    let clean_data: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = clean_data.len();

    if n < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Jarque-Bera test requires at least 3 observations".into(),
        ));
    }

    // Compute mean
    let mean: f64 = clean_data.iter().sum::<f64>() / n as f64;

    // Compute central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &x in &clean_data {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    m2 /= n as f64;
    m3 /= n as f64;
    m4 /= n as f64;

    // Check for zero variance
    if m2 <= 0.0 {
        return Err(StatsError::InvalidInput("Data has zero variance".into()));
    }

    // Compute skewness and kurtosis
    let std_dev = m2.sqrt();
    let skewness = m3 / (std_dev * std_dev * std_dev);
    let kurtosis = m4 / (m2 * m2) - 3.0; // Excess kurtosis

    // Jarque-Bera statistic: JB = n/6 * (S^2 + K^2/4)
    // where S is skewness and K is excess kurtosis
    let jb_stat = (n as f64 / 6.0) * (skewness * skewness + kurtosis * kurtosis / 4.0);

    // p-value from chi-squared distribution with 2 degrees of freedom
    // Using approximation: P(X > x) ≈ exp(-x/2) for chi-squared(2)
    let p_value = (-jb_stat / 2.0).exp();

    Ok(JarqueBeraResult {
        statistic: jb_stat,
        p_value,
        skewness,
        kurtosis,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jarque_bera_normal() {
        // Data that's roughly normal should have low JB statistic
        let data: Vec<f64> = vec![
            -1.0, -0.5, 0.0, 0.5, 1.0, -0.8, -0.3, 0.2, 0.7, 1.2, -1.2, -0.7, -0.2, 0.3, 0.8, -0.9,
            -0.4, 0.1, 0.6, 1.1,
        ];

        let result = jarque_bera(&data).unwrap();
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        println!(
            "Normal-ish data: JB={:.4}, p={:.4}, skew={:.4}, kurt={:.4}",
            result.statistic, result.p_value, result.skewness, result.kurtosis
        );
    }

    #[test]
    fn test_jarque_bera_skewed() {
        // Heavily right-skewed data
        let data: Vec<f64> = vec![
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0,
            50.0,
        ];

        let result = jarque_bera(&data).unwrap();
        assert!(result.skewness > 1.0); // Should be positively skewed
        println!(
            "Skewed data: JB={:.4}, p={:.4}, skew={:.4}, kurt={:.4}",
            result.statistic, result.p_value, result.skewness, result.kurtosis
        );
    }

    #[test]
    fn test_jarque_bera_insufficient_data() {
        let data = vec![1.0, 2.0];
        assert!(jarque_bera(&data).is_err());
    }

    #[test]
    fn test_jarque_bera_with_nan() {
        let data = vec![1.0, f64::NAN, 2.0, 3.0, f64::NAN, 4.0, 5.0];
        let result = jarque_bera(&data).unwrap();
        assert_eq!(result.n, 5);
    }
}
