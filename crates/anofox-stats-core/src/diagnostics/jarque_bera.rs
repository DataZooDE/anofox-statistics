//! Jarque-Bera test for normality
//!
//! The Jarque-Bera test is used to check if sample data has skewness and kurtosis
//! matching a normal distribution.

use crate::errors::{StatsError, StatsResult};
use crate::tests::convert_error;

pub use anofox_tests::JarqueBeraResult;

/// Jarque-Bera test for normality (delegates to `anofox_statistics::jarque_bera`,
/// R `tseries::jarque.bera.test`). NaN values are dropped first.
///
/// # Arguments
/// * `data` - Sample data (typically residuals)
///
/// # Returns
/// JarqueBeraResult with test statistic, p-value, skewness, and excess kurtosis
pub fn jarque_bera(data: &[f64]) -> StatsResult<JarqueBeraResult> {
    let clean_data: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();

    if clean_data.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Jarque-Bera test requires at least 3 observations".into(),
        ));
    }

    anofox_tests::jarque_bera(&clean_data).map_err(convert_error)
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
