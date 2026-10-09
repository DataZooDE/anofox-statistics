//! Distributional tests
//!
//! - Shapiro-Wilk test (normality)
//! - D'Agostino K-squared test (normality)

use super::{convert_error, filter_nan, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    dagostino_k_squared as lib_dagostino_k_squared, shapiro_wilk as lib_shapiro_wilk, Alternative,
};

/// Shapiro-Wilk test for normality
///
/// Tests whether a sample comes from a normal distribution.
/// Valid for sample sizes between 3 and 5000.
pub fn shapiro_wilk(data: &[f64]) -> StatsResult<TestResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Shapiro-Wilk test requires at least 3 observations".into(),
        ));
    }
    if filtered.len() > 5000 {
        return Err(StatsError::InvalidInput(
            "Shapiro-Wilk test is limited to n <= 5000".into(),
        ));
    }

    let result = lib_shapiro_wilk(&filtered).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: f64::NAN,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: filtered.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "Shapiro-Wilk test".into(),
    })
}

/// D'Agostino-Pearson K-squared test for normality
///
/// Omnibus test combining skewness and kurtosis tests.
pub fn dagostino_k_squared(data: &[f64]) -> StatsResult<TestResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 8 {
        return Err(StatsError::InsufficientDataMsg(
            "D'Agostino K-squared test requires at least 8 observations".into(),
        ));
    }

    let result = lib_dagostino_k_squared(&filtered).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: 2.0, // Chi-squared with 2 df
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: filtered.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "D'Agostino K-squared test".into(),
    })
}

/// Extended result for D'Agostino test including skewness and kurtosis
#[derive(Debug, Clone)]
pub struct DAgostinoResult {
    /// K-squared statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Skewness z-score
    pub z_skewness: f64,
    /// Kurtosis z-score
    pub z_kurtosis: f64,
    /// Sample size
    pub n: usize,
}

/// D'Agostino K-squared test with detailed results
pub fn dagostino_k_squared_detailed(data: &[f64]) -> StatsResult<DAgostinoResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 8 {
        return Err(StatsError::InsufficientDataMsg(
            "D'Agostino K-squared test requires at least 8 observations".into(),
        ));
    }

    let result = lib_dagostino_k_squared(&filtered).map_err(convert_error)?;

    Ok(DAgostinoResult {
        statistic: result.statistic,
        p_value: result.p_value,
        z_skewness: result.z_skewness,
        z_kurtosis: result.z_kurtosis,
        n: filtered.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapiro_wilk_normal() {
        let data = vec![
            -0.5, 0.1, -0.3, 0.8, 0.2, -0.1, 0.4, -0.2, 0.3, 0.0, -0.4, 0.5, 0.1, -0.6, 0.2, -0.1,
            0.3, -0.3, 0.4, 0.0,
        ];
        let result = shapiro_wilk(&data).unwrap();

        assert!(result.statistic > 0.9);
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_dagostino_k_squared() {
        let data = vec![
            -0.5, 0.1, -0.3, 0.8, 0.2, -0.1, 0.4, -0.2, 0.3, 0.0, -0.4, 0.5, 0.1, -0.6, 0.2, -0.1,
            0.3, -0.3, 0.4, 0.0,
        ];
        let result = dagostino_k_squared(&data).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    /// R shapiro.test(); small-sample branch (n <= 11) and n >= 12.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_shapiro_wilk_matches_r() {
        let r = shapiro_wilk(&[4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1]).unwrap();
        assert!((r.statistic - 0.88790406283629908).abs() < 1e-10);
        assert!((r.p_value - 0.16058545199963783).abs() < 1e-10);
        let r = shapiro_wilk(&[1.2, 3.4, 2.2, 5.9]).unwrap();
        assert!((r.p_value - 0.74255791273476768).abs() < 1e-10);
        let r = shapiro_wilk(&[1.0, 2.0, 4.0]).unwrap();
        assert!((r.p_value - 0.6368868450289632).abs() < 1e-10);
    }
}
