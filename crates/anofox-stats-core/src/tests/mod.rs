//! Statistical hypothesis testing
//!
//! This module provides wrappers around the anofox-tests crate (anofox-statistics)
//! for statistical hypothesis testing in DuckDB.

pub mod categorical;
pub mod correlation;
pub mod distributional;
pub mod equivalence;
pub mod forecast;
pub mod modern;
pub mod nonparametric;
pub mod parametric;
pub mod resampling;

// Re-export common types from anofox-tests
pub use anofox_tests::Alternative;

use crate::StatsError;

/// Generic test result structure for all statistical tests
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic (t, U, chi2, F, etc.)
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom (f64::NAN if not applicable)
    pub df: f64,
    /// Effect size (Cohen's d, r, etc.) (f64::NAN if not applicable)
    pub effect_size: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Total sample size
    pub n: usize,
    /// Group 1 sample size (for two-sample tests)
    pub n1: usize,
    /// Group 2 sample size (for two-sample tests)
    pub n2: usize,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Test method/name
    pub method: String,
}

impl Default for TestResult {
    fn default() -> Self {
        Self {
            statistic: f64::NAN,
            p_value: f64::NAN,
            df: f64::NAN,
            effect_size: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            confidence_level: 0.95,
            n: 0,
            n1: 0,
            n2: 0,
            alternative: Alternative::TwoSided,
            method: String::new(),
        }
    }
}

/// Extended test result for ANOVA
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F statistic
    pub f_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Between-groups degrees of freedom
    pub df_between: usize,
    /// Within-groups degrees of freedom
    pub df_within: usize,
    /// Between-groups sum of squares
    pub ss_between: f64,
    /// Within-groups sum of squares
    pub ss_within: f64,
    /// Number of groups
    pub n_groups: usize,
    /// Total sample size
    pub n: usize,
    /// Test method
    pub method: String,
}

/// Correlation test result
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Correlation coefficient
    pub r: f64,
    /// Test statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Sample size
    pub n: usize,
    /// Method name
    pub method: String,
}

/// Chi-square test result
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Method name
    pub method: String,
}

/// Convert anofox_tests StatError to our StatsError
fn convert_error(e: anofox_tests::StatError) -> StatsError {
    StatsError::InvalidInput(e.to_string())
}

/// Filter NaN values from a slice
fn filter_nan(data: &[f64]) -> Vec<f64> {
    data.iter().copied().filter(|x| !x.is_nan()).collect()
}
