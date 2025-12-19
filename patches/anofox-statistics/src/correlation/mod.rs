//! Correlation functions and significance tests.
//!
//! This module provides various correlation measures:
//! - Pearson's product-moment correlation (parametric)
//! - Spearman's rank correlation (nonparametric)
//! - Kendall's tau correlation (nonparametric)
//! - Partial correlation (controlling for covariates)
//! - Distance correlation (detecting non-linear dependence)
//! - Intraclass correlation (ICC) for reliability

mod distance;
mod icc;
mod kendall;
mod partial;
mod pearson;
mod spearman;

pub use distance::{distance_cor, distance_cor_test, DistanceCorResult};
pub use icc::{icc, ICCResult, ICCType};
pub use kendall::{kendall, KendallVariant};
pub use partial::{partial_cor, semi_partial_cor, PartialCorResult};
pub use pearson::pearson;
pub use spearman::spearman;

use crate::error::{Result, StatError};

/// Correlation method used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}

impl std::fmt::Display for CorrelationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrelationMethod::Pearson => write!(f, "Pearson"),
            CorrelationMethod::Spearman => write!(f, "Spearman"),
            CorrelationMethod::Kendall => write!(f, "Kendall"),
        }
    }
}

/// Confidence interval for correlation coefficient
#[derive(Debug, Clone)]
pub struct CorrelationConfInt {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub conf_level: f64,
}

/// Result of a correlation test
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Correlation coefficient estimate
    pub estimate: f64,
    /// Test statistic (t for Pearson/Spearman, z for Kendall)
    pub statistic: f64,
    /// Degrees of freedom (n-2 for Pearson/Spearman, None for Kendall)
    pub df: Option<f64>,
    /// p-value for the test
    pub p_value: f64,
    /// Confidence interval (if requested)
    pub conf_int: Option<CorrelationConfInt>,
    /// Correlation method used
    pub method: CorrelationMethod,
    /// Sample size
    pub n: usize,
}

/// Validate input data for correlation tests
pub(crate) fn validate_correlation_input(x: &[f64], y: &[f64]) -> Result<usize> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "x and y must have the same length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }

    // Check for NaN or infinite values
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatError::InvalidParameter(format!(
                "non-finite value at index {}: x={}, y={}",
                i, xi, yi
            )));
        }
    }

    Ok(n)
}

/// Compute mean of a slice
pub(crate) fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute variance (sample variance with n-1 denominator)
pub(crate) fn variance(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Compute standard deviation
pub(crate) fn std_dev(data: &[f64], mean: f64) -> f64 {
    variance(data, mean).sqrt()
}
