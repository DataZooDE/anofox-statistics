//! Equivalence testing using TOST (Two One-Sided Tests).
//!
//! This module provides functions for testing equivalence hypotheses, where the goal
//! is to demonstrate that an effect is small enough to be practically equivalent to zero
//! (or some other value), rather than simply testing if it differs from zero.
//!
//! # Overview
//!
//! Traditional null hypothesis significance testing (NHST) tests whether an effect
//! differs from zero. However, failing to reject the null hypothesis does not prove
//! equivalence - it may simply reflect insufficient statistical power.
//!
//! TOST addresses this by testing two one-sided hypotheses:
//! - H01: effect ≤ -Δ (effect is too negative to be equivalent)
//! - H02: effect ≥ +Δ (effect is too positive to be equivalent)
//!
//! If both null hypotheses are rejected, we conclude the effect is practically equivalent
//! to zero (i.e., falls within the equivalence bounds [-Δ, +Δ]).
//!
//! # Available Tests
//!
//! - **t-test TOST**: For comparing means (one-sample, two-sample, paired)
//! - **Correlation TOST**: For testing if a correlation is practically zero
//! - **Proportion TOST**: For comparing proportions
//! - **Wilcoxon TOST**: Non-parametric alternative using ranks
//! - **Bootstrap TOST**: Resampling-based approach
//! - **Yuen TOST**: Robust test using trimmed means
//!
//! # R Equivalent
//!
//! The TOSTER package in R provides similar functionality.

mod bootstrap_tost;
mod bounds;
mod correlation_tost;
mod proportion_tost;
mod ttest_tost;
mod wilcoxon_tost;
mod yuen_tost;

pub use bootstrap_tost::tost_bootstrap;
pub use bounds::EquivalenceBounds;
pub use correlation_tost::{tost_correlation, CorrelationTostMethod};
pub use proportion_tost::{tost_prop_one, tost_prop_two};
pub use ttest_tost::{tost_t_test_one_sample, tost_t_test_paired, tost_t_test_two_sample};
pub use wilcoxon_tost::{tost_wilcoxon_paired, tost_wilcoxon_two_sample};
pub use yuen_tost::tost_yuen;

/// Result of a one-sided test within TOST.
#[derive(Debug, Clone)]
pub struct OneSidedTestResult {
    /// Null hypothesis description (e.g., "H0: effect <= -0.5")
    pub hypothesis: String,
    /// Test statistic
    pub statistic: f64,
    /// p-value for this one-sided test
    pub p_value: f64,
    /// Whether this null hypothesis was rejected at the given alpha
    pub rejected: bool,
}

/// Result of a TOST (Two One-Sided Tests) equivalence test.
#[derive(Debug, Clone)]
pub struct TostResult {
    /// Point estimate of the effect (mean difference, correlation, etc.)
    pub estimate: f64,
    /// Confidence interval at (1 - 2α) level
    pub ci: (f64, f64),
    /// Equivalence bounds used (lower, upper)
    pub bounds: (f64, f64),
    /// Result of the lower bound test (testing H0: effect ≤ lower_bound)
    pub lower_test: OneSidedTestResult,
    /// Result of the upper bound test (testing H0: effect ≥ upper_bound)
    pub upper_test: OneSidedTestResult,
    /// TOST p-value: max(p_lower, p_upper)
    pub tost_p_value: f64,
    /// Whether equivalence was established (CI within bounds)
    pub equivalent: bool,
    /// Significance level used
    pub alpha: f64,
    /// Sample size (or total n for two-sample tests)
    pub n: usize,
    /// Degrees of freedom (if applicable)
    pub df: Option<f64>,
    /// Name of the test method
    pub method: String,
}

impl TostResult {
    /// Check if the effect is statistically equivalent at the given alpha level.
    pub fn is_equivalent(&self) -> bool {
        self.equivalent
    }

    /// Get a text summary of the result.
    pub fn summary(&self) -> String {
        let conclusion = if self.equivalent {
            "Equivalence established"
        } else {
            "Equivalence not established"
        };

        format!(
            "{}: estimate = {:.4}, 90% CI [{:.4}, {:.4}], bounds [{:.4}, {:.4}], TOST p = {:.4}",
            conclusion,
            self.estimate,
            self.ci.0,
            self.ci.1,
            self.bounds.0,
            self.bounds.1,
            self.tost_p_value
        )
    }
}
