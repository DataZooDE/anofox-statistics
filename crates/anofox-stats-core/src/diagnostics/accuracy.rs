//! Predictive accuracy indexes (issue #308): concordance, `Hmisc::somers2`,
//! `Hmisc::rcorr.cens`, Gini's mean difference, `Hmisc::R2Measures`, the
//! `rms::lrm` stats vector and `rms::val.prob`.
//!
//! Thin delegation to `anofox_regression::diagnostics::accuracy`; this module
//! only maps errors (too few rows / no usable pairs are degenerate, bad
//! lengths, labels and weights are input errors).

use crate::errors::{StatsError, StatsResult};
use anofox_regression::diagnostics::accuracy::{self as acc, AccuracyError};

pub use anofox_regression::diagnostics::accuracy::{
    CensoredConcordance, Concordance, LrmStats, R2Measures, Somers2, ValProb,
};

fn convert(e: AccuracyError) -> StatsError {
    match e {
        AccuracyError::Length(a, b) => StatsError::DimensionMismatch {
            y_len: a,
            x_rows: b,
        },
        AccuracyError::TooFew(_) | AccuracyError::NoPairs => {
            StatsError::InsufficientDataMsg(e.to_string())
        }
        AccuracyError::NotBinary | AccuracyError::InvalidWeights => {
            StatsError::InvalidInput(e.to_string())
        }
        // `AccuracyError` is `#[non_exhaustive]`.
        other => StatsError::RegressError(other.to_string()),
    }
}

/// Pair counts of the prediction `x` against the ordered response `y`.
pub fn concordance(x: &[f64], y: &[f64], weights: Option<&[f64]>) -> StatsResult<Concordance> {
    acc::concordance(x, y, weights).map_err(convert)
}

/// ROC area and Somers' `Dxy` of `p` for a 0/1 response (`Hmisc::somers2`).
pub fn somers2(p: &[f64], y: &[f64], weights: Option<&[f64]>) -> StatsResult<Somers2> {
    acc::somers2(p, y, weights).map_err(convert)
}

/// Harrell's `C` for right-censored data (`Hmisc::rcorr.cens`).
pub fn rcorr_cens(x: &[f64], time: &[f64], event: &[bool]) -> StatsResult<CensoredConcordance> {
    acc::rcorr_cens(x, time, event).map_err(convert)
}

/// Gini's mean difference (`Hmisc::GiniMd`); NaN for fewer than 2 values.
pub fn gini_md(x: &[f64]) -> f64 {
    acc::gini_md(x)
}

/// `Hmisc::R2Measures(lr, p, n, ess)`.
pub fn r2_measures(lr: f64, p: f64, n: f64, ess: Option<f64>) -> R2Measures {
    acc::r2_measures(lr, p, n, ess)
}

/// Effective sample size `n (1 - sum P_k^3)` of category frequencies `freqs`.
pub fn effective_sample_size(n: f64, freqs: &[f64]) -> f64 {
    acc::effective_sample_size(n, freqs)
}

/// `rms::lrm` stats of a binary model from its response and linear predictor.
pub fn lrm_stats(y: &[f64], lp: &[f64], weights: Option<&[f64]>, df: f64) -> StatsResult<LrmStats> {
    acc::lrm_stats(y, lp, weights, df).map_err(convert)
}

/// `rms::val.prob(p, y, pl = FALSE)`.
pub fn val_prob(p: &[f64], y: &[f64]) -> StatsResult<ValProb> {
    acc::val_prob(p, y).map_err(convert)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn somers2_and_errors() {
        let s = somers2(&[0.1, 0.4, 0.35, 0.8], &[0.0, 0.0, 1.0, 1.0], None).unwrap();
        assert!((s.c - 0.75).abs() < 1e-15);
        assert!((s.dxy - 0.5).abs() < 1e-15);
        assert!(matches!(
            somers2(&[0.1, 0.2], &[0.0, 2.0], None),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            rcorr_cens(&[1.0, 2.0], &[1.0, 2.0], &[false, false]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            concordance(&[1.0], &[1.0, 2.0], None),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }
}
