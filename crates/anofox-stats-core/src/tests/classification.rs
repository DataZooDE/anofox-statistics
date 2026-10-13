//! Evaluation of binary classifiers (issue #153): ROC, precision-recall and
//! calibration curves, ROC AUC, Brier score and log loss.
//!
//! Thin delegation to `anofox_tests::classification`, which follows
//! scikit-learn (`roc_curve`, `precision_recall_curve`,
//! `calibration_curve(strategy = "uniform")`). This module only maps errors.

use super::convert_error;
use crate::StatsResult;

pub use anofox_tests::{CalibrationBin, CalibrationCurve, PrCurve, RocCurve};

/// ROC curve; `drop_intermediate` as sklearn (true = its default).
pub fn roc_curve(
    scores: &[f64],
    labels: &[bool],
    drop_intermediate: bool,
) -> StatsResult<RocCurve> {
    anofox_tests::roc_curve(scores, labels, drop_intermediate).map_err(convert_error)
}

/// Precision-recall curve (increasing thresholds, then the `(recall 0, precision 1)` point).
pub fn pr_curve(scores: &[f64], labels: &[bool]) -> StatsResult<PrCurve> {
    anofox_tests::pr_curve(scores, labels).map_err(convert_error)
}

/// Calibration curve over `n_bins` equal-width bins with Wilson intervals.
pub fn calibration_curve(
    probs: &[f64],
    labels: &[bool],
    n_bins: usize,
    conf_level: f64,
) -> StatsResult<CalibrationCurve> {
    anofox_tests::calibration_curve_with_conf_level(probs, labels, n_bins, conf_level)
        .map_err(convert_error)
}

/// Area under the ROC curve (ties count one half).
pub fn auc(scores: &[f64], labels: &[bool]) -> StatsResult<f64> {
    anofox_tests::auc(scores, labels).map_err(convert_error)
}

/// Brier score (mean squared error of the predicted probability).
pub fn brier_score(probs: &[f64], labels: &[bool]) -> StatsResult<f64> {
    anofox_tests::brier_score(probs, labels).map_err(convert_error)
}

/// Mean binary cross-entropy, probabilities clipped to `[eps, 1 - eps]`.
pub fn log_loss(probs: &[f64], labels: &[bool]) -> StatsResult<f64> {
    anofox_tests::log_loss(probs, labels).map_err(convert_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StatsError;

    #[test]
    fn delegates_curves() {
        let s = [0.1, 0.4, 0.35, 0.8];
        let y = [false, false, true, true];
        let roc = roc_curve(&s, &y, false).unwrap();
        assert_eq!(roc.fpr, vec![0.0, 0.0, 0.5, 0.5, 1.0]);
        assert_eq!(roc.tpr, vec![0.0, 0.5, 0.5, 1.0, 1.0]);
        assert!((auc(&s, &y).unwrap() - 0.75).abs() < 1e-15);
        let pr = pr_curve(&s, &y).unwrap();
        assert_eq!(pr.precision.len(), pr.thresholds.len() + 1);
        let cal = calibration_curve(&s, &y, 2, 0.95).unwrap();
        assert_eq!(cal.bins.len(), 2);
    }

    #[test]
    fn maps_errors() {
        // One class only: AUC is undefined -> degenerate (NULL in SQL).
        assert!(matches!(
            auc(&[0.1, 0.2], &[true, true]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            roc_curve(&[], &[], true),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        // NaN score: degenerate, like the other aggregates.
        assert!(matches!(
            roc_curve(&[f64::NAN], &[true], true),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        // Probability outside [0, 1] and n_bins = 0 are argument errors.
        assert!(matches!(
            brier_score(&[1.5], &[true]),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            calibration_curve(&[0.5], &[true], 0, 0.95),
            Err(StatsError::InvalidInput(_))
        ));
    }
}
