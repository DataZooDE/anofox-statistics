//! Confidence-based performance estimation of a binary classifier (#365):
//! upstream anofox-statistics 0.10 `monitoring::cbpe` (NannyML's
//! `CBPE(problem_type = "classification_binary")`).
//!
//! The model fitted on a labelled reference period is stored by the SQL layer
//! as a STRUCT of its parts (calibrator, threshold, sampling-error
//! components) and rebuilt with [`cbpe_model`] (`CbpeModel::from_parts`).
//!
//! Predictions: `None` (or all entries `None`) uses upstream's threshold path
//! (`score >= threshold`); otherwise a row's given prediction is used and a
//! missing one falls back to `score >= threshold`, with upstream's
//! `*_with_predictions` functions.

use super::convert_error;
use crate::{StatsError, StatsResult};
use anofox_tests::classification::{IsotonicCalibrator, PlattCalibrator};
use anofox_tests::monitoring::{fit_reference, fit_reference_with_predictions, CbpeOptions};
pub use anofox_tests::monitoring::{
    Calibrator, CbpeCalibration, CbpeEstimate, CbpeModel, CbpeSamplingComponents, MetricEstimate,
    RealizedPerformance,
};

/// The model's parts in the order the SQL layer stores them.
pub const SAMPLING_FIELDS: usize = 15;

/// A reference with one class is degenerate data (NULL in SQL), not an
/// invalid argument.
fn cbpe_error(e: anofox_tests::StatError) -> StatsError {
    match convert_error(e) {
        StatsError::InvalidInput(msg) if msg.contains("both classes") => {
            StatsError::InsufficientDataMsg(msg)
        }
        other => other,
    }
}

/// The predictions to pass upstream: `None` when every row uses the
/// threshold.
fn resolve_predictions(
    scores: &[f64],
    predictions: Option<&[Option<bool>]>,
    threshold: f64,
) -> StatsResult<Option<Vec<bool>>> {
    let Some(p) = predictions else {
        return Ok(None);
    };
    if p.len() != scores.len() {
        return Err(StatsError::InvalidInput(format!(
            "predictions has {} values, expected {}",
            p.len(),
            scores.len()
        )));
    }
    if p.iter().all(Option::is_none) {
        return Ok(None);
    }
    Ok(Some(
        p.iter()
            .zip(scores)
            .map(|(p, &s)| p.unwrap_or(s >= threshold))
            .collect(),
    ))
}

/// Fits CBPE on a labelled reference period (`fit_reference` /
/// `fit_reference_with_predictions`). The model's threshold is `threshold`
/// in both cases.
pub fn fit_cbpe(
    scores: &[f64],
    labels: &[bool],
    predictions: Option<&[Option<bool>]>,
    calibration: CbpeCalibration,
    threshold: f64,
) -> StatsResult<CbpeModel> {
    if !threshold.is_finite() {
        return Err(StatsError::InvalidInput(format!(
            "threshold must be finite, got {threshold}"
        )));
    }
    match resolve_predictions(scores, predictions, threshold)? {
        None => fit_reference(
            scores,
            labels,
            &CbpeOptions {
                calibration,
                threshold,
            },
        )
        .map_err(cbpe_error),
        Some(p) => {
            let m = fit_reference_with_predictions(scores, &p, labels, calibration)
                .map_err(cbpe_error)?;
            Ok(CbpeModel::from_parts(m.calibrator, threshold, m.sampling))
        }
    }
}

/// The sampling components in the stored order.
pub fn sampling_to_array(s: &CbpeSamplingComponents) -> [f64; SAMPLING_FIELDS] {
    [
        s.accuracy_std,
        s.precision_std,
        s.precision_fraction,
        s.recall_std,
        s.recall_fraction,
        s.specificity_std,
        s.specificity_fraction,
        s.f1_std,
        s.f1_fraction,
        s.roc_auc_std,
        s.roc_auc_fraction,
        s.true_positive_std,
        s.false_positive_std,
        s.false_negative_std,
        s.true_negative_std,
    ]
}

fn sampling_from_array(v: &[f64; SAMPLING_FIELDS]) -> CbpeSamplingComponents {
    CbpeSamplingComponents {
        accuracy_std: v[0],
        precision_std: v[1],
        precision_fraction: v[2],
        recall_std: v[3],
        recall_fraction: v[4],
        specificity_std: v[5],
        specificity_fraction: v[6],
        f1_std: v[7],
        f1_fraction: v[8],
        roc_auc_std: v[9],
        roc_auc_fraction: v[10],
        true_positive_std: v[11],
        false_positive_std: v[12],
        false_negative_std: v[13],
        true_negative_std: v[14],
    }
}

/// Rebuilds a stored model (`CbpeModel::from_parts`; the isotonic knots
/// through `IsotonicCalibrator::from_parts`, which validates them).
pub fn cbpe_model(
    calibration: CbpeCalibration,
    knots_x: &[f64],
    knots_y: &[f64],
    platt_a: f64,
    platt_b: f64,
    threshold: f64,
    sampling: &[f64; SAMPLING_FIELDS],
) -> StatsResult<CbpeModel> {
    if !threshold.is_finite() {
        return Err(StatsError::InvalidInput(format!(
            "the model's threshold must be finite, got {threshold}"
        )));
    }
    let calibrator = match calibration {
        CbpeCalibration::None => Calibrator::Identity,
        CbpeCalibration::Isotonic => Calibrator::Isotonic(
            IsotonicCalibrator::from_parts(knots_x.to_vec(), knots_y.to_vec())
                .map_err(convert_error)?,
        ),
        CbpeCalibration::Platt => {
            if !platt_a.is_finite() || !platt_b.is_finite() {
                return Err(StatsError::InvalidInput(
                    "the model's Platt parameters must be finite".into(),
                ));
            }
            Calibrator::Platt(PlattCalibrator {
                a: platt_a,
                b: platt_b,
            })
        }
    };
    Ok(CbpeModel::from_parts(
        calibrator,
        threshold,
        sampling_from_array(sampling),
    ))
}

/// Estimated performance of an unlabelled chunk (`estimate` /
/// `estimate_with_predictions`).
pub fn cbpe_estimate(
    model: &CbpeModel,
    scores: &[f64],
    predictions: Option<&[Option<bool>]>,
) -> StatsResult<CbpeEstimate> {
    match resolve_predictions(scores, predictions, model.threshold)? {
        None => model.estimate(scores),
        Some(p) => model.estimate_with_predictions(scores, &p),
    }
    .map_err(cbpe_error)
}

/// Realized performance of a labelled chunk (`CbpeModel::realized` /
/// `realized_performance`).
pub fn cbpe_realized(
    model: &CbpeModel,
    scores: &[f64],
    labels: &[bool],
    predictions: Option<&[Option<bool>]>,
) -> StatsResult<RealizedPerformance> {
    match resolve_predictions(scores, predictions, model.threshold)? {
        None => model.realized(scores, labels),
        Some(p) => anofox_tests::monitoring::realized_performance(scores, &p, labels),
    }
    .map_err(cbpe_error)
}

#[cfg(test)]
mod unit {
    use super::*;

    fn data() -> (Vec<f64>, Vec<bool>) {
        let s: Vec<f64> = (0..40).map(|i| (i as f64 + 0.5) / 40.0).collect();
        let y = (0..40).map(|i| (i * 7) % 10 < (i / 4)).collect();
        (s, y)
    }

    #[test]
    fn rebuilt_model_estimates_like_the_fitted_one() {
        let (s, y) = data();
        for cal in [
            CbpeCalibration::Isotonic,
            CbpeCalibration::Platt,
            CbpeCalibration::None,
        ] {
            let m = fit_cbpe(&s, &y, None, cal, 0.5).unwrap();
            let (kx, ky, a, b) = match &m.calibrator {
                Calibrator::Isotonic(c) => (c.x.clone(), c.y.clone(), f64::NAN, f64::NAN),
                Calibrator::Platt(c) => (vec![], vec![], c.a, c.b),
                Calibrator::Identity => (vec![], vec![], f64::NAN, f64::NAN),
            };
            let r = cbpe_model(
                cal,
                &kx,
                &ky,
                a,
                b,
                m.threshold,
                &sampling_to_array(&m.sampling),
            )
            .unwrap();
            assert_eq!(r, m);
            assert_eq!(
                cbpe_estimate(&r, &s, None).unwrap(),
                m.estimate(&s).unwrap()
            );
        }
    }

    #[test]
    fn predictions_fall_back_to_the_threshold() {
        let (s, y) = data();
        let none: Vec<Option<bool>> = vec![None; s.len()];
        let a = fit_cbpe(&s, &y, None, CbpeCalibration::Isotonic, 0.4).unwrap();
        let b = fit_cbpe(&s, &y, Some(&none), CbpeCalibration::Isotonic, 0.4).unwrap();
        assert_eq!(a, b);
        let given: Vec<Option<bool>> = s.iter().map(|&v| Some(v >= 0.4)).collect();
        let c = fit_cbpe(&s, &y, Some(&given), CbpeCalibration::Isotonic, 0.4).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn one_class_reference_is_degenerate() {
        let e = fit_cbpe(
            &[0.1, 0.9],
            &[true, true],
            None,
            CbpeCalibration::Isotonic,
            0.5,
        );
        assert!(matches!(e, Err(StatsError::InsufficientDataMsg(_))));
    }
}
