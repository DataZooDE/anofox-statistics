//! Model monitoring without labels (#366, #367), from upstream
//! anofox-statistics 0.10 `monitoring`:
//!
//! - multivariate drift by PCA reconstruction error, as NannyML's
//!   `DataReconstructionDriftCalculator` (`fit_reconstruction`,
//!   `ReconstructionModel::{from_parts, row_errors, chunk_error}`,
//!   `reconstruction_thresholds`)
//! - alert thresholds of a chunked metric from its reference values, as
//!   NannyML's `StandardDeviationThreshold` / `ConstantThreshold` and
//!   `calculate_threshold_values` (`threshold_band`)
//!
//! The wrapper only converts a fitted model to its parts (to be stored in a
//! model STRUCT) and rebuilds it from them with upstream `from_parts`.

use super::convert_error;
use crate::StatsResult;
use anofox_tests::monitoring::{
    fit_reconstruction, reconstruction_thresholds, threshold_band, ReconstructionModel,
    ReconstructionOptions,
};
pub use anofox_tests::monitoring::{
    ComponentSelection, ReconstructionChunk, ThresholdBand, ThresholdMethod, ValueLimits,
};
use anofox_tests::multivariate::Pca;

/// The parts of a fitted reconstruction model ([`ReconstructionModel`] with
/// its [`Pca`]), as stored in the model STRUCT.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionParts {
    /// Reference column means (also the imputation values).
    pub center: Vec<f64>,
    /// Population standard deviations the columns were divided by; `None`
    /// for an unscaled model.
    pub scale: Option<Vec<f64>>,
    /// Standard deviations of the kept components.
    pub sdev: Vec<f64>,
    /// Share of the total variance per kept component.
    pub proportion_of_variance: Vec<f64>,
    /// `rotation[i][j]`: loading of variable `i` on component `j`.
    pub rotation: Vec<Vec<f64>>,
    /// Components used for the reconstruction.
    pub n_components: usize,
    /// NaN inputs are imputed by `center`.
    pub impute_nan: bool,
}

impl From<ReconstructionModel> for ReconstructionParts {
    fn from(m: ReconstructionModel) -> Self {
        Self {
            center: m.pca.center,
            scale: m.pca.scale,
            sdev: m.pca.sdev,
            proportion_of_variance: m.pca.explained_variance_ratio,
            rotation: m.pca.rotation,
            n_components: m.n_components,
            impute_nan: m.impute_nan,
        }
    }
}

impl ReconstructionParts {
    /// Rebuilds the upstream model (`ReconstructionModel::from_parts`).
    fn model(&self) -> StatsResult<ReconstructionModel> {
        let pca = Pca {
            center: self.center.clone(),
            scale: self.scale.clone(),
            sdev: self.sdev.clone(),
            rotation: self.rotation.clone(),
            explained_variance_ratio: self.proportion_of_variance.clone(),
        };
        ReconstructionModel::from_parts(pca, self.n_components, self.impute_nan)
            .map_err(convert_error)
    }
}

/// Fits the reconstruction model on the reference rows (observations x
/// numeric variables; NaN is a missing value).
pub fn reconstruction_fit(
    reference: &[Vec<f64>],
    n_components: ComponentSelection,
    scale: bool,
    impute_nan: bool,
) -> StatsResult<ReconstructionParts> {
    let options = ReconstructionOptions {
        n_components,
        scale,
        impute_nan,
    };
    fit_reconstruction(reference, &options)
        .map(ReconstructionParts::from)
        .map_err(convert_error)
}

/// Reconstruction error of each row.
pub fn reconstruction_row_errors(
    model: &ReconstructionParts,
    rows: &[Vec<f64>],
) -> StatsResult<Vec<f64>> {
    model.model()?.row_errors(rows).map_err(convert_error)
}

/// Mean reconstruction error of a chunk with its sampling error and
/// NannyML's confidence band.
pub fn reconstruction_chunk(
    model: &ReconstructionParts,
    rows: &[Vec<f64>],
) -> StatsResult<ReconstructionChunk> {
    model.model()?.chunk_error(rows).map_err(convert_error)
}

/// Alert band of the reconstruction error from its reference chunk values
/// (a lower threshold at or below 0 dropped, as NannyML).
pub fn reconstruction_band(
    reference_chunk_values: &[f64],
    method: &ThresholdMethod,
) -> StatsResult<ThresholdBand> {
    reconstruction_thresholds(reference_chunk_values, method).map_err(convert_error)
}

/// Alert band of a metric from its reference values, limited by `limits`.
pub fn alert_band(
    reference_values: &[f64],
    method: &ThresholdMethod,
    limits: &ValueLimits,
) -> StatsResult<ThresholdBand> {
    threshold_band(reference_values, method, limits).map_err(convert_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Vec<f64>> {
        (0..40)
            .map(|i| {
                let t = i as f64;
                vec![t.sin(), (0.7 * t).cos() + 0.1 * t, (1.3 * t).sin() * 2.0]
            })
            .collect()
    }

    #[test]
    fn parts_round_trip_through_from_parts() {
        let x = data();
        let parts =
            reconstruction_fit(&x, ComponentSelection::VarianceShare(0.65), true, true).unwrap();
        assert_eq!(parts.sdev.len(), parts.n_components);
        let direct = fit_reconstruction(&x, &ReconstructionOptions::default()).unwrap();
        assert_eq!(parts.model().unwrap(), direct);
        assert_eq!(
            reconstruction_row_errors(&parts, &x[..5]).unwrap(),
            direct.row_errors(&x[..5]).unwrap()
        );
        assert_eq!(
            reconstruction_chunk(&parts, &x).unwrap(),
            direct.chunk_error(&x).unwrap()
        );
    }

    #[test]
    fn bands_match_upstream() {
        let v = [0.91, 0.93, f64::NAN, 0.92, 0.95, 0.90];
        let band = alert_band(
            &v,
            &ThresholdMethod::NANNYML_DEFAULT,
            &ValueLimits::UNIT_INTERVAL,
        )
        .unwrap();
        assert!((band.lower.unwrap() - 0.8703860483977444).abs() < 1e-12);
        let r = reconstruction_band(&[0.1, 0.9], &ThresholdMethod::NANNYML_DEFAULT).unwrap();
        assert_eq!(r.lower, None);
        assert!(alert_band(
            &v,
            &ThresholdMethod::Constant {
                lower: Some(1.0),
                upper: Some(0.0)
            },
            &ValueLimits::NONE
        )
        .is_err());
    }
}
