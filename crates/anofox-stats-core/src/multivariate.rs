//! Principal component analysis (#303): upstream anofox-statistics
//! `multivariate::pca_with_options` (R's `prcomp(x, center = TRUE, scale. =
//! FALSE | TRUE)`) and `Pca::transform` (`predict.prcomp`).
//!
//! The wrapper only keeps the first `n_components` components of the
//! upstream fit (`prcomp`'s `rank.`) and rebuilds an upstream [`Pca`] from a
//! model's pieces to score new rows.

use crate::tests::convert_error;
use crate::StatsResult;
use anofox_tests::multivariate::{pca_with_options, Pca, PcaOptions};

/// A fitted PCA, truncated to `sdev.len()` components.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaFit {
    /// Column means of the fitted rows.
    pub center: Vec<f64>,
    /// Column standard deviations (divisor `n - 1`) the rows were divided by
    /// (`prcomp`'s `scale`); `None` for an unscaled PCA.
    pub scale: Option<Vec<f64>>,
    /// Standard deviations of the kept components (divisor `n - 1`), decreasing.
    pub sdev: Vec<f64>,
    /// Share of the total variance per kept component.
    pub proportion_of_variance: Vec<f64>,
    /// `rotation[i][j]`: loading of variable `i` on component `j`. Each
    /// component's entry of largest absolute value is positive (upstream's
    /// sign convention).
    pub rotation: Vec<Vec<f64>>,
    /// Number of rows fitted.
    pub n_observations: usize,
}

/// PCA of the rows of `data` (observations x variables), keeping the first
/// `n_components` components (`None`: all `min(n - 1, d)`; more than that
/// keeps all, like `prcomp`'s `rank.`). `scale`: standardise the columns
/// (`prcomp(scale. = TRUE)`; a constant column is an error).
pub fn pca_fit(data: &[Vec<f64>], n_components: Option<usize>, scale: bool) -> StatsResult<PcaFit> {
    let p = pca_with_options(data, &PcaOptions { scale }).map_err(convert_error)?;
    let k = n_components.map_or(p.n_components(), |k| k.min(p.n_components()));
    Ok(PcaFit {
        center: p.center,
        scale: p.scale,
        sdev: p.sdev[..k].to_vec(),
        proportion_of_variance: p.explained_variance_ratio[..k].to_vec(),
        rotation: p.rotation.iter().map(|row| row[..k].to_vec()).collect(),
        n_observations: data.len(),
    })
}

/// Scores of `rows` on every component of a fitted model given by its
/// `center`, `scale` (`None`: unscaled), `sdev` and `rotation`
/// (`rotation[i][j]`, as [`PcaFit`]).
pub fn pca_transform(
    center: &[f64],
    scale: Option<&[f64]>,
    sdev: &[f64],
    rotation: &[Vec<f64>],
    rows: &[Vec<f64>],
) -> StatsResult<Vec<Vec<f64>>> {
    let model = Pca {
        center: center.to_vec(),
        sdev: sdev.to_vec(),
        rotation: rotation.to_vec(),
        explained_variance_ratio: vec![f64::NAN; sdev.len()],
        scale: scale.map(<[f64]>::to_vec),
    };
    model
        .transform(rows, model.n_components())
        .map_err(convert_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_truncates_and_transform_reproduces_scores() {
        let data = vec![
            vec![2.5, 2.4, 1.0],
            vec![0.5, 0.7, 2.0],
            vec![2.2, 2.9, 0.5],
            vec![1.9, 2.2, 1.5],
            vec![3.1, 3.0, 0.2],
        ];
        let full = pca_fit(&data, None, false).unwrap();
        assert_eq!(full.sdev.len(), 3);
        let two = pca_fit(&data, Some(2), false).unwrap();
        assert_eq!(two.sdev, full.sdev[..2].to_vec());
        assert_eq!(two.rotation[0].len(), 2);
        // more than available: all of them
        assert_eq!(pca_fit(&data, Some(10), false).unwrap().sdev.len(), 3);
        let scores = pca_transform(&two.center, None, &two.sdev, &two.rotation, &data).unwrap();
        let direct = pca_with_options(&data, &PcaOptions::default())
            .unwrap()
            .transform(&data, 2)
            .unwrap();
        assert_eq!(scores, direct);
        // Scaled: the stored sds are applied to new rows.
        let scaled = pca_fit(&data, None, true).unwrap();
        let sds = scaled.scale.clone().unwrap();
        assert_eq!(sds.len(), 3);
        let s = pca_transform(
            &scaled.center,
            Some(&sds),
            &scaled.sdev,
            &scaled.rotation,
            &data,
        )
        .unwrap();
        let direct = pca_with_options(&data, &PcaOptions { scale: true })
            .unwrap()
            .transform(&data, 3)
            .unwrap();
        assert_eq!(s, direct);
        // A constant column cannot be scaled (prcomp).
        let constant = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 5.0]];
        assert!(pca_fit(&constant, None, true).is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(pca_fit(&[vec![1.0, 2.0]], None, false).is_err());
        let m = pca_fit(
            &[vec![1.0, 2.0], vec![2.0, 1.0], vec![0.0, 0.5]],
            None,
            false,
        )
        .unwrap();
        assert!(pca_transform(&m.center, None, &m.sdev, &m.rotation, &[vec![1.0]]).is_err());
    }
}
