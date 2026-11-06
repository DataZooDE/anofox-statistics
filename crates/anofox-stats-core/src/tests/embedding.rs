//! Embedding drift and multivariate two-sample tests (#302), from upstream
//! anofox-statistics 0.7:
//!
//! - evidently's embedding drift methods `distance`, `ratio` and `mmd`
//!   (`drift::embedding_distance_drift`, `embedding_ratio_drift`,
//!   `embedding_mmd_drift`, with a fixed or a bootstrapped threshold)
//! - the multivariate energy distance and MMD permutation tests
//!   (`modern::energy_distance_test`, `modern::mmd_test`)
//!
//! Every sample is a list of rows (one embedding per observation) of equal
//! length. The defaults of the options are evidently's, as upstream documents
//! them.

use super::{convert_error, UnifiedTestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::drift::{
    defaults, embedding_distance_drift, embedding_mmd_bandwidth, embedding_mmd_drift,
    embedding_ratio_drift, EmbeddingThreshold,
};
pub use anofox_tests::drift::{DriftTest, EmbeddingDistance};
use anofox_tests::modern::median_heuristic_bandwidth;
pub use anofox_tests::Kernel;
use anofox_tests::{energy_distance_test, mmd_test, Alternative};

/// evidently's embedding drift method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingMethod {
    /// Distance between the mean embeddings.
    Distance,
    /// Share of drifted components.
    Ratio,
    /// Maximum mean discrepancy with evidently's median bandwidth.
    Mmd,
}

/// Options of [`embedding_drift`]; each method reads only its own.
#[derive(Debug, Clone)]
pub struct EmbeddingDriftOptions {
    pub method: EmbeddingMethod,
    /// Distance between the mean embeddings (`distance`).
    pub metric: EmbeddingDistance,
    /// Fixed threshold; `None`: evidently's default (0.2 distance and ratio,
    /// 0.015 mmd). Not used with `bootstrap`.
    pub threshold: Option<f64>,
    /// Bootstrap the threshold from reference resamples (`distance`, `mmd`).
    pub bootstrap: bool,
    /// Bootstrap resamples.
    pub n_bootstrap: usize,
    /// Quantile of the bootstrap scores; `None`: evidently's default (0.95
    /// distance, 0.05 mmd).
    pub quantile: Option<f64>,
    /// Per-component test (`ratio`); evidently's default Wasserstein.
    pub component_test: DriftTest,
    /// Per-component threshold (`ratio`); `None`: the test's default.
    pub component_threshold: Option<f64>,
    /// Seed of the bootstrap and of permutation component tests.
    pub seed: Option<u64>,
}

impl Default for EmbeddingDriftOptions {
    fn default() -> Self {
        Self {
            method: EmbeddingMethod::Distance,
            metric: EmbeddingDistance::Euclidean,
            threshold: None,
            bootstrap: false,
            n_bootstrap: defaults::EMBEDDING_BOOTSTRAP_RESAMPLES,
            quantile: None,
            component_test: defaults::EMBEDDING_RATIO_COMPONENT_TEST,
            component_threshold: None,
            seed: None,
        }
    }
}

/// Result of [`embedding_drift`].
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDrift {
    /// evidently's method name: `"distance"`, `"ratio"` or `"mmd"`.
    pub method: &'static str,
    pub score: f64,
    /// The threshold the score was compared with (the bootstrap quantile when
    /// bootstrapped).
    pub threshold: f64,
    pub drifted: bool,
    pub n1: usize,
    pub n2: usize,
    pub dim: usize,
    /// Kernel bandwidth of the `mmd` method (upstream
    /// `drift::embedding_mmd_bandwidth`, evidently's median convention);
    /// `None` for the other methods.
    pub bandwidth: Option<f64>,
}

/// Embedding drift of `current` against `reference` (rows of equal length).
pub fn embedding_drift(
    reference: &[Vec<f64>],
    current: &[Vec<f64>],
    options: &EmbeddingDriftOptions,
) -> StatsResult<EmbeddingDrift> {
    // evidently's defaults (upstream `drift::defaults`): the fixed threshold
    // and the bootstrap of each method; options override single parts.
    let threshold = |fixed: EmbeddingThreshold, quantile: f64| -> StatsResult<EmbeddingThreshold> {
        if !options.bootstrap {
            return Ok(match options.threshold {
                Some(t) => EmbeddingThreshold::Fixed(t),
                None => fixed,
            });
        }
        if options.threshold.is_some() {
            return Err(StatsError::InvalidInput(
                "threshold and bootstrap exclude each other".into(),
            ));
        }
        Ok(EmbeddingThreshold::Bootstrap {
            n_bootstrap: options.n_bootstrap,
            quantile: options.quantile.unwrap_or(quantile),
            seed: options.seed,
        })
    };
    let r = match options.method {
        EmbeddingMethod::Distance => embedding_distance_drift(
            reference,
            current,
            options.metric,
            threshold(
                EmbeddingThreshold::EVIDENTLY_DISTANCE,
                defaults::EMBEDDING_DISTANCE_QUANTILE,
            )?,
        ),
        EmbeddingMethod::Mmd => embedding_mmd_drift(
            reference,
            current,
            threshold(
                EmbeddingThreshold::EVIDENTLY_MMD,
                defaults::EMBEDDING_MMD_QUANTILE,
            )?,
        ),
        EmbeddingMethod::Ratio => {
            if options.bootstrap {
                return Err(StatsError::InvalidInput(
                    "the ratio method has no bootstrap threshold".into(),
                ));
            }
            embedding_ratio_drift(
                reference,
                current,
                options.component_test,
                options.component_threshold,
                options
                    .threshold
                    .unwrap_or(defaults::EMBEDDING_RATIO_THRESHOLD),
                options.seed,
            )
        }
    }
    .map_err(convert_error)?;
    let bandwidth = match options.method {
        EmbeddingMethod::Mmd => {
            Some(embedding_mmd_bandwidth(reference, current).map_err(convert_error)?)
        }
        _ => None,
    };
    Ok(EmbeddingDrift {
        bandwidth,
        method: r.method,
        score: r.score,
        threshold: r.threshold,
        drifted: r.drifted,
        n1: reference.len(),
        n2: current.len(),
        dim: reference.first().map_or(0, Vec::len),
    })
}

/// Which multivariate two-sample test [`multivariate_two_sample`] runs.
#[derive(Debug, Clone, Copy)]
pub enum MultivariateTest {
    /// Energy distance permutation test.
    Energy,
    /// Unbiased MMD² permutation test with the kernel.
    Mmd(Kernel),
}

/// Result of [`multivariate_two_sample`].
#[derive(Debug, Clone)]
pub struct MultivariateTestResult {
    /// Energy distance or unbiased MMD².
    pub statistic: f64,
    /// Permutation p-value `(1 + #{perm >= observed}) / (1 + B)`.
    pub p_value: f64,
    pub n1: usize,
    pub n2: usize,
    pub dim: usize,
    pub n_permutations: usize,
    /// Bandwidth of the gaussian / laplacian MMD kernel (with
    /// `Kernel::GaussianMedianHeuristic` the median heuristic's,
    /// `modern::median_heuristic_bandwidth`); `None` otherwise.
    pub bandwidth: Option<f64>,
    pub unified: UnifiedTestResult,
}

/// Multivariate two-sample test of `x` against `y` (rows of equal length).
pub fn multivariate_two_sample(
    test: MultivariateTest,
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: Option<u64>,
) -> StatsResult<MultivariateTestResult> {
    let n = x.len() + y.len();
    let bandwidth = match test {
        MultivariateTest::Mmd(Kernel::GaussianMedianHeuristic) => {
            Some(median_heuristic_bandwidth(x, y).map_err(convert_error)?)
        }
        MultivariateTest::Mmd(Kernel::Gaussian { bandwidth })
        | MultivariateTest::Mmd(Kernel::Laplacian { bandwidth }) => Some(bandwidth),
        _ => None,
    };
    let (statistic, p_value, unified) = match test {
        MultivariateTest::Energy => {
            let r = energy_distance_test(x, y, n_permutations, seed).map_err(convert_error)?;
            (r.statistic, r.p_value, UnifiedTestResult::from(&r))
        }
        MultivariateTest::Mmd(kernel) => {
            let r = mmd_test(x, y, kernel, n_permutations, seed).map_err(convert_error)?;
            (r.statistic, r.p_value, UnifiedTestResult::from(&r))
        }
    };
    Ok(MultivariateTestResult {
        statistic,
        p_value,
        n1: x.len(),
        n2: y.len(),
        dim: x.first().map_or(0, Vec::len),
        n_permutations,
        bandwidth,
        unified: unified.with_alternative(Alternative::TwoSided).with_n(n),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// validation/generators/make_reference_embedding_drift.R (samples A, B).
    fn data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let a = (0..30)
            .map(|i| {
                let i = i as f64;
                vec![(0.7 * i).sin(), (1.3 * i).cos(), 0.1 * (i % 5.0)]
            })
            .collect();
        let b = (0..25)
            .map(|i| {
                let i = i as f64;
                vec![
                    (0.9 * i).sin() + 0.4,
                    (1.1 * i).cos() - 0.2,
                    0.1 * (i % 7.0) + 0.1,
                ]
            })
            .collect();
        (a, b)
    }

    #[test]
    fn embedding_drift_methods() {
        let (a, b) = data();
        let d = embedding_drift(&a, &b, &EmbeddingDriftOptions::default()).unwrap();
        assert_eq!((d.method, d.n1, d.n2, d.dim), ("distance", 30, 25, 3));
        assert_eq!(d.threshold, 0.2);
        assert!((d.score - 0.497432149080502).abs() < 1e-12);
        assert!(d.drifted);
        let cosine = EmbeddingDriftOptions {
            metric: EmbeddingDistance::Cosine,
            ..Default::default()
        };
        let c = embedding_drift(&a, &b, &cosine).unwrap();
        assert!((c.score - 0.262802179832803).abs() < 1e-12);
        let mmd = EmbeddingDriftOptions {
            method: EmbeddingMethod::Mmd,
            ..Default::default()
        };
        let m = embedding_drift(&a, &b, &mmd).unwrap();
        assert_eq!((m.method, m.threshold), ("mmd", 0.015));
        assert!((m.score - 0.0232831661825482).abs() < 1e-12);
        let ratio = EmbeddingDriftOptions {
            method: EmbeddingMethod::Ratio,
            ..Default::default()
        };
        let r = embedding_drift(&a, &b, &ratio).unwrap();
        assert_eq!((r.method, r.threshold), ("ratio", 0.2));
        assert_eq!(r.score, 1.0);

        let boot = EmbeddingDriftOptions {
            bootstrap: true,
            seed: Some(7),
            ..Default::default()
        };
        let b1 = embedding_drift(&a, &b, &boot).unwrap();
        let b2 = embedding_drift(&a, &b, &boot).unwrap();
        assert_eq!(b1, b2);
        assert_eq!(b1.score, d.score);
        let both = EmbeddingDriftOptions {
            threshold: Some(0.1),
            ..boot.clone()
        };
        assert!(embedding_drift(&a, &b, &both).is_err());
        let ratio_boot = EmbeddingDriftOptions {
            method: EmbeddingMethod::Ratio,
            bootstrap: true,
            ..Default::default()
        };
        assert!(embedding_drift(&a, &b, &ratio_boot).is_err());
        let ragged = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(embedding_drift(&ragged, &b, &EmbeddingDriftOptions::default()).is_err());
    }

    #[test]
    fn multivariate_tests() {
        let (a, b) = data();
        let e = multivariate_two_sample(MultivariateTest::Energy, &a, &b, 199, Some(1)).unwrap();
        assert_eq!((e.n1, e.n2, e.dim, e.n_permutations), (30, 25, 3, 199));
        assert!(e.p_value > 0.0 && e.p_value <= 1.0);
        assert!((e.statistic - 0.0883702929758567).abs() < 1e-12);
        assert_eq!(e.unified.n, Some(55));
        let kernel = Kernel::Gaussian { bandwidth: 1.0 };
        let m =
            multivariate_two_sample(MultivariateTest::Mmd(kernel), &a, &b, 199, Some(1)).unwrap();
        assert!(m.p_value > 0.0 && m.p_value <= 1.0);
        assert!((m.statistic - 0.0240892109612384).abs() < 1e-12);
        let one = vec![vec![0.0, 1.0]];
        assert!(multivariate_two_sample(MultivariateTest::Mmd(kernel), &one, &b, 9, None).is_err());
    }
}
