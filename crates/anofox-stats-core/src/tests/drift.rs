//! Data-drift tests (#245-#255): evidently's drift statistics from upstream
//! anofox-statistics 0.5.
//!
//! - Two-sample Kolmogorov-Smirnov, Cramér-von Mises and Epps-Singleton tests
//! - k-sample Anderson-Darling test
//! - 1-D Wasserstein distance
//! - Binned PSI, Kullback-Leibler, Jensen-Shannon and Hellinger divergences
//! - Total variation distance with a permutation p-value
//!
//! Categorical data reaches these wrappers as numeric category codes; the
//! divergences then bin with [`Binning::Categorical`] (one bin per code).

use super::{convert_error, UnifiedTestResult};
use crate::StatsResult;
pub use anofox_tests::drift::Binning;
use anofox_tests::{
    ad_test_ksamp, cvm_test_2sample, epps_singleton_test, hellinger, jensen_shannon, kl_divergence,
    ks_test_2sample, psi, tvd_test_numeric, wasserstein_distance_1d, wasserstein_test_1d,
};
pub use anofox_tests::{Alternative, CvmMethod};

/// Which drift statistic [`drift_test`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftKind {
    Ks,
    Cvm,
    EppsSingleton,
    Wasserstein,
    Psi,
    KlDivergence,
    JensenShannon,
    Hellinger,
    Tvd,
}

/// Options of [`drift_test`]; each kind reads only its own.
#[derive(Debug, Clone)]
pub struct DriftOptions {
    /// KS alternative.
    pub alternative: Alternative,
    /// KS exact p-value (`None`: R's rule).
    pub exact: Option<bool>,
    /// Cramér-von Mises p-value method.
    pub cvm_method: CvmMethod,
    /// Epps-Singleton evaluation points (`None`: `[0.4, 0.8]`).
    pub t: Option<Vec<f64>>,
    /// Wasserstein order.
    pub p: f64,
    /// Permutations (Wasserstein, TVD).
    pub n_permutations: usize,
    /// Permutation seed.
    pub seed: Option<u64>,
    /// Binning of the divergences.
    pub binning: Binning,
    /// Logarithm base of KL and Jensen-Shannon (`None`: e).
    pub base: Option<f64>,
}

impl Default for DriftOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            exact: None,
            cvm_method: CvmMethod::Auto,
            t: None,
            p: 1.0,
            n_permutations: 0,
            seed: None,
            binning: Binning::Auto,
            base: None,
        }
    }
}

/// Result of a drift statistic. A field that does not apply is NaN (numbers)
/// or 0 (`n_bins`).
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// The statistic (D, T, W, distance or divergence).
    pub statistic: f64,
    /// A second value: the normed Wasserstein distance, the Jensen-Shannon
    /// divergence (the squared distance), or the raw Anderson-Darling criterion.
    pub statistic2: f64,
    pub p_value: f64,
    /// Epps-Singleton degrees of freedom.
    pub df: f64,
    /// Reference (or, for Anderson-Darling, total) sample size.
    pub n1: usize,
    /// Current sample size (Anderson-Darling: number of groups).
    pub n2: usize,
    /// Bins or categories of the divergences and TVD.
    pub n_bins: usize,
    /// Whether the p-value is exact (KS, Cramér-von Mises).
    pub exact: bool,
    pub method: String,
    pub unified: UnifiedTestResult,
}

impl DriftResult {
    fn new(statistic: f64, n1: usize, n2: usize, unified: UnifiedTestResult) -> Self {
        Self {
            statistic,
            statistic2: f64::NAN,
            p_value: unified.p_value.unwrap_or(f64::NAN),
            df: f64::NAN,
            n1,
            n2,
            n_bins: 0,
            exact: false,
            method: unified.method.clone(),
            unified,
        }
    }
}

/// Two-sample drift statistic of `current` against `reference`.
pub fn drift_test(
    kind: DriftKind,
    reference: &[f64],
    current: &[f64],
    options: &DriftOptions,
) -> StatsResult<DriftResult> {
    let (m, n) = (reference.len(), current.len());
    let r = match kind {
        DriftKind::Ks => {
            let r = ks_test_2sample(reference, current, options.alternative, options.exact)
                .map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.exact = r.exact;
            d
        }
        DriftKind::Cvm => {
            let r =
                cvm_test_2sample(reference, current, options.cvm_method).map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.exact = r.exact;
            d
        }
        DriftKind::EppsSingleton => {
            let r = epps_singleton_test(reference, current, options.t.as_deref())
                .map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.df = r.df as f64;
            d
        }
        DriftKind::Wasserstein => {
            if options.p == 1.0 {
                let r =
                    wasserstein_test_1d(reference, current, options.n_permutations, options.seed)
                        .map_err(convert_error)?;
                let unified = UnifiedTestResult::from(&r).with_n(m + n);
                let mut d = DriftResult::new(r.distance, m, n, unified);
                d.statistic2 = r.distance_normed;
                d
            } else {
                // Upstream norms and permutes only W_1.
                if options.n_permutations > 0 {
                    return Err(crate::StatsError::InvalidInput(
                        "wasserstein: a permutation p-value needs p = 1".into(),
                    ));
                }
                let w = wasserstein_distance_1d(reference, current, options.p)
                    .map_err(convert_error)?;
                let mut unified =
                    UnifiedTestResult::new(format!("Wasserstein-{} distance", options.p))
                        .with_n(m + n);
                unified.statistic = Some(w);
                DriftResult::new(w, m, n, unified)
            }
        }
        DriftKind::Psi
        | DriftKind::KlDivergence
        | DriftKind::JensenShannon
        | DriftKind::Hellinger => {
            let r = match kind {
                DriftKind::Psi => psi(reference, current, options.binning),
                DriftKind::KlDivergence => {
                    kl_divergence(reference, current, options.binning, options.base)
                }
                DriftKind::JensenShannon => {
                    jensen_shannon(reference, current, options.binning, options.base)
                }
                _ => hellinger(reference, current, options.binning),
            }
            .map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.n_bins = r.n_bins;
            if kind == DriftKind::JensenShannon {
                d.statistic2 = r.statistic * r.statistic;
            }
            d
        }
        DriftKind::Tvd => {
            let r = tvd_test_numeric(reference, current, options.n_permutations, options.seed)
                .map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.n_bins = r.n_categories;
            d
        }
    };
    Ok(r)
}

/// k-sample Anderson-Darling test (Scholz and Stephens). `n1` is the total
/// sample size, `n2` the number of groups, `statistic2` the raw AD criterion.
pub fn anderson_darling_ksamp(groups: &[Vec<f64>], midrank: bool) -> StatsResult<DriftResult> {
    let slices: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    let r = ad_test_ksamp(&slices, midrank).map_err(convert_error)?;
    let mut d = DriftResult::new(r.statistic, r.n, r.k, UnifiedTestResult::from(&r));
    d.statistic2 = r.ad;
    d.exact = false;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
        let y = (0..40)
            .map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5)
            .collect();
        let xc = (0..50).map(|i| (i % 4) as f64).collect();
        let yc = (0..40).map(|i| ((i * 3) % 5) as f64).collect();
        (x, y, xc, yc)
    }

    fn run(kind: DriftKind, a: &[f64], b: &[f64]) -> DriftResult {
        drift_test(kind, a, b, &DriftOptions::default()).unwrap()
    }

    #[test]
    fn reference_values() {
        let (x, y, xc, yc) = data();
        let ks = run(DriftKind::Ks, &x, &y);
        assert!((ks.statistic - 0.275).abs() < 1e-12);
        assert!((ks.p_value - 0.0574636491581).abs() < 1e-9);
        assert!(ks.exact);
        let cvm = run(DriftKind::Cvm, &x, &y);
        assert!((cvm.p_value - 0.0468933196396).abs() < 1e-9);
        let es = run(DriftKind::EppsSingleton, &x, &y);
        assert!((es.statistic - 19.25205304182).abs() < 1e-8);
        assert_eq!(es.df, 4.0);
        let w = run(DriftKind::Wasserstein, &x, &y);
        assert!((w.statistic - 0.625792792352).abs() < 1e-10);
        assert!((w.statistic2 - 0.43940285495).abs() < 1e-10);
        assert!(w.p_value.is_nan());
        assert!((run(DriftKind::Psi, &x, &y).statistic - 0.570692693084).abs() < 1e-10);
        assert!((run(DriftKind::Psi, &xc, &yc).statistic - 1.56548983794).abs() < 1e-10);
        assert!((run(DriftKind::KlDivergence, &xc, &yc).statistic - 0.223061363402).abs() < 1e-10);
        let js = run(DriftKind::JensenShannon, &x, &y);
        assert!((js.statistic - 0.257847900615).abs() < 1e-10);
        assert!((js.statistic2 - js.statistic.powi(2)).abs() < 1e-15);
        assert!((run(DriftKind::Hellinger, &x, &y).statistic - 0.262287362723).abs() < 1e-10);
        let opts = DriftOptions {
            n_permutations: 1000,
            seed: Some(42),
            ..Default::default()
        };
        let tvd = drift_test(DriftKind::Tvd, &xc, &yc, &opts).unwrap();
        assert!((tvd.statistic - 0.2).abs() < 1e-15);
        assert_eq!(tvd.n_bins, 5);
        let ad = anderson_darling_ksamp(&[x, y], true).unwrap();
        assert!((ad.statistic - 3.6104).abs() < 1e-4);
        assert_eq!((ad.n1, ad.n2), (90, 2));
    }
}
