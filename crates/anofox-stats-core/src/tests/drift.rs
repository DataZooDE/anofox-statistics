//! Data-drift tests (#245-#255): evidently's drift statistics from upstream
//! anofox-statistics 0.5.
//!
//! - Two-sample Kolmogorov-Smirnov, Cramér-von Mises and Epps-Singleton tests
//! - k-sample Anderson-Darling test
//! - 1-D Wasserstein distance
//! - Binned PSI, Kullback-Leibler, Jensen-Shannon and Hellinger divergences
//! - Total variation distance with a permutation p-value
//! - Two-proportion z test, chi-squared and G goodness-of-fit drift tests,
//!   evidently's per-column drift (default test choice, thresholds, drift
//!   rules) and the dataset drift share (#300, anofox-statistics 0.7)
//!
//! Categorical data reaches these wrappers as numeric category codes; the
//! divergences then bin with [`Binning::Categorical`] (one bin per code).

use super::{convert_error, UnifiedTestResult};
use crate::StatsResult;
use anofox_tests::drift::{
    chisq_drift_numeric, column_drift_categorical, column_drift_numeric, g_drift_numeric,
    z_test_numeric,
};
pub use anofox_tests::drift::{
    Binning, ColumnDrift, DatasetDrift, DriftOptions as ColumnDriftOptions, DriftTest, ZeroFill,
};
use anofox_tests::{
    ad_test_ksamp, cvm_test_2sample, epps_singleton_test, hellinger, jensen_shannon, kl_divergence,
    ks_test_2sample, psi, tvd_test_numeric, wasserstein_test_1d,
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
    /// Two-proportion z test (evidently "z").
    Z,
    /// Chi-squared goodness-of-fit drift test (evidently "chisquare").
    ChiSquare,
    /// G goodness-of-fit drift test (evidently "g_test").
    GTest,
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
    /// Zero fill of PSI and KL.
    pub zero_fill: ZeroFill,
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
            zero_fill: ZeroFill::Evidently,
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
    /// divergence, or the raw Anderson-Darling criterion.
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
    /// Shares of the reference and the current sample that are not the
    /// smallest category (z test).
    pub estimate_reference: f64,
    pub estimate_current: f64,
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
            estimate_reference: f64::NAN,
            estimate_current: f64::NAN,
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
            let r = wasserstein_test_1d(
                reference,
                current,
                options.p,
                options.n_permutations,
                options.seed,
            )
            .map_err(convert_error)?;
            let unified = UnifiedTestResult::from(&r).with_n(m + n);
            let mut d = DriftResult::new(r.distance, m, n, unified);
            d.statistic2 = r.distance_normed;
            d
        }
        DriftKind::Psi
        | DriftKind::KlDivergence
        | DriftKind::JensenShannon
        | DriftKind::Hellinger => {
            let r = match kind {
                DriftKind::Psi => psi(reference, current, options.binning, options.zero_fill),
                DriftKind::KlDivergence => kl_divergence(
                    reference,
                    current,
                    options.binning,
                    options.base,
                    options.zero_fill,
                ),
                DriftKind::JensenShannon => {
                    jensen_shannon(reference, current, options.binning, options.base)
                }
                _ => hellinger(reference, current, options.binning),
            }
            .map_err(convert_error)?;
            let mut d = DriftResult::new(r.statistic, m, n, UnifiedTestResult::from(&r));
            d.n_bins = r.n_bins;
            if let Some(divergence) = r.divergence {
                d.statistic2 = divergence;
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
        DriftKind::Z => {
            let r =
                z_test_numeric(reference, current, options.alternative).map_err(convert_error)?;
            let unified = UnifiedTestResult {
                method: "Two-proportion z test (drift)".to_string(),
                alternative: Some(options.alternative),
                statistic: Some(r.statistic),
                p_value: Some(r.p_value),
                estimate: Some(r.estimate[0] - r.estimate[1]),
                n: Some(m + n),
                ..Default::default()
            };
            let mut d = DriftResult::new(r.statistic, m, n, unified);
            d.n_bins = r.n_categories;
            d.estimate_reference = r.estimate[0];
            d.estimate_current = r.estimate[1];
            d
        }
        DriftKind::ChiSquare | DriftKind::GTest => {
            let r = if kind == DriftKind::ChiSquare {
                chisq_drift_numeric(reference, current)
            } else {
                g_drift_numeric(reference, current)
            }
            .map_err(convert_error)?;
            let unified = UnifiedTestResult::from(&r).with_n(n);
            let mut d = DriftResult::new(r.statistic, m, n, unified);
            d.df = r.df;
            d.n_bins = r.expected.first().map_or(0, Vec::len);
            d
        }
    };
    Ok(r)
}

/// evidently's drift of one column of `current` against `reference`
/// (upstream `column_drift_numeric` / `column_drift_categorical`).
/// `categorical`: the values are category codes whose order is the label
/// order, tested with evidently's categorical rules.
pub fn column_drift(
    reference: &[f64],
    current: &[f64],
    categorical: bool,
    options: &ColumnDriftOptions,
) -> StatsResult<ColumnDrift> {
    if categorical {
        // Category codes are small non-negative integers.
        let codes = |v: &[f64]| v.iter().map(|&c| c as i64).collect::<Vec<i64>>();
        column_drift_categorical(&codes(reference), &codes(current), options)
    } else {
        column_drift_numeric(reference, current, options)
    }
    .map_err(convert_error)
}

/// evidently's dataset drift: drifted when the share of drifted columns is
/// at least `drift_share`.
pub fn dataset_drift(drifted: &[bool], drift_share: f64) -> StatsResult<DatasetDrift> {
    anofox_tests::drift::dataset_drift(drifted, drift_share).map_err(convert_error)
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
        assert!((js.statistic2 - 0.0664855398517632).abs() < 1e-10);
        assert!((run(DriftKind::Hellinger, &x, &y).statistic - 0.262287362723).abs() < 1e-10);
        // Integer codes: Auto bins per category for every divergence (upstream #45).
        assert!((run(DriftKind::Hellinger, &xc, &yc).statistic - 0.325194993744).abs() < 1e-10);
        let w2 = DriftOptions {
            p: 2.0,
            ..Default::default()
        };
        let w2 = drift_test(DriftKind::Wasserstein, &x, &y, &w2).unwrap();
        assert!((w2.statistic2 - 0.446953341978072).abs() < 1e-10);
        let eps = DriftOptions {
            zero_fill: ZeroFill::Constant(1e-3),
            ..Default::default()
        };
        let psi_eps = drift_test(DriftKind::Psi, &xc, &yc, &eps).unwrap();
        assert!((psi_eps.statistic - 1.10043459222267).abs() < 1e-10);
        let kl_eps = drift_test(DriftKind::KlDivergence, &xc, &yc, &eps).unwrap();
        assert!((kl_eps.statistic - 0.217427520064165).abs() < 1e-10);
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

    #[test]
    fn evidently_tests() {
        // validation/generators/make_reference_drift_evidently.R
        let xb: Vec<f64> = (0..50).map(|i| f64::from(i % 3 == 0)).collect();
        let yb: Vec<f64> = (0..40).map(|i| f64::from(i % 2 == 0)).collect();
        let z = run(DriftKind::Z, &xb, &yb);
        assert!((z.statistic + 1.53291275669323).abs() < 1e-12);
        assert!((z.p_value - 0.12529735230714).abs() < 1e-12);
        assert_eq!((z.estimate_reference, z.estimate_current), (0.34, 0.5));
        let xs: Vec<f64> = (0..50).map(|i| (i % 3) as f64).collect();
        let ys: Vec<f64> = (0..40).map(|i| ((i * 7) % 5 % 3) as f64).collect();
        let c = run(DriftKind::ChiSquare, &xs, &ys);
        assert!((c.statistic - 2.64705882352941).abs() < 1e-12);
        assert!((c.p_value - 0.266194133351106).abs() < 1e-12);
        assert_eq!((c.df, c.n_bins), (2.0, 3));
        let g = run(DriftKind::GTest, &xs, &ys);
        assert!((g.statistic - 2.88115341992583).abs() < 1e-12);
        assert!((g.p_value - 0.236791159476047).abs() < 1e-12);

        let (x, y, _, _) = data();
        let auto = column_drift(&x, &y, false, &ColumnDriftOptions::default()).unwrap();
        assert_eq!(auto.test, DriftTest::Ks);
        assert!((auto.score - 0.0574636491581379).abs() < 1e-9);
        assert!(!auto.drifted);
        let cat = column_drift(&xs, &ys, true, &ColumnDriftOptions::default()).unwrap();
        assert_eq!(cat.test, DriftTest::ChiSquare);
        assert!((cat.score - 0.266194133351106).abs() < 1e-12);
        let ed = ColumnDriftOptions {
            test: Some(DriftTest::EnergyDistance),
            ..Default::default()
        };
        let ed = column_drift(&x, &y, false, &ed).unwrap();
        assert!((ed.score - 0.435949389837772).abs() < 1e-12);
        assert_eq!(ed.threshold, 0.1);
        assert!(ed.drifted);

        let d = dataset_drift(&[true, false, true], 0.5).unwrap();
        assert_eq!((d.n_columns, d.n_drifted), (3, 2));
        assert!(d.dataset_drift);
        assert!(dataset_drift(&[], 0.5).is_err());
    }
}
