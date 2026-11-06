//! Multiple imputation and Rubin pooling (#321): thin delegation to
//! anofox-regression `imputation::areg_impute` (`Hmisc::aregImpute`: additive
//! regression, bootstrap and predictive mean matching, replaying R's random
//! draws for a seed) and `inference::pool` (`pool_rubin`, Rubin's rules with
//! the Barnard–Rubin d.f. of `mice::pool`; `pooled_wald_d1`, the D1 Wald test
//! of `mice::D1` with Reiter's d.f.).
//!
//! The wrapper only validates the shapes, converts between row vectors and
//! faer matrices, and computes the per-coefficient t statistics, p-values and
//! intervals with upstream's `CoefficientInference` on each coefficient's
//! pooled d.f. (`summary(mice::pool(...), conf.int = TRUE)`).

use crate::{StatsError, StatsResult};
use anofox_regression::imputation::{areg_impute as upstream_areg_impute, AregImputeOptions};
pub use anofox_regression::imputation::{DonorMatch, ImputeVariable};
use anofox_regression::inference::pool::{pool_rubin as upstream_pool_rubin, pooled_wald_d1};
use anofox_regression::inference::CoefficientInference;
use faer::{Col, Mat};

/// Options of [`areg_impute`] (`Hmisc::aregImpute` argument names in brackets).
#[derive(Debug, Clone, PartialEq)]
pub struct AregImputeConfig {
    /// Number of imputations (`n.impute`).
    pub n_impute: usize,
    /// Spline knots (`nk`; 0 makes every variable linear).
    pub nk: usize,
    /// Model only the linear part of a spline target (`tlinear`).
    pub tlinear: bool,
    /// Iterations before the first kept imputation (`burnin`); `None`: Hmisc's default.
    pub burnin: Option<usize>,
    /// Donor selection (`match`, `fweighted`).
    pub matching: DonorMatch,
    /// Seed of R's generator (`set.seed`).
    pub seed: i32,
}

impl Default for AregImputeConfig {
    fn default() -> Self {
        Self {
            n_impute: 5,
            nk: 3,
            tlinear: true,
            burnin: None,
            matching: DonorMatch::Weighted(0.2),
            seed: 1,
        }
    }
}

/// Result of [`areg_impute`].
#[derive(Debug, Clone, PartialEq)]
pub struct AregImputeResult {
    /// `completed[m][i]`: row `i` of completed data set `m` (observed values
    /// kept, missing ones imputed).
    pub completed: Vec<Vec<Vec<f64>>>,
    /// R² of the last imputation model of each column (NaN for complete columns).
    pub r_squared: Vec<f64>,
}

/// `aregImpute(~ ., data)` of `rows` (NaN = missing), one kind per column.
pub fn areg_impute(
    rows: &[Vec<f64>],
    kinds: &[ImputeVariable],
    config: &AregImputeConfig,
) -> StatsResult<AregImputeResult> {
    let n = rows.len();
    let p = kinds.len();
    if p < 2 {
        return Err(StatsError::InvalidInput(
            "multiple imputation needs two or more variables".into(),
        ));
    }
    if let Some(r) = rows.iter().find(|r| r.len() != p) {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "every row must have {p} values, got {}",
            r.len()
        )));
    }
    if config.n_impute == 0 {
        return Err(StatsError::InvalidInput(
            "n_impute must be at least 1".into(),
        ));
    }
    if let DonorMatch::Weighted(f) = config.matching {
        if !(f.is_finite() && f > 0.0) {
            return Err(StatsError::InvalidInput(
                "fweighted must be positive".into(),
            ));
        }
    }
    if rows.iter().flatten().any(|v| v.is_infinite()) {
        return Err(StatsError::InvalidInput(
            "values must be finite (NULL or NaN marks a missing value)".into(),
        ));
    }
    if n < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "multiple imputation needs two or more rows".into(),
        ));
    }
    let x = Mat::from_fn(n, p, |i, j| rows[i][j]);
    let options = AregImputeOptions::default()
        .n_impute(config.n_impute)
        .nk(config.nk)
        .tlinear(config.tlinear)
        .burnin(config.burnin)
        .matching(config.matching)
        .seed(config.seed);
    // The shapes and options are valid here, so what upstream rejects is the
    // data (a constant or entirely missing column, a bootstrap sample without
    // every level of a categorical variable): a degenerate outcome.
    let fit = upstream_areg_impute(&x, kinds, &options).map_err(|e| match StatsError::from(e) {
        StatsError::InvalidInput(msg) => StatsError::InsufficientDataMsg(msg),
        other => other,
    })?;
    let completed = (0..config.n_impute)
        .map(|m| {
            let c = fit.complete(&x, m);
            (0..n)
                .map(|i| (0..p).map(|j| c[(i, j)]).collect())
                .collect()
        })
        .collect();
    Ok(AregImputeResult {
        completed,
        r_squared: fit.r_squared,
    })
}

/// One pooled coefficient (`summary(mice::pool(fit), conf.int = TRUE)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PooledTerm {
    /// Mean of the estimates, `Q̄`.
    pub estimate: f64,
    /// `√T`, the total standard error.
    pub std_error: f64,
    /// `Q̄ / √T`.
    pub statistic: f64,
    /// Barnard–Rubin d.f. with a complete-data d.f., Rubin's otherwise.
    pub df: f64,
    /// Two-sided p-value of the t statistic on `df`.
    pub p_value: f64,
    pub conf_low: f64,
    pub conf_high: f64,
    /// Relative increase in variance `r = (1 + 1/m) B / Ū`.
    pub riv: f64,
    /// Proportion of the variance due to missingness `λ = (1 + 1/m) B / T`.
    pub lambda: f64,
    /// Fraction of missing information.
    pub fmi: f64,
}

/// The D1 pooled Wald test (`mice::D1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PooledWaldTest {
    pub f: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    /// Average relative increase in variance `r₁`.
    pub riv: f64,
}

/// Fits pooled by [`pool_rubin`].
#[derive(Debug, Clone, PartialEq)]
pub struct PooledResult {
    pub terms: Vec<PooledTerm>,
    /// Total covariance `T = Ū + (1 + 1/m) B`, row-major.
    pub vcov: Vec<f64>,
    /// D1 test of the coefficients `wald_from..dim`; `None` when there are
    /// none or the within-imputation covariance of them is singular.
    pub wald: Option<PooledWaldTest>,
}

/// Rubin's rules over `m ≥ 2` fits: `estimates[i]` (`dim` values) and
/// `vcovs[i]` (`dim × dim`, row-major) of imputation `i`. `df_complete` is
/// the complete-data residual d.f. (`mice`'s `dfcom`) for the Barnard–Rubin
/// d.f.; `None` gives Rubin's large-sample d.f.
pub fn pool_rubin(
    estimates: &[Vec<f64>],
    vcovs: &[Vec<f64>],
    df_complete: Option<f64>,
    confidence_level: f64,
    wald_from: usize,
) -> StatsResult<PooledResult> {
    let m = estimates.len();
    if m < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "pooling needs two or more imputations".into(),
        ));
    }
    if vcovs.len() != m {
        return Err(StatsError::DimensionMismatchMsg(
            "one covariance per imputation is needed".into(),
        ));
    }
    let dim = estimates[0].len();
    if dim == 0 {
        return Err(StatsError::InvalidInput(
            "the fits have no coefficients".into(),
        ));
    }
    if estimates.iter().any(|e| e.len() != dim) || vcovs.iter().any(|v| v.len() != dim * dim) {
        return Err(StatsError::DimensionMismatchMsg(
            "every fit must have the same coefficients".into(),
        ));
    }
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(StatsError::InvalidInput(
            "confidence_level must be in (0, 1)".into(),
        ));
    }
    let cols: Vec<Col<f64>> = estimates
        .iter()
        .map(|e| Col::from_fn(dim, |j| e[j]))
        .collect();
    let mats: Vec<Mat<f64>> = vcovs
        .iter()
        .map(|v| Mat::from_fn(dim, dim, |a, b| v[a * dim + b]))
        .collect();
    let pooled = upstream_pool_rubin(&cols, &mats, df_complete)?;
    let se = pooled.std_errors();
    let df = pooled
        .df_barnard_rubin
        .clone()
        .unwrap_or_else(|| pooled.df_rubin.clone());
    let terms = (0..dim)
        .map(|j| {
            let est = Col::from_fn(1, |_| pooled.coefficients[j]);
            let sej = Col::from_fn(1, |_| se[j]);
            let t = CoefficientInference::t_statistics(&est, &sej);
            // A d.f. that is not finite (the imputations agree exactly, B = 0)
            // leaves the p-value and interval undefined (NaN): upstream's
            // t quantile does not terminate for an infinite d.f.
            let nan = Col::from_fn(1, |_| f64::NAN);
            let (p, lo, hi) = if df[j].is_finite() {
                let p = CoefficientInference::p_values(&t, df[j]);
                let (lo, hi) =
                    CoefficientInference::confidence_intervals(&est, &sej, df[j], confidence_level);
                (p, lo, hi)
            } else {
                (nan.clone(), nan.clone(), nan)
            };
            PooledTerm {
                estimate: pooled.coefficients[j],
                std_error: se[j],
                statistic: t[0],
                df: df[j],
                p_value: p[0],
                conf_low: lo[0],
                conf_high: hi[0],
                riv: pooled.riv[j],
                lambda: pooled.lambda[j],
                fmi: pooled.fmi[j],
            }
        })
        .collect();
    let vcov = (0..dim * dim)
        .map(|k| pooled.vcov[(k / dim, k % dim)])
        .collect();
    let indices: Vec<usize> = (wald_from..dim).collect();
    let wald = if indices.is_empty() {
        None
    } else {
        pooled_wald_d1(&pooled, &indices)
            .ok()
            .map(|w| PooledWaldTest {
                f: w.f,
                df1: w.df1,
                df2: w.df2,
                p_value: w.p_value,
                riv: w.riv,
            })
    };
    Ok(PooledResult { terms, vcov, wald })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Vec<f64>> {
        // y, x with a few missing x
        (0..40)
            .map(|i| {
                let x = (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.1;
                let y = 1.0 + 0.5 * x + ((i * 7 % 11) as f64 - 5.0) * 0.2;
                vec![y, if i % 6 == 2 { f64::NAN } else { x }]
            })
            .collect()
    }

    #[test]
    fn areg_impute_completes_and_keeps_observed() {
        let rows = data();
        let kinds = vec![ImputeVariable::Numeric; 2];
        let cfg = AregImputeConfig {
            n_impute: 3,
            ..Default::default()
        };
        let r = areg_impute(&rows, &kinds, &cfg).unwrap();
        assert_eq!(r.completed.len(), 3);
        for c in &r.completed {
            for (i, row) in c.iter().enumerate() {
                assert!(row.iter().all(|v| v.is_finite()));
                assert_eq!(row[0], rows[i][0]);
                if !rows[i][1].is_nan() {
                    assert_eq!(row[1], rows[i][1]);
                }
            }
        }
        // same seed, same draws
        assert_eq!(
            r.completed,
            areg_impute(&rows, &kinds, &cfg).unwrap().completed
        );
        assert!(r.r_squared[0].is_nan() && r.r_squared[1].is_finite());
    }

    #[test]
    fn areg_impute_rejects_bad_input() {
        let rows = data();
        assert!(matches!(
            areg_impute(&rows, &[ImputeVariable::Numeric], &Default::default()),
            Err(StatsError::InvalidInput(_))
        ));
        let mut constant = rows.clone();
        constant.iter_mut().for_each(|r| r[0] = 1.0);
        let kinds = vec![ImputeVariable::Numeric; 2];
        assert!(matches!(
            areg_impute(&constant, &kinds, &Default::default()),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }

    #[test]
    fn pool_matches_upstream() {
        let est = vec![vec![1.0, 2.0], vec![1.2, 1.9], vec![0.9, 2.2]];
        let v = vec![vec![0.04, 0.01, 0.01, 0.09]; 3];
        let r = pool_rubin(&est, &v, Some(50.0), 0.95, 1).unwrap();
        assert!((r.terms[0].estimate - 31.0 / 30.0).abs() < 1e-12);
        let cols: Vec<Col<f64>> = est.iter().map(|e| Col::from_fn(2, |j| e[j])).collect();
        let mats: Vec<Mat<f64>> = v
            .iter()
            .map(|x| Mat::from_fn(2, 2, |a, b| x[a * 2 + b]))
            .collect();
        let up = upstream_pool_rubin(&cols, &mats, Some(50.0)).unwrap();
        assert_eq!(r.terms[1].df, up.df_barnard_rubin.unwrap()[1]);
        assert_eq!(r.terms[1].fmi, up.fmi[1]);
        assert_eq!(r.vcov[1], up.vcov[(0, 1)]);
        assert!(r.terms[1].conf_low < r.terms[1].estimate);
        let w = r.wald.unwrap();
        assert_eq!(w.df1, 1.0);
        assert!(pool_rubin(&est[..1], &v[..1], None, 0.95, 0).is_err());
        assert!(pool_rubin(&est, &v, None, 0.95, 2).unwrap().wald.is_none());
    }

    #[test]
    fn pool_identical_imputations_terminates() {
        // B = 0: Rubin's d.f. is infinite; p-value and interval are left undefined
        let est = vec![vec![1.0, 2.0]; 3];
        let v = vec![vec![0.04, 0.0, 0.0, 0.09]; 3];
        let r = pool_rubin(&est, &v, None, 0.95, 1).unwrap();
        assert!(r.terms[1].df.is_infinite());
        assert!(r.terms[1].p_value.is_nan() && r.terms[1].conf_low.is_nan());
        assert!((r.terms[1].std_error - 0.3).abs() < 1e-12);
    }
}
