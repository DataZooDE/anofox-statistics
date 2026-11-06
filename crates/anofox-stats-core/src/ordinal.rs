//! Ordinal regression (#313): upstream anofox-regression's cumulative link
//! model `OrdinalRegressor` (`rms::orm` / `MASS::polr`, five links, case
//! weights, an offset, a quadratic penalty, the proportional-odds score test)
//! and the continuation-ratio model `ContinuationRatioRegressor`
//! (`rms::cr.setup` + `lrm`).
//!
//! The wrappers fit upstream and hand back the estimates and fit statistics.
//!
//! Prediction from a stored model ([`ordinal_predict`],
//! [`continuation_ratio_predict`]): upstream's `FittedOrdinal` and
//! `FittedContinuationRatio` cannot be rebuilt from their parts (private
//! fields, no constructor), so a stored model is scored here by composing
//! upstream's public link functions (`OrdinalLink::cdf` / `sf`) with the
//! stored intercepts and slopes, exactly as upstream's
//! `exceedance_probabilities` / `category_probabilities` / `predict_mean` /
//! `conditional_probabilities` do. The unit tests check the composition
//! against those upstream methods on the fitted model. Predicted quantiles
//! (`predict_quantile[_ci]`) and the delta-method SE of the mean
//! (`predict_mean_with_se`) need the fitted object (its covariance and the
//! private `quantile_of` interpolation) and are not offered for stored models.

use crate::{StatsError, StatsResult};
use anofox_regression::solvers::{
    ContinuationRatioRegressor, FittedRegressor, IterativeFit, OrdinalLink, OrdinalRegressor,
};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal};

/// Response levels up to which the proportional-odds score test is computed
/// by default (its information matrix is dense in the number of levels).
pub const PO_TEST_DEFAULT_MAX_LEVELS: usize = 50;
/// Response levels up to which the intercepts' standard errors are reported
/// (they need the dense `(k - 1 + p)²` covariance).
pub const INTERCEPT_SE_MAX_LEVELS: usize = 1000;

/// The link of a cumulative link model by its `rms::orm` family name
/// (`logistic`, `probit`, `loglog`, `cloglog`, `cauchit`; `logit` is an
/// alias of `logistic`).
pub fn parse_ordinal_link(name: &str) -> Option<OrdinalLink> {
    match name.to_ascii_lowercase().as_str() {
        "logistic" | "logit" => Some(OrdinalLink::Logistic),
        "probit" => Some(OrdinalLink::Probit),
        "loglog" => Some(OrdinalLink::LogLog),
        "cloglog" => Some(OrdinalLink::CLogLog),
        "cauchit" => Some(OrdinalLink::Cauchit),
        _ => None,
    }
}

/// The FFI code of a link: 0 logistic, 1 probit, 2 loglog, 3 cloglog,
/// 4 cauchit.
pub fn ordinal_link_from_code(code: u32) -> Option<OrdinalLink> {
    match code {
        0 => Some(OrdinalLink::Logistic),
        1 => Some(OrdinalLink::Probit),
        2 => Some(OrdinalLink::LogLog),
        3 => Some(OrdinalLink::CLogLog),
        4 => Some(OrdinalLink::Cauchit),
        _ => None,
    }
}

/// Fit options of [`fit_ordinal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdinalOptions {
    pub link: OrdinalLink,
    /// `orm`'s `penalty = λ`: `λ · diag(var(x_j))` on the slopes; 0: none.
    pub penalty: f64,
    /// 0-based column of `x` used as an offset (coefficient fixed at 1,
    /// removed from the design).
    pub offset_column: Option<usize>,
    pub max_iterations: usize,
    pub tolerance: f64,
    /// Compute the proportional-odds score test: `None` when the response has
    /// at most [`PO_TEST_DEFAULT_MAX_LEVELS`] levels.
    pub po_test: Option<bool>,
}

impl Default for OrdinalOptions {
    fn default() -> Self {
        Self {
            link: OrdinalLink::Logistic,
            penalty: 0.0,
            offset_column: None,
            max_iterations: 100,
            tolerance: 1e-10,
            po_test: None,
        }
    }
}

/// A fitted cumulative link model (`rms::orm` parametrisation
/// `P(Y ≥ y_{j+1} | x) = F(α_j + xᵀβ)`).
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalFit {
    pub levels: Vec<f64>,
    pub counts: Vec<f64>,
    pub intercepts: Vec<f64>,
    /// `None` above [`INTERCEPT_SE_MAX_LEVELS`] levels.
    pub intercept_std_errors: Option<Vec<f64>>,
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub log_likelihood: f64,
    pub penalized_log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub lr_chisq: f64,
    /// Effective d.f. of the slopes (their number without a penalty).
    pub lr_df: f64,
    pub lr_p_value: f64,
    pub aic: f64,
    /// `(statistic, df, p_value)` of the proportional-odds score test.
    pub po_test: Option<(f64, f64, f64)>,
    pub n_observations: usize,
    /// Slopes (the columns of `x` without the offset column).
    pub n_features: usize,
    pub iterations: usize,
    pub converged: bool,
}

fn check_columns(y: &[f64], x_columns: &[Vec<f64>], weights: Option<&[f64]>) -> StatsResult<()> {
    let n = y.len();
    if x_columns.iter().any(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "every x column must have one value per y".into(),
        ));
    }
    if weights.is_some_and(|w| w.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "weights must have one value per y".into(),
        ));
    }
    Ok(())
}

fn distinct_levels(y: &[f64]) -> usize {
    let mut v = y.to_vec();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v.len()
}

/// Wald `z = β / SE` and two-sided normal p-values of the upstream estimates
/// and standard errors.
fn wald(coefficients: &[f64], std_errors: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let normal = Normal::standard();
    coefficients
        .iter()
        .zip(std_errors)
        .map(|(b, se)| {
            let z = if *se > 0.0 && se.is_finite() {
                b / se
            } else {
                f64::NAN
            };
            let p = if z.is_finite() {
                2.0 * normal.sf(z.abs())
            } else {
                f64::NAN
            };
            (z, p)
        })
        .unzip()
}

/// Fit a cumulative link model to the ordered response `y` (its distinct
/// values are the levels) and the columns `x_columns`, with optional case
/// `weights`. Fewer than two rows or levels is
/// [`StatsError::InsufficientDataMsg`] (a NULL in SQL).
pub fn fit_ordinal(
    y: &[f64],
    x_columns: &[Vec<f64>],
    weights: Option<&[f64]>,
    options: &OrdinalOptions,
) -> StatsResult<OrdinalFit> {
    check_columns(y, x_columns, weights)?;
    let n = y.len();
    let k = distinct_levels(y);
    if n < 2 || k < 2 {
        return Err(StatsError::InsufficientDataMsg(format!(
            "ordinal regression needs at least two rows and two response levels, got {n} rows and {k} levels"
        )));
    }
    if !(options.penalty.is_finite() && options.penalty >= 0.0) {
        return Err(StatsError::InvalidInput(format!(
            "penalty must be finite and >= 0, got {}",
            options.penalty
        )));
    }
    if let Some(c) = options.offset_column {
        if c >= x_columns.len() {
            return Err(StatsError::InvalidInput(format!(
                "offset column {} is out of range for {} x columns",
                c + 1,
                x_columns.len()
            )));
        }
    }
    let features: Vec<&Vec<f64>> = x_columns
        .iter()
        .enumerate()
        .filter(|(j, _)| Some(*j) != options.offset_column)
        .map(|(_, c)| c)
        .collect();
    let p = features.len();
    let x = Mat::from_fn(n, p, |i, j| features[j][i]);
    let yc = Col::from_fn(n, |i| y[i]);
    let mut builder = OrdinalRegressor::builder()
        .link(options.link)
        .max_iterations(options.max_iterations)
        .tolerance(options.tolerance);
    if let Some(w) = weights {
        builder = builder.weights(Col::from_fn(n, |i| w[i]));
    }
    if let Some(c) = options.offset_column {
        builder = builder.offset(Col::from_fn(n, |i| x_columns[c][i]));
    }
    if options.penalty > 0.0 {
        builder = builder.penalty(options.penalty);
    }
    let fitted = builder.build().fit(&x, &yc)?;

    let coefficients: Vec<f64> = fitted.coefficients().iter().copied().collect();
    let std_errors: Vec<f64> = fitted.std_errors().iter().copied().collect();
    let (z_values, p_values) = wald(&coefficients, &std_errors);
    let m = fitted.intercepts().nrows();
    let intercept_std_errors = (m < INTERCEPT_SE_MAX_LEVELS).then(|| {
        let v = fitted.vcov();
        (0..m).map(|j| v[(j, j)].max(0.0).sqrt()).collect()
    });
    let want_po = options.po_test.unwrap_or(m < PO_TEST_DEFAULT_MAX_LEVELS);
    // The test has (k - 2) d.f. per slope: none with two levels or no slopes.
    let po_test = if want_po && m >= 2 && p > 0 {
        let t = fitted.proportional_odds_score_test(&x, &yc, &[])?;
        Some((t.statistic, t.df as f64, t.p_value))
    } else {
        None
    };
    Ok(OrdinalFit {
        levels: fitted.levels().to_vec(),
        counts: fitted.counts().to_vec(),
        intercepts: fitted.intercepts().iter().copied().collect(),
        intercept_std_errors,
        coefficients,
        std_errors,
        z_values,
        p_values,
        log_likelihood: fitted.log_likelihood(),
        penalized_log_likelihood: fitted.penalized_log_likelihood(),
        null_log_likelihood: fitted.null_log_likelihood(),
        lr_chisq: fitted.lr_chisq(),
        lr_df: fitted.effective_df(),
        lr_p_value: fitted.lr_p_value(),
        aic: fitted.aic(),
        po_test,
        n_observations: fitted.n_observations(),
        n_features: p,
        iterations: fitted.iterations(),
        converged: fitted.converged(),
    })
}

/// A fitted continuation-ratio model.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationRatioFit {
    pub levels: Vec<f64>,
    /// Coefficients of the logistic fit on the expanded data (`lrm(y ~ cohort
    /// + x + cohort:x_j)` on `cr.setup` data): the intercept θ₁, the cohort
    /// effects `θ_j − θ₁` (`j = 2, …, k − 1`), the slopes γ, then for each
    /// cohort-specific column its slope differences in cohorts `2, …, k − 1`.
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    /// Subjects (rows of the original data).
    pub n_observations: usize,
    /// Records of the expanded data.
    pub n_records: usize,
    pub n_features: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// Fit the continuation-ratio model to the ordered response `y` (at least
/// three levels) and the columns `x_columns`, with optional case `weights`;
/// the 0-based columns `cohort_specific` get a slope per cohort. Fewer than
/// three levels is [`StatsError::InsufficientDataMsg`] (a NULL in SQL).
pub fn fit_continuation_ratio(
    y: &[f64],
    x_columns: &[Vec<f64>],
    weights: Option<&[f64]>,
    cohort_specific: &[usize],
) -> StatsResult<ContinuationRatioFit> {
    check_columns(y, x_columns, weights)?;
    let n = y.len();
    let k = distinct_levels(y);
    if k < 3 {
        return Err(StatsError::InsufficientDataMsg(format!(
            "a continuation-ratio model needs at least three response levels, got {k}"
        )));
    }
    let p = x_columns.len();
    if let Some(&c) = cohort_specific.iter().find(|&&c| c >= p) {
        return Err(StatsError::InvalidInput(format!(
            "cohort_specific column {} is out of range for {p} x columns",
            c + 1
        )));
    }
    let x = Mat::from_fn(n, p, |i, j| x_columns[j][i]);
    let yc = Col::from_fn(n, |i| y[i]);
    let mut builder = ContinuationRatioRegressor::builder().cohort_specific(cohort_specific);
    if let Some(w) = weights {
        builder = builder.weights(w);
    }
    let fitted = builder.build().fit(&x, &yc)?;
    let logistic = fitted.logistic();
    let r = logistic.result();
    let coefficients: Vec<f64> = fitted.coefficients().iter().copied().collect();
    let nan = |v: Option<f64>| v.unwrap_or(f64::NAN);
    let col = |v: &Option<Col<f64>>| -> Vec<f64> {
        let mut out = vec![f64::NAN; coefficients.len() - 1];
        if let Some(c) = v {
            for (o, x) in out.iter_mut().zip(c.iter()) {
                *o = *x;
            }
        }
        out
    };
    let mut std_errors = vec![nan(r.intercept_std_error)];
    std_errors.extend(col(&r.std_errors));
    let mut z_values = vec![nan(r.intercept_t_statistic)];
    z_values.extend(col(&r.t_statistics));
    let mut p_values = vec![nan(r.intercept_p_value)];
    p_values.extend(col(&r.p_values));
    Ok(ContinuationRatioFit {
        levels: fitted.levels().to_vec(),
        coefficients,
        std_errors,
        z_values,
        p_values,
        log_likelihood: fitted.log_likelihood(),
        aic: r.aic,
        n_observations: n,
        n_records: r.n_observations,
        n_features: p,
        iterations: fitted.iterations(),
        converged: fitted.converged(),
    })
}

/// Predictions of one row from a stored model.
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalPrediction {
    /// `P(Y = y_j | x)`, `j = 1, …, k`.
    pub probabilities: Vec<f64>,
    /// `P(Y ≥ y_{j+1} | x)`, `j = 1, …, k − 1`.
    pub exceedance: Vec<f64>,
    /// `Σ_j y_j P(Y = y_j | x)` (`rms::Mean.orm`).
    pub mean: f64,
    /// Index of the most probable level (the first on ties; `polr`'s
    /// `predict(type = "class")`).
    pub class: usize,
}

fn finish(levels: &[f64], probabilities: Vec<f64>, exceedance: Vec<f64>) -> OrdinalPrediction {
    let mean = probabilities.iter().zip(levels).map(|(p, y)| p * y).sum();
    let class = (0..probabilities.len()).fold(0, |b, j| {
        if probabilities[j] > probabilities[b] {
            j
        } else {
            b
        }
    });
    OrdinalPrediction {
        probabilities,
        exceedance,
        mean,
        class,
    }
}

/// Score one row `x` (the columns of the fit, including an offset column)
/// under a stored cumulative link model: `F(α_j + xᵀβ + offset)` with
/// upstream's link `F`, differenced from the tail that keeps precision as
/// upstream's `category_probabilities`.
pub fn ordinal_predict(
    link: OrdinalLink,
    levels: &[f64],
    intercepts: &[f64],
    coefficients: &[f64],
    offset_column: Option<usize>,
    x: &[f64],
) -> StatsResult<OrdinalPrediction> {
    let k = levels.len();
    if k < 2 || intercepts.len() + 1 != k {
        return Err(StatsError::InvalidInput(format!(
            "an ordinal model needs at least two levels and one intercept fewer than levels, got {k} levels and {} intercepts",
            intercepts.len()
        )));
    }
    let expected = coefficients.len() + usize::from(offset_column.is_some());
    if x.len() != expected {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "the model has {expected} x columns but x has {} values",
            x.len()
        )));
    }
    let mut eta = 0.0;
    let mut slope = coefficients.iter();
    for (j, v) in x.iter().enumerate() {
        if Some(j) == offset_column {
            eta += v;
        } else if let Some(b) = slope.next() {
            eta += b * v;
        }
    }
    let exceedance: Vec<f64> = intercepts.iter().map(|a| link.cdf(a + eta)).collect();
    let probabilities = (0..k)
        .map(|j| {
            let a = if j == 0 {
                f64::INFINITY
            } else {
                intercepts[j - 1] + eta
            };
            let b = if j + 1 == k {
                f64::NEG_INFINITY
            } else {
                intercepts[j] + eta
            };
            if b > 0.0 {
                link.sf(b) - link.sf(a)
            } else {
                link.cdf(a) - link.cdf(b)
            }
        })
        .collect();
    Ok(finish(levels, probabilities, exceedance))
}

/// Score one row `x` under a stored continuation-ratio model
/// ([`ContinuationRatioFit::coefficients`] layout, 0-based
/// `cohort_specific` columns): the conditional probabilities
/// `h_j = logit⁻¹(θ_j + xᵀγ + xᵀγ_j)` (upstream's logistic `F`) and
/// `P(Y = y_j) = h_j ∏_{c<j} (1 − h_c)` as upstream's
/// `category_probabilities`.
pub fn continuation_ratio_predict(
    levels: &[f64],
    coefficients: &[f64],
    cohort_specific: &[usize],
    x: &[f64],
) -> StatsResult<OrdinalPrediction> {
    let k = levels.len();
    let p = x.len();
    if k < 3 {
        return Err(StatsError::InvalidInput(format!(
            "a continuation-ratio model needs at least three levels, got {k}"
        )));
    }
    let m = k - 1;
    let expected = m + p + cohort_specific.len() * (m - 1);
    if coefficients.len() != expected {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "a continuation-ratio model of {k} levels with {} cohort-specific columns and an x of {p} values needs {expected} coefficients, got {}",
            cohort_specific.len(),
            coefficients.len()
        )));
    }
    if let Some(&c) = cohort_specific.iter().find(|&&c| c >= p) {
        return Err(StatsError::InvalidInput(format!(
            "cohort_specific column {} is out of range for an x of {p} values",
            c + 1
        )));
    }
    let slopes = &coefficients[m..m + p];
    let common: f64 = slopes.iter().zip(x).map(|(b, v)| b * v).sum();
    let link = OrdinalLink::Logistic;
    let h: Vec<f64> = (0..m)
        .map(|j| {
            let mut eta = coefficients[0] + common;
            if j > 0 {
                eta += coefficients[j];
                for (s, &c) in cohort_specific.iter().enumerate() {
                    eta += coefficients[m + p + s * (m - 1) + (j - 1)] * x[c];
                }
            }
            link.cdf(eta)
        })
        .collect();
    let mut probabilities = Vec::with_capacity(k);
    let mut survive = 1.0;
    for hj in &h {
        probabilities.push(hj * survive);
        survive *= 1.0 - hj;
    }
    probabilities.push(survive);
    // P(Y ≥ y_{j+1}) = ∏_{c ≤ j} (1 − h_c)
    let mut exceedance = Vec::with_capacity(m);
    let mut s = 1.0;
    for hj in &h {
        s *= 1.0 - hj;
        exceedance.push(s);
    }
    Ok(finish(levels, probabilities, exceedance))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 90;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7) as f64 * 0.37).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 3) as f64 * 0.91).cos()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s = x1[i] - 0.5 * x2[i] + 0.8 * ((i * 11) as f64).sin();
                if s < -0.6 {
                    1.0
                } else if s < 0.0 {
                    2.0
                } else if s < 0.6 {
                    4.0
                } else {
                    7.0
                }
            })
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn ordinal_predictions_match_upstream() {
        let (y, x) = data();
        let n = y.len();
        let xm = Mat::from_fn(n, 2, |i, j| x[j][i]);
        let yc = Col::from_fn(n, |i| y[i]);
        for link in [
            OrdinalLink::Logistic,
            OrdinalLink::Probit,
            OrdinalLink::LogLog,
            OrdinalLink::CLogLog,
            OrdinalLink::Cauchit,
        ] {
            let opts = OrdinalOptions {
                link,
                ..Default::default()
            };
            let fit = fit_ordinal(&y, &x, None, &opts).unwrap();
            assert!(fit.converged);
            assert_eq!(fit.levels, vec![1.0, 2.0, 4.0, 7.0]);
            assert!(fit.po_test.is_some());
            let up = OrdinalRegressor::builder()
                .link(link)
                .build()
                .fit(&xm, &yc)
                .unwrap();
            let probs = up.category_probabilities(&xm);
            let exceed = up.exceedance_probabilities(&xm);
            let mean = up.predict_mean(&xm);
            for i in 0..n {
                let pr = ordinal_predict(
                    link,
                    &fit.levels,
                    &fit.intercepts,
                    &fit.coefficients,
                    None,
                    &[x[0][i], x[1][i]],
                )
                .unwrap();
                for j in 0..4 {
                    assert!((pr.probabilities[j] - probs[(i, j)]).abs() < 1e-12);
                }
                for j in 0..3 {
                    assert!((pr.exceedance[j] - exceed[(i, j)]).abs() < 1e-12);
                }
                assert!((pr.mean - mean[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn offset_column_is_added_to_the_linear_predictor() {
        let (y, x) = data();
        let off: Vec<f64> = (0..y.len()).map(|i| 0.1 * (i % 5) as f64).collect();
        let cols = vec![x[0].clone(), off.clone(), x[1].clone()];
        let opts = OrdinalOptions {
            offset_column: Some(1),
            ..Default::default()
        };
        let fit = fit_ordinal(&y, &cols, None, &opts).unwrap();
        assert_eq!(fit.n_features, 2);
        let a = ordinal_predict(
            OrdinalLink::Logistic,
            &fit.levels,
            &fit.intercepts,
            &fit.coefficients,
            Some(1),
            &[0.3, 0.5, -0.2],
        )
        .unwrap();
        let b = ordinal_predict(
            OrdinalLink::Logistic,
            &fit.levels,
            &fit.intercepts.iter().map(|v| v + 0.5).collect::<Vec<_>>(),
            &fit.coefficients,
            None,
            &[0.3, -0.2],
        )
        .unwrap();
        for j in 0..4 {
            assert!((a.probabilities[j] - b.probabilities[j]).abs() < 1e-14);
        }
    }

    #[test]
    fn continuation_ratio_predictions_match_upstream() {
        let (y, x) = data();
        let n = y.len();
        let xm = Mat::from_fn(n, 2, |i, j| x[j][i]);
        let yc = Col::from_fn(n, |i| y[i]);
        for cs in [vec![], vec![1usize]] {
            let fit = fit_continuation_ratio(&y, &x, None, &cs).unwrap();
            assert_eq!(fit.coefficients.len(), 3 + 2 + cs.len() * 2);
            let up = ContinuationRatioRegressor::builder()
                .cohort_specific(&cs)
                .build()
                .fit(&xm, &yc)
                .unwrap();
            let probs = up.category_probabilities(&xm);
            for i in 0..n {
                let pr = continuation_ratio_predict(
                    &fit.levels,
                    &fit.coefficients,
                    &cs,
                    &[x[0][i], x[1][i]],
                )
                .unwrap();
                for j in 0..4 {
                    assert!((pr.probabilities[j] - probs[(i, j)]).abs() < 1e-12);
                }
                assert!((pr.exceedance[2] - pr.probabilities[3]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn degenerate_and_invalid_inputs() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        assert!(matches!(
            fit_ordinal(&[1.0, 1.0, 1.0], &x, None, &Default::default()),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            fit_continuation_ratio(&[1.0, 2.0, 1.0], &x, None, &[]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        let bad = OrdinalOptions {
            offset_column: Some(3),
            ..Default::default()
        };
        assert!(matches!(
            fit_ordinal(&[1.0, 2.0, 1.0], &x, None, &bad),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(fit_continuation_ratio(&[1.0, 2.0, 3.0], &x, None, &[2]).is_err());
        assert!(ordinal_predict(
            OrdinalLink::Logistic,
            &[1.0, 2.0],
            &[0.0],
            &[1.0],
            None,
            &[1.0, 2.0]
        )
        .is_err());
        assert!(continuation_ratio_predict(&[1.0, 2.0, 3.0], &[0.0, 0.0], &[], &[1.0]).is_err());
    }

    #[test]
    fn weights_equal_row_replication() {
        let (y, x) = data();
        let w: Vec<f64> = (0..y.len()).map(|i| (i % 3 + 1) as f64).collect();
        let a = fit_ordinal(&y, &x, Some(&w), &Default::default()).unwrap();
        let (mut yr, mut xr) = (Vec::new(), vec![Vec::new(), Vec::new()]);
        for (i, yi) in y.iter().enumerate() {
            for _ in 0..(i % 3 + 1) {
                yr.push(*yi);
                xr[0].push(x[0][i]);
                xr[1].push(x[1][i]);
            }
        }
        let b = fit_ordinal(&yr, &xr, None, &Default::default()).unwrap();
        for j in 0..2 {
            assert!((a.coefficients[j] - b.coefficients[j]).abs() < 1e-8);
            assert!((a.std_errors[j] - b.std_errors[j]).abs() < 1e-8);
        }
        assert!((a.log_likelihood - b.log_likelihood).abs() < 1e-8);
    }
}
