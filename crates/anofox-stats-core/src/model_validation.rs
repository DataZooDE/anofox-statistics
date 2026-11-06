//! Resampling model validation and calibration (#318): `rms::validate`,
//! `rms::calibrate` (linear and binary logistic models, and `calibrate.cph`
//! for Cox models) and `rms::bootcov`.
//!
//! Everything statistical — the resampling engine, the indexes, the
//! calibration curves, the bootstrap covariance and the seeded resample draws —
//! lives in [`anofox_regression::validation`]. This module only lays the
//! column-major rows out as upstream's design matrix, builds the resamples
//! from a seed, forwards the call and maps the error type onto [`StatsError`].
//!
//! The resamples are drawn by row position, so callers pass the rows in a
//! canonical order (the extension sorts them) to make a seed reproducible
//! whatever the thread count.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::{FittedRegressor, LogisticRegressor, OlsRegressor, Regressor};
use anofox_regression::survival::cox::{CoxOptions, Ties};
use anofox_regression::validation::{
    self as upstream, BootCov, Calibration, Resamples, SurvivalCalibration, ValidationError,
    ValidationTable,
};
use faer::{Col, Mat};

/// The model being validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationModel {
    /// Linear model (`rms::ols`).
    Ols,
    /// Binary logistic model (`rms::lrm`), `y` in {0, 1}.
    Logistic,
}

/// The resampling scheme (`rms::validate(method = ...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleScheme {
    /// Efron–Gong optimism bootstrap (`"boot"`).
    Bootstrap,
    /// B-fold cross-validation (`"crossvalidation"`).
    Crossvalidation,
    /// Efron's .632 bootstrap (`".632"`).
    Point632,
}

/// How many resamples and with which seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleSpec {
    pub scheme: ResampleScheme,
    /// Bootstrap repetitions, or the number of cross-validation groups.
    pub b: usize,
    pub seed: u64,
}

fn map_err(e: ValidationError) -> StatsError {
    match e {
        ValidationError::OriginalFit => {
            StatsError::InsufficientDataMsg("the model could not be fitted on the full data".into())
        }
        // Too few rows for the requested groups or grid: the data cannot
        // support the validation (NULL in SQL).
        ValidationError::Resamples(msg) if msg.contains("B <= n/2") => {
            StatsError::InsufficientDataMsg(format!("{msg} (too few rows for B groups)"))
        }
        ValidationError::Resamples(msg) if msg.contains("fewer than two") => {
            StatsError::InsufficientDataMsg(msg)
        }
        ValidationError::Data(msg)
            if msg.contains("need n >") || msg.contains("two or more cuts") =>
        {
            StatsError::InsufficientDataMsg(msg)
        }
        other => StatsError::InvalidInput(other.to_string()),
    }
}

/// Upstream's design matrix from column-major feature columns.
fn design(x: &[Vec<f64>], n: usize) -> StatsResult<Mat<f64>> {
    crate::validation::validate_x_columns(n, x)?;
    Ok(Mat::from_fn(n, x.len(), |i, j| x[j][i]))
}

fn check_input(y: &[f64], x: &[Vec<f64>], model: ValidationModel) -> StatsResult<Mat<f64>> {
    let n = y.len();
    if n == 0 {
        return Err(StatsError::NoValidData);
    }
    if x.is_empty() {
        return Err(StatsError::InvalidInput(
            "x must hold at least one feature".into(),
        ));
    }
    if model == ValidationModel::Logistic && y.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err(StatsError::InvalidInput(
            "the logistic model needs y in {0, 1}".into(),
        ));
    }
    if n <= x.len() + 1 {
        return Err(StatsError::InsufficientData {
            rows: n,
            cols: x.len(),
        });
    }
    design(x, n)
}

/// Seeded resamples of `n` rows.
pub fn resamples(n: usize, spec: &ResampleSpec) -> StatsResult<Resamples> {
    if spec.b == 0 {
        return Err(StatsError::InvalidInput("B must be at least 1".into()));
    }
    match spec.scheme {
        ResampleScheme::Bootstrap => Resamples::bootstrap_seeded(n, spec.b, spec.seed),
        ResampleScheme::Crossvalidation => Resamples::crossvalidation_seeded(n, spec.b, spec.seed),
        ResampleScheme::Point632 => Resamples::point632_seeded(n, spec.b, spec.seed),
    }
    .map_err(map_err)
}

/// `rms::validate` of a linear (`validate.ols`) or binary logistic
/// (`validate.lrm`) model of `y` on the columns of `x` (an intercept is
/// always fitted).
pub fn validate(
    model: ValidationModel,
    y: &[f64],
    x: &[Vec<f64>],
    spec: &ResampleSpec,
) -> StatsResult<ValidationTable> {
    let xm = check_input(y, x, model)?;
    let r = resamples(y.len(), spec)?;
    match model {
        ValidationModel::Ols => upstream::validate_ols(&xm, y, &r),
        ValidationModel::Logistic => upstream::validate_lrm(&xm, y, &r),
    }
    .map_err(map_err)
}

/// `rms::calibrate` of a linear or binary logistic model: the
/// overfitting-corrected lowess calibration curve at `predy` (default: 50
/// points between the 5th smallest and 5th largest prediction).
pub fn calibrate(
    model: ValidationModel,
    y: &[f64],
    x: &[Vec<f64>],
    spec: &ResampleSpec,
    predy: Option<Vec<f64>>,
) -> StatsResult<Calibration> {
    let xm = check_input(y, x, model)?;
    if let Some(p) = &predy {
        if p.is_empty() || p.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::InvalidInput(
                "predy must be a non-empty list of finite values".into(),
            ));
        }
    }
    let r = resamples(y.len(), spec)?;
    match model {
        ValidationModel::Ols => upstream::calibrate_ols(&xm, y, &r, predy),
        ValidationModel::Logistic => upstream::calibrate_lrm(&xm, y, &r, predy),
    }
    .map_err(map_err)
}

/// Options of [`calibrate_cph`].
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateCphSpec {
    /// Time horizon at which survival is calibrated (`u`).
    pub u: f64,
    /// About `m` subjects per group of predicted survival.
    pub m: usize,
    /// Explicit group boundaries on the survival scale (`cuts`).
    pub cuts: Option<Vec<f64>>,
    /// Breslow (`true`) or Efron ties.
    pub breslow: bool,
    pub max_iter: usize,
    pub eps: f64,
}

/// `rms::calibrate.cph(cmethod = "KM")`: calibration of a Cox model's
/// predicted survival at `u` against Kaplan–Meier estimates, corrected for
/// overfitting. `event` is 1 for an event and 0 for censoring.
pub fn calibrate_cph(
    time: &[f64],
    x: &[Vec<f64>],
    event: &[f64],
    spec: &ResampleSpec,
    cph: &CalibrateCphSpec,
) -> StatsResult<SurvivalCalibration> {
    let n = time.len();
    if n == 0 {
        return Err(StatsError::NoValidData);
    }
    if x.is_empty() {
        return Err(StatsError::InvalidInput(
            "x must hold at least one feature".into(),
        ));
    }
    if event.len() != n {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "time has {n} values, event has {}",
            event.len()
        )));
    }
    if event.iter().any(|&e| e != 0.0 && e != 1.0) {
        return Err(StatsError::InvalidInput(
            "event must be 1 (event) or 0 (censored)".into(),
        ));
    }
    if !(cph.u.is_finite() && cph.u > 0.0) {
        return Err(StatsError::InvalidInput(format!(
            "u must be a positive time, got {}",
            cph.u
        )));
    }
    if cph.m == 0 {
        return Err(StatsError::InvalidInput("m must be at least 1".into()));
    }
    let xm = design(x, n)?;
    let ev: Vec<bool> = event.iter().map(|&e| e == 1.0).collect();
    let r = resamples(n, spec)?;
    let options = CoxOptions::default()
        .ties(if cph.breslow {
            Ties::Breslow
        } else {
            Ties::Efron
        })
        .max_iter(cph.max_iter)
        .eps(cph.eps);
    upstream::calibrate_cph(&xm, time, &ev, cph.u, &r, cph.m, cph.cuts.clone(), &options)
        .map_err(map_err)
}

/// Bootstrap covariance of a fit's coefficients (`rms::bootcov`), with the
/// original coefficients and percentile intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct BootcovResult {
    /// Coefficients of the fit on all rows: intercept first, then one per
    /// column of `x`.
    pub estimate: Vec<f64>,
    /// Upstream's bootstrap replicates, mean and covariance.
    pub boot: BootCov,
    /// Percentile intervals of every coefficient.
    pub conf_low: Vec<f64>,
    pub conf_high: Vec<f64>,
}

/// Coefficients (intercept first) of upstream's linear or logistic fit on
/// `rows` (with repetitions); `None` when the fit fails or a column is aliased.
fn coefficients(
    model: ValidationModel,
    x: &Mat<f64>,
    y: &[f64],
    rows: &[usize],
) -> Option<Vec<f64>> {
    let xr = Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]);
    let yr = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let (intercept, beta) = match model {
        ValidationModel::Ols => {
            let f = OlsRegressor::builder().build().fit(&xr, &yr).ok()?;
            let r = f.result();
            if r.aliased.iter().any(|&a| a) {
                return None;
            }
            (r.intercept?, r.coefficients.clone())
        }
        // The settings of upstream's own logistic refits in validate_lrm.
        ValidationModel::Logistic => {
            let f = LogisticRegressor::builder()
                .tolerance(1e-12)
                .max_iterations(100)
                .build()
                .fit(&xr, &yr)
                .ok()?;
            (f.intercept()?, f.coefficients().clone())
        }
    };
    let mut out = Vec::with_capacity(beta.nrows() + 1);
    out.push(intercept);
    out.extend(beta.iter().copied());
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// `rms::bootcov` of a linear or binary logistic model: `b` seeded ordinary
/// bootstrap resamples, or, with `cluster` (a dense code per row), cluster
/// bootstrap resamples of whole clusters.
pub fn bootcov(
    model: ValidationModel,
    y: &[f64],
    x: &[Vec<f64>],
    cluster: Option<&[usize]>,
    b: usize,
    seed: u64,
    conf_level: f64,
) -> StatsResult<BootcovResult> {
    let xm = check_input(y, x, model)?;
    if b < 2 {
        return Err(StatsError::InvalidInput("B must be at least 2".into()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(StatsError::InvalidInput(format!(
            "conf_level must be in (0, 1), got {conf_level}"
        )));
    }
    let n = y.len();
    let samples = match cluster {
        Some(c) => {
            if c.len() != n {
                return Err(StatsError::DimensionMismatchMsg(format!(
                    "{n} rows but {} cluster codes",
                    c.len()
                )));
            }
            upstream::cluster_bootstrap_rows(c, b, seed)
        }
        None => upstream::bootstrap_rows(n, b, seed),
    };
    let all: Vec<usize> = (0..n).collect();
    let estimate = coefficients(model, &xm, y, &all).ok_or(StatsError::SingularMatrix)?;
    let boot =
        upstream::bootcov(&samples, |rows| coefficients(model, &xm, y, rows)).map_err(map_err)?;
    let (conf_low, conf_high) = boot.percentile_ci(conf_level).into_iter().unzip();
    Ok(BootcovResult {
        estimate,
        boot,
        conf_low,
        conf_high,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let x1: Vec<f64> = (0..n).map(|i| ((i * 37) % 23) as f64 / 7.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 11) % 17) as f64 / 5.0).collect();
        let yc: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.8 * x1[i] - 0.5 * x2[i] + ((i * 7) % 13) as f64 / 6.0)
            .collect();
        let yb: Vec<f64> = (0..n)
            .map(|i| f64::from(x1[i] - 0.5 * x2[i] + ((i * 5) % 9) as f64 / 3.0 > 1.5))
            .collect();
        (yc, yb, vec![x1, x2])
    }

    fn spec(scheme: ResampleScheme, b: usize) -> ResampleSpec {
        ResampleSpec { scheme, b, seed: 7 }
    }

    #[test]
    fn validate_reproduces_and_corrects() {
        let (yc, yb, x) = data(120);
        let s = spec(ResampleScheme::Bootstrap, 30);
        let a = validate(ValidationModel::Ols, &yc, &x, &s).unwrap();
        assert_eq!(a, validate(ValidationModel::Ols, &yc, &x, &s).unwrap());
        let (orig, _, _, opt, corr, n) = a.get("R-square").unwrap();
        assert!((orig - opt - corr).abs() < 1e-12);
        assert_eq!(n, 30);
        let l = validate(ValidationModel::Logistic, &yb, &x, &s).unwrap();
        assert_eq!(l.names.len(), 11);
        assert!(validate(
            ValidationModel::Ols,
            &yc,
            &x,
            &spec(ResampleScheme::Crossvalidation, 10)
        )
        .is_ok());
        assert!(validate(
            ValidationModel::Ols,
            &yc,
            &x,
            &spec(ResampleScheme::Point632, 40)
        )
        .is_ok());
        // .632 needs every row left out of some sample: an input error with too few samples.
        assert!(matches!(
            validate(
                ValidationModel::Ols,
                &yc,
                &x,
                &spec(ResampleScheme::Point632, 2)
            ),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn invalid_and_degenerate_inputs() {
        let (yc, _, x) = data(40);
        assert!(matches!(
            validate(
                ValidationModel::Logistic,
                &yc,
                &x,
                &spec(ResampleScheme::Bootstrap, 5)
            ),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            validate(
                ValidationModel::Ols,
                &yc,
                &x,
                &spec(ResampleScheme::Crossvalidation, 30)
            ),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            validate(
                ValidationModel::Ols,
                &yc[..3],
                &[x[0][..3].to_vec(), x[1][..3].to_vec()],
                &spec(ResampleScheme::Bootstrap, 5)
            ),
            Err(StatsError::InsufficientData { .. })
        ));
        assert!(validate(
            ValidationModel::Ols,
            &yc,
            &x,
            &spec(ResampleScheme::Bootstrap, 0)
        )
        .is_err());
    }

    #[test]
    fn calibrate_and_bootcov_run() {
        let (yc, yb, x) = data(100);
        let s = spec(ResampleScheme::Bootstrap, 20);
        let c = calibrate(ValidationModel::Logistic, &yb, &x, &s, None).unwrap();
        assert_eq!(c.predy.len(), 50);
        let g = calibrate(ValidationModel::Ols, &yc, &x, &s, Some(vec![1.0, 2.0, 3.0])).unwrap();
        assert_eq!(g.predy, vec![1.0, 2.0, 3.0]);
        let b = bootcov(ValidationModel::Ols, &yc, &x, None, 50, 3, 0.95).unwrap();
        assert_eq!(b.estimate.len(), 3);
        assert_eq!(b.boot.vcov.nrows(), 3);
        assert!(b.conf_low.iter().zip(&b.conf_high).all(|(l, h)| l <= h));
        let cl: Vec<usize> = (0..100).map(|i| i / 4).collect();
        let bc = bootcov(ValidationModel::Logistic, &yb, &x, Some(&cl), 30, 3, 0.9).unwrap();
        assert_eq!(bc.boot.failed + bc.boot.replicates.nrows(), 30);
    }

    #[test]
    fn calibrate_cph_runs() {
        let n = 120;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 37) % 23) as f64 / 7.0).collect();
        let time: Vec<f64> = (0..n)
            .map(|i| 1.0 + ((i * 13) % 29) as f64 / (1.0 + x1[i]))
            .collect();
        let event: Vec<f64> = (0..n).map(|i| f64::from(i % 4 != 0)).collect();
        let cph = CalibrateCphSpec {
            u: 5.0,
            m: 40,
            cuts: None,
            breslow: false,
            max_iter: 20,
            eps: 1e-9,
        };
        let c = calibrate_cph(
            &time,
            &[x1],
            &event,
            &spec(ResampleScheme::Bootstrap, 10),
            &cph,
        )
        .unwrap();
        assert_eq!(c.km.len() + 1, c.cuts.len());
        let bad = vec![2.0; n];
        assert!(calibrate_cph(
            &time,
            &[vec![0.0; n]],
            &bad,
            &spec(ResampleScheme::Bootstrap, 10),
            &cph
        )
        .is_err());
    }
}
