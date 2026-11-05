//! Generalized least squares for grouped / longitudinal data (#315,
//! `nlme::gls`).
//!
//! The model — `y = Xβ + ε` with `Var(ε_g) = σ² A_g R_g A_g` per subject `g`,
//! a within-subject correlation structure `R_g` (compound symmetry, AR(1),
//! continuous-time AR(1), exponential / Gaussian / linear / spherical with an
//! optional nugget, unstructured) and a variance function `A_g` (`varIdent`,
//! `varPower`, `varExp`), fitted by REML or ML — lives in
//! [`anofox_regression::solvers::gls`]. This module only drops the rows the fit
//! cannot use, forwards the call, maps the error type onto [`StatsError`] and
//! reshapes the fit.

use crate::errors::{StatsError, StatsResult};
pub use anofox_regression::solvers::gls::GlsResidualType;
use anofox_regression::solvers::gls::{
    FittedGls, GlsCorrelation, GlsMethod, GlsRegressor, GlsVariance,
};
use anofox_regression::solvers::{IterativeFit, RegressionError};
use faer::{Col, Mat};

/// Within-subject correlation structure (nlme `corStruct`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlsCorrelationKind {
    #[default]
    Independent,
    /// `corCompSymm`.
    CompoundSymmetry,
    /// `corAR1(form = ~ time | subject)`: integer times.
    Ar1,
    /// `corCAR1`: continuous times.
    ContinuousAr1,
    /// `corExp`.
    Exponential,
    /// `corGaus`.
    Gaussian,
    /// `corLin`.
    Linear,
    /// `corSpher`.
    Spherical,
    /// `corSymm`: a free correlation between every two distinct times.
    Unstructured,
}

impl GlsCorrelationKind {
    /// The nlme-style name used in the SQL result.
    pub fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::CompoundSymmetry => "compound_symmetry",
            Self::Ar1 => "ar1",
            Self::ContinuousAr1 => "car1",
            Self::Exponential => "exponential",
            Self::Gaussian => "gaussian",
            Self::Linear => "linear",
            Self::Spherical => "spherical",
            Self::Unstructured => "unstructured",
        }
    }

    fn is_spatial(self) -> bool {
        matches!(
            self,
            Self::Exponential | Self::Gaussian | Self::Linear | Self::Spherical
        )
    }
}

/// Variance function (nlme `varFunc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlsVarianceKind {
    #[default]
    Constant,
    /// `varIdent(form = ~ 1 | stratum)`: the covariate is a stratum code.
    Ident,
    /// `varPower(form = ~ v)`.
    Power,
    /// `varExp(form = ~ v)`.
    Exponential,
}

impl GlsVarianceKind {
    /// The name used in the SQL result.
    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Ident => "ident",
            Self::Power => "power",
            Self::Exponential => "exp",
        }
    }
}

/// Options of [`fit_gls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlsSpec {
    pub correlation: GlsCorrelationKind,
    /// Estimate a nugget (spatial structures only).
    pub nugget: bool,
    pub variance: GlsVarianceKind,
    /// REML (default) or ML.
    pub reml: bool,
    pub fit_intercept: bool,
    pub confidence_level: f64,
    pub max_iterations: usize,
}

impl Default for GlsSpec {
    fn default() -> Self {
        Self {
            correlation: GlsCorrelationKind::Independent,
            nugget: false,
            variance: GlsVarianceKind::Constant,
            reml: true,
            fit_intercept: true,
            confidence_level: 0.95,
            max_iterations: 500,
        }
    }
}

/// Per-row data of a GLS fit, one entry per input row (`x` column-major).
///
/// `subject` holds the caller's dense subject codes (negative: no subject);
/// `variance_covariate` the stratum code (`varIdent`) or the covariate
/// (`varPower` / `varExp`) and is ignored for a constant variance. Rows with a
/// non-finite value in a column the fit uses are left out. The upstream fit
/// sees the remaining rows in the order given: the subjects in order of first
/// appearance, each subject's rows in input order (the order of the normalized
/// residuals) and, for `varIdent`, the stratum of the first row as reference.
#[derive(Debug, Clone, Copy)]
pub struct GlsRows<'a> {
    pub y: &'a [f64],
    pub x: &'a [Vec<f64>],
    pub subject: &'a [i32],
    pub time: &'a [f64],
    pub variance_covariate: Option<&'a [f64]>,
}

/// One class of an empirical semivariogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlsVariogramBin {
    pub dist: f64,
    pub variog: f64,
    pub n_pairs: usize,
}

/// A fitted GLS model and the bookkeeping that maps it back onto the input rows.
#[derive(Debug, Clone)]
pub struct GlsModel {
    pub fit: FittedGls,
    pub spec: GlsSpec,
    /// Input row of each fitted row.
    pub used: Vec<usize>,
    /// Distinct subjects among the fitted rows.
    pub n_groups: usize,
    /// `varIdent` strata codes in parameter order: the reference first, then
    /// the stratum of every entry of [`FittedGls::variance_params`]. Empty for
    /// the other variance functions.
    pub strata: Vec<f64>,
    pub n_input: usize,
    n_features: usize,
}

/// Point estimates of a fit, flattened for the FFI.
#[derive(Debug, Clone)]
pub struct GlsSummary {
    /// Coefficients, intercept first when fitted.
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_low: Vec<f64>,
    pub conf_high: Vec<f64>,
    /// Row-major `p × p` covariance of the coefficients.
    pub vcov: Vec<f64>,
    pub sigma: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_params: usize,
    pub df_residual: f64,
    pub correlation_params: Vec<f64>,
    pub variance_params: Vec<f64>,
    /// Var-cov intervals (`intervals(fit, which = "var-cov")`); `None` when the
    /// Hessian is not negative definite.
    pub correlation_conf: Option<(Vec<f64>, Vec<f64>)>,
    pub variance_conf: Option<(Vec<f64>, Vec<f64>)>,
    pub sigma_conf: Option<(f64, f64)>,
    pub n_observations: usize,
    pub n_groups: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// Per-input-row results; NaN for rows the value does not exist for.
#[derive(Debug, Clone)]
pub struct GlsAugment {
    /// `x'β̂` for every row with finite x (training or not).
    pub fitted: Vec<f64>,
    /// Standard error of `x'β̂`.
    pub se_fit: Vec<f64>,
    pub residual: Vec<f64>,
    pub pearson: Vec<f64>,
    pub normalized: Vec<f64>,
    pub is_training: Vec<bool>,
}

/// Map upstream errors. Data that cannot support the fit (a structure without
/// the pairs it needs, a failed variance–covariance optimisation, a singular
/// design) becomes a degenerate error (NULL in SQL); the rest stay errors.
fn map_err(e: RegressionError) -> StatsError {
    match e {
        RegressionError::InvalidInput(msg)
            if msg.contains("needs a group with two or more")
                || msg.contains("needs two or more times") =>
        {
            StatsError::InsufficientDataMsg(msg)
        }
        RegressionError::InvalidInput(msg) if msg.contains("optimisation failed") => {
            StatsError::ConvergenceFailure {
                iterations: 0,
                tolerance: f64::NAN,
            }
        }
        other => StatsError::from(other),
    }
}

fn correlation(spec: &GlsSpec) -> StatsResult<GlsCorrelation> {
    if spec.nugget && !spec.correlation.is_spatial() {
        return Err(StatsError::InvalidValue {
            field: "nugget",
            message: format!(
                "only the spatial structures (exponential, gaussian, linear, spherical) have a \
                 nugget, not '{}'",
                spec.correlation.name()
            ),
        });
    }
    let nugget = spec.nugget;
    Ok(match spec.correlation {
        GlsCorrelationKind::Independent => GlsCorrelation::Independent,
        GlsCorrelationKind::CompoundSymmetry => GlsCorrelation::CompoundSymmetry,
        GlsCorrelationKind::Ar1 => GlsCorrelation::Ar1,
        GlsCorrelationKind::ContinuousAr1 => GlsCorrelation::ContinuousAr1,
        GlsCorrelationKind::Exponential => GlsCorrelation::Exponential { nugget },
        GlsCorrelationKind::Gaussian => GlsCorrelation::Gaussian { nugget },
        GlsCorrelationKind::Linear => GlsCorrelation::Linear { nugget },
        GlsCorrelationKind::Spherical => GlsCorrelation::Spherical { nugget },
        GlsCorrelationKind::Unstructured => GlsCorrelation::Unstructured,
    })
}

/// Fit `nlme::gls` on the usable rows of `rows`.
pub fn fit_gls(rows: &GlsRows, spec: &GlsSpec) -> StatsResult<GlsModel> {
    let n = rows.y.len();
    if rows.subject.len() != n
        || rows.time.len() != n
        || rows.x.iter().any(|c| c.len() != n)
        || rows.variance_covariate.is_some_and(|v| v.len() != n)
    {
        return Err(StatsError::DimensionMismatchMsg(
            "y, x, subject, time and the variance covariate must have one value per row".into(),
        ));
    }
    if !(spec.confidence_level > 0.0 && spec.confidence_level < 1.0) {
        return Err(StatsError::InvalidValue {
            field: "confidence_level",
            message: format!("must be in (0, 1), got {}", spec.confidence_level),
        });
    }
    if spec.max_iterations == 0 {
        return Err(StatsError::InvalidValue {
            field: "max_iterations",
            message: "must be at least 1".into(),
        });
    }
    let cor = correlation(spec)?;
    let var_cov = match spec.variance {
        GlsVarianceKind::Constant => None,
        _ => Some(rows.variance_covariate.ok_or_else(|| {
            StatsError::InvalidInput(format!(
                "variance '{}' needs a variance covariate",
                spec.variance.name()
            ))
        })?),
    };

    let used: Vec<usize> = (0..n)
        .filter(|&i| {
            rows.y[i].is_finite()
                && rows.subject[i] >= 0
                && rows.time[i].is_finite()
                && rows.x.iter().all(|c| c[i].is_finite())
                && var_cov.is_none_or(|v| v[i].is_finite())
        })
        .collect();
    if used.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let m = used.len();
    let p = rows.x.len();
    let xm = Mat::from_fn(m, p, |r, c| rows.x[c][used[r]]);
    let yv = Col::from_fn(m, |r| rows.y[used[r]]);
    let subject: Vec<usize> = used.iter().map(|&i| rows.subject[i] as usize).collect();
    let time: Vec<f64> = used.iter().map(|&i| rows.time[i]).collect();
    let mut n_groups_seen = subject.clone();
    n_groups_seen.sort_unstable();
    n_groups_seen.dedup();

    let mut strata = Vec::new();
    let variance = match (spec.variance, var_cov) {
        (GlsVarianceKind::Ident, Some(v)) => {
            let codes: Vec<f64> = used.iter().map(|&i| v[i]).collect();
            if codes.iter().any(|c| *c < 0.0 || c.fract() != 0.0) {
                return Err(StatsError::InvalidInput(
                    "varIdent stratum codes must be non-negative integers".into(),
                ));
            }
            // Strata in order of first appearance: upstream's parameter order.
            for &c in &codes {
                if !strata.contains(&c) {
                    strata.push(c);
                }
            }
            GlsVariance::Ident(codes.iter().map(|&c| c as usize).collect())
        }
        (GlsVarianceKind::Power, Some(v)) => {
            GlsVariance::Power(used.iter().map(|&i| v[i]).collect())
        }
        (GlsVarianceKind::Exponential, Some(v)) => {
            GlsVariance::Exponential(used.iter().map(|&i| v[i]).collect())
        }
        _ => GlsVariance::Constant,
    };

    let reg = GlsRegressor::builder()
        .correlation(cor)
        .variance(variance)
        .method(if spec.reml {
            GlsMethod::Reml
        } else {
            GlsMethod::Ml
        })
        .with_intercept(spec.fit_intercept)
        .confidence_level(spec.confidence_level)
        .max_iterations(spec.max_iterations)
        .build();
    let fit = reg.fit(&xm, &yv, &subject, Some(&time)).map_err(map_err)?;
    Ok(GlsModel {
        fit,
        spec: *spec,
        used,
        n_groups: n_groups_seen.len(),
        strata,
        n_input: n,
        n_features: p,
    })
}

fn col_vec(c: faer::ColRef<'_, f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

impl GlsModel {
    /// Coefficients, their inference, the variance–covariance parameters and
    /// the fit statistics.
    pub fn summary(&self) -> GlsSummary {
        let f = &self.fit;
        let p = f.coefficients().nrows();
        let (lo, hi) = f.conf_int();
        let vcov = f.vcov();
        let iv = f.var_cov_intervals();
        let split = |v: &[anofox_regression::solvers::gls::Interval]| {
            (
                v.iter().map(|i| i.lower).collect::<Vec<_>>(),
                v.iter().map(|i| i.upper).collect::<Vec<_>>(),
            )
        };
        GlsSummary {
            coefficients: col_vec(f.coefficients().as_ref()),
            std_errors: col_vec(f.std_errors().as_ref()),
            t_values: col_vec(f.t_values().as_ref()),
            p_values: col_vec(f.p_values().as_ref()),
            conf_low: col_vec(lo.as_ref()),
            conf_high: col_vec(hi.as_ref()),
            vcov: (0..p * p).map(|k| vcov[(k / p, k % p)]).collect(),
            sigma: f.sigma(),
            log_likelihood: f.log_likelihood(),
            aic: f.aic(),
            bic: f.bic(),
            n_params: f.n_params(),
            df_residual: f.df_residual(),
            correlation_params: col_vec(f.correlation_params()),
            variance_params: col_vec(f.variance_params()),
            correlation_conf: iv.map(|i| split(&i.correlation)),
            variance_conf: iv.map(|i| split(&i.variance)),
            sigma_conf: iv.map(|i| (i.sigma.lower, i.sigma.upper)),
            n_observations: self.used.len(),
            n_groups: self.n_groups,
            iterations: f.iterations(),
            converged: f.converged(),
        }
    }

    /// Empirical semivariogram of the residuals against the within-subject time
    /// distance (`nlme::Variogram`), pooled into `nint` quantile classes.
    pub fn variogram(
        &self,
        time: &[f64],
        residuals: GlsResidualType,
        nint: usize,
    ) -> StatsResult<Vec<GlsVariogramBin>> {
        if time.len() != self.n_input {
            return Err(StatsError::DimensionMismatchMsg(
                "time must have one value per input row".into(),
            ));
        }
        let t: Vec<f64> = self.used.iter().map(|&i| time[i]).collect();
        let bins = self.fit.variogram(&t, residuals, nint).map_err(map_err)?;
        Ok(bins
            .iter()
            .map(|b| GlsVariogramBin {
                dist: b.dist,
                variog: b.variog,
                n_pairs: b.n_pairs,
            })
            .collect())
    }

    /// Per-input-row fitted values with their standard errors (every row with
    /// finite x) and the residuals of the training rows.
    pub fn augment(&self, x: &[Vec<f64>]) -> StatsResult<GlsAugment> {
        let n = self.n_input;
        if x.len() != self.n_features || x.iter().any(|c| c.len() != n) {
            return Err(StatsError::DimensionMismatchMsg(
                "x must have the fitted features and one value per input row".into(),
            ));
        }
        let rows: Vec<usize> = (0..n)
            .filter(|&i| x.iter().all(|c| c[i].is_finite()))
            .collect();
        let xm = Mat::from_fn(rows.len(), self.n_features, |r, c| x[c][rows[r]]);
        let (mean, se) = self.fit.predict_with_se(&xm);
        let mut out = GlsAugment {
            fitted: vec![f64::NAN; n],
            se_fit: vec![f64::NAN; n],
            residual: vec![f64::NAN; n],
            pearson: vec![f64::NAN; n],
            normalized: vec![f64::NAN; n],
            is_training: vec![false; n],
        };
        for (r, &i) in rows.iter().enumerate() {
            out.fitted[i] = mean[r];
            out.se_fit[i] = se[r];
        }
        let resp = self.fit.residuals(GlsResidualType::Response);
        let pear = self.fit.residuals(GlsResidualType::Pearson);
        let norm = self.fit.residuals(GlsResidualType::Normalized);
        for (r, &i) in self.used.iter().enumerate() {
            out.residual[i] = resp[r];
            out.pearson[i] = pear[r];
            out.normalized[i] = norm[r];
            out.is_training[i] = true;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> (Vec<f64>, Vec<Vec<f64>>, Vec<i32>, Vec<f64>) {
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let subject: Vec<i32> = (0..n).map(|i| (i / 6) as i32).collect();
        let time: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x[i] + 0.2 * subject[i] as f64 + ((i * 13) % 7) as f64 / 10.0)
            .collect();
        (y, vec![x], subject, time)
    }

    #[test]
    fn ar1_fit_and_augment() {
        let (y, x, subject, time) = panel();
        let rows = GlsRows {
            y: &y,
            x: &x,
            subject: &subject,
            time: &time,
            variance_covariate: None,
        };
        let spec = GlsSpec {
            correlation: GlsCorrelationKind::Ar1,
            ..GlsSpec::default()
        };
        let m = fit_gls(&rows, &spec).unwrap();
        let s = m.summary();
        assert_eq!(s.coefficients.len(), 2);
        assert_eq!(s.correlation_params.len(), 1);
        assert_eq!(s.n_groups, 10);
        let a = m.augment(&x).unwrap();
        assert!(a.is_training.iter().all(|&t| t));
        assert!(a.se_fit.iter().all(|v| *v > 0.0));
        let v = m.variogram(&time, GlsResidualType::Pearson, 20).unwrap();
        assert!(!v.is_empty());
    }

    #[test]
    fn rows_with_missing_values_are_dropped() {
        let (mut y, x, mut subject, time) = panel();
        y[0] = f64::NAN;
        subject[1] = -1;
        let rows = GlsRows {
            y: &y,
            x: &x,
            subject: &subject,
            time: &time,
            variance_covariate: None,
        };
        let m = fit_gls(&rows, &GlsSpec::default()).unwrap();
        assert_eq!(m.used.len(), 58);
        let a = m.augment(&x).unwrap();
        assert!(!a.is_training[0] && a.fitted[0].is_finite() && a.residual[0].is_nan());
    }

    #[test]
    fn ident_needs_a_covariate_and_nugget_needs_spatial() {
        let (y, x, subject, time) = panel();
        let rows = GlsRows {
            y: &y,
            x: &x,
            subject: &subject,
            time: &time,
            variance_covariate: None,
        };
        let spec = GlsSpec {
            variance: GlsVarianceKind::Ident,
            ..GlsSpec::default()
        };
        assert!(matches!(
            fit_gls(&rows, &spec),
            Err(StatsError::InvalidInput(_))
        ));
        let spec = GlsSpec {
            nugget: true,
            correlation: GlsCorrelationKind::Ar1,
            ..GlsSpec::default()
        };
        assert!(matches!(
            fit_gls(&rows, &spec),
            Err(StatsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn ident_strata_in_first_appearance_order() {
        let (y, x, subject, time) = panel();
        let strata: Vec<f64> = (0..60).map(|i| ((i + 1) % 3) as f64).collect();
        let rows = GlsRows {
            y: &y,
            x: &x,
            subject: &subject,
            time: &time,
            variance_covariate: Some(&strata),
        };
        let spec = GlsSpec {
            variance: GlsVarianceKind::Ident,
            ..GlsSpec::default()
        };
        let m = fit_gls(&rows, &spec).unwrap();
        assert_eq!(m.strata, vec![1.0, 2.0, 0.0]);
        assert_eq!(m.summary().variance_params.len(), 2);
    }
}
