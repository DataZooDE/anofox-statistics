//! Cox proportional hazards regression (#312).
//!
//! The model — partial-likelihood Newton–Raphson with Efron / Breslow ties,
//! case weights, offsets, strata, counting-process `(entry, stop]` data, the
//! ridge penalty, Harrell's C, residuals, the robust covariance, the
//! Grambsch–Therneau test and predicted survival — lives in
//! [`anofox_regression::survival::cox`] (`survival::coxph`). This module only
//! drops the rows the fit cannot use, codes the strata and clusters, forwards
//! the call and maps the error type onto [`StatsError`].

use crate::errors::{StatsError, StatsResult};
use anofox_regression::survival::cox::{coxph, CoxFit, CoxInput, CoxOptions};
use anofox_regression::survival::SurvivalError;
use faer::Mat;
use statrs::distribution::{ContinuousCDF, Normal};

pub use anofox_regression::survival::cox::{Ties as CoxTies, ZphTransform};

/// Map upstream survival errors: no events, a singular information matrix or
/// no rows are data that cannot support a fit (NULL in SQL); the rest are
/// invalid arguments.
pub(crate) fn map_err(e: SurvivalError) -> StatsError {
    match e {
        SurvivalError::Empty => StatsError::NoValidData,
        SurvivalError::Invalid("no events") => {
            StatsError::InsufficientDataMsg("no events: every row is censored".into())
        }
        SurvivalError::Invalid(msg) if msg.contains("singular") => StatsError::SingularMatrix,
        SurvivalError::Length(a, b) => {
            StatsError::DimensionMismatchMsg(format!("inputs have lengths {a} and {b}"))
        }
        other => StatsError::InvalidInput(other.to_string()),
    }
}

/// Per-row data of a Cox fit, as the caller collected it (one entry per input
/// row; `x` column-major). Rows with a non-finite value anywhere are left out
/// of the fit.
#[derive(Debug, Clone, Copy)]
pub struct CoxRows<'a> {
    pub time: &'a [f64],
    pub event: &'a [f64],
    pub x: &'a [Vec<f64>],
    pub weights: Option<&'a [f64]>,
    pub entry: Option<&'a [f64]>,
    pub offset: Option<&'a [f64]>,
    pub strata: Option<&'a [f64]>,
    pub cluster: Option<&'a [f64]>,
}

/// Options of [`fit_cox`].
#[derive(Debug, Clone, Copy)]
pub struct CoxSpec {
    pub ties: CoxTies,
    pub max_iter: usize,
    pub eps: f64,
    pub ridge: f64,
    pub conf_level: f64,
    pub timefix: bool,
    /// Robust (sandwich) covariance, each row its own cluster unless the rows
    /// carry cluster codes (`coxph(robust = TRUE)` / `cluster = id`).
    pub robust: bool,
}

impl Default for CoxSpec {
    fn default() -> Self {
        Self {
            ties: CoxTies::Efron,
            max_iter: 20,
            eps: 1e-9,
            ridge: 0.0,
            conf_level: 0.95,
            timefix: true,
            robust: false,
        }
    }
}

/// A fitted Cox model with the bookkeeping that maps it back onto the input rows.
#[derive(Debug, Clone)]
pub struct CoxModel {
    pub fit: CoxFit,
    /// Input row of each fitted row.
    pub used: Vec<usize>,
    /// Stratum value of each upstream stratum code (`code -> value`); empty
    /// for an unstratified fit.
    pub strata_values: Vec<f64>,
    /// Stratum code of each fitted row (all 0 when unstratified).
    strata_codes: Vec<usize>,
    /// Robust covariance when requested.
    pub robust_var: Option<Mat<f64>>,
    pub conf_level: f64,
    pub ridge: f64,
    pub n_input: usize,
}

/// Dense codes of `values` (sorted distinct values -> 0, 1, ...).
pub(crate) fn codes(values: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let mut levels: Vec<f64> = values.to_vec();
    levels.sort_by(|a, b| a.total_cmp(b));
    levels.dedup();
    let c = values
        .iter()
        .map(|v| levels.partition_point(|l| l.total_cmp(v).is_lt()))
        .collect();
    (c, levels)
}

/// Fit `coxph` on the usable rows of `rows`.
pub fn fit_cox(rows: &CoxRows, spec: &CoxSpec) -> StatsResult<CoxModel> {
    let n = rows.time.len();
    let lens = [
        Some(rows.event.len()),
        rows.weights.map(<[f64]>::len),
        rows.entry.map(<[f64]>::len),
        rows.offset.map(<[f64]>::len),
        rows.strata.map(<[f64]>::len),
        rows.cluster.map(<[f64]>::len),
    ];
    if lens.iter().flatten().any(|&l| l != n) || rows.x.iter().any(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "every Cox input must have one value per row".into(),
        ));
    }
    if rows.x.is_empty() {
        return Err(StatsError::InvalidInput(
            "the Cox model needs at least one covariate".into(),
        ));
    }
    if !(spec.conf_level > 0.0 && spec.conf_level < 1.0) {
        return Err(StatsError::InvalidValue {
            field: "confidence_level",
            message: format!("must be in (0, 1), got {}", spec.conf_level),
        });
    }
    if !(spec.ridge >= 0.0 && spec.ridge.is_finite()) {
        return Err(StatsError::InvalidValue {
            field: "ridge",
            message: format!("must be finite and >= 0, got {}", spec.ridge),
        });
    }
    let finite = |o: Option<&[f64]>, i: usize| o.is_none_or(|v| v[i].is_finite());
    let used: Vec<usize> = (0..n)
        .filter(|&i| {
            rows.time[i].is_finite()
                && rows.event[i].is_finite()
                && rows.x.iter().all(|c| c[i].is_finite())
                && finite(rows.weights, i)
                && finite(rows.entry, i)
                && finite(rows.offset, i)
                && finite(rows.strata, i)
                && finite(rows.cluster, i)
        })
        .collect();
    if used.is_empty() {
        return Err(StatsError::NoValidData);
    }
    if let Some(&i) = used
        .iter()
        .find(|&&i| rows.event[i] != 0.0 && rows.event[i] != 1.0)
    {
        return Err(StatsError::InvalidValue {
            field: "event",
            message: format!("must be 0 (censored) or 1 (event), got {}", rows.event[i]),
        });
    }
    let pick = |v: &[f64]| -> Vec<f64> { used.iter().map(|&i| v[i]).collect() };
    let time = pick(rows.time);
    let event: Vec<bool> = used.iter().map(|&i| rows.event[i] == 1.0).collect();
    let p = rows.x.len();
    let x = Mat::from_fn(used.len(), p, |r, j| rows.x[j][used[r]]);
    let weights = rows.weights.map(pick);
    let entry = rows.entry.map(pick);
    let offset = rows.offset.map(pick);
    let (strata_codes, strata_values) = match rows.strata {
        Some(s) => codes(&pick(s)),
        None => (vec![0; used.len()], Vec::new()),
    };
    let clusters: Vec<usize> = match rows.cluster {
        Some(c) => codes(&pick(c)).0,
        None => (0..used.len()).collect(),
    };

    let mut input = CoxInput::new(&time, &event);
    if let Some(w) = &weights {
        input = input.weights(w);
    }
    if let Some(e) = &entry {
        input = input.entry(e);
    }
    if let Some(o) = &offset {
        input = input.offset(o);
    }
    if rows.strata.is_some() {
        input = input.strata(&strata_codes);
    }
    let options = CoxOptions::default()
        .ties(spec.ties)
        .max_iter(spec.max_iter)
        .eps(spec.eps)
        .ridge(spec.ridge)
        .conf_level(spec.conf_level)
        .timefix(spec.timefix);
    let fit = coxph(&x, &input, &options).map_err(map_err)?;
    let robust_var = if spec.robust || rows.cluster.is_some() {
        Some(
            fit.robust_vcov(&clusters)
                .map_err(|e| StatsError::InvalidInput(e.to_string()))?,
        )
    } else {
        None
    };
    Ok(CoxModel {
        fit,
        used,
        strata_values,
        strata_codes,
        robust_var,
        conf_level: spec.conf_level,
        ridge: spec.ridge,
        n_input: n,
    })
}

/// Coefficient table and fit statistics of a Cox fit.
#[derive(Debug, Clone)]
pub struct CoxSummary {
    pub coefficients: Vec<f64>,
    /// Standard errors of the reported covariance (robust when requested).
    pub std_errors: Vec<f64>,
    /// Model-based standard errors (inverse information).
    pub naive_std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_low: Vec<f64>,
    pub conf_high: Vec<f64>,
    pub hazard_ratios: Vec<f64>,
    pub hr_conf_low: Vec<f64>,
    pub hr_conf_high: Vec<f64>,
    /// Reported covariance, row-major.
    pub vcov: Vec<f64>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub lr_statistic: f64,
    pub lr_p_value: f64,
    pub wald_statistic: f64,
    pub wald_p_value: f64,
    pub score_statistic: f64,
    pub score_p_value: f64,
    pub df: usize,
    pub concordance: f64,
    pub r_squared: f64,
    /// `-2 loglik + 2 df` (R `AIC(coxph)`); NaN for a ridge fit, whose
    /// degrees of freedom are not `df`.
    pub aic: f64,
    /// `-2 loglik + df log(n_events)` (R `BIC(coxph)`, whose `nobs` is the
    /// number of events); NaN for a ridge fit.
    pub bic: f64,
    pub n_observations: usize,
    pub n_events: usize,
    pub n_strata: usize,
    pub iterations: usize,
    pub converged: bool,
    pub robust: bool,
}

impl CoxModel {
    /// The coefficient table (Wald z on the reported covariance, as
    /// `summary(coxph)`) and the fit statistics.
    pub fn summary(&self) -> CoxSummary {
        let f = &self.fit;
        let p = f.coefficients.len();
        let var = self.robust_var.as_ref().unwrap_or(&f.var);
        let se: Vec<f64> = (0..p).map(|j| var[(j, j)].sqrt()).collect();
        let norm = Normal::new(0.0, 1.0).expect("standard normal");
        let zc = norm.inverse_cdf(1.0 - (1.0 - self.conf_level) / 2.0);
        let z: Vec<f64> = (0..p).map(|j| f.coefficients[j] / se[j]).collect();
        let pv = z.iter().map(|v| 2.0 * norm.cdf(-v.abs())).collect();
        let lo: Vec<f64> = (0..p).map(|j| f.coefficients[j] - zc * se[j]).collect();
        let hi: Vec<f64> = (0..p).map(|j| f.coefficients[j] + zc * se[j]).collect();
        let ll = f.loglik.1;
        let penalised = self.ridge > 0.0;
        let df = f.df as f64;
        CoxSummary {
            hazard_ratios: f.coefficients.iter().map(|b| b.exp()).collect(),
            hr_conf_low: lo.iter().map(|v| v.exp()).collect(),
            hr_conf_high: hi.iter().map(|v| v.exp()).collect(),
            coefficients: f.coefficients.clone(),
            naive_std_errors: f.std_errors.clone(),
            std_errors: se,
            z_values: z,
            p_values: pv,
            conf_low: lo,
            conf_high: hi,
            vcov: (0..p * p).map(|k| var[(k / p, k % p)]).collect(),
            log_likelihood: ll,
            null_log_likelihood: f.loglik.0,
            lr_statistic: f.lr_test,
            lr_p_value: f.test_p_value(f.lr_test),
            wald_statistic: f.wald_test,
            wald_p_value: f.test_p_value(f.wald_test),
            score_statistic: f.score_test,
            score_p_value: f.test_p_value(f.score_test),
            df: f.df,
            concordance: f.concordance(),
            r_squared: f.r2(),
            aic: if penalised {
                f64::NAN
            } else {
                -2.0 * ll + 2.0 * df
            },
            bic: if penalised {
                f64::NAN
            } else {
                -2.0 * ll + df * (f.n_events as f64).ln()
            },
            n_observations: f.n,
            n_events: f.n_events,
            n_strata: self.strata_values.len().max(1),
            iterations: f.iterations,
            converged: f.converged,
            robust: self.robust_var.is_some(),
        }
    }

    /// Baseline cumulative hazard at covariates 0 (R `basehaz(fit, centered =
    /// FALSE)`), per stratum: `(stratum value, time, cumhaz)` at every distinct
    /// time of the stratum, from upstream `predict_survival`. The stratum
    /// value is NaN for an unstratified fit.
    pub fn baseline(&self) -> StatsResult<Vec<(f64, f64, f64)>> {
        let zeros = vec![0.0; self.fit.coefficients.len()];
        let mut out = Vec::new();
        if self.strata_values.is_empty() {
            let c = self.fit.predict_survival(&zeros, None).map_err(map_err)?;
            out.extend(
                c.time
                    .iter()
                    .zip(&c.cumhaz)
                    .map(|(&t, &h)| (f64::NAN, t, h)),
            );
        } else {
            for (code, &value) in self.strata_values.iter().enumerate() {
                let c = self
                    .fit
                    .predict_survival(&zeros, Some(code))
                    .map_err(map_err)?;
                out.extend(c.time.iter().zip(&c.cumhaz).map(|(&t, &h)| (value, t, h)));
            }
        }
        Ok(out)
    }

    /// Grambsch–Therneau test per covariate plus the global test (upstream
    /// `cox_zph`, `survival::cox.zph`): `(chisq, df, p)` per column, global last.
    pub fn zph(&self, transform: ZphTransform) -> StatsResult<Vec<(f64, usize, f64)>> {
        let z = self.fit.cox_zph(transform, None).map_err(map_err)?;
        Ok(z.terms
            .iter()
            .chain(std::iter::once(&z.global))
            .map(|t| (t.chisq, t.df, t.p_value))
            .collect())
    }

    /// Per-input-row results (NaN for rows left out of the fit).
    pub fn augment(&self) -> CoxAugment {
        let f = &self.fit;
        let n = self.n_input;
        let p = f.coefficients.len();
        let mut a = CoxAugment {
            n_features: p,
            is_training: vec![false; n],
            linear_predictor: vec![f64::NAN; n],
            risk: vec![f64::NAN; n],
            martingale: vec![f64::NAN; n],
            deviance: vec![f64::NAN; n],
            score: vec![f64::NAN; n * p],
            dfbeta: vec![f64::NAN; n * p],
            dfbetas: vec![f64::NAN; n * p],
            schoenfeld: vec![f64::NAN; n * p],
            scaled_schoenfeld: vec![f64::NAN; n * p],
            stratum: vec![f64::NAN; n],
        };
        let lp = f.linear_predictors();
        let mart = f.martingale_residuals();
        let dev = f.deviance_residuals();
        let score = f.score_residuals();
        let dfb = f.dfbeta();
        let dfbs = f.dfbetas();
        for (r, &i) in self.used.iter().enumerate() {
            a.is_training[i] = true;
            a.linear_predictor[i] = lp[r];
            a.risk[i] = lp[r].exp();
            a.martingale[i] = mart[r];
            a.deviance[i] = dev[r];
            if !self.strata_values.is_empty() {
                a.stratum[i] = self.strata_values[self.strata_codes[r]];
            }
            for j in 0..p {
                a.score[i * p + j] = score[(r, j)];
                a.dfbeta[i * p + j] = dfb[(r, j)];
                a.dfbetas[i * p + j] = dfbs[(r, j)];
            }
        }
        for (scaled, out) in [(false, &mut a.schoenfeld), (true, &mut a.scaled_schoenfeld)] {
            let (rows, _, res) = f.schoenfeld_residuals(scaled);
            for (k, &r) in rows.iter().enumerate() {
                let i = self.used[r];
                for j in 0..p {
                    out[i * p + j] = res[(k, j)];
                }
            }
        }
        a
    }
}

/// Per-row Cox results; the `n × p` arrays are row-major.
#[derive(Debug, Clone)]
pub struct CoxAugment {
    pub n_features: usize,
    pub is_training: Vec<bool>,
    pub linear_predictor: Vec<f64>,
    pub risk: Vec<f64>,
    pub martingale: Vec<f64>,
    pub deviance: Vec<f64>,
    pub score: Vec<f64>,
    pub dfbeta: Vec<f64>,
    pub dfbetas: Vec<f64>,
    /// NaN except on event rows.
    pub schoenfeld: Vec<f64>,
    pub scaled_schoenfeld: Vec<f64>,
    pub stratum: Vec<f64>,
}

/// Predicted survival `S(t | x)` of a fitted Cox model at each of `times`
/// from its stored baseline (as the model STRUCT carries it): the baseline
/// cumulative hazard at covariates 0, `base_cumhaz[k]` at `base_time[k]`
/// (increasing), scaled by the relative risk `exp(x'β)`, exactly upstream
/// `predict_survival`'s point estimate (`survfit(fit, newdata)`).
///
/// `S = 1` before the first baseline time and stays at its last value after
/// the last one; NaN for a NaN time, a row of the wrong length or a
/// non-finite linear predictor.
pub fn cox_survival_curve(
    coefficients: &[f64],
    base_time: &[f64],
    base_cumhaz: &[f64],
    x_row: &[f64],
    times: &[f64],
) -> Vec<f64> {
    let lp = if x_row.len() == coefficients.len() && base_time.len() == base_cumhaz.len() {
        coefficients
            .iter()
            .zip(x_row)
            .map(|(b, x)| b * x)
            .sum::<f64>()
    } else {
        f64::NAN
    };
    let risk = lp.exp();
    times
        .iter()
        .map(|&t| {
            if t.is_nan() || !risk.is_finite() {
                return f64::NAN;
            }
            let k = base_time.partition_point(|&s| s <= t);
            if k == 0 {
                1.0
            } else {
                (-base_cumhaz[k - 1] * risk).exp()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let time = vec![
            6.0, 7.0, 10.0, 15.0, 19.0, 25.0, 3.0, 9.0, 11.0, 20.0, 4.0, 13.0,
        ];
        let event = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let x = vec![vec![
            0.5, 1.2, -0.3, 0.8, 2.0, -1.0, 1.5, 0.1, -0.7, 0.4, 1.1, -0.2,
        ]];
        (time, event, x)
    }

    fn rows<'a>(t: &'a [f64], e: &'a [f64], x: &'a [Vec<f64>]) -> CoxRows<'a> {
        CoxRows {
            time: t,
            event: e,
            x,
            weights: None,
            entry: None,
            offset: None,
            strata: None,
            cluster: None,
        }
    }

    #[test]
    fn survival_from_baseline_matches_upstream_prediction() {
        let (t, e, x) = data();
        let m = fit_cox(&rows(&t, &e, &x), &CoxSpec::default()).unwrap();
        let base = m.baseline().unwrap();
        let bt: Vec<f64> = base.iter().map(|b| b.1).collect();
        let bh: Vec<f64> = base.iter().map(|b| b.2).collect();
        for x0 in [-1.0, 0.0, 0.7, 2.0] {
            let up = m.fit.predict_survival(&[x0], None).unwrap();
            let s = cox_survival_curve(&m.fit.coefficients, &bt, &bh, &[x0], &up.time);
            for (a, b) in s.iter().zip(&up.surv) {
                assert!((a - b).abs() < 1e-12, "{a} vs {b}");
            }
        }
        let s = cox_survival_curve(&m.fit.coefficients, &bt, &bh, &[0.0], &[0.0, 1e9]);
        assert_eq!(s[0], 1.0);
        assert!(s[1] < 1.0);
    }

    #[test]
    fn rows_with_non_finite_values_are_left_out() {
        let (mut t, e, x) = data();
        t[3] = f64::NAN;
        let m = fit_cox(&rows(&t, &e, &x), &CoxSpec::default()).unwrap();
        assert_eq!(m.fit.n, 11);
        let a = m.augment();
        assert!(!a.is_training[3] && a.martingale[3].is_nan());
        assert!(a.is_training[0]);
    }

    #[test]
    fn errors_map_onto_the_sql_policy() {
        let (t, _, x) = data();
        let none = vec![0.0; t.len()];
        assert!(matches!(
            fit_cox(&rows(&t, &none, &x), &CoxSpec::default()),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        let mut bad = none.clone();
        bad[0] = 2.0;
        assert!(matches!(
            fit_cox(&rows(&t, &bad, &x), &CoxSpec::default()),
            Err(StatsError::InvalidValue { field: "event", .. })
        ));
    }

    #[test]
    fn strata_are_coded_by_value() {
        let (t, e, x) = data();
        let s: Vec<f64> = (0..t.len())
            .map(|i| if i < 6 { 5.0 } else { 2.0 })
            .collect();
        let r = CoxRows {
            strata: Some(&s),
            ..rows(&t, &e, &x)
        };
        let m = fit_cox(&r, &CoxSpec::default()).unwrap();
        assert_eq!(m.strata_values, vec![2.0, 5.0]);
        let base = m.baseline().unwrap();
        assert_eq!(base[0].0, 2.0);
        assert_eq!(base.last().unwrap().0, 5.0);
        assert_eq!(m.augment().stratum[0], 5.0);
    }
}
