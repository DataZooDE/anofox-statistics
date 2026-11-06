//! Non-parametric survival analysis (#156)
//!
//! - Kaplan-Meier estimator with Greenwood standard errors
//! - k-group log-rank test
//!
//! Both delegate to `anofox_statistics::survival` (R `survival::survfit`,
//! `survival::survdiff`).
//!
//! The extended forms (#322) — case weights, left truncation, robust standard
//! errors, the Nelson–Aalen / Fleming–Harrington cumulative hazard, and the
//! stratified G-rho test — delegate to `anofox_regression::survival`
//! ([`survfit_curve`], [`survdiff_test`]).

use super::{convert_error, UnifiedTestResult};
use crate::models::cox::{codes, map_err};
use crate::{StatsError, StatsResult};
use anofox_regression::survival::{survdiff, survfit, SurvInput, SurvfitOptions};

pub use anofox_regression::survival::{ConfType as SurvConfType, HazardType, SurvCurve};
pub use anofox_tests::{KaplanMeierResult, KmConfType};

/// Kaplan-Meier curve of right-censored `time` (`event = true` observed).
pub fn kaplan_meier(
    time: &[f64],
    event: &[bool],
    conf_type: KmConfType,
    conf_level: f64,
) -> StatsResult<KaplanMeierResult> {
    if time.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Kaplan-Meier requires at least 1 observation".into(),
        ));
    }
    anofox_tests::kaplan_meier(time, event, conf_type, conf_level).map_err(convert_error)
}

/// Log-rank test of equal survival across the groups coded by `group`.
/// Fewer than two groups is insufficient data.
pub fn log_rank(time: &[f64], event: &[bool], group: &[u64]) -> StatsResult<UnifiedTestResult> {
    let mut levels: Vec<u64> = group.to_vec();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "log-rank test needs at least 2 groups".into(),
        ));
    }
    let result = anofox_tests::log_rank(time, event, group).map_err(convert_error)?;
    Ok(UnifiedTestResult::from(&result))
}

/// Per-row data of [`survfit_curve`], one entry per input row. `event` is
/// non-zero for an observed event. Rows with a non-finite value in any given
/// column are left out.
#[derive(Debug, Clone, Copy)]
pub struct SurvfitRows<'a> {
    pub time: &'a [f64],
    pub event: &'a [f64],
    pub entry: Option<&'a [f64]>,
    pub weights: Option<&'a [f64]>,
    pub cluster: Option<&'a [f64]>,
}

/// Options of [`survfit_curve`] (`survfit(conf.int, conf.type, ctype, robust)`
/// and `Surv(timefix)`); `robust: None` is survfit's default (robust with
/// non-integer weights or a cluster).
#[derive(Debug, Clone, Copy)]
pub struct SurvfitSpec {
    pub conf_level: f64,
    pub conf_type: SurvConfType,
    pub hazard: HazardType,
    pub timefix: bool,
    pub robust: Option<bool>,
}

impl Default for SurvfitSpec {
    fn default() -> Self {
        Self {
            conf_level: 0.95,
            conf_type: SurvConfType::Log,
            hazard: HazardType::NelsonAalen,
            timefix: true,
            robust: None,
        }
    }
}

/// A Kaplan–Meier / cumulative-hazard curve from upstream `survfit`, with
/// `std_err` reported as the standard error of `S` (R's
/// `summary(survfit)$std.err`, as [`kaplan_meier`] reports it).
#[derive(Debug, Clone)]
pub struct SurvfitCurve {
    pub curve: SurvCurve,
    /// Standard error of `surv`; NaN where `surv` is 0 (R gives `NA`).
    pub std_err: Vec<f64>,
}

/// Kaplan–Meier curve with Greenwood or robust standard errors and the
/// Nelson–Aalen / Fleming–Harrington cumulative hazard, as
/// `survfit(Surv(entry, time, event) ~ 1, weights, id)`.
pub fn survfit_curve(rows: &SurvfitRows, spec: &SurvfitSpec) -> StatsResult<SurvfitCurve> {
    let n = rows.time.len();
    let lens = [
        Some(rows.event.len()),
        rows.entry.map(<[f64]>::len),
        rows.weights.map(<[f64]>::len),
        rows.cluster.map(<[f64]>::len),
    ];
    if lens.iter().flatten().any(|&l| l != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "every survfit input must have one value per row".into(),
        ));
    }
    if !(spec.conf_level > 0.0 && spec.conf_level < 1.0) {
        return Err(StatsError::InvalidValue {
            field: "conf_level",
            message: format!("must be in (0, 1), got {}", spec.conf_level),
        });
    }
    let finite = |o: Option<&[f64]>, i: usize| o.is_none_or(|v| v[i].is_finite());
    let used: Vec<usize> = (0..n)
        .filter(|&i| {
            rows.time[i].is_finite()
                && rows.event[i].is_finite()
                && finite(rows.entry, i)
                && finite(rows.weights, i)
                && finite(rows.cluster, i)
        })
        .collect();
    if used.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Kaplan-Meier requires at least 1 observation".into(),
        ));
    }
    let pick = |v: &[f64]| -> Vec<f64> { used.iter().map(|&i| v[i]).collect() };
    let time = pick(rows.time);
    let event: Vec<bool> = used.iter().map(|&i| rows.event[i] != 0.0).collect();
    let entry = rows.entry.map(pick);
    let weights = rows.weights.map(pick);
    let cluster = rows.cluster.map(|c| codes(&pick(c)).0);
    let mut input = SurvInput::new(&time, &event);
    if let Some(e) = &entry {
        input = input.entry(e);
    }
    if let Some(w) = &weights {
        input = input.weights(w);
    }
    if let Some(c) = &cluster {
        input = input.cluster(c);
    }
    let options = SurvfitOptions::default()
        .conf_level(spec.conf_level)
        .conf_type(spec.conf_type)
        .hazard(spec.hazard)
        .timefix(spec.timefix)
        .robust(spec.robust);
    let mut curves = survfit(&input, None, &options).map_err(map_err)?;
    let curve = curves.remove(0);
    // Upstream's std_err is on the -log S scale; R's summary reports S * se.
    let std_err = curve
        .surv
        .iter()
        .zip(&curve.std_err)
        .map(|(&s, &se)| if s == 0.0 { f64::NAN } else { s * se })
        .collect();
    Ok(SurvfitCurve { curve, std_err })
}

/// G-rho test of equal survival across the groups coded by `group`
/// (`survdiff(Surv(time, event) ~ group + strata(s), rho)`): `rho = 0` is the
/// log-rank test, `rho = 1` the Peto–Peto test. Rows with a non-finite time or
/// stratum are left out; fewer than two groups is insufficient data.
pub fn survdiff_test(
    time: &[f64],
    event: &[bool],
    group: &[u64],
    strata: Option<&[f64]>,
    rho: f64,
) -> StatsResult<UnifiedTestResult> {
    let n = time.len();
    if event.len() != n || group.len() != n || strata.is_some_and(|s| s.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "time, event, group and strata must have one value per row".into(),
        ));
    }
    if !rho.is_finite() {
        return Err(StatsError::InvalidValue {
            field: "rho",
            message: format!("must be finite, got {rho}"),
        });
    }
    let used: Vec<usize> = (0..n)
        .filter(|&i| time[i].is_finite() && strata.is_none_or(|s| s[i].is_finite()))
        .collect();
    let mut levels: Vec<u64> = used.iter().map(|&i| group[i]).collect();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "log-rank test needs at least 2 groups".into(),
        ));
    }
    let t: Vec<f64> = used.iter().map(|&i| time[i]).collect();
    let e: Vec<bool> = used.iter().map(|&i| event[i]).collect();
    let g: Vec<usize> = used.iter().map(|&i| group[i] as usize).collect();
    let s: Option<Vec<usize>> =
        strata.map(|s| codes(&used.iter().map(|&i| s[i]).collect::<Vec<f64>>()).0);
    let r = survdiff(&t, &e, &g, s.as_deref(), rho).map_err(map_err)?;
    let method = if rho == 0.0 {
        "Log-rank test"
    } else if rho == 1.0 {
        "Peto-Peto test"
    } else {
        "G-rho test"
    };
    let mut out = UnifiedTestResult::new(method);
    out.statistic = Some(r.chisq);
    out.df1 = Some(r.df as f64);
    out.p_value = Some(r.p_value);
    out.n = Some(r.n.iter().sum());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn km_steps_at_event_times() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0];
        let event = [true, true, false, true, false];
        let km = kaplan_meier(&time, &event, KmConfType::Log, 0.95).unwrap();
        assert_eq!(km.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(km.survival[0], 0.8);
        assert_relative_eq!(km.survival[1], 0.6);
        assert_relative_eq!(km.survival[2], 0.3);
        assert_eq!(km.n_censor, vec![0, 1, 0, 1]);
    }

    #[test]
    fn log_rank_needs_two_groups() {
        let time = [1.0, 2.0, 3.0];
        let event = [true, true, false];
        assert!(matches!(
            log_rank(&time, &event, &[0, 0, 0]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        let t = log_rank(&time, &event, &[0, 1, 0]).unwrap();
        assert_eq!(t.df1, Some(1.0));
        assert_eq!(t.n, Some(3));
    }

    #[test]
    fn survfit_matches_the_legacy_curve_without_options() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0];
        let event = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ev: Vec<bool> = event.iter().map(|&e| e != 0.0).collect();
        let old = kaplan_meier(&time, &ev, KmConfType::Log, 0.95).unwrap();
        let rows = SurvfitRows {
            time: &time,
            event: &event,
            entry: None,
            weights: None,
            cluster: None,
        };
        let new = survfit_curve(&rows, &SurvfitSpec::default()).unwrap();
        assert_eq!(new.curve.time, old.time);
        for k in 0..old.time.len() {
            assert_eq!(new.curve.surv[k], old.survival[k]);
            assert_eq!(new.curve.n_risk[k], old.n_risk[k] as f64);
            let both_nan = new.std_err[k].is_nan() && old.std_err[k].is_nan();
            assert!(both_nan || (new.std_err[k] - old.std_err[k]).abs() < 1e-15);
        }
    }

    #[test]
    fn survdiff_rho_zero_is_the_log_rank_test() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = [true, true, false, true, true, true, false, true];
        let group = [0, 1, 0, 1, 0, 1, 0, 1];
        let a = log_rank(&time, &event, &group).unwrap();
        let b = survdiff_test(&time, &event, &group, None, 0.0).unwrap();
        assert_relative_eq!(a.statistic.unwrap(), b.statistic.unwrap(), epsilon = 1e-12);
        assert_eq!(a.df1, b.df1);
        assert_eq!(a.n, b.n);
        assert!(matches!(
            survdiff_test(&time, &event, &[0; 8], None, 1.0),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }
}
