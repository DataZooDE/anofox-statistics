//! rms-style Wald ANOVA (chunk tests) and fast backward step-down from a fitted
//! model's coefficients and covariance (#311): thin delegation to
//! `anofox_regression::inference::anova` (`rms::anova.rms`) and
//! `anofox_regression::inference::fastbw` (`rms::fastbw`).
//!
//! The model STRUCT stores the slopes in feature order and the covariance
//! row-major, intercept first when the model has one. A term is the set of
//! slope columns that carry the same label; columns without a label belong to
//! no term (the ANOVA leaves them out, the step-down never deletes them). An
//! interaction term names its two main-effect factors; its columns are
//! expected in rms order (the full product `a * b` with the first factor
//! varying fastest, or the restricted `a %ia% b` without doubly nonlinear
//! products, recognised by its column count).
//!
//! This module only validates the term specification, builds the upstream
//! `TermMap` (main effects first, then interactions, each in first-appearance
//! order), permutes `β̂` / `V̂` into its column order and maps the results back
//! to the model's columns and the caller's term numbering.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::design::TermMap;
use anofox_regression::inference::{
    anova, fastbw as upstream_fastbw, AnovaOptions, AnovaRowKind, StepRule, StepType,
    WaldDistribution, WaldError,
};
use faer::{Col, Mat};

impl From<WaldError> for StatsError {
    fn from(err: WaldError) -> Self {
        match err {
            WaldError::Dimension(msg) => StatsError::DimensionMismatchMsg(msg),
            WaldError::Column(j) => StatsError::InvalidInput(format!("column {j} is out of range")),
            WaldError::Singular => StatsError::SingularMatrix,
            WaldError::Empty => StatsError::InvalidInput("nothing to test".into()),
            // `WaldError` is `#[non_exhaustive]`: surface a new variant.
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// One term of the specification, in the caller's numbering.
#[derive(Debug, Clone)]
pub struct TermSpec {
    pub name: String,
    /// The two main-effect factors (caller's term indices) of an interaction.
    pub factors: Option<(usize, usize)>,
}

/// Coefficients and covariance of a fitted model, as its STRUCT stores them.
#[derive(Debug, Clone)]
pub struct ModelCovariance<'a> {
    /// Slopes in feature order.
    pub coefficients: &'a [f64],
    /// Intercept (NaN without one).
    pub intercept: f64,
    /// Row-major `dim x dim`, `dim = p` or `p + 1` (intercept first).
    pub vcov: &'a [f64],
    pub dim: usize,
}

impl ModelCovariance<'_> {
    fn p(&self) -> usize {
        self.coefficients.len()
    }

    fn has_intercept(&self) -> bool {
        self.dim == self.p() + 1
    }

    fn validate(&self) -> StatsResult<()> {
        let p = self.p();
        if p == 0 {
            return Err(StatsError::InvalidInput(
                "the model has no slope coefficients".into(),
            ));
        }
        if self.vcov.len() != self.dim * self.dim || (self.dim != p && self.dim != p + 1) {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "vcov has {} entries for {p} coefficients (expected {} or {})",
                self.vcov.len(),
                p * p,
                (p + 1) * (p + 1)
            )));
        }
        Ok(())
    }

    /// Covariance entry of slopes `i`, `j` (model order).
    fn v(&self, i: usize, j: usize) -> f64 {
        let off = usize::from(self.has_intercept());
        self.vcov[(i + off) * self.dim + j + off]
    }
}

/// The upstream term map of a specification and the model column of every
/// term-map column, plus the caller's index of every term-map term.
pub(crate) struct Mapped {
    pub(crate) terms: TermMap,
    /// Term-map column -> model slope column.
    pub(crate) columns: Vec<usize>,
    /// Term-map term -> caller's term index.
    pub(crate) term_index: Vec<usize>,
}

fn invalid(msg: String) -> StatsError {
    StatsError::InvalidInput(msg)
}

/// Columns (model order) of every caller term; checks `term_of_column`.
fn columns_by_term(
    term_of_column: &[Option<usize>],
    n_terms: usize,
) -> StatsResult<Vec<Vec<usize>>> {
    let mut by_term = vec![Vec::new(); n_terms];
    for (j, t) in term_of_column.iter().enumerate() {
        if let Some(t) = *t {
            by_term
                .get_mut(t)
                .ok_or_else(|| invalid(format!("term {t} of column {} is out of range", j + 1)))?
                .push(j);
        }
    }
    if let Some(t) = by_term.iter().position(|c| c.is_empty()) {
        return Err(invalid(format!("term {t} has no columns")));
    }
    Ok(by_term)
}

pub(crate) fn build_term_map(
    term_of_column: &[Option<usize>],
    terms: &[TermSpec],
    nonlinear: &[bool],
) -> StatsResult<Mapped> {
    let by_term = columns_by_term(term_of_column, terms.len())?;
    if by_term.is_empty() {
        return Err(invalid("no coefficient belongs to a term".into()));
    }
    let mut map = TermMap::new();
    let mut columns = Vec::new();
    let mut term_index = Vec::new();
    // Caller term -> term-map term, for the main effects.
    let mut main_of = vec![None; terms.len()];
    for (t, spec) in terms.iter().enumerate() {
        if spec.factors.is_none() {
            let nl: Vec<bool> = by_term[t].iter().map(|&j| nonlinear[j]).collect();
            main_of[t] = Some(map.add_main(&spec.name, &nl));
            columns.extend(&by_term[t]);
            term_index.push(t);
        }
    }
    for (t, spec) in terms.iter().enumerate() {
        let Some((a, b)) = spec.factors else { continue };
        let factor = |f: usize| {
            main_of.get(f).copied().flatten().ok_or_else(|| {
                invalid(format!(
                    "interaction '{}': its factors must be main-effect terms",
                    spec.name
                ))
            })
        };
        let (ma, mb) = (factor(a)?, factor(b)?);
        let (na, nb) = (by_term[a].len(), by_term[b].len());
        let nl_a = by_term[a].iter().filter(|&&j| nonlinear[j]).count();
        let nl_b = by_term[b].iter().filter(|&&j| nonlinear[j]).count();
        let full = na * nb;
        let restricted = full - nl_a * nl_b;
        let k = by_term[t].len();
        let is_restricted = if k == full {
            false
        } else if k == restricted {
            true
        } else {
            return Err(invalid(format!(
                "interaction '{}' has {k} columns; '{}' ({na} columns) x '{}' ({nb} columns) needs {full} \
                 (full product) or {restricted} (restricted, without doubly nonlinear products)",
                spec.name, terms[a].name, terms[b].name
            )));
        };
        map.add_interaction(ma, mb, is_restricted)
            .map_err(|e| invalid(format!("interaction '{}': {e}", spec.name)))?;
        columns.extend(&by_term[t]);
        term_index.push(t);
    }
    Ok(Mapped {
        terms: map,
        columns,
        term_index,
    })
}

fn check_estimated(model: &ModelCovariance, columns: &[usize]) -> StatsResult<()> {
    if columns.iter().any(|&j| !model.coefficients[j].is_finite()) {
        // An aliased (not estimated) coefficient: the data do not support the test.
        return Err(StatsError::SingularMatrix);
    }
    Ok(())
}

/// What an ANOVA row tests, as the SQL `kind` label.
pub fn anova_row_kind_name(kind: AnovaRowKind) -> &'static str {
    match kind {
        AnovaRowKind::MainEffect => "main_effect",
        AnovaRowKind::CombinedEffect => "combined_effect",
        AnovaRowKind::Interaction => "interaction",
        AnovaRowKind::CombinedInteractions => "all_interactions",
        AnovaRowKind::Nonlinear => "nonlinear",
        AnovaRowKind::NonlinearInteraction => "nonlinear_interaction",
        AnovaRowKind::DoublyNonlinearInteraction => "doubly_nonlinear_interaction",
        AnovaRowKind::NonlinearInteractionFirst => "nonlinear_interaction_first",
        AnovaRowKind::NonlinearInteractionSecond => "nonlinear_interaction_second",
        AnovaRowKind::TotalNonlinear => "total_nonlinear",
        AnovaRowKind::TotalInteraction => "total_interaction",
        AnovaRowKind::TotalNonlinearInteraction => "total_nonlinear_interaction",
        AnovaRowKind::Total => "total",
        _ => "other",
    }
}

/// One row of [`wald_anova`].
#[derive(Debug, Clone)]
pub struct WaldAnovaRow {
    /// Label as `anova.rms` prints it.
    pub label: String,
    pub kind: &'static str,
    /// Caller's term index (`None` for the totals).
    pub term: Option<usize>,
    /// Model slope columns tested (0-based, ascending).
    pub columns: Vec<usize>,
    pub chi_square: f64,
    pub df: usize,
    /// `χ²/df` for an F test, else NaN.
    pub f: f64,
    /// Residual d.f. of an F test, else NaN.
    pub df_residual: f64,
    pub p_value: f64,
    /// `χ²·σ²` when σ² is given, else NaN.
    pub partial_ss: f64,
}

/// Result of [`wald_anova`].
#[derive(Debug, Clone)]
pub struct WaldAnovaResult {
    pub rows: Vec<WaldAnovaRow>,
    /// `(d.f., SS, MS)` of the residual for an F test with σ² given.
    pub error: Option<(f64, f64, f64)>,
}

/// `rms::anova.rms` table of the model. `df_residual` selects the F test
/// (`Some`) or the χ² test; `sigma2` adds partial sums of squares.
pub fn wald_anova(
    model: &ModelCovariance,
    term_of_column: &[Option<usize>],
    terms: &[TermSpec],
    nonlinear: &[bool],
    df_residual: Option<f64>,
    sigma2: Option<f64>,
) -> StatsResult<WaldAnovaResult> {
    model.validate()?;
    let p = model.p();
    if term_of_column.len() != p || nonlinear.len() != p {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "terms has {} and nonlinear {} entries for {p} coefficients",
            term_of_column.len(),
            nonlinear.len()
        )));
    }
    let mapped = build_term_map(term_of_column, terms, nonlinear)?;
    check_estimated(model, &mapped.columns)?;
    let cols = &mapped.columns;
    let n = cols.len();
    let beta = Col::from_fn(n, |a| model.coefficients[cols[a]]);
    let vcov = Mat::from_fn(n, n, |a, b| model.v(cols[a], cols[b]));
    let distribution = match df_residual {
        Some(df) if df.is_finite() && df > 0.0 => WaldDistribution::F { df_residual: df },
        Some(df) => {
            return Err(StatsError::InsufficientDataMsg(format!(
                "the F test needs positive residual degrees of freedom, got {df}"
            )))
        }
        None => WaldDistribution::ChiSquare,
    };
    let options = AnovaOptions::default()
        .distribution(distribution)
        .sigma2(sigma2.filter(|s| s.is_finite()));
    let table = anova(&beta, &vcov, &mapped.terms, &options)?;
    let rows = table
        .rows
        .iter()
        .map(|r| {
            let mut columns: Vec<usize> = r.columns.iter().map(|&c| cols[c]).collect();
            columns.sort_unstable();
            WaldAnovaRow {
                label: r.label.clone(),
                kind: anova_row_kind_name(r.kind),
                term: r.term.map(|t| mapped.term_index[t]),
                columns,
                chi_square: r.test.chi_square,
                df: r.test.df,
                f: r.test.f.unwrap_or(f64::NAN),
                df_residual: r.test.df_residual.unwrap_or(f64::NAN),
                p_value: r.test.p_value,
                partial_ss: r.partial_ss.unwrap_or(f64::NAN),
            }
        })
        .collect();
    Ok(WaldAnovaResult {
        rows,
        error: table.error,
    })
}

/// Stopping rule of [`fastbw`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FastbwRule {
    /// Delete while `χ² − k·d.f. ≤ aics`.
    Aic { k: f64, aics: f64 },
    /// Delete while the p-value exceeds `sls`.
    P { sls: f64 },
}

/// One deletion of [`fastbw`], in the caller's term numbering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastbwStep {
    pub term: usize,
    pub chi_square: f64,
    pub df: usize,
    pub p_value: f64,
    pub residual_chi_square: f64,
    pub residual_df: usize,
    pub residual_p_value: f64,
    pub aic: f64,
}

/// Result of [`fastbw`].
#[derive(Debug, Clone, PartialEq)]
pub struct FastbwResult {
    pub deleted: Vec<FastbwStep>,
    /// Caller's indices of the kept terms, ascending.
    pub kept_terms: Vec<usize>,
    /// Approximate slopes of the reduced model, model order (0 when deleted).
    pub coefficients: Vec<f64>,
    /// Approximate intercept (NaN without one).
    pub intercept: f64,
    /// Approximate covariance of the reduced model (`fastbw`'s `var`,
    /// upstream `StepDown::vcov`), row-major in the layout of the input
    /// covariance (`model.dim` square, intercept first) with zero rows and
    /// columns for deleted parameters.
    pub vcov: Vec<f64>,
    /// Kept parameters (`fastbw`'s `parms.kept`), 0-based positions in that
    /// layout (the intercept is 0 when there is one), ascending.
    pub kept_parameters: Vec<usize>,
}

/// `rms::fastbw` of the model. Columns without a term are forced into the
/// model (as `fastbw(force = ...)`), like the intercept.
pub fn fastbw(
    model: &ModelCovariance,
    term_of_column: &[Option<usize>],
    term_names: &[String],
    rule: FastbwRule,
    individual: bool,
) -> StatsResult<FastbwResult> {
    model.validate()?;
    let p = model.p();
    if term_of_column.len() != p {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "terms has {} entries for {p} coefficients",
            term_of_column.len()
        )));
    }
    match rule {
        FastbwRule::Aic { k, aics } if !(k.is_finite() && k >= 0.0 && aics.is_finite()) => {
            return Err(invalid(format!(
                "k_aic must be finite and >= 0 and aics finite, got {k} and {aics}"
            )))
        }
        FastbwRule::P { sls } if !(sls > 0.0 && sls < 1.0) => {
            return Err(invalid(format!("sls must be in (0, 1), got {sls}")))
        }
        _ => {}
    }
    let specs: Vec<TermSpec> = term_names
        .iter()
        .map(|n| TermSpec {
            name: n.clone(),
            factors: None,
        })
        .collect();
    let mapped = build_term_map(term_of_column, &specs, &vec![false; p])?;
    check_estimated(model, &(0..p).collect::<Vec<_>>())?;
    let has_int = model.has_intercept();
    if has_int && !model.intercept.is_finite() {
        return Err(invalid(
            "the covariance has an intercept row but the model's intercept is NULL".into(),
        ));
    }
    // Parameter order: intercept, forced columns, then the term-map columns.
    let forced: Vec<usize> = (0..p).filter(|&j| term_of_column[j].is_none()).collect();
    let n_fixed = usize::from(has_int) + forced.len();
    // Parameter -> model slope column (None for the intercept).
    let params: Vec<Option<usize>> = has_int
        .then_some(None)
        .into_iter()
        .chain(forced.iter().chain(&mapped.columns).map(|&j| Some(j)))
        .collect();
    let off = usize::from(has_int);
    let full = |a: Option<usize>| a.map_or(0, |j| j + off);
    let n = params.len();
    let beta = Col::from_fn(n, |a| match params[a] {
        None => model.intercept,
        Some(j) => model.coefficients[j],
    });
    let vcov = Mat::from_fn(n, n, |a, b| {
        model.vcov[full(params[a]) * model.dim + full(params[b])]
    });
    let (step_rule, step_type) = (
        match rule {
            FastbwRule::Aic { k, aics } => StepRule::Aic { k, threshold: aics },
            FastbwRule::P { sls } => StepRule::P { sls },
        },
        if individual {
            StepType::Individual
        } else {
            StepType::Residual
        },
    );
    let s = upstream_fastbw(&beta, &vcov, n_fixed, &mapped.terms, step_rule, step_type);
    let mut coefficients = vec![0.0; p];
    let mut intercept = f64::NAN;
    for (a, &param) in params.iter().enumerate() {
        match param {
            None => intercept = s.coefficients[a],
            Some(j) => coefficients[j] = s.coefficients[a],
        }
    }
    let mut kept_terms: Vec<usize> = s.kept_terms.iter().map(|&t| mapped.term_index[t]).collect();
    kept_terms.sort_unstable();
    // Upstream's covariance is in the parameter order above; put it back in
    // the model's layout.
    let dim = model.dim;
    let mut vcov_out = vec![0.0; dim * dim];
    for a in 0..n {
        for b in 0..n {
            vcov_out[full(params[a]) * dim + full(params[b])] = s.vcov[(a, b)];
        }
    }
    let mut kept_parameters: Vec<usize> =
        s.kept_parameters.iter().map(|&a| full(params[a])).collect();
    kept_parameters.sort_unstable();
    Ok(FastbwResult {
        deleted: s
            .deleted
            .iter()
            .map(|d| FastbwStep {
                term: mapped.term_index[d.term],
                chi_square: d.chisq,
                df: d.df,
                p_value: d.p_value,
                residual_chi_square: d.residual_chisq,
                residual_df: d.residual_df,
                residual_p_value: d.residual_p,
                aic: d.aic,
            })
            .collect(),
        kept_terms,
        coefficients,
        intercept,
        vcov: vcov_out,
        kept_parameters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model<'a>(beta: &'a [f64], vcov: &'a [f64], dim: usize) -> ModelCovariance<'a> {
        ModelCovariance {
            coefficients: beta,
            intercept: 0.5,
            vcov,
            dim,
        }
    }

    #[test]
    fn single_column_terms_are_z_squared() {
        // Intercept + 2 slopes, diagonal covariance.
        let beta = [1.5, -0.4];
        let vcov = [1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.04];
        let specs = vec![
            TermSpec {
                name: "a".into(),
                factors: None,
            },
            TermSpec {
                name: "b".into(),
                factors: None,
            },
        ];
        let r = wald_anova(
            &model(&beta, &vcov, 3),
            &[Some(0), Some(1)],
            &specs,
            &[false, false],
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.rows.len(), 3);
        assert_eq!(r.rows[0].label, "a");
        assert!((r.rows[0].chi_square - 9.0).abs() < 1e-12);
        assert!((r.rows[1].chi_square - 4.0).abs() < 1e-12);
        assert_eq!(r.rows[2].label, "TOTAL");
        assert!((r.rows[2].chi_square - 13.0).abs() < 1e-12);
        assert_eq!(r.rows[2].term, None);
    }

    #[test]
    fn interaction_columns_are_checked() {
        let beta = [1.0, 2.0, 3.0];
        let vcov: Vec<f64> = (0..9).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let specs = vec![
            TermSpec {
                name: "a".into(),
                factors: None,
            },
            TermSpec {
                name: "a * b".into(),
                factors: Some((0, 2)),
            },
            TermSpec {
                name: "b".into(),
                factors: None,
            },
        ];
        // a, b one column each: the product has one column.
        let ok = wald_anova(
            &model(&beta, &vcov, 3),
            &[Some(0), Some(2), Some(1)],
            &specs,
            &[false; 3],
            None,
            None,
        )
        .unwrap();
        assert!(ok.rows.iter().any(|r| r.kind == "interaction"));
        let ia = ok.rows.iter().find(|r| r.kind == "interaction").unwrap();
        assert_eq!(ia.columns, vec![2]);
        let bad = wald_anova(
            &model(&beta, &vcov, 3),
            &[Some(0), Some(1), Some(1)],
            &specs[..2],
            &[false; 3],
            None,
            None,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn fastbw_deletes_the_weak_term_and_keeps_forced_columns() {
        let beta = [2.0, 0.01, 3.0];
        let vcov: Vec<f64> = (0..16)
            .map(|i| if i % 5 == 0 { 0.1 } else { 0.0 })
            .collect();
        let names = vec!["a".to_string(), "b".to_string()];
        let r = fastbw(
            &model(&beta, &vcov, 4),
            &[Some(0), Some(1), None],
            &names,
            FastbwRule::Aic { k: 2.0, aics: 0.0 },
            false,
        )
        .unwrap();
        assert_eq!(r.deleted.len(), 1);
        assert_eq!(r.deleted[0].term, 1);
        assert_eq!(r.kept_terms, vec![0]);
        assert_eq!(r.coefficients[1], 0.0);
        assert!((r.coefficients[2] - 3.0).abs() < 1e-12);
        assert!((r.intercept - 0.5).abs() < 1e-12);
    }
}
