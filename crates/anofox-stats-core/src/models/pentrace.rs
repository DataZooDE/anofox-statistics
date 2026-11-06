//! Penalized maximum likelihood à la rms (#323): thin delegation to
//! `anofox_regression::solvers::pentrace` — `rms::pentrace` (the search for
//! the penalty by AIC / BIC / corrected AIC with the effective d.f.) for
//! least squares, binary and ordinal logistic models, and
//! `rms::ols(penalty = )` (`penalized_ols`).
//!
//! The penalty is rms's differential penalty by column type (simple,
//! nonlinear, interaction, nonlinear interaction) on standardised
//! coefficients; the term map comes from the same specification as
//! `wald_anova` (#311): one term label per column, nonlinear flags, the two
//! factors of every interaction, plus the categorical terms (whose dummy
//! columns get rms's `I − 1/(L+1)` block). This module validates the
//! specification, builds the upstream `TermMap` (via `wald_anova`'s
//! builder), permutes the design columns into its order, calls upstream and
//! maps the results back to the caller's columns and terms.

use super::wald_anova::{build_term_map, Mapped, TermSpec};
use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::{
    effective_df_by_term, penalized_ols as upstream_penalized_ols, penalty_grid,
    penalty_matrix as upstream_penalty_matrix, pentrace as upstream_pentrace,
    pentrace_ols as upstream_pentrace_ols, OrdinalLink, PenaltySpec, PentraceCriterion,
};
use faer::{Col, Mat};

/// The term specification of a penalized fit, in the caller's numbering.
#[derive(Debug, Clone)]
pub struct PenaltyTerms {
    /// Term of every column (all columns must belong to a term).
    pub term_of_column: Vec<usize>,
    /// The terms (name and, for an interaction, its two main-effect factors).
    pub terms: Vec<TermSpec>,
    /// Per column: nonlinear (ignored for interaction columns).
    pub nonlinear: Vec<bool>,
    /// Caller's indices of the categorical terms.
    pub categorical: Vec<usize>,
}

impl PenaltyTerms {
    /// Every column its own linear term `x1 .. xk`.
    pub fn columns(p: usize) -> Self {
        Self {
            term_of_column: (0..p).collect(),
            terms: (0..p)
                .map(|j| TermSpec {
                    name: format!("x{}", j + 1),
                    factors: None,
                })
                .collect(),
            nonlinear: vec![false; p],
            categorical: Vec::new(),
        }
    }
}

/// The penalties of a trace: every combination of the given values
/// (`rms::pentrace(fit, list(simple = .., nonlinear = .., ...))`); an empty
/// list leaves that component to rms's fallback.
#[derive(Debug, Clone, Default)]
pub struct PenaltyGrid {
    pub simple: Vec<f64>,
    pub nonlinear: Vec<f64>,
    pub interaction: Vec<f64>,
    pub nonlinear_interaction: Vec<f64>,
}

/// One penalty (`ols(penalty = list(...))`); `None` falls back as in rms.
#[derive(Debug, Clone, Copy, Default)]
pub struct Penalty {
    pub simple: f64,
    pub nonlinear: Option<f64>,
    pub interaction: Option<f64>,
    pub nonlinear_interaction: Option<f64>,
}

impl Penalty {
    fn spec(&self) -> PenaltySpec {
        PenaltySpec::uniform(self.simple)
            .nonlinear(self.nonlinear)
            .interaction(self.interaction)
            .nonlinear_interaction(self.nonlinear_interaction)
    }
}

/// The model `pentrace` fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PentraceModel {
    /// Least squares (`ols`).
    Ols,
    /// Binary logistic (`lrm` on a two-level response), with its link.
    Binary(OrdinalLink),
    /// Ordinal cumulative-link model (`lrm` / `orm`), with its link.
    Ordinal(OrdinalLink),
}

/// One penalty of a trace; the penalty components are resolved (fallbacks applied).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PentraceRowResult {
    pub penalty: [f64; 4],
    pub df: f64,
    pub lr_chisq: f64,
    pub aic: f64,
    pub bic: f64,
    pub aic_c: f64,
}

/// Result of [`pentrace`].
#[derive(Debug, Clone)]
pub struct PentraceResult {
    pub rows: Vec<PentraceRowResult>,
    /// Index of the best row.
    pub best: usize,
    pub n_observations: usize,
}

/// Result of [`penalized_ols`], in the caller's column and term order.
#[derive(Debug, Clone)]
pub struct PenalizedOlsResult {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    /// Row-major `(p+1) x (p+1)`, intercept first.
    pub vcov: Vec<f64>,
    /// Row-major `(p+1) x (p+1)`, intercept first (`var.penalty = "sandwich"`).
    pub vcov_sandwich: Vec<f64>,
    pub sigma: f64,
    pub r_squared: f64,
    pub lr_chisq: f64,
    pub effective_df: f64,
    /// Intercept first.
    pub effective_df_diagonal: Vec<f64>,
    /// By caller term.
    pub effective_df_by_term: Vec<f64>,
    pub penalty: [f64; 4],
    pub n_observations: usize,
}

fn invalid(msg: impl Into<String>) -> StatsError {
    StatsError::InvalidInput(msg.into())
}

/// Validate the specification and build the upstream term map.
fn mapped_terms(spec: &PenaltyTerms, p: usize) -> StatsResult<Mapped> {
    if spec.term_of_column.len() != p || spec.nonlinear.len() != p {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "terms has {} labels and nonlinear {} flags for {p} columns",
            spec.term_of_column.len(),
            spec.nonlinear.len()
        )));
    }
    let term_of_column: Vec<Option<usize>> = spec.term_of_column.iter().map(|&t| Some(t)).collect();
    let mapped = build_term_map(&term_of_column, &spec.terms, &spec.nonlinear)?;
    for &t in &spec.categorical {
        let spec_t = spec
            .terms
            .get(t)
            .ok_or_else(|| invalid(format!("categorical term {t} is out of range")))?;
        if spec_t.factors.is_some() {
            return Err(invalid(format!(
                "categorical: '{}' is an interaction, not a main-effect term",
                spec_t.name
            )));
        }
    }
    Ok(mapped)
}

/// The categorical terms in term-map numbering.
fn categorical_in_map(mapped: &Mapped, categorical: &[usize]) -> Vec<usize> {
    mapped
        .term_index
        .iter()
        .enumerate()
        .filter(|(_, t)| categorical.contains(t))
        .map(|(m, _)| m)
        .collect()
}

/// The complete rows, the design in term-map column order and `y`.
fn design(y: &[f64], x: &[Vec<f64>], mapped: &Mapped) -> StatsResult<(Mat<f64>, Col<f64>)> {
    let n = y.len();
    if x.iter().any(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "every x column must have the length of y".into(),
        ));
    }
    let rows: Vec<usize> = (0..n)
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let cols = &mapped.columns;
    let xm = Mat::from_fn(rows.len(), cols.len(), |i, j| x[cols[j]][rows[i]]);
    let yc = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let p = cols.len();
    if rows.len() <= p + 1 {
        return Err(StatsError::InsufficientData {
            rows: rows.len(),
            cols: p,
        });
    }
    // A constant column is aliased with the intercept: the penalized normal
    // equations are singular (its standardised penalty is zero).
    for j in 0..p {
        let first = xm[(0, j)];
        if (0..xm.nrows()).all(|i| xm[(i, j)] == first) {
            return Err(StatsError::SingularMatrix);
        }
    }
    Ok((xm, yc))
}

fn check_penalty(values: &[f64], what: &str) -> StatsResult<()> {
    if let Some(v) = values.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
        return Err(invalid(format!(
            "{what} penalties must be finite and >= 0, got {v}"
        )));
    }
    Ok(())
}

fn ordinal_levels(y: &Col<f64>) -> usize {
    let mut v: Vec<f64> = y.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup();
    v.len()
}

/// `rms::pentrace`: fit `model` at every penalty of `grid` and score it by
/// the effective d.f.; the best row maximises `criterion`.
pub fn pentrace(
    y: &[f64],
    x: &[Vec<f64>],
    spec: &PenaltyTerms,
    grid: &PenaltyGrid,
    model: PentraceModel,
    criterion: PentraceCriterion,
) -> StatsResult<PentraceResult> {
    let p = x.len();
    if p == 0 {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    check_penalty(&grid.simple, "simple")?;
    check_penalty(&grid.nonlinear, "nonlinear")?;
    check_penalty(&grid.interaction, "interaction")?;
    check_penalty(&grid.nonlinear_interaction, "nonlinear_interaction")?;
    if grid.simple.is_empty() {
        return Err(invalid("penalties: no simple penalties to trace"));
    }
    let mapped = mapped_terms(spec, p)?;
    let categorical = categorical_in_map(&mapped, &spec.categorical);
    let (xm, yc) = design(y, x, &mapped)?;
    let penalties = penalty_grid(
        &grid.simple,
        &grid.nonlinear,
        &grid.interaction,
        &grid.nonlinear_interaction,
    );
    if penalties.is_empty() {
        return Err(invalid(
            "penalties: no combination has nonlinear >= simple, interaction >= nonlinear and \
             nonlinear_interaction >= interaction (rms complex.more)",
        ));
    }
    let (rows, best) = match model {
        PentraceModel::Ols => {
            let t = upstream_pentrace_ols(
                &xm,
                &yc,
                &mapped.terms,
                &categorical,
                &penalties,
                criterion,
            )?;
            (t.rows, t.best)
        }
        PentraceModel::Binary(link) | PentraceModel::Ordinal(link) => {
            let levels = ordinal_levels(&yc);
            if matches!(model, PentraceModel::Binary(_)) && levels != 2 {
                return Err(invalid(format!(
                    "model 'logistic' needs a binary response (two distinct values), got {levels}; \
                     use model 'ordinal' for an ordered response"
                )));
            }
            if levels < 2 {
                return Err(StatsError::InsufficientDataMsg(
                    "the response has a single value".into(),
                ));
            }
            let t = upstream_pentrace(
                &xm,
                &yc,
                &mapped.terms,
                &categorical,
                &penalties,
                link,
                criterion,
            )?;
            (t.rows, t.best)
        }
    };
    let rows: Vec<PentraceRowResult> = rows
        .iter()
        .map(|r| PentraceRowResult {
            penalty: r.penalty.resolved(),
            df: r.df,
            lr_chisq: r.lr_chisq,
            aic: r.aic,
            bic: r.bic,
            aic_c: r.aic_c,
        })
        .collect();
    if rows
        .iter()
        .any(|r| !(r.df.is_finite() && r.lr_chisq.is_finite()))
    {
        return Err(StatsError::NumericalFailure(
            "non-finite effective d.f. or likelihood ratio".into(),
        ));
    }
    Ok(PentraceResult {
        rows,
        best,
        n_observations: yc.nrows(),
    })
}

/// `rms::ols(penalty = )`: least squares with rms's penalty matrix for `penalty`.
pub fn penalized_ols(
    y: &[f64],
    x: &[Vec<f64>],
    spec: &PenaltyTerms,
    penalty: &Penalty,
) -> StatsResult<PenalizedOlsResult> {
    let p = x.len();
    if p == 0 {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    let given: Vec<f64> = [
        Some(penalty.simple),
        penalty.nonlinear,
        penalty.interaction,
        penalty.nonlinear_interaction,
    ]
    .into_iter()
    .flatten()
    .collect();
    check_penalty(&given, "the")?;
    let mapped = mapped_terms(spec, p)?;
    let categorical = categorical_in_map(&mapped, &spec.categorical);
    let (xm, yc) = design(y, x, &mapped)?;
    let pspec = penalty.spec();
    let pm = upstream_penalty_matrix(&xm, &mapped.terms, &categorical, &pspec)?;
    let fit = upstream_penalized_ols(&xm, &yc, &pm)?;
    if !(fit.sigma.is_finite() && fit.coefficients.iter().all(|c| c.is_finite())) {
        return Err(StatsError::SingularMatrix);
    }
    let cols = &mapped.columns;
    // Parameter position (intercept first) of every caller column.
    let mut pos = vec![0usize; p];
    for (m, &j) in cols.iter().enumerate() {
        pos[j] = m + 1;
    }
    let param = |a: usize| if a == 0 { 0 } else { pos[a - 1] };
    let k = p + 1;
    let reorder = |m: &Mat<f64>| -> Vec<f64> {
        (0..k * k)
            .map(|e| m[(param(e / k), param(e % k))])
            .collect()
    };
    let diag = &fit.effective_df_diagonal;
    let by_map = effective_df_by_term(&mapped.terms, &Col::from_fn(p, |m| diag[m + 1]));
    let mut by_term = vec![0.0; spec.terms.len()];
    for (m, &t) in mapped.term_index.iter().enumerate() {
        by_term[t] = by_map[m];
    }
    Ok(PenalizedOlsResult {
        intercept: fit.coefficients[0],
        coefficients: (0..p).map(|j| fit.coefficients[pos[j]]).collect(),
        vcov: reorder(&fit.vcov),
        vcov_sandwich: reorder(&fit.vcov_sandwich),
        sigma: fit.sigma,
        r_squared: fit.r_squared,
        lr_chisq: fit.lr_chisq,
        effective_df: fit.effective_df,
        effective_df_diagonal: (0..k).map(|a| diag[param(a)]).collect(),
        effective_df_by_term: by_term,
        penalty: pspec.resolved(),
        n_observations: yc.nrows(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 60;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x1[i] - 0.2 * x2[i] + (i as f64 * 1.3).cos())
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn zero_penalty_is_least_squares() {
        let (y, x) = data();
        let fit = penalized_ols(&y, &x, &PenaltyTerms::columns(2), &Penalty::default()).unwrap();
        assert!((fit.effective_df - 2.0).abs() < 1e-8);
        assert!((fit.effective_df_by_term[0] - 1.0).abs() < 1e-8);
        assert_eq!(fit.vcov.len(), 9);
    }

    #[test]
    fn column_order_is_restored() {
        // Interaction listed before a main effect: the term map reorders the
        // columns, the result must come back in the caller's order.
        let (y, x) = data();
        let x3: Vec<f64> = x[0].iter().zip(&x[1]).map(|(a, b)| a * b).collect();
        let xs = vec![x3.clone(), x[0].clone(), x[1].clone()];
        let spec = PenaltyTerms {
            term_of_column: vec![2, 0, 1],
            terms: vec![
                TermSpec {
                    name: "a".into(),
                    factors: None,
                },
                TermSpec {
                    name: "b".into(),
                    factors: None,
                },
                TermSpec {
                    name: "a * b".into(),
                    factors: Some((0, 1)),
                },
            ],
            nonlinear: vec![false; 3],
            categorical: vec![],
        };
        let pen = Penalty {
            simple: 3.0,
            ..Penalty::default()
        };
        let r = penalized_ols(&y, &xs, &spec, &pen).unwrap();
        let ordered = vec![x[0].clone(), x[1].clone(), x3];
        let spec2 = PenaltyTerms {
            term_of_column: vec![0, 1, 2],
            ..spec.clone()
        };
        let r2 = penalized_ols(&y, &ordered, &spec2, &pen).unwrap();
        assert!((r.coefficients[0] - r2.coefficients[2]).abs() < 1e-10);
        assert!((r.coefficients[1] - r2.coefficients[0]).abs() < 1e-10);
        // vcov entry (x3, x1) in caller order vs. (x3, x1) in the ordered fit.
        assert!((r.vcov[4 + 2] - r2.vcov[3 * 4 + 1]).abs() < 1e-10);
        assert!((r.effective_df_by_term[2] - r2.effective_df_by_term[2]).abs() < 1e-10);
    }

    #[test]
    fn trace_picks_a_row() {
        let (y, x) = data();
        let grid = PenaltyGrid {
            simple: vec![0.0, 1.0, 10.0],
            ..PenaltyGrid::default()
        };
        let t = pentrace(
            &y,
            &x,
            &PenaltyTerms::columns(2),
            &grid,
            PentraceModel::Ols,
            PentraceCriterion::CorrectedAic,
        )
        .unwrap();
        assert_eq!(t.rows.len(), 3);
        assert!(t.best < 3);
        assert!((t.rows[0].df - 2.0).abs() < 1e-8);
    }

    #[test]
    fn binary_model_needs_two_levels() {
        let (y, x) = data();
        let grid = PenaltyGrid {
            simple: vec![0.0],
            ..PenaltyGrid::default()
        };
        assert!(pentrace(
            &y,
            &x,
            &PenaltyTerms::columns(2),
            &grid,
            PentraceModel::Binary(OrdinalLink::Logistic),
            PentraceCriterion::Aic,
        )
        .is_err());
    }
}
