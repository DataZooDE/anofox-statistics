//! Per-observation diagnostics (broom `augment`) for OLS, WLS, Ridge and the
//! GLMs.
//!
//! This module only marshals data: it fits the model exactly as
//! [`crate::models::fit_ols`] / [`crate::models::fit_wls`] do (same row
//! filtering, same non-estimable-column handling) and hands the fit to
//! upstream `anofox_regression::diagnostics::augment`, which computes every
//! measure. The GLMs are fitted by upstream's `PenalizedGlmRegressor`, which
//! runs the same `penalized_glm::fit_*` as [`crate::models::fit_poisson`] and
//! friends, and augmented by its `augment` (`diagnostics::augment_glm`). Ridge
//! has the leverage and leave-one-out columns only (`FittedRidge::loo`). Rows
//! that are not used for fitting get the fit's prediction as `fitted` and NaN
//! everywhere else.

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    BinomialOptions, GammaOptions, GlmPriorOptions, LogisticOptions, NegBinomialOptions,
    OlsOptions, PoissonOptions, RidgeOptions, SolverType, TweedieOptions, WlsOptions,
};
use anofox_regression::diagnostics::{augment, Augment, Augmentable};
use anofox_regression::prelude::*;
use anofox_regression::solvers::penalized_glm::{
    FittedPenalizedGlm, PenalizedGlmFamily, PenalizedGlmRegressor,
};
use faer::{Col, Mat};

/// Per-row diagnostics of an OLS/WLS fit, one entry per input row.
#[derive(Debug, Clone)]
pub struct AugmentResult {
    /// Fitted value (training rows) or prediction (other rows); NaN when the
    /// row's features are missing.
    pub fitted: Vec<f64>,
    /// Residual of type `resid_type`; NaN for rows not used in the fit.
    pub residual: Vec<f64>,
    /// Standardised residual (R `rstandard`); NaN for rows not used in the fit.
    pub std_residual: Vec<f64>,
    /// Externally studentised residual (R `rstudent`); NaN off the fit.
    pub stud_residual: Vec<f64>,
    /// Leverage (R `hatvalues`); NaN for rows not used in the fit.
    pub leverage: Vec<f64>,
    /// Cook's distance; NaN for rows not used in the fit.
    pub cooks_d: Vec<f64>,
    /// DFFITS; NaN for rows not used in the fit.
    pub dffits: Vec<f64>,
    /// Leave-one-out (PRESS) residual `y − ŷ₍ᵢ₎` (#220); NaN for the GLMs,
    /// rows with leverage 1 and rows not used in the fit.
    pub loo_residual: Vec<f64>,
    /// Leave-one-out prediction `ŷ₍ᵢ₎`; NaN where `loo_residual` is.
    pub loo_fitted: Vec<f64>,
    /// One-step approximation of the leave-one-out fitted mean of a GLM
    /// (#235, upstream `Augment::approx_loo_fitted`); NaN for OLS / WLS /
    /// Ridge, rows with leverage 1 and rows not used in the fit.
    pub approx_loo_fitted: Vec<f64>,
    /// `y − approx_loo_fitted`; NaN where `approx_loo_fitted` is.
    pub approx_loo_residual: Vec<f64>,
    /// Whether the row was used to fit the model.
    pub used: Vec<bool>,
    /// Name of the residual type (`"response"` for OLS/WLS, `"deviance"` for
    /// the GLMs).
    pub resid_type: &'static str,
    /// Number of estimated coefficients, intercept included (upstream's
    /// `n_parameters`, aliased columns excluded).
    pub n_params: usize,
}

impl AugmentResult {
    /// `predicted` as `fitted` for every row and NaN diagnostics, to be filled
    /// in for the training rows.
    fn predictions_only(
        predicted: &Col<f64>,
        used: Vec<bool>,
        resid_type: &'static str,
        n_params: usize,
    ) -> Self {
        let n_obs = used.len();
        AugmentResult {
            fitted: (0..n_obs).map(|i| predicted[i]).collect(),
            residual: vec![f64::NAN; n_obs],
            std_residual: vec![f64::NAN; n_obs],
            stud_residual: vec![f64::NAN; n_obs],
            leverage: vec![f64::NAN; n_obs],
            cooks_d: vec![f64::NAN; n_obs],
            dffits: vec![f64::NAN; n_obs],
            loo_residual: vec![f64::NAN; n_obs],
            loo_fitted: vec![f64::NAN; n_obs],
            approx_loo_fitted: vec![f64::NAN; n_obs],
            approx_loo_residual: vec![f64::NAN; n_obs],
            used,
            resid_type,
            n_params,
        }
    }
}

fn convert_solver(solver: SolverType) -> anofox_regression::SolverType {
    match solver {
        SolverType::Qr => anofox_regression::SolverType::Qr,
        SolverType::Svd => anofox_regression::SolverType::Svd,
        SolverType::Cholesky => anofox_regression::SolverType::Cholesky,
    }
}

/// The training design shared by the OLS and WLS paths.
struct Design {
    used: Vec<bool>,
    valid: Vec<usize>,
    kept: Vec<usize>,
}

/// Same row filtering and column dropping as `fit_ols` / `fit_wls`.
fn prepare(
    y: &[f64],
    x: &[Vec<f64>],
    weights: Option<&[f64]>,
    fit_intercept: bool,
) -> StatsResult<Design> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    let n_obs = y.len();
    let n_features = x.len();
    crate::validation::validate_x_columns(n_obs, x)?;
    if let Some(w) = weights {
        if w.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: w.len(),
            });
        }
    }

    let used: Vec<bool> = (0..n_obs)
        .map(|i| {
            y[i].is_finite()
                && weights.is_none_or(|w| w[i].is_finite() && w[i] > 0.0)
                && x.iter().all(|col| col[i].is_finite())
        })
        .collect();
    let valid: Vec<usize> = (0..n_obs).filter(|&i| used[i]).collect();
    if valid.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let dropped = crate::validation::droppable_columns(x, &valid, fit_intercept);
    let kept: Vec<usize> = (0..n_features).filter(|&j| !dropped[j]).collect();
    // Two rows at least: upstream needs them to fit, and a single row has no
    // residual variance to standardise by.
    if (kept.is_empty() && !fit_intercept)
        || valid.len() < 2
        || valid.len() < kept.len() + usize::from(fit_intercept)
    {
        return Err(StatsError::InsufficientData {
            rows: valid.len(),
            cols: n_features,
        });
    }
    Ok(Design { used, valid, kept })
}

/// Fit with `fit`, augment the training rows upstream and scatter the results
/// back to one entry per input row.
fn run<M: Augmentable + FittedRegressor>(
    y: &[f64],
    x: &[Vec<f64>],
    design: Design,
    fit: impl FnOnce(&Mat<f64>, &Col<f64>) -> StatsResult<M>,
) -> StatsResult<AugmentResult> {
    let Design { used, valid, kept } = design;
    let n_obs = y.len();
    let y_col = Col::from_fn(valid.len(), |i| y[valid[i]]);
    let x_mat = Mat::from_fn(valid.len(), kept.len(), |i, j| x[kept[j]][valid[i]]);
    let fitted_model = fit(&x_mat, &y_col)?;
    let aug: Augment = augment(&fitted_model, &x_mat, &y_col, None).map_err(StatsError::from)?;

    // Predictions for every row (NaN features give NaN).
    let x_all = Mat::from_fn(n_obs, kept.len(), |i, j| x[kept[j]][i]);
    let predicted = fitted_model.predict(&x_all);

    let mut out = AugmentResult::predictions_only(
        &predicted,
        used,
        aug.resid_type.as_str(),
        fitted_model.result().n_parameters,
    );
    for (r, &i) in valid.iter().enumerate() {
        out.fitted[i] = aug.fitted[r];
        out.residual[i] = aug.residual[r];
        out.std_residual[i] = aug.std_residual[r];
        out.stud_residual[i] = aug.stud_residual[r];
        out.leverage[i] = aug.leverage[r];
        out.cooks_d[i] = aug.cooks_d[r];
        out.dffits[i] = aug.dffits[r];
        out.loo_residual[i] = aug.loo_residual[r];
        out.loo_fitted[i] = aug.loo_fitted[r];
    }
    Ok(out)
}

/// Fit OLS on the usable rows and return per-row diagnostics for every row.
///
/// A row is used for fitting when `y` and every feature are finite; pass NaN
/// as `y` for rows that must only be predicted.
pub fn augment_ols(y: &[f64], x: &[Vec<f64>], options: &OlsOptions) -> StatsResult<AugmentResult> {
    let design = prepare(y, x, None, options.fit_intercept)?;
    run(y, x, design, |x_mat, y_col| {
        let mut builder = OlsRegressor::builder()
            .with_intercept(options.fit_intercept)
            .compute_inference(false)
            .solve_method(convert_solver(options.solver));
        if let Some(tol) = options.alias_tolerance {
            builder = builder.alias_tolerance(tol);
        }
        builder.build().fit(x_mat, y_col).map_err(StatsError::from)
    })
}

/// Fit WLS on the usable rows and return per-row diagnostics for every row.
///
/// A row is used for fitting when `y`, its weight and every feature are
/// finite and the weight is positive.
pub fn augment_wls(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
) -> StatsResult<AugmentResult> {
    if weights.is_empty() {
        return Err(StatsError::EmptyInput { field: "weights" });
    }
    let design = prepare(y, x, Some(weights), options.fit_intercept)?;
    let w_col = Col::from_fn(design.valid.len(), |i| weights[design.valid[i]]);
    run(y, x, design, |x_mat, y_col| {
        let mut builder = WlsRegressor::builder()
            .with_intercept(options.fit_intercept)
            .weights(w_col)
            .compute_inference(false)
            .solve_method(convert_solver(options.solver));
        if let Some(tol) = options.alias_tolerance {
            builder = builder.alias_tolerance(tol);
        }
        builder.build().fit(x_mat, y_col).map_err(StatsError::from)
    })
}

/// Fit Ridge on the usable rows (as `fit_ridge`: finite `y` and features,
/// non-estimable columns dropped) and return, for every row, the fitted value
/// (prediction off the fit), the response residual, the ridge leverage and the
/// exact leave-one-out residual / prediction (#235, upstream
/// `FittedRidge::loo`). The standardised / studentised residuals, Cook's
/// distance and DFFITS have no ridge counterpart in R and are NaN.
pub fn augment_ridge(
    y: &[f64],
    x: &[Vec<f64>],
    options: &RidgeOptions,
) -> StatsResult<AugmentResult> {
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }
    let Design { used, valid, kept } = prepare(y, x, None, options.fit_intercept)?;
    let n_obs = y.len();
    let y_col = Col::from_fn(valid.len(), |i| y[valid[i]]);
    let x_mat = Mat::from_fn(valid.len(), kept.len(), |i, j| x[kept[j]][valid[i]]);
    let ridge_options = RidgeOptions {
        compute_inference: false,
        ..options.clone()
    };
    let fitted_model = crate::models::ridge::ridge_regressor(&ridge_options)
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let loo = fitted_model.loo(&x_mat, &y_col).map_err(StatsError::from)?;
    let x_all = Mat::from_fn(n_obs, kept.len(), |i, j| x[kept[j]][i]);
    let result = fitted_model.result();
    let mut out = AugmentResult::predictions_only(
        &fitted_model.predict(&x_all),
        used,
        "response",
        result.n_parameters,
    );
    for (r, &i) in valid.iter().enumerate() {
        out.fitted[i] = result.fitted_values[r];
        out.residual[i] = result.residuals[r];
        out.leverage[i] = loo.leverage[r];
        out.loo_residual[i] = loo.loo_residual[r];
        out.loo_fitted[i] = loo.loo_fitted[r];
    }
    Ok(out)
}

/// Partial-residual (CCPR) and added-variable plot data, one entry per input
/// row and x column (term-major: entry `j * len + i` is column `j`, row `i`).
///
/// Computed by `anofox_regression::diagnostics::{partial_residuals,
/// partial_residuals_glm, added_variable}`. Rows not used for fitting, columns
/// the fit dropped (aliased / constant) and a GLM offset column are NaN; `x`
/// is the raw feature value. The added-variable coordinates exist for OLS /
/// WLS only (`has_added_variable`); for the GLMs they are NaN.
#[derive(Debug, Clone)]
pub struct PartialResidualsResult {
    pub n_terms: usize,
    pub len: usize,
    pub x: Vec<f64>,
    /// `b_j (x_j - mean(x_j))` (uncentred without an intercept).
    pub component: Vec<f64>,
    /// residual (working residual for GLMs) + component.
    pub partial_residual: Vec<f64>,
    /// Residual of `x_j` on the other columns (`car::avPlots`).
    pub x_residual: Vec<f64>,
    /// Residual of `y` on the other columns.
    pub y_residual: Vec<f64>,
    pub has_added_variable: bool,
}

impl PartialResidualsResult {
    fn new(x: &[Vec<f64>], has_added_variable: bool) -> Self {
        let len = x.first().map_or(0, Vec::len);
        let nan = vec![f64::NAN; x.len() * len];
        Self {
            n_terms: x.len(),
            len,
            x: x.iter().flat_map(|c| c.iter().copied()).collect(),
            component: nan.clone(),
            partial_residual: nan.clone(),
            x_residual: nan.clone(),
            y_residual: nan,
            has_added_variable,
        }
    }
}

/// OLS / WLS partial residuals and added-variable coordinates on the usable rows.
fn partial_residuals_linear(
    y: &[f64],
    x: &[Vec<f64>],
    weights: Option<&[f64]>,
    fit_intercept: bool,
    solver: SolverType,
) -> StatsResult<PartialResidualsResult> {
    use anofox_regression::diagnostics::{added_variable, partial_residuals};
    let Design { valid, kept, .. } = prepare(y, x, weights, fit_intercept)?;
    let y_col = Col::from_fn(valid.len(), |i| y[valid[i]]);
    let x_mat = Mat::from_fn(valid.len(), kept.len(), |i, j| x[kept[j]][valid[i]]);
    let w_col = weights.map(|w| Col::from_fn(valid.len(), |i| w[valid[i]]));
    let terms = match &w_col {
        Some(w) => {
            let fit = WlsRegressor::builder()
                .with_intercept(fit_intercept)
                .weights(w.clone())
                .compute_inference(false)
                .solve_method(convert_solver(solver))
                .build()
                .fit(&x_mat, &y_col)
                .map_err(StatsError::from)?;
            partial_residuals(&fit, &x_mat, &y_col)
        }
        None => {
            let fit = OlsRegressor::builder()
                .with_intercept(fit_intercept)
                .compute_inference(false)
                .solve_method(convert_solver(solver))
                .build()
                .fit(&x_mat, &y_col)
                .map_err(StatsError::from)?;
            partial_residuals(&fit, &x_mat, &y_col)
        }
    }
    .map_err(StatsError::from)?;
    let av =
        added_variable(&x_mat, &y_col, w_col.as_ref(), fit_intercept).map_err(StatsError::from)?;

    let mut out = PartialResidualsResult::new(x, true);
    let len = out.len;
    for t in &terms {
        let j = kept[t.term];
        for (r, &i) in valid.iter().enumerate() {
            out.component[j * len + i] = t.component[r];
            out.partial_residual[j * len + i] = t.partial_residual[r];
        }
    }
    for a in &av {
        let j = kept[a.term];
        // An aliased column has no coefficient; leave its coordinates NaN too.
        if terms.iter().all(|t| t.term != a.term) {
            continue;
        }
        for (r, &i) in valid.iter().enumerate() {
            out.x_residual[j * len + i] = a.x_residual[r];
            out.y_residual[j * len + i] = a.y_residual[r];
        }
    }
    Ok(out)
}

/// Partial residuals (R `residuals(lm, type = "partial")`, `car::crPlots`) and
/// added-variable coordinates (`car::avPlots`) of an OLS fit.
pub fn partial_residuals_ols(
    y: &[f64],
    x: &[Vec<f64>],
    options: &OlsOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_linear(y, x, None, options.fit_intercept, options.solver)
}

/// As [`partial_residuals_ols`] for a WLS fit (`lm(weights = w)`).
pub fn partial_residuals_wls(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
) -> StatsResult<PartialResidualsResult> {
    if weights.is_empty() {
        return Err(StatsError::EmptyInput { field: "weights" });
    }
    partial_residuals_linear(y, x, Some(weights), options.fit_intercept, options.solver)
}

/// The fit settings shared by the six GLM families.
struct GlmSpec<'a> {
    family: PenalizedGlmFamily,
    fit_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
    lambda: f64,
    prior_opts: &'a GlmPriorOptions,
    /// 1-based column of `x` holding the offset, as in the `fit_*` options.
    offset_column: Option<usize>,
    estimate_dispersion: bool,
}

/// A GLM fit with the feature matrix over all rows, `y` and the offset column.
type GlmFit = (FittedPenalizedGlm, Mat<f64>, Col<f64>, Option<Col<f64>>);

/// The GLM fit shared by the augment and partial-residual entry points.
fn fit_glm_spec(y: &[f64], x: &[Vec<f64>], spec: GlmSpec<'_>) -> StatsResult<GlmFit> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    let n_obs = y.len();
    crate::validation::validate_x_columns(n_obs, x)?;
    let offset_idx = match spec.offset_column {
        None => None,
        Some(c) if c >= 1 && c <= x.len() => Some(c - 1),
        Some(c) => {
            return Err(StatsError::InvalidValue {
                field: "offset",
                message: format!("offset column {c} is outside the {} x columns", x.len()),
            })
        }
    };
    let features: Vec<&Vec<f64>> = x
        .iter()
        .enumerate()
        .filter(|(j, _)| Some(*j) != offset_idx)
        .map(|(_, c)| c)
        .collect();
    let x_mat = Mat::from_fn(n_obs, features.len(), |i, j| features[j][i]);
    let y_col = Col::from_fn(n_obs, |i| y[i]);
    let offset = offset_idx.map(|k| Col::from_fn(n_obs, |i| x[k][i]));

    let mut builder = PenalizedGlmRegressor::builder()
        .family(spec.family)
        .with_intercept(spec.fit_intercept)
        .max_iterations(spec.max_iterations)
        .tolerance(spec.tolerance)
        .compute_inference(false)
        .lambda(spec.lambda)
        .priors(spec.prior_opts.priors.clone())
        .vcov_type(spec.prior_opts.vcov)
        .estimate_dispersion(spec.estimate_dispersion);
    if let Some(off) = &offset {
        builder = builder.offset(off.clone());
    }
    let fitted_model = builder
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    Ok((fitted_model, x_mat, y_col, offset))
}

/// Fit a GLM on the usable rows and return per-row diagnostics for every row.
///
/// `x` is laid out as for the `fit_*` functions (the offset column, if any,
/// included). The fit drops rows with a non-finite `y`, feature or offset;
/// those rows get the fit's prediction (NaN if a feature or the offset is
/// missing) and NaN diagnostics.
fn augment_glm_spec(y: &[f64], x: &[Vec<f64>], spec: GlmSpec<'_>) -> StatsResult<AugmentResult> {
    let n_obs = y.len();
    let (fitted_model, x_mat, y_col, offset) = fit_glm_spec(y, x, spec)?;
    let aug = fitted_model
        .augment(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let predicted = match &offset {
        Some(off) => fitted_model.predict_with_offset(&x_mat, off),
        None => fitted_model.predict(&x_mat),
    };

    let result = fitted_model.result();
    let mut used = vec![false; n_obs];
    for &i in &result.valid_rows {
        used[i] = true;
    }
    let n_params = result
        .core
        .coefficients
        .iter()
        .filter(|b| b.is_finite())
        .count()
        + usize::from(result.core.intercept.is_some());
    let col = |c: &Col<f64>, i: usize| if used[i] { c[i] } else { f64::NAN };
    Ok(AugmentResult {
        fitted: (0..n_obs)
            .map(|i| if used[i] { aug.fitted[i] } else { predicted[i] })
            .collect(),
        residual: (0..n_obs).map(|i| col(&aug.residual, i)).collect(),
        std_residual: (0..n_obs).map(|i| col(&aug.std_residual, i)).collect(),
        stud_residual: (0..n_obs).map(|i| col(&aug.stud_residual, i)).collect(),
        leverage: (0..n_obs).map(|i| col(&aug.leverage, i)).collect(),
        cooks_d: (0..n_obs).map(|i| col(&aug.cooks_d, i)).collect(),
        dffits: (0..n_obs).map(|i| col(&aug.dffits, i)).collect(),
        loo_residual: (0..n_obs).map(|i| col(&aug.loo_residual, i)).collect(),
        loo_fitted: (0..n_obs).map(|i| col(&aug.loo_fitted, i)).collect(),
        approx_loo_fitted: (0..n_obs).map(|i| col(&aug.approx_loo_fitted, i)).collect(),
        approx_loo_residual: (0..n_obs)
            .map(|i| col(&aug.approx_loo_residual, i))
            .collect(),
        resid_type: aug.resid_type.as_str(),
        n_params,
        used,
    })
}

/// GLM partial residuals (working residual + component, R
/// `residuals(glm, type = "partial")`); no added-variable coordinates.
fn partial_residuals_glm_spec(
    y: &[f64],
    x: &[Vec<f64>],
    spec: GlmSpec<'_>,
) -> StatsResult<PartialResidualsResult> {
    let offset_idx = spec.offset_column.and_then(|c| c.checked_sub(1));
    let (fitted_model, ..) = fit_glm_spec(y, x, spec)?;
    let terms = anofox_regression::diagnostics::partial_residuals_glm(fitted_model.result(), y, x)
        .map_err(StatsError::from)?;
    let mut out = PartialResidualsResult::new(x, false);
    let len = out.len;
    // Terms are numbered over the features, the offset column excluded.
    let positions: Vec<usize> = (0..x.len()).filter(|&j| Some(j) != offset_idx).collect();
    if let Some(o) = offset_idx {
        out.x[o * len..(o + 1) * len].fill(f64::NAN);
    }
    for t in &terms {
        let j = positions[t.term];
        for i in 0..len {
            out.component[j * len + i] = t.component[i];
            out.partial_residual[j * len + i] = t.partial_residual[i];
        }
    }
    Ok(out)
}

impl<'a> From<&'a PoissonOptions> for GlmSpec<'a> {
    fn from(o: &'a PoissonOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::Poisson(o.link),
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: o.estimate_dispersion,
        }
    }
}

/// Per-row diagnostics of a Poisson fit (options as [`crate::models::fit_poisson`]).
pub fn augment_poisson(
    y: &[f64],
    x: &[Vec<f64>],
    o: &PoissonOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a Poisson fit (options as [`crate::models::fit_poisson`]).
pub fn partial_residuals_poisson(
    y: &[f64],
    x: &[Vec<f64>],
    o: &PoissonOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

impl<'a> From<&'a BinomialOptions> for GlmSpec<'a> {
    fn from(o: &'a BinomialOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::Binomial(o.link),
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        }
    }
}

/// Per-row diagnostics of a Binomial fit (options as [`crate::models::fit_binomial`]).
pub fn augment_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &BinomialOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a Binomial fit (options as [`crate::models::fit_binomial`]).
pub fn partial_residuals_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &BinomialOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

impl<'a> From<&'a LogisticOptions> for GlmSpec<'a> {
    fn from(o: &'a LogisticOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::Logistic {
                threshold: o.threshold,
            },
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        }
    }
}

/// Per-row diagnostics of a logistic fit (options as [`crate::models::fit_logistic`]).
pub fn augment_logistic(
    y: &[f64],
    x: &[Vec<f64>],
    o: &LogisticOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a logistic fit (options as [`crate::models::fit_logistic`]).
pub fn partial_residuals_logistic(
    y: &[f64],
    x: &[Vec<f64>],
    o: &LogisticOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

impl<'a> From<&'a NegBinomialOptions> for GlmSpec<'a> {
    fn from(o: &'a NegBinomialOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::NegativeBinomial { theta: o.alpha },
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        }
    }
}

/// Per-row diagnostics of a Negative Binomial fit (options as [`crate::models::fit_negbinomial`]; `alpha` is the fixed `theta`).
pub fn augment_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &NegBinomialOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a Negative Binomial fit (options as [`crate::models::fit_negbinomial`]; `alpha` is the fixed `theta`).
pub fn partial_residuals_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &NegBinomialOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

impl<'a> From<&'a GammaOptions> for GlmSpec<'a> {
    fn from(o: &'a GammaOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::Gamma,
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        }
    }
}

/// Per-row diagnostics of a Gamma fit (options as [`crate::models::fit_gamma`]).
pub fn augment_gamma(y: &[f64], x: &[Vec<f64>], o: &GammaOptions) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a Gamma fit (options as [`crate::models::fit_gamma`]).
pub fn partial_residuals_gamma(
    y: &[f64],
    x: &[Vec<f64>],
    o: &GammaOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

impl<'a> From<&'a TweedieOptions> for GlmSpec<'a> {
    fn from(o: &'a TweedieOptions) -> Self {
        GlmSpec {
            family: PenalizedGlmFamily::Tweedie { power: o.power },
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        }
    }
}

/// Per-row diagnostics of a Tweedie fit (options as [`crate::models::fit_tweedie`]).
pub fn augment_tweedie(
    y: &[f64],
    x: &[Vec<f64>],
    o: &TweedieOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(y, x, o.into())
}

/// Partial residuals of a Tweedie fit (options as [`crate::models::fit_tweedie`]).
pub fn partial_residuals_tweedie(
    y: &[f64],
    x: &[Vec<f64>],
    o: &TweedieOptions,
) -> StatsResult<PartialResidualsResult> {
    partial_residuals_glm_spec(y, x, o.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_augment_scatters_back_to_input_rows() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]];
        let y = vec![1.1, 2.3, 2.9, 4.2, f64::NAN, 6.1, 9.7];
        let r = augment_ols(&y, &x, &OlsOptions::default()).unwrap();
        assert_eq!(r.fitted.len(), 7);
        assert_eq!(r.used, vec![true, true, true, true, false, true, true]);
        assert!(r.fitted[4].is_finite());
        assert!(r.residual[4].is_nan() && r.leverage[4].is_nan());
        assert!(r.leverage[6] > r.leverage[2]);
        let h: f64 = r.leverage.iter().filter(|v| v.is_finite()).sum();
        assert!((h - 2.0).abs() < 1e-10);
        assert_eq!(r.n_params, 2);
        assert_eq!(r.resid_type, "response");
        // Leave-one-out (#220): e / (1 - h), and y minus it.
        assert!(r.loo_residual[4].is_nan() && r.loo_fitted[4].is_nan());
        for i in [0, 1, 2, 3, 5, 6] {
            let want = r.residual[i] / (1.0 - r.leverage[i]);
            assert!((r.loo_residual[i] - want).abs() < 1e-12);
            assert!((r.loo_fitted[i] - (y[i] - want)).abs() < 1e-12);
        }
    }

    #[test]
    fn wls_augment_uses_prior_weights() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let y = vec![1.0, 2.5, 2.8, 4.4, 4.9, 6.3];
        let w = vec![1.0, 2.0, 1.0, 0.0, 3.0, 1.0];
        let r = augment_wls(&y, &x, &w, &WlsOptions::default()).unwrap();
        assert!(!r.used[3]);
        assert!(r.residual[3].is_nan());
        let h: f64 = r.leverage.iter().filter(|v| v.is_finite()).sum();
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn poisson_augment_predicts_held_out_rows() {
        let x = vec![
            vec![0.1, 0.4, 0.9, 1.3, 1.8, 2.2, 2.9, 3.1, 1.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ];
        let y = vec![1.0, 1.0, 3.0, 2.0, 6.0, 5.0, 14.0, 11.0, f64::NAN];
        let r = augment_poisson(&y, &x, &PoissonOptions::default()).unwrap();
        assert_eq!(r.resid_type, "deviance");
        assert_eq!(r.n_params, 3);
        assert!(!r.used[8] && r.used[0]);
        assert!(r.fitted[8].is_finite() && r.fitted[8] > 0.0);
        assert!(r.residual[8].is_nan() && r.leverage[8].is_nan());
        let h: f64 = r.leverage.iter().filter(|v| v.is_finite()).sum();
        assert!((h - 3.0).abs() < 1e-8);
        // No exact leave-one-out residuals for a GLM; the one-step
        // approximation (#235) for the training rows only.
        assert!(r.loo_residual.iter().all(|v| v.is_nan()));
        assert!(r.approx_loo_fitted[8].is_nan() && r.approx_loo_residual[8].is_nan());
        for (i, &yi) in y.iter().enumerate().take(8) {
            assert!(r.approx_loo_fitted[i] > 0.0);
            assert!((r.approx_loo_residual[i] - (yi - r.approx_loo_fitted[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn ridge_augment_matches_refits_without_each_row() {
        let x = vec![
            vec![0.1, 0.4, 0.9, 1.3, 1.8, 2.2, 2.9, 3.1, 1.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, f64::NAN],
        ];
        let y = vec![1.2, 1.1, 3.0, 2.4, 5.6, 4.9, 8.1, 7.0, 3.0];
        let options = RidgeOptions {
            alpha: 0.7,
            ..Default::default()
        };
        let r = augment_ridge(&y, &x, &options).unwrap();
        assert_eq!(r.resid_type, "response");
        assert_eq!(r.n_params, 3);
        assert!(!r.used[8] && r.fitted[8].is_nan() && r.loo_residual[8].is_nan());
        assert!(r.std_residual.iter().all(|v| v.is_nan()));
        assert!(r.approx_loo_fitted.iter().all(|v| v.is_nan()));
        for i in 0..8 {
            let keep: Vec<usize> = (0..8).filter(|&k| k != i).collect();
            let xi: Vec<Vec<f64>> = x
                .iter()
                .map(|c| keep.iter().map(|&k| c[k]).collect())
                .collect();
            let yi: Vec<f64> = keep.iter().map(|&k| y[k]).collect();
            let fit = crate::models::fit_ridge(&yi, &xi, &options).unwrap();
            let c = &fit.core.coefficients;
            let pred = fit.core.intercept.unwrap() + c[0] * x[0][i] + c[1] * x[1][i];
            assert!((r.loo_fitted[i] - pred).abs() < 1e-10, "row {i}");
            assert!((r.loo_residual[i] - (y[i] - pred)).abs() < 1e-10);
            assert!(r.leverage[i] > 0.0 && r.leverage[i] < 1.0);
        }
    }

    #[test]
    fn penalized_glm_augment_is_rejected() {
        let x = vec![vec![0.1, 0.4, 0.9, 1.3, 1.8, 2.2]];
        let y = vec![1.0, 1.0, 3.0, 2.0, 6.0, 5.0];
        let o = PoissonOptions {
            lambda: 1.0,
            ..Default::default()
        };
        assert!(augment_poisson(&y, &x, &o).is_err());
    }

    #[test]
    fn ols_partial_residuals_add_component_to_residual() {
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0],
        ];
        let y = vec![1.1, 2.3, 2.9, 4.2, f64::NAN, 6.1, 9.7];
        let r = partial_residuals_ols(&y, &x, &OlsOptions::default()).unwrap();
        assert_eq!((r.n_terms, r.len), (2, 7));
        assert!(r.has_added_variable);
        assert!(r.partial_residual[4].is_nan() && r.x[4] == 5.0);
        // Components of a term are centred over the rows used.
        let c: f64 = (0..7)
            .map(|i| r.component[i])
            .filter(|v| v.is_finite())
            .sum();
        assert!(c.abs() < 1e-10);
        // Frisch-Waugh-Lovell: the AV slope is the coefficient, i.e. the
        // component's slope in x.
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for i in 0..7 {
            if r.x_residual[i].is_finite() {
                sxy += r.x_residual[i] * r.y_residual[i];
                sxx += r.x_residual[i] * r.x_residual[i];
            }
        }
        let b = (r.component[1] - r.component[0]) / (x[0][1] - x[0][0]);
        assert!((sxy / sxx - b).abs() < 1e-10);
    }

    #[test]
    fn glm_partial_residuals_skip_the_offset() {
        let x = vec![
            vec![0.1, 0.4, 0.9, 1.3, 1.8, 2.2, 2.9, 3.1],
            vec![0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1],
        ];
        let y = vec![1.0, 1.0, 3.0, 2.0, 6.0, 5.0, 14.0, 11.0];
        let o = PoissonOptions {
            offset_column: Some(2),
            ..Default::default()
        };
        let r = partial_residuals_poisson(&y, &x, &o).unwrap();
        assert!(!r.has_added_variable);
        assert!(r.partial_residual[0].is_finite());
        assert!(r.partial_residual[8].is_nan() && r.x[8].is_nan());
    }

    #[test]
    fn augment_needs_two_rows() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0, f64::NAN];
        assert!(augment_ols(&y, &x, &OlsOptions::default()).is_err());
    }
}
