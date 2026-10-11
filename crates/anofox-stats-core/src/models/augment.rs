//! Per-observation diagnostics (broom `augment`) for OLS, WLS and the GLMs.
//!
//! This module only marshals data: it fits the model exactly as
//! [`crate::models::fit_ols`] / [`crate::models::fit_wls`] do (same row
//! filtering, same non-estimable-column handling) and hands the fit to
//! upstream `anofox_regression::diagnostics::augment`, which computes every
//! measure. The GLMs are fitted by upstream's `PenalizedGlmRegressor`, which
//! runs the same `penalized_glm::fit_*` as [`crate::models::fit_poisson`] and
//! friends, and augmented by its `augment` (`diagnostics::augment_glm`). Rows
//! that are not used for fitting get the fit's prediction as `fitted` and NaN
//! everywhere else.

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    BinomialOptions, GammaOptions, GlmPriorOptions, LogisticOptions, NegBinomialOptions,
    OlsOptions, PoissonOptions, SolverType, TweedieOptions, WlsOptions,
};
use anofox_regression::diagnostics::{augment, Augment, Augmentable};
use anofox_regression::prelude::*;
use anofox_regression::solvers::penalized_glm::{PenalizedGlmFamily, PenalizedGlmRegressor};
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
    /// Whether the row was used to fit the model.
    pub used: Vec<bool>,
    /// Name of the residual type (`"response"` for OLS/WLS, `"deviance"` for
    /// the GLMs).
    pub resid_type: &'static str,
    /// Number of estimated coefficients, intercept included (upstream's
    /// `n_parameters`, aliased columns excluded).
    pub n_params: usize,
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

    let mut out = AugmentResult {
        fitted: (0..n_obs).map(|i| predicted[i]).collect(),
        residual: vec![f64::NAN; n_obs],
        std_residual: vec![f64::NAN; n_obs],
        stud_residual: vec![f64::NAN; n_obs],
        leverage: vec![f64::NAN; n_obs],
        cooks_d: vec![f64::NAN; n_obs],
        dffits: vec![f64::NAN; n_obs],
        used,
        resid_type: aug.resid_type.as_str(),
        n_params: fitted_model.result().n_parameters,
    };
    for (r, &i) in valid.iter().enumerate() {
        out.fitted[i] = aug.fitted[r];
        out.residual[i] = aug.residual[r];
        out.std_residual[i] = aug.std_residual[r];
        out.stud_residual[i] = aug.stud_residual[r];
        out.leverage[i] = aug.leverage[r];
        out.cooks_d[i] = aug.cooks_d[r];
        out.dffits[i] = aug.dffits[r];
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
        OlsRegressor::builder()
            .with_intercept(options.fit_intercept)
            .compute_inference(false)
            .solve_method(convert_solver(options.solver))
            .build()
            .fit(x_mat, y_col)
            .map_err(StatsError::from)
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
        WlsRegressor::builder()
            .with_intercept(options.fit_intercept)
            .weights(w_col)
            .compute_inference(false)
            .solve_method(convert_solver(options.solver))
            .build()
            .fit(x_mat, y_col)
            .map_err(StatsError::from)
    })
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

/// Fit a GLM on the usable rows and return per-row diagnostics for every row.
///
/// `x` is laid out as for the `fit_*` functions (the offset column, if any,
/// included). The fit drops rows with a non-finite `y`, feature or offset;
/// those rows get the fit's prediction (NaN if a feature or the offset is
/// missing) and NaN diagnostics.
fn augment_glm_spec(y: &[f64], x: &[Vec<f64>], spec: GlmSpec<'_>) -> StatsResult<AugmentResult> {
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
        resid_type: aug.resid_type.as_str(),
        n_params,
        used,
    })
}

/// Per-row diagnostics of a Poisson fit (options as [`crate::models::fit_poisson`]).
pub fn augment_poisson(
    y: &[f64],
    x: &[Vec<f64>],
    o: &PoissonOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
        GlmSpec {
            family: PenalizedGlmFamily::Poisson(o.link),
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: o.estimate_dispersion,
        },
    )
}

/// Per-row diagnostics of a Binomial fit (options as [`crate::models::fit_binomial`]).
pub fn augment_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &BinomialOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
        GlmSpec {
            family: PenalizedGlmFamily::Binomial(o.link),
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        },
    )
}

/// Per-row diagnostics of a logistic fit (options as [`crate::models::fit_logistic`]).
pub fn augment_logistic(
    y: &[f64],
    x: &[Vec<f64>],
    o: &LogisticOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
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
        },
    )
}

/// Per-row diagnostics of a Negative Binomial fit (options as
/// [`crate::models::fit_negbinomial`]; `alpha` is the fixed `theta`).
pub fn augment_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    o: &NegBinomialOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
        GlmSpec {
            family: PenalizedGlmFamily::NegativeBinomial { theta: o.alpha },
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        },
    )
}

/// Per-row diagnostics of a Gamma fit (options as [`crate::models::fit_gamma`]).
pub fn augment_gamma(y: &[f64], x: &[Vec<f64>], o: &GammaOptions) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
        GlmSpec {
            family: PenalizedGlmFamily::Gamma,
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        },
    )
}

/// Per-row diagnostics of a Tweedie fit (options as [`crate::models::fit_tweedie`]).
pub fn augment_tweedie(
    y: &[f64],
    x: &[Vec<f64>],
    o: &TweedieOptions,
) -> StatsResult<AugmentResult> {
    augment_glm_spec(
        y,
        x,
        GlmSpec {
            family: PenalizedGlmFamily::Tweedie { power: o.power },
            fit_intercept: o.fit_intercept,
            max_iterations: o.max_iterations,
            tolerance: o.tolerance,
            lambda: o.lambda,
            prior_opts: &o.prior_opts,
            offset_column: o.offset_column,
            estimate_dispersion: false,
        },
    )
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
    fn augment_needs_two_rows() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0, f64::NAN];
        assert!(augment_ols(&y, &x, &OlsOptions::default()).is_err());
    }
}
