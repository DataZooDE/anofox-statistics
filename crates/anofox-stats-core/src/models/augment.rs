//! Per-observation diagnostics (broom `augment`) for OLS and WLS.
//!
//! This module only marshals data: it fits the model exactly as
//! [`crate::models::fit_ols`] / [`crate::models::fit_wls`] do (same row
//! filtering, same non-estimable-column handling) and hands the fit to
//! upstream `anofox_regression::diagnostics::augment`, which computes every
//! measure. Rows that are not used for fitting get the fit's prediction as
//! `fitted` and NaN everywhere else.

use crate::errors::{StatsError, StatsResult};
use crate::types::{OlsOptions, SolverType, WlsOptions};
use anofox_regression::diagnostics::{augment, Augment, Augmentable};
use anofox_regression::prelude::*;
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
    /// Name of the residual type (`"response"`).
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
    fn augment_needs_two_rows() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0, f64::NAN];
        assert!(augment_ols(&y, &x, &OlsOptions::default()).is_err());
    }
}
