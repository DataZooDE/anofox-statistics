//! Compact, mergeable least-squares states (#266).
//!
//! A state is upstream's [`SufficientStats`] in its versioned byte encoding: the
//! row counts, `Σw`, the weighted means and the centred co-moments, `O(p²)` bytes
//! whatever the number of rows. States built per partition can be stored,
//! merged later and fitted without the rows (OLS, WLS, ridge with λ applied at
//! fit time).
//!
//! Determinism (#226): rows are added in a canonical order (lexicographic by
//! `x_0, …, x_{p-1}, y, w`) and states are merged in the order of their bytes, so
//! the bytes of a state, and every fit from it, do not depend on the row order or
//! the number of threads that produced the inputs.
//!
//! Row-level outputs (HC standard errors, leave-one-out statistics) need the
//! rows and are rejected with an error instead of being approximated.

use crate::errors::{StatsError, StatsResult};
use crate::types::{
    FitResult, FitResultCore, FitResultInference, ModelSummary, OlsOptions, RidgeOptions,
    SolverType, WlsOptions,
};
use anofox_regression::core::{HasModelInfo, ModelInfo, RegressionResult};
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, SufficientStats, WlsRegressor};
use std::cmp::Ordering;

// Same mapping as the direct fits (models/ols.rs).
fn convert_solver(solver: SolverType) -> anofox_regression::SolverType {
    match solver {
        SolverType::Qr => anofox_regression::SolverType::Qr,
        SolverType::Svd => anofox_regression::SolverType::Svd,
        SolverType::Cholesky => anofox_regression::SolverType::Cholesky,
    }
}

fn upstream(e: anofox_regression::solvers::RegressionError) -> StatsError {
    StatsError::from(e)
}

/// Decode a state. A buffer that is not a state is invalid input (an error in SQL),
/// never a degenerate fit (NULL).
fn decode(bytes: &[u8]) -> StatsResult<SufficientStats> {
    SufficientStats::from_bytes(bytes)
        .map_err(|e| StatsError::InvalidInput(format!("not a valid least-squares state: {e}")))
}

/// The state of the rows of `y` / `x` (one inner vector per feature) and the
/// optional `weights`, as bytes. Rows with a non-finite value are skipped, as in
/// the direct fits; a negative weight is an error.
pub fn ls_state_from_rows(
    y: &[f64],
    x: &[Vec<f64>],
    weights: Option<&[f64]>,
) -> StatsResult<Vec<u8>> {
    let n = y.len();
    let p = x.len();
    crate::validation::validate_x_columns(n, x)?;
    if let Some(w) = weights {
        if w.len() != n {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "weights has {} entries, expected {n}",
                w.len()
            )));
        }
    }
    let w_at = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mut rows: Vec<usize> = (0..n)
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()) && w_at(i).is_finite())
        .collect();
    let key_cmp = |&a: &usize, &b: &usize| -> Ordering {
        for col in x {
            match col[a].total_cmp(&col[b]) {
                Ordering::Equal => {}
                o => return o,
            }
        }
        y[a].total_cmp(&y[b]).then(w_at(a).total_cmp(&w_at(b)))
    };
    rows.sort_by(key_cmp);

    let mut state = SufficientStats::new(p);
    let mut x_row = vec![0.0; p];
    for i in rows {
        for (j, col) in x.iter().enumerate() {
            x_row[j] = col[i];
        }
        if weights.is_some() {
            state
                .push_weighted(&x_row, y[i], w_at(i))
                .map_err(upstream)?;
        } else {
            state.push(&x_row, y[i]).map_err(upstream)?;
        }
    }
    Ok(state.to_bytes())
}

/// Merge states (in the order of their bytes, so the result does not depend on
/// the input order). All states must have the same number of features.
pub fn ls_state_merge(states: &[&[u8]]) -> StatsResult<Vec<u8>> {
    let mut sorted: Vec<&[u8]> = states.to_vec();
    sorted.sort();
    let mut it = sorted.into_iter();
    let first = it.next().ok_or(StatsError::NoValidData)?;
    let mut merged = decode(first)?;
    for bytes in it {
        let other = decode(bytes)?;
        if other.n_features() != merged.n_features() {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "cannot merge states with {} and {} features",
                merged.n_features(),
                other.n_features()
            )));
        }
        merged.merge(&other).map_err(upstream)?;
    }
    Ok(merged.to_bytes())
}

/// What a state holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LsStateInfo {
    /// Rows added (including zero-weight rows)
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
    /// Whether any row had a weight other than 1
    pub weighted: bool,
}

/// Row count, feature count and weighting of a state.
pub fn ls_state_info(bytes: &[u8]) -> StatsResult<LsStateInfo> {
    let s = decode(bytes)?;
    Ok(LsStateInfo {
        n_observations: s.n_observations(),
        n_features: s.n_features(),
        weighted: s.is_weighted(),
    })
}

/// A fit from a state: the result plus, with inference, the coefficient
/// covariance `σ̂²·M` (row-major, intercept first; zero rows / columns for
/// aliased features), the same matrix the direct aggregates report.
#[derive(Debug, Clone)]
pub struct StateFit {
    /// Same fields as the direct fit
    pub fit: FitResult,
    /// `(vcov, dim)`, `None` without inference
    pub vcov: Option<(Vec<f64>, usize)>,
}

fn vcov_from(m: Option<faer::Mat<f64>>, rmse: f64, inference: bool) -> Option<(Vec<f64>, usize)> {
    if !inference {
        return None;
    }
    let m = m?;
    let dim = m.nrows();
    let s2 = rmse * rmse;
    let mut out = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            out[a * dim + b] = m[(a, b)] * s2;
        }
    }
    Some((out, dim))
}

fn load(bytes: &[u8]) -> StatsResult<SufficientStats> {
    let s = decode(bytes)?;
    if s.n_positive() == 0 {
        return Err(StatsError::NoValidData);
    }
    Ok(s)
}

fn reject_row_options(hc: bool, loo: bool) -> StatsResult<()> {
    if hc {
        return Err(StatsError::InvalidInput(
            "hc_type needs the rows (Σ e²xxᵀ); it is not available from a state".into(),
        ));
    }
    if loo {
        return Err(StatsError::InvalidInput(
            "compute_loo needs the rows (leverage); it is not available from a state".into(),
        ));
    }
    Ok(())
}

/// The extension's fit result from an upstream result: the same fields and
/// inference as the direct fits (NaN coefficients for aliased columns).
fn to_fit_result(
    info: ModelInfo,
    result: &RegressionResult,
    n_features: usize,
    compute_inference: bool,
    confidence_level: f64,
    with_f: bool,
) -> FitResult {
    let full = |c: Option<&faer::Col<f64>>| -> Vec<f64> {
        (0..n_features)
            .map(|j| c.map_or(f64::NAN, |c| c[j]))
            .collect()
    };
    let core = FitResultCore {
        coefficients: full(Some(&result.coefficients)),
        intercept: result.intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: result.n_observations,
        n_features,
    };
    let inference = compute_inference.then(|| FitResultInference {
        std_errors: full(result.std_errors.as_ref()),
        t_values: full(result.t_statistics.as_ref()),
        p_values: full(result.p_values.as_ref()),
        ci_lower: full(result.conf_interval_lower.as_ref()),
        ci_upper: full(result.conf_interval_upper.as_ref()),
        confidence_level,
        f_statistic: with_f.then_some(result.f_statistic),
        f_pvalue: with_f.then_some(result.f_pvalue),
    });
    FitResult {
        core,
        inference,
        diagnostics: None,
        summary: ModelSummary::from_result(info, result, compute_inference),
    }
}

/// OLS from a state. A weighted state gives the WLS fit.
pub fn fit_ols_from_state(bytes: &[u8], options: &OlsOptions) -> StatsResult<StateFit> {
    reject_row_options(options.hc_type.is_some(), options.compute_loo)?;
    let state = load(bytes)?;
    let mut builder = OlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .compute_inference(options.compute_inference)
        .confidence_level(options.confidence_level)
        .solve_method(convert_solver(options.solver));
    if let Some(tol) = options.alias_tolerance {
        builder = builder.alias_tolerance(tol);
    }
    let fitted = builder.build().fit_from_state(&state).map_err(upstream)?;
    Ok(StateFit {
        fit: to_fit_result(
            fitted.model_info(),
            fitted.result(),
            state.n_features(),
            options.compute_inference,
            options.confidence_level,
            true,
        ),
        vcov: vcov_from(
            fitted.variance_factor(),
            fitted.result().rmse,
            options.compute_inference,
        ),
    })
}

/// WLS from a state (the weights are the ones recorded when the state was built).
pub fn fit_wls_from_state(bytes: &[u8], options: &WlsOptions) -> StatsResult<StateFit> {
    reject_row_options(options.hc_type.is_some(), options.compute_loo)?;
    let state = load(bytes)?;
    let mut builder = WlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .compute_inference(options.compute_inference)
        .confidence_level(options.confidence_level)
        .solve_method(convert_solver(options.solver));
    if let Some(tol) = options.alias_tolerance {
        builder = builder.alias_tolerance(tol);
    }
    let fitted = builder.build().fit_from_state(&state).map_err(upstream)?;
    Ok(StateFit {
        fit: to_fit_result(
            fitted.model_info(),
            fitted.result(),
            state.n_features(),
            options.compute_inference,
            options.confidence_level,
            true,
        ),
        vcov: vcov_from(
            fitted.variance_factor(),
            fitted.result().rmse,
            options.compute_inference,
        ),
    })
}

/// Ridge from a state; `alpha` (with its `lambda_scaling`) is applied here, so one
/// state serves any λ.
pub fn fit_ridge_from_state(bytes: &[u8], options: &RidgeOptions) -> StatsResult<StateFit> {
    if options.alpha < 0.0 {
        return Err(StatsError::InvalidAlpha(options.alpha));
    }
    reject_row_options(false, options.compute_loo)?;
    let state = load(bytes)?;
    let fitted = crate::models::ridge::ridge_regressor(options)
        .fit_from_state(&state)
        .map_err(upstream)?;
    Ok(StateFit {
        fit: to_fit_result(
            fitted.model_info(),
            fitted.result(),
            state.n_features(),
            options.compute_inference,
            options.confidence_level,
            options.alpha == 0.0,
        ),
        vcov: vcov_from(
            fitted.variance_factor().cloned(),
            fitted.result().rmse,
            options.compute_inference,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{fit_ols, fit_ridge, fit_wls};

    fn data(n: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let x1: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + 1e3)
            .collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + 0.5 * x1[i] - 1.5 * x2[i] + (i as f64 * 7.1).sin() * 0.3)
            .collect();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64 * 0.25).collect();
        (y, vec![x1, x2], w)
    }

    fn close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    fn assert_same(a: &FitResult, b: &FitResult) {
        for (u, v) in a.core.coefficients.iter().zip(&b.core.coefficients) {
            assert!(close(*u, *v), "{u} vs {v}");
        }
        assert!(close(a.core.intercept.unwrap(), b.core.intercept.unwrap()));
        assert!(close(a.core.r_squared, b.core.r_squared));
        assert!(close(a.core.residual_std_error, b.core.residual_std_error));
        assert_eq!(a.core.n_observations, b.core.n_observations);
        if let (Some(ia), Some(ib)) = (&a.inference, &b.inference) {
            for (u, v) in ia.std_errors.iter().zip(&ib.std_errors) {
                assert!(close(*u, *v));
            }
            match (ia.f_statistic, ib.f_statistic) {
                (Some(u), Some(v)) => assert!(close(u, v)),
                (u, v) => assert_eq!(u.is_some(), v.is_some()),
            }
        }
    }

    fn split(y: &[f64], x: &[Vec<f64>], w: Option<&[f64]>, cuts: &[usize]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut lo = 0;
        for &hi in cuts.iter().chain(std::iter::once(&y.len())) {
            let xs: Vec<Vec<f64>> = x.iter().map(|c| c[lo..hi].to_vec()).collect();
            out.push(ls_state_from_rows(&y[lo..hi], &xs, w.map(|w| &w[lo..hi])).unwrap());
            lo = hi;
        }
        out
    }

    #[test]
    fn merged_parts_match_direct_fits() {
        let (y, x, w) = data(200);
        let parts = split(&y, &x, None, &[13, 90, 91, 150]);
        let refs: Vec<&[u8]> = parts.iter().map(|v| v.as_slice()).collect();
        let st = ls_state_merge(&refs).unwrap();
        let ols = OlsOptions {
            compute_inference: true,
            ..OlsOptions::default()
        };
        let sf = fit_ols_from_state(&st, &ols).unwrap();
        assert_same(&sf.fit, &fit_ols(&y, &x, &ols).unwrap());
        // vcov = σ̂²·M equals the direct aggregates' matrix (interval_matrix · s²)
        let (v, dim) = sf.vcov.unwrap();
        let (m, mdim) =
            crate::models::interval_matrix(&x, &sf.fit.core.coefficients, true, None, 0.0).unwrap();
        let s2 = sf.fit.core.residual_std_error.powi(2);
        assert_eq!(dim, mdim);
        for (a, b) in v.iter().zip(&m) {
            assert!(close(*a, b * s2), "{a} vs {}", b * s2);
        }
        let ridge = RidgeOptions {
            alpha: 0.7,
            compute_inference: true,
            ..RidgeOptions::default()
        };
        assert_same(
            &fit_ridge_from_state(&st, &ridge).unwrap().fit,
            &fit_ridge(&y, &x, &ridge).unwrap(),
        );

        let wparts = split(&y, &x, Some(&w), &[50, 120]);
        let wrefs: Vec<&[u8]> = wparts.iter().map(|v| v.as_slice()).collect();
        let wst = ls_state_merge(&wrefs).unwrap();
        let wls = WlsOptions {
            compute_inference: true,
            ..WlsOptions::default()
        };
        assert_same(
            &fit_wls_from_state(&wst, &wls).unwrap().fit,
            &fit_wls(&y, &x, &w, &wls).unwrap(),
        );
    }

    #[test]
    fn bytes_do_not_depend_on_row_or_merge_order() {
        let (y, x, _) = data(120);
        let whole = ls_state_from_rows(&y, &x, None).unwrap();
        let rev_y: Vec<f64> = y.iter().rev().copied().collect();
        let rev_x: Vec<Vec<f64>> = x
            .iter()
            .map(|c| c.iter().rev().copied().collect())
            .collect();
        assert_eq!(whole, ls_state_from_rows(&rev_y, &rev_x, None).unwrap());
        let parts = split(&y, &x, None, &[30, 70]);
        let a: Vec<&[u8]> = parts.iter().map(|v| v.as_slice()).collect();
        let b: Vec<&[u8]> = parts.iter().rev().map(|v| v.as_slice()).collect();
        assert_eq!(ls_state_merge(&a).unwrap(), ls_state_merge(&b).unwrap());
    }

    #[test]
    fn non_finite_rows_skipped_and_row_options_rejected() {
        let (mut y, x, _) = data(40);
        y[3] = f64::NAN;
        let st = ls_state_from_rows(&y, &x, None).unwrap();
        assert_eq!(ls_state_info(&st).unwrap().n_observations, 39);
        let hc = OlsOptions {
            hc_type: Some(crate::types::HcType::HC1),
            ..OlsOptions::default()
        };
        assert!(fit_ols_from_state(&st, &hc).is_err());
        let empty = ls_state_from_rows(&[], &[vec![], vec![]], None).unwrap();
        assert!(matches!(
            fit_ols_from_state(&empty, &OlsOptions::default()),
            Err(StatsError::NoValidData)
        ));
        let other = ls_state_from_rows(&y[..5], &[x[0][..5].to_vec()], None).unwrap();
        assert!(ls_state_merge(&[&st, &other]).is_err());
        assert!(matches!(
            fit_ols_from_state(&[0, 1, 2], &OlsOptions::default()),
            Err(StatsError::InvalidInput(_))
        ));
    }
}
