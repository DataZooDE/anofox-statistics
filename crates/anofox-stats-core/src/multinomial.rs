//! Multinomial (softmax) logistic regression with an elastic-net penalty
//! (#314): upstream anofox-regression `MultinomialLogisticRegressor`, the
//! objective of scikit-learn's `LogisticRegression` with `lambda = 1 / C`:
//!
//! ```text
//! Σᵢ sᵢ ℓᵢ(B, b) + λ · ( α ‖B‖₁ + (1 − α)/2 ‖B‖²₂ )
//! ```
//!
//! The wrapper takes the class of each row as an index `0..K` (the caller
//! maps its labels onto their sorted order), fits upstream and hands back
//! the coefficient matrix, the intercepts and the fit statistics.
//!
//! [`multinomial_predict_proba`] scores a row from a stored model: it
//! rebuilds upstream's `FittedMultinomialLogistic` from the coefficient
//! matrix and intercepts (`from_parts`, classes `0..K`) and returns its
//! `predict_proba` and `predict`, so a stored model predicts exactly as the
//! fit.

use crate::{StatsError, StatsResult};
use anofox_regression::solvers::{
    FittedMultinomialLogistic, IterativeFit, MultinomialLogisticRegressor,
};
use faer::{Col, Mat};

/// Fit options (sklearn `LogisticRegression` names in brackets).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultinomialOptions {
    /// Penalty strength `λ = 1 / C` (sklearn default `C = 1`); 0: no penalty.
    pub lambda: f64,
    /// Elastic-net mixing in `[0, 1]`: 0 ridge, 1 lasso (`l1_ratio`).
    pub l1_ratio: f64,
    /// Fit unpenalized intercepts (`fit_intercept`).
    pub fit_intercept: bool,
    /// Maximum proximal Newton iterations (`max_iter`).
    pub max_iterations: usize,
    /// Convergence tolerance on the largest parameter change.
    pub tolerance: f64,
}

impl Default for MultinomialOptions {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.0,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }
}

/// A fitted multinomial logistic regression.
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialFit {
    /// Number of classes `K` (>= 2).
    pub n_classes: usize,
    /// Coefficients, one row per linear predictor (sklearn `coef_`): one row
    /// (the log-odds of class 1 against class 0) for two classes, `K` rows
    /// otherwise.
    pub coefficients: Vec<Vec<f64>>,
    /// Intercepts, one per coefficient row (sklearn `intercept_`; 0 without
    /// an intercept). With `K >= 3` classes they sum to zero.
    pub intercepts: Vec<f64>,
    /// Rows fitted.
    pub n_observations: usize,
    /// Features.
    pub n_features: usize,
    /// Proximal Newton iterations.
    pub iterations: usize,
    /// Whether the fit converged within `max_iterations`.
    pub converged: bool,
    /// Weighted log-likelihood at the fit.
    pub log_likelihood: f64,
    /// The minimised objective: `-log_likelihood` plus the penalty.
    pub objective: f64,
    /// Share of training rows whose most probable class is their class.
    pub accuracy: f64,
}

/// Fit to the classes `y` (indices `0..K`, every class present at least once)
/// of the rows of `x_columns` (one vector per feature), with optional
/// non-negative sample `weights`.
///
/// Fewer than two rows or classes is [`StatsError::InsufficientDataMsg`]
/// (a NULL in SQL); a fit that does not converge is returned with
/// `converged = false`.
pub fn fit_multinomial(
    y: &[f64],
    x_columns: &[Vec<f64>],
    weights: Option<&[f64]>,
    options: &MultinomialOptions,
) -> StatsResult<MultinomialFit> {
    let n = y.len();
    let p = x_columns.len();
    if x_columns.iter().any(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatchMsg(
            "every x column must have one value per y".into(),
        ));
    }
    let mut distinct: Vec<f64> = y.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    if n < 2 || distinct.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(format!(
            "multinomial regression needs at least two rows and two classes, got {n} rows and {} classes",
            distinct.len()
        )));
    }
    let x = Mat::from_fn(n, p, |i, j| x_columns[j][i]);
    let yc = Col::from_fn(n, |i| y[i]);
    let mut builder = MultinomialLogisticRegressor::builder()
        .lambda(options.lambda)
        .l1_ratio(options.l1_ratio)
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations)
        .tolerance(options.tolerance)
        .error_on_non_convergence(false);
    if let Some(w) = weights {
        if w.len() != n {
            return Err(StatsError::DimensionMismatchMsg(
                "weights must have one value per y".into(),
            ));
        }
        builder = builder.weights(Col::from_fn(n, |i| w[i]));
    }
    let fitted = builder.build().fit(&x, &yc)?;
    let coef = fitted.coefficients();
    let coefficients = (0..coef.nrows())
        .map(|k| (0..coef.ncols()).map(|j| coef[(k, j)]).collect())
        .collect();
    let intercepts = fitted.intercepts().iter().copied().collect();
    Ok(MultinomialFit {
        n_classes: fitted.classes().len(),
        coefficients,
        intercepts,
        n_observations: n,
        n_features: p,
        iterations: fitted.iterations(),
        converged: fitted.converged(),
        log_likelihood: fitted.log_likelihood(),
        objective: fitted.objective(),
        accuracy: fitted.score(&x, &yc),
    })
}

/// Class probabilities (in class order) of one row `x` under a model with
/// `n_classes` classes, its coefficient rows and intercepts as
/// [`MultinomialFit`] (one row for two classes, `n_classes` rows otherwise),
/// and the index of the most probable class (the first on ties). Upstream's
/// model rebuilt from these parts does the scoring.
pub fn multinomial_predict_proba(
    coefficients: &[Vec<f64>],
    intercepts: &[f64],
    n_classes: usize,
    x: &[f64],
) -> StatsResult<(Vec<f64>, usize)> {
    if let Some(row) = coefficients.iter().find(|r| r.len() != x.len()) {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "the model has {} coefficients per class but x has {} values",
            row.len(),
            x.len()
        )));
    }
    let p = x.len();
    let coef = Mat::from_fn(coefficients.len(), p, |k, j| coefficients[k][j]);
    let icpt = Col::from_fn(intercepts.len(), |k| intercepts[k]);
    let classes: Vec<f64> = (0..n_classes).map(|k| k as f64).collect();
    let model = FittedMultinomialLogistic::from_parts(classes, coef, icpt)?;
    let xm = Mat::from_fn(1, p, |_, j| x[j]);
    let proba = model.predict_proba(&xm);
    let class = model.predict(&xm)[0];
    Ok((
        (0..n_classes).map(|k| proba[(0, k)]).collect(),
        // Classes are the indices 0..K.
        class as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 60;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7) as f64 * 0.37).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 3) as f64 * 0.91).cos()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s = x1[i] - 0.5 * x2[i] + 0.3 * ((i * 11) as f64).sin();
                if s < -0.3 {
                    0.0
                } else if s < 0.4 {
                    1.0
                } else {
                    2.0
                }
            })
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn probabilities_match_upstream_predict_proba() {
        let (y, x) = data();
        for (lambda, l1) in [(1.0, 0.0), (0.5, 0.5), (0.0, 0.0)] {
            let opts = MultinomialOptions {
                lambda,
                l1_ratio: l1,
                ..Default::default()
            };
            let fit = fit_multinomial(&y, &x, None, &opts).unwrap();
            assert_eq!(fit.n_classes, 3);
            let n = y.len();
            let xm = Mat::from_fn(n, 2, |i, j| x[j][i]);
            let up = MultinomialLogisticRegressor::builder()
                .lambda(lambda)
                .l1_ratio(l1)
                .error_on_non_convergence(false)
                .build()
                .fit(&xm, &Col::from_fn(n, |i| y[i]))
                .unwrap();
            let proba = up.predict_proba(&xm);
            let pred = up.predict(&xm);
            for i in 0..n {
                let (p, class) = multinomial_predict_proba(
                    &fit.coefficients,
                    &fit.intercepts,
                    3,
                    &[x[0][i], x[1][i]],
                )
                .unwrap();
                for (k, pk) in p.iter().enumerate() {
                    assert!((pk - proba[(i, k)]).abs() < 1e-12);
                }
                assert_eq!(class as f64, pred[i]);
            }
        }
    }

    #[test]
    fn binary_has_one_row_and_matches_upstream() {
        let (y, x) = data();
        let yb: Vec<f64> = y.iter().map(|v| f64::from(*v >= 1.0)).collect();
        let fit = fit_multinomial(&yb, &x, None, &MultinomialOptions::default()).unwrap();
        assert_eq!(fit.n_classes, 2);
        assert_eq!(fit.coefficients.len(), 1);
        let (p, _) =
            multinomial_predict_proba(&fit.coefficients, &fit.intercepts, 2, &[0.2, -0.1]).unwrap();
        let eta = fit.intercepts[0] + 0.2 * fit.coefficients[0][0] - 0.1 * fit.coefficients[0][1];
        assert!((p[1] - 1.0 / (1.0 + (-eta).exp())).abs() < 1e-14);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn degenerate_and_invalid_inputs() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let err = fit_multinomial(&[0.0, 0.0, 0.0], &x, None, &MultinomialOptions::default());
        assert!(matches!(err, Err(StatsError::InsufficientDataMsg(_))));
        let bad = MultinomialOptions {
            l1_ratio: 2.0,
            ..Default::default()
        };
        assert!(matches!(
            fit_multinomial(&[0.0, 1.0, 0.0], &x, None, &bad),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(fit_multinomial(
            &[0.0, 1.0, 0.0],
            &x,
            Some(&[1.0, -1.0, 1.0]),
            &Default::default()
        )
        .is_err());
        assert!(multinomial_predict_proba(&[vec![1.0]], &[0.0], 3, &[1.0]).is_err());
        assert!(multinomial_predict_proba(&[vec![1.0]], &[0.0], 2, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn weights_equal_row_replication() {
        let (y, x) = data();
        let w: Vec<f64> = (0..y.len()).map(|i| (i % 3 + 1) as f64).collect();
        let a = fit_multinomial(&y, &x, Some(&w), &Default::default()).unwrap();
        let mut yr = Vec::new();
        let mut xr = vec![Vec::new(), Vec::new()];
        for (i, yi) in y.iter().enumerate() {
            for _ in 0..(i % 3 + 1) {
                yr.push(*yi);
                xr[0].push(x[0][i]);
                xr[1].push(x[1][i]);
            }
        }
        let b = fit_multinomial(&yr, &xr, None, &Default::default()).unwrap();
        for k in 0..3 {
            for j in 0..2 {
                assert!((a.coefficients[k][j] - b.coefficients[k][j]).abs() < 1e-7);
            }
        }
    }
}
