//! Leverage-aware prediction / confidence intervals for linear models.
//!
//! For a linear predictor `yhat0 = x0' beta` (with `x0` augmented by a leading 1
//! when the model has an intercept) the variance of `yhat0` is
//! `sigma^2 * x0' M x0`, where `M` depends on the estimator:
//!
//! * OLS:   `M = (X'X)^-1`
//! * WLS:   `M = (X'WX)^-1`
//! * Ridge: `M = A X'X A` with `A = (X'X + P)^-1`, `P = lambda * I` except that
//!   the intercept is not penalised (`P[0,0] = 0`). This is exactly the
//!   estimator used by `fit_ridge` / upstream `RidgeRegressor`, which centers
//!   `X` and `y` and penalises only the slopes — algebraically identical to the
//!   augmented formulation with an unpenalised intercept column.
//!   (Weights and ridge may be combined: `M = A X'WX A`, `A = (X'WX + P)^-1`.)
//!
//! The prediction interval is `yhat0 ± t_{df} * s * sqrt(1 + x0' M x0)`, the
//! confidence interval for the mean `yhat0 ± t_{df} * s * sqrt(x0' M x0)`,
//! `df = n - p_effective`.

use crate::errors::{StatsError, StatsResult};
use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// A column is excluded from `M` when its coefficient is NaN (aliased /
/// constant, dropped from the fit) or exactly 0.0 (inactive in a sparse fit
/// such as elastic net / LASSO / NNLS at the bound).
fn is_excluded(coef: f64) -> bool {
    coef.is_nan() || coef == 0.0
}

/// Compute the variance-factor matrix `M` (row-major, `dim x dim`).
///
/// * `x` – feature columns (same layout as the fit), all of equal length.
/// * `coefficients` – fitted coefficients, one per column of `x`.
/// * `fit_intercept` – if true, `dim = x.len() + 1` and index 0 is the intercept.
/// * `weights` – optional observation weights (WLS).
/// * `ridge_lambda` – effective (raw) ridge penalty; `0` for none. For glmnet
///   lambda scaling pass `lambda * n`.
///
/// Rows with any non-finite feature value (or non-finite / negative weight) are
/// skipped, matching the fit's row filtering. Rows/columns of excluded features
/// are zero in the result. Returns `SingularMatrix` if the (penalised) Gram
/// matrix of the retained columns is numerically singular.
pub fn interval_matrix(
    x: &[Vec<f64>],
    coefficients: &[f64],
    fit_intercept: bool,
    weights: Option<&[f64]>,
    ridge_lambda: f64,
) -> StatsResult<(Vec<f64>, usize)> {
    let p = x.len();
    if p == 0 && !fit_intercept {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if coefficients.len() != p {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "coefficients has {} entries, expected one per x column ({p})",
            coefficients.len()
        )));
    }
    if !(ridge_lambda.is_finite() && ridge_lambda >= 0.0) {
        return Err(StatsError::InvalidInput(format!(
            "ridge_lambda must be finite and >= 0, got {ridge_lambda}"
        )));
    }
    let n = x.first().map(|c| c.len()).unwrap_or(0);
    crate::validation::validate_x_columns(n, x)?;
    if let Some(w) = weights {
        if w.len() != n && !(p == 0 && fit_intercept) {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "weights has {} entries, expected {n}",
                w.len()
            )));
        }
    }
    // Intercept-only model: the number of rows comes from the weights (if any).
    let n = if p == 0 {
        weights.map(|w| w.len()).unwrap_or(0)
    } else {
        n
    };

    let off = usize::from(fit_intercept);
    let dim = p + off;
    // Indices (into the augmented design) of retained parameters.
    let retained: Vec<usize> = (0..dim)
        .filter(|&j| j < off || !is_excluded(coefficients[j - off]))
        .collect();
    let q = retained.len();
    if q == 0 {
        return Err(StatsError::InvalidInput(
            "all coefficients are NaN or zero; nothing to build".into(),
        ));
    }

    let value = |i: usize, j: usize| -> f64 {
        if j < off {
            1.0
        } else {
            x[j - off][i]
        }
    };

    // Weighted Gram matrix over valid rows and retained columns.
    let mut g = Mat::<f64>::zeros(q, q);
    let mut n_used = 0usize;
    for i in 0..n {
        if x.iter().any(|col| !col[i].is_finite()) {
            continue;
        }
        let w = match weights {
            Some(w) => {
                if !w[i].is_finite() || w[i] < 0.0 {
                    continue;
                }
                w[i]
            }
            None => 1.0,
        };
        n_used += 1;
        for a in 0..q {
            let va = w * value(i, retained[a]);
            for b in a..q {
                g[(a, b)] += va * value(i, retained[b]);
            }
        }
    }
    if n_used == 0 {
        return Err(StatsError::NoValidData);
    }
    for a in 0..q {
        for b in (a + 1)..q {
            g[(b, a)] = g[(a, b)];
        }
    }

    // Penalised matrix H = G + P (intercept unpenalised).
    let mut h = g.clone();
    if ridge_lambda > 0.0 {
        for (a, &j) in retained.iter().enumerate() {
            if j >= off {
                h[(a, a)] += ridge_lambda;
            }
        }
    }

    let a_inv = invert_spd_checked(&h)?;
    let m_reduced = if ridge_lambda > 0.0 {
        &(&a_inv * &g) * &a_inv
    } else {
        a_inv
    };

    let mut out = vec![0.0; dim * dim];
    for (a, &ja) in retained.iter().enumerate() {
        for (b, &jb) in retained.iter().enumerate() {
            out[ja * dim + jb] = m_reduced[(a, b)];
        }
    }
    Ok((out, dim))
}

/// Invert a symmetric positive-definite matrix via Cholesky after diagonal
/// equilibration, failing with `SingularMatrix` when it is numerically singular.
fn invert_spd_checked(h: &Mat<f64>) -> StatsResult<Mat<f64>> {
    let q = h.nrows();
    let mut d = vec![0.0; q];
    for (j, dj) in d.iter_mut().enumerate() {
        let hjj = h[(j, j)];
        if !(hjj.is_finite() && hjj > 0.0) {
            return Err(StatsError::SingularMatrix);
        }
        *dj = 1.0 / hjj.sqrt();
    }
    // Equilibrated S = D H D has unit diagonal, so its Cholesky pivots are a
    // scale-free conditioning measure.
    let s = Mat::<f64>::from_fn(q, q, |i, j| h[(i, j)] * d[i] * d[j]);
    let llt = s.llt(Side::Lower).map_err(|_| StatsError::SingularMatrix)?;
    let l = llt.L();
    let min_pivot = (0..q).map(|j| l[(j, j)]).fold(f64::INFINITY, f64::min);
    // Pivot^2 bounds the smallest eigenvalue-ish of S (1 on the diagonal);
    // below ~1e-12 the inverse carries no reliable digits.
    if !(min_pivot.is_finite() && min_pivot * min_pivot > 1e-12) {
        return Err(StatsError::SingularMatrix);
    }
    let s_inv = llt.inverse();
    Ok(Mat::from_fn(q, q, |i, j| s_inv[(i, j)] * d[i] * d[j]))
}

/// Interval kind for [`predict_with_interval_matrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalType {
    /// Interval for a new observation: `sqrt(1 + x0' M x0)`.
    Prediction,
    /// Interval for the conditional mean: `sqrt(x0' M x0)`.
    Confidence,
}

/// Point prediction with a leverage-aware interval.
///
/// * `intercept` – NaN if the model has no intercept.
/// * `matrix`/`dim` – output of [`interval_matrix`]; `dim` must be
///   `x_new.len() + 1` if the model has an intercept, else `x_new.len()`.
/// * `n_params_effective` – number of estimated parameters including the
///   intercept (`df = n_observations - n_params_effective`).
///
/// Returns `(yhat, lower, upper)`. When the interval is undefined (`df == 0` or
/// a non-finite / non-positive `residual_std_error`) the bounds are NaN.
#[allow(clippy::too_many_arguments)]
pub fn predict_with_interval_matrix(
    coefficients: &[f64],
    intercept: f64,
    x_new: &[f64],
    matrix: &[f64],
    dim: usize,
    n_observations: usize,
    n_params_effective: usize,
    residual_std_error: f64,
    confidence_level: f64,
    interval_type: IntervalType,
) -> StatsResult<(f64, f64, f64)> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(StatsError::InvalidInput(format!(
            "confidence_level must be in (0, 1), got {confidence_level}"
        )));
    }
    let p = x_new.len();
    if coefficients.len() != p {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "x_new has {p} entries but there are {} coefficients",
            coefficients.len()
        )));
    }
    let has_intercept = !intercept.is_nan();
    let off = usize::from(has_intercept);
    if dim != p + off {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "matrix dim {dim} does not match {} (x_new length {p}{})",
            p + off,
            if has_intercept { " + intercept" } else { "" }
        )));
    }
    if dim.checked_mul(dim) != Some(matrix.len()) {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "matrix has {} entries, expected {dim}x{dim}",
            matrix.len()
        )));
    }

    // Point prediction; NaN coefficients contribute nothing (as anofox_predict).
    let mut yhat = if has_intercept { intercept } else { 0.0 };
    for (c, v) in coefficients.iter().zip(x_new) {
        if !c.is_nan() {
            yhat += c * v;
        }
    }

    let df = n_observations.saturating_sub(n_params_effective);
    if df == 0 || !(residual_std_error.is_finite() && residual_std_error > 0.0) {
        return Ok((yhat, f64::NAN, f64::NAN));
    }

    // x0 augmented; excluded features are zero (their rows/cols of M are zero).
    let x0: Vec<f64> = (0..dim)
        .map(|j| {
            if j < off {
                1.0
            } else if is_excluded(coefficients[j - off]) {
                0.0
            } else {
                x_new[j - off]
            }
        })
        .collect();
    let mut quad = 0.0;
    for a in 0..dim {
        if x0[a] == 0.0 {
            continue;
        }
        for b in 0..dim {
            quad += x0[a] * matrix[a * dim + b] * x0[b];
        }
    }
    let quad = quad.max(0.0);
    let factor = match interval_type {
        IntervalType::Prediction => (1.0 + quad).sqrt(),
        IntervalType::Confidence => quad.sqrt(),
    };
    let t = StudentsT::new(0.0, 1.0, df as f64)
        .map_err(|e| StatsError::InvalidInput(e.to_string()))?
        .inverse_cdf((1.0 + confidence_level) / 2.0);
    let margin = t * residual_std_error * factor;
    Ok((yhat, yhat - margin, yhat + margin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::fit_ols;
    use crate::types::OlsOptions;

    fn data() -> (Vec<f64>, Vec<f64>) {
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1],
        )
    }

    /// R: m <- lm(y ~ x); predict(m, data.frame(x = c(4.5, 12)),
    ///    interval = 'prediction' | 'confidence', level = 0.95 | 0.90)
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_r_lm_prediction_and_confidence_intervals() {
        let (x, y) = data();
        let fit = fit_ols(&y, std::slice::from_ref(&x), &OlsOptions::default()).unwrap();
        let coef = fit.core.coefficients.clone();
        let intercept = fit.core.intercept.unwrap();
        let s = fit.core.residual_std_error;
        assert!((s - 0.18016747059421578).abs() < 1e-10);

        let (m, dim) = interval_matrix(&[x], &coef, true, None, 0.0).unwrap();
        assert_eq!(dim, 2);

        // (x0, level, type, fit, lwr, upr)
        #[rustfmt::skip]
        let cases = [
            (4.5, 0.95, IntervalType::Prediction, 9.0250000000000004, 8.5574038065565929, 9.4925961934434078),
            (12.0, 0.95, IntervalType::Prediction, 24.00714285714286, 23.315088455013225, 24.699197259272495),
            (4.5, 0.95, IntervalType::Confidence, 9.0250000000000004, 8.8691346021855306, 9.1808653978144701),
            (12.0, 0.95, IntervalType::Confidence, 24.00714285714286, 23.473675784475667, 24.540609929810053),
            (4.5, 0.90, IntervalType::Prediction, 9.0250000000000004, 8.6536651266501163, 9.3963348733498844),
            (12.0, 0.90, IntervalType::Prediction, 24.00714285714286, 23.457557686498948, 24.556728027786772),
        ];
        for (x0, cl, kind, fit_r, lwr, upr) in cases {
            let (yhat, lo, hi) =
                predict_with_interval_matrix(&coef, intercept, &[x0], &m, dim, 8, 2, s, cl, kind)
                    .unwrap();
            assert!((yhat - fit_r).abs() < 1e-8, "fit {x0} {cl} {kind:?}");
            assert!(
                (lo - lwr).abs() < 1e-8,
                "lwr {x0} {cl} {kind:?}: {lo} vs {lwr}"
            );
            assert!(
                (hi - upr).abs() < 1e-8,
                "upr {x0} {cl} {kind:?}: {hi} vs {upr}"
            );
        }
    }

    /// R: m0 <- lm(y ~ 0 + x); predict(m0, ..., interval = 'prediction')
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_r_no_intercept() {
        let (x, _) = data();
        let coef = [2.0039215686274505];
        let s = 0.16769987865147581;
        let (m, dim) = interval_matrix(&[x], &coef, false, None, 0.0).unwrap();
        assert_eq!(dim, 1);
        let (_, lo, hi) = predict_with_interval_matrix(
            &coef,
            f64::NAN,
            &[12.0],
            &m,
            dim,
            8,
            1,
            s,
            0.95,
            IntervalType::Prediction,
        )
        .unwrap();
        assert!((lo - 23.529130780939248).abs() < 1e-8);
        assert!((hi - 24.564986866119561).abs() < 1e-8);
    }

    /// R: solve(t(X) %*% diag(w) %*% X) and
    /// predict(lm(y ~ x, weights = w), ..., interval = 'confidence')
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_r_weighted() {
        let (x, _) = data();
        let w = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let coef = [2.0138297872340427];
        let intercept = -0.04787234042553349;
        let (m, dim) = interval_matrix(&[x], &coef, true, Some(&w), 0.0).unwrap();
        let r = [
            0.43085106382978716,
            -0.074468085106382961,
            -0.074468085106382947,
            0.015957446808510634,
        ];
        for (a, b) in m.iter().zip(r) {
            assert!((a - b).abs() < 1e-12);
        }
        let (_, lo, hi) = predict_with_interval_matrix(
            &coef,
            intercept,
            &[12.0],
            &m,
            dim,
            8,
            2,
            0.21782288334938041,
            0.95,
            IntervalType::Confidence,
        )
        .unwrap();
        assert!((lo - 23.600919618290312).abs() < 1e-8);
        assert!((hi - 24.635250594475643).abs() < 1e-8);
    }

    /// R: A <- solve(G + diag(c(0, 2))); A %*% G %*% A (intercept unpenalised)
    #[test]
    #[allow(clippy::excessive_precision)]
    fn ridge_matrix_matches_r() {
        let (x, _) = data();
        let (m, _) = interval_matrix(std::slice::from_ref(&x), &[1.9], true, None, 2.0).unwrap();
        let r = [
            0.56430785123966953,
            -0.097623966942148713,
            -0.097623966942148782,
            0.021694214876033045,
        ];
        for (a, b) in m.iter().zip(r) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        let (m0, _) = interval_matrix(&[x], &[1.9], false, None, 2.0).unwrap();
        assert!((m0[0] - 0.0048072391365821462).abs() < 1e-14);
    }

    #[test]
    fn excluded_columns_and_nonfinite_rows() {
        let (x, _) = data();
        let mut x2 = x.clone();
        x2[3] = f64::NAN; // row 3 skipped
        let junk = vec![5.0; 8];
        let (m, dim) = interval_matrix(
            &[x2.clone(), junk.clone()],
            &[2.0, f64::NAN],
            true,
            None,
            0.0,
        )
        .unwrap();
        assert_eq!(dim, 3);
        // Excluded column (index 2) is all zero.
        for k in 0..3 {
            assert_eq!(m[2 * 3 + k], 0.0);
            assert_eq!(m[k * 3 + 2], 0.0);
        }
        // Same as fitting without the junk column on the 7 finite rows.
        let (m1, _) = interval_matrix(&[x2], &[2.0], true, None, 0.0).unwrap();
        assert!((m[0] - m1[0]).abs() < 1e-12 && (m[4] - m1[3]).abs() < 1e-12);
        // Exactly-zero coefficient is excluded too.
        let (mz, _) = interval_matrix(&[x, junk], &[2.0, 0.0], true, None, 0.0).unwrap();
        assert_eq!(mz[8], 0.0);
    }

    #[test]
    fn singular_and_invalid_inputs() {
        // Two identical columns -> singular.
        let (x, _) = data();
        let r = interval_matrix(&[x.clone(), x.clone()], &[1.0, 1.0], true, None, 0.0);
        assert!(matches!(r, Err(StatsError::SingularMatrix)));
        // ... but fine with a ridge penalty.
        assert!(interval_matrix(&[x.clone(), x.clone()], &[1.0, 1.0], true, None, 1.0).is_ok());
        // Mismatched lengths.
        assert!(interval_matrix(&[x.clone(), vec![1.0]], &[1.0, 1.0], true, None, 0.0).is_err());
        assert!(interval_matrix(std::slice::from_ref(&x), &[1.0, 2.0], true, None, 0.0).is_err());
        assert!(
            interval_matrix(std::slice::from_ref(&x), &[1.0], true, Some(&[1.0]), 0.0).is_err()
        );
        assert!(interval_matrix(std::slice::from_ref(&x), &[1.0], true, None, -1.0).is_err());
        // All rows non-finite.
        assert!(matches!(
            interval_matrix(&[vec![f64::NAN; 3]], &[1.0], true, None, 0.0),
            Err(StatsError::NoValidData)
        ));
        // Predict: bad level / dims; df = 0 gives NaN bounds.
        let m = [1.0, 0.0, 0.0, 1.0];
        let kind = IntervalType::Prediction;
        assert!(
            predict_with_interval_matrix(&[1.0], 0.0, &[1.0], &m, 2, 8, 2, 1.0, 1.0, kind).is_err()
        );
        assert!(
            predict_with_interval_matrix(&[1.0], 0.0, &[1.0], &m, 1, 8, 2, 1.0, 0.9, kind).is_err()
        );
        let (y, lo, hi) =
            predict_with_interval_matrix(&[1.0], 0.0, &[1.0], &m, 2, 2, 2, 1.0, 0.9, kind).unwrap();
        assert_eq!(y, 1.0);
        assert!(lo.is_nan() && hi.is_nan());
    }
}
