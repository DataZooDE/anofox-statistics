//! Leverage-aware prediction / confidence intervals for linear models.
//!
//! Thin marshalling layer over `anofox_regression::inference`
//! ([`compute_variance_factor`], [`intervals_from_variance_factor`]): the
//! variance factor `M` (`(X'X)^-1`, `(X'WX)^-1`, or the ridge sandwich
//! `A X'(W)X A` with an unpenalised intercept) and the interval
//! `yhat ± t_df · s · sqrt(1 + x0'Mx0)` (prediction) or `sqrt(x0'Mx0)`
//! (confidence) are computed upstream. This module only converts the
//! column-major `Vec<Vec<f64>>` layout, filters rows the fit skipped, maps the
//! NaN/zero-coefficient convention to the `excluded` mask and maps errors.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::core::IntervalType as UpIntervalType;
use anofox_regression::inference::{compute_variance_factor, intervals_from_variance_factor};
use faer::{Col, Mat};

/// A column is excluded from `M` when its coefficient is NaN (aliased /
/// constant, dropped from the fit) or exactly 0.0 (inactive in a sparse fit
/// such as elastic net / LASSO / NNLS at the bound).
fn is_excluded(coef: f64) -> bool {
    coef.is_nan() || coef == 0.0
}

fn map_upstream_err(msg: &'static str) -> StatsError {
    if msg.contains("singular") {
        StatsError::SingularMatrix
    } else {
        StatsError::InvalidInput(msg.to_string())
    }
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

    // Keep the rows the fit used.
    let rows: Vec<usize> = (0..n)
        .filter(|&i| x.iter().all(|col| col[i].is_finite()))
        .filter(|&i| weights.is_none_or(|w| w[i].is_finite() && w[i] >= 0.0))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let excluded: Vec<bool> = coefficients.iter().map(|&c| is_excluded(c)).collect();
    if !fit_intercept && excluded.iter().all(|&e| e) {
        return Err(StatsError::InvalidInput(
            "all coefficients are NaN or zero; nothing to build".into(),
        ));
    }
    let xm = Mat::from_fn(rows.len(), p, |i, j| x[j][rows[i]]);
    let wc = weights.map(|w| Col::from_fn(rows.len(), |i| w[rows[i]]));
    let m = compute_variance_factor(&xm, wc.as_ref(), fit_intercept, &excluded, ridge_lambda)
        .map_err(map_upstream_err)?;

    let dim = m.nrows();
    let mut out = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            out[a * dim + b] = m[(a, b)];
        }
    }
    Ok((out, dim))
}

/// Interval kind for [`predict_with_interval_matrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalType {
    /// Interval for a new observation: `sqrt(1 + x0' M x0)`.
    Prediction,
    /// Interval for the conditional mean: `sqrt(x0' M x0)`.
    Confidence,
}

impl From<IntervalType> for UpIntervalType {
    fn from(t: IntervalType) -> Self {
        match t {
            IntervalType::Prediction => UpIntervalType::Prediction,
            IntervalType::Confidence => UpIntervalType::Confidence,
        }
    }
}

fn validate_level(confidence_level: f64) -> StatsResult<()> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(StatsError::InvalidInput(format!(
            "confidence_level must be in (0, 1), got {confidence_level}"
        )));
    }
    Ok(())
}

/// Linear predictor; NaN coefficients contribute nothing (as `anofox_predict`).
fn linear_predictor(coefficients: &[f64], intercept: f64, x_new: &[f64]) -> f64 {
    let base = if intercept.is_nan() { 0.0 } else { intercept };
    coefficients
        .iter()
        .zip(x_new)
        .filter(|(c, _)| !c.is_nan())
        .fold(base, |acc, (c, v)| acc + c * v)
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
    validate_level(confidence_level)?;
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

    let yhat = linear_predictor(coefficients, intercept, x_new);
    let df = n_observations.saturating_sub(n_params_effective);
    if df == 0 || !(residual_std_error.is_finite() && residual_std_error > 0.0) {
        return Ok((yhat, f64::NAN, f64::NAN));
    }

    // Excluded features have zero rows/cols in M; zero their x so a NaN there
    // cannot leak into the leverage.
    let m = Mat::from_fn(dim, dim, |a, b| matrix[a * dim + b]);
    let x0 = Mat::from_fn(1, p, |_, j| {
        if is_excluded(coefficients[j]) {
            0.0
        } else {
            x_new[j]
        }
    });
    let r = intervals_from_variance_factor(
        &x0,
        &m,
        &Col::from_fn(1, |_| yhat),
        residual_std_error * residual_std_error,
        df as f64,
        confidence_level,
        interval_type.into(),
        has_intercept,
    );
    Ok((yhat, r.lower[0], r.upper[0]))
}

/// Two-sided Student-t critical value `t_{(1+level)/2, df}`, or NaN for
/// invalid inputs. Delegates to upstream's interval routine (a zero-leverage
/// prediction interval with unit variance has half-width exactly `t`).
pub fn t_critical(confidence_level: f64, df: usize) -> f64 {
    if df == 0 || !(confidence_level > 0.0 && confidence_level < 1.0) {
        return f64::NAN;
    }
    let r = intervals_from_variance_factor(
        &Mat::zeros(1, 0),
        &Mat::zeros(1, 1),
        &Col::zeros(1),
        1.0,
        df as f64,
        confidence_level,
        UpIntervalType::Prediction,
        true,
    );
    r.upper[0]
}

/// Prediction interval without the training design: assumes the leverage of
/// the new row equals that of the centroid (`h = 1/n`), i.e.
/// `yhat ± t_df · s · sqrt(1 + 1/n)` with `df = n - p` (`p` counts the
/// intercept). Returns `(yhat, lower, upper)`; when the interval is undefined
/// the bounds equal `yhat`.
pub fn predict_with_centroid_interval(
    coefficients: &[f64],
    intercept: f64,
    x_new: &[f64],
    residual_std_error: f64,
    n_observations: usize,
    confidence_level: f64,
) -> (f64, f64, f64) {
    let yhat = linear_predictor(coefficients, intercept, x_new);
    let p = coefficients.len() + usize::from(!intercept.is_nan());
    let df = n_observations.saturating_sub(p);
    if residual_std_error.is_nan()
        || residual_std_error <= 0.0
        || n_observations <= coefficients.len() + 1
        || df == 0
        || !(confidence_level > 0.0 && confidence_level < 1.0)
    {
        return (yhat, yhat, yhat);
    }
    let r = intervals_from_variance_factor(
        &Mat::zeros(1, 0),
        &Mat::from_fn(1, 1, |_, _| 1.0 / n_observations as f64),
        &Col::from_fn(1, |_| yhat),
        residual_std_error * residual_std_error,
        df as f64,
        confidence_level,
        UpIntervalType::Prediction,
        true,
    );
    (yhat, r.lower[0], r.upper[0])
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

    /// R: qt(0.975, 10); 5 + qt(0.95, 6) * 0.5 * sqrt(1 + 1/8)
    #[test]
    #[allow(clippy::excessive_precision)]
    fn t_critical_and_centroid_interval() {
        assert!((t_critical(0.95, 10) - 2.2281388519649385).abs() < 1e-8);
        assert!(t_critical(0.95, 0).is_nan() && t_critical(1.0, 5).is_nan());
        let (y, lo, hi) = predict_with_centroid_interval(&[2.0], 1.0, &[2.0], 0.5, 8, 0.90);
        assert_eq!(y, 5.0);
        let half = 1.9431802803927816 * 0.5 * (1.0f64 + 1.0 / 8.0).sqrt();
        assert!((hi - (5.0 + half)).abs() < 1e-8 && (lo - (5.0 - half)).abs() < 1e-8);
        let (_, lo, hi) = predict_with_centroid_interval(&[2.0], 1.0, &[2.0], f64::NAN, 8, 0.9);
        assert!(lo == 5.0 && hi == 5.0);
    }
}
