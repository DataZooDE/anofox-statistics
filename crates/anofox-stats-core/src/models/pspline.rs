//! P-spline (penalized B-spline) smoother wrapper (#316)
//!
//! Upstream's `PSplineRegressor`, the Eilers & Marx P-spline as
//! `mgcv::gam(y ~ s(x, bs = "ps", k = n_basis, m = c(2, penalty_order)),
//! method = "GCV.Cp")`: cubic B-splines on evenly spaced knots, an
//! order-`penalty_order` difference penalty, the smoothing parameter fixed or
//! chosen by GCV, Bayesian standard errors of the mean and a linear extension
//! outside the data range.
//!
//! Upstream keeps the fitted basis (knots, coefficients, Cholesky factor)
//! private, so a fitted smoother cannot be rebuilt from stored parts. The
//! extension therefore fits and predicts in one call ([`fit_predict_pspline`]).

use super::interval::IntervalType;
use crate::errors::{StatsError, StatsResult};
use crate::types::ModelSummary;
use anofox_regression::core::{HasModelInfo, IntervalType as UpIntervalType};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Options of the P-spline smoother.
#[derive(Debug, Clone, Copy)]
pub struct PSplineOptions {
    /// Number of B-spline basis functions (mgcv's `k`); 0 lets upstream
    /// choose (`n / 4 + 4`, clamped to [6, 40]).
    pub n_basis: usize,
    /// Order of the difference penalty (mgcv's `m[2]`, default 2).
    pub penalty_order: usize,
    /// Fixed smoothing parameter; `None` selects it by GCV.
    pub lambda: Option<f64>,
}

impl Default for PSplineOptions {
    fn default() -> Self {
        Self {
            n_basis: 0,
            penalty_order: 2,
            lambda: None,
        }
    }
}

/// The fitted smoother's summary.
#[derive(Debug, Clone)]
pub struct PSplineFit {
    /// B-spline coefficients (`n_basis` values).
    pub coefficients: Vec<f64>,
    /// Smoothing parameter of the difference penalty (mgcv: `sp / S.scale`).
    pub lambda: f64,
    /// Effective degrees of freedom, intercept included (`sum(gam$edf)`).
    pub edf: f64,
    /// Residual variance `RSS / (n - edf)` (`gam$sig2`).
    pub sigma2: f64,
    pub r_squared: f64,
    pub n_observations: usize,
    pub penalty_order: usize,
    pub summary: ModelSummary,
}

/// Predictions at the requested points.
#[derive(Debug, Clone, Default)]
pub struct PSplinePredictions {
    pub fit: Vec<f64>,
    /// Standard error of the mean (`predict.gam(se.fit = TRUE)`).
    pub se_fit: Vec<f64>,
    /// Interval bounds; NaN when no interval was requested.
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Fit the P-spline on the complete rows of (`x`, `y`) and predict at `x_new`.
///
/// Rows with a non-finite `x` or `y` are dropped from the fit; a non-finite
/// `x_new` gets NaN predictions. `interval` selects the band (confidence: the
/// mean; prediction: a new observation) at `level`; `None` leaves the bounds NaN.
pub fn fit_predict_pspline(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    options: &PSplineOptions,
    interval: Option<IntervalType>,
    level: f64,
) -> StatsResult<(PSplineFit, PSplinePredictions)> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: y.len(),
            x_rows: x.len(),
        });
    }
    if options.penalty_order == 0 {
        return Err(StatsError::InvalidValue {
            field: "penalty_order",
            message: "penalty_order must be at least 1".to_string(),
        });
    }
    if let Some(lambda) = options.lambda {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(StatsError::InvalidValue {
                field: "lambda",
                message: format!("lambda must be finite and >= 0, got {lambda}"),
            });
        }
    }
    if interval.is_some() && !(level > 0.0 && level < 1.0) {
        return Err(StatsError::InvalidValue {
            field: "confidence_level",
            message: format!("confidence_level must be in (0, 1), got {level}"),
        });
    }
    let rows: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n = rows.len();
    let x_mat = Mat::from_fn(n, 1, |i, _| x[rows[i]]);
    let y_col = Col::from_fn(n, |i| y[rows[i]]);

    let mut model = PSplineRegressor::new()
        .with_n_basis(options.n_basis)
        .with_penalty_order(options.penalty_order);
    if let Some(lambda) = options.lambda {
        model = model.with_lambda(lambda);
    }
    let fitted = model.fit(&x_mat, &y_col).map_err(StatsError::from)?;
    let result = fitted.result();

    // The fit has no likelihood upstream (its RegressionResult leaves
    // log_likelihood / AIC / BIC at 0), so the summary only names the model.
    let fit = PSplineFit {
        coefficients: result.coefficients.iter().copied().collect(),
        lambda: fitted.lambda(),
        edf: fitted.edf(),
        sigma2: fitted.sigma2(),
        r_squared: result.r_squared,
        n_observations: n,
        penalty_order: options.penalty_order,
        summary: ModelSummary::new(fitted.model_info()),
    };

    let points: Vec<usize> = (0..x_new.len()).filter(|&i| x_new[i].is_finite()).collect();
    let mut pred = PSplinePredictions {
        fit: vec![f64::NAN; x_new.len()],
        se_fit: vec![f64::NAN; x_new.len()],
        lower: vec![f64::NAN; x_new.len()],
        upper: vec![f64::NAN; x_new.len()],
    };
    if !points.is_empty() {
        let new_mat = Mat::from_fn(points.len(), 1, |i, _| x_new[points[i]]);
        // The confidence call carries the standard error of the mean; the
        // requested band comes from its own call at `level`.
        let mean = fitted.predict_with_interval(&new_mat, Some(UpIntervalType::Confidence), 0.95);
        let band =
            interval.map(|kind| fitted.predict_with_interval(&new_mat, Some(kind.into()), level));
        for (k, &i) in points.iter().enumerate() {
            pred.fit[i] = mean.fit[k];
            pred.se_fit[i] = mean.se[k];
            if let Some(b) = &band {
                pred.lower[i] = b.lower[k];
                pred.upper[i] = b.upper[k];
            }
        }
    }
    Ok((fit, pred))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..60).map(|i| ((i * 37) % 60) as f64 / 6.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 7 % 5) as f64 - 2.0))
            .collect();
        (x, y)
    }

    #[test]
    fn test_pspline_fit_predict() {
        let (x, y) = data();
        let (fit, pred) = fit_predict_pspline(
            &x,
            &y,
            &x,
            &PSplineOptions {
                n_basis: 12,
                ..Default::default()
            },
            Some(IntervalType::Prediction),
            0.9,
        )
        .unwrap();
        assert_eq!(fit.coefficients.len(), 12);
        assert_eq!(fit.n_observations, 60);
        assert!(fit.edf > 2.0 && fit.edf < 12.0);
        assert!(fit.lambda > 0.0 && fit.sigma2 > 0.0);
        assert_eq!(fit.summary.model_type, "pspline");
        assert!(fit.summary.aic.is_nan());
        for (i, xi) in x.iter().enumerate() {
            assert!((pred.fit[i] - xi.sin()).abs() < 0.3);
            assert!(pred.lower[i] < pred.fit[i] && pred.fit[i] < pred.upper[i]);
            // A prediction interval is wider than the mean's standard error.
            assert!(pred.upper[i] - pred.fit[i] > pred.se_fit[i]);
        }
    }

    #[test]
    fn test_pspline_no_interval_and_nan_points() {
        let (x, y) = data();
        let (_, pred) = fit_predict_pspline(
            &x,
            &y,
            &[f64::NAN, 2.0],
            &PSplineOptions::default(),
            None,
            0.95,
        )
        .unwrap();
        assert!(pred.fit[0].is_nan() && pred.se_fit[0].is_nan());
        assert!(pred.fit[1].is_finite() && pred.se_fit[1] > 0.0);
        assert!(pred.lower[1].is_nan() && pred.upper[1].is_nan());
    }

    #[test]
    fn test_pspline_fixed_lambda_and_errors() {
        let (x, y) = data();
        let opts = PSplineOptions {
            lambda: Some(10.0),
            ..Default::default()
        };
        let (fit, _) = fit_predict_pspline(&x, &y, &[], &opts, None, 0.95).unwrap();
        assert_eq!(fit.lambda, 10.0);

        let bad = PSplineOptions {
            lambda: Some(-1.0),
            ..Default::default()
        };
        assert!(fit_predict_pspline(&x, &y, &[], &bad, None, 0.95).is_err());
        let bad = PSplineOptions {
            penalty_order: 0,
            ..Default::default()
        };
        assert!(fit_predict_pspline(&x, &y, &[], &bad, None, 0.95).is_err());
        assert!(fit_predict_pspline(&x, &y[..3], &[], &opts, None, 0.95).is_err());
        assert!(fit_predict_pspline(&[f64::NAN], &[1.0], &[], &opts, None, 0.95).is_err());
        // Too few points for a cubic basis.
        assert!(fit_predict_pspline(&x[..4], &y[..4], &[], &opts, None, 0.95).is_err());
        // Constant x.
        let c = vec![1.0; 10];
        assert!(fit_predict_pspline(&c, &y[..10], &[], &opts, None, 0.95).is_err());
    }
}
