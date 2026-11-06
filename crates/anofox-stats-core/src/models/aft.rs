//! Accelerated failure time (AFT) regression with right censoring.
//!
//! The model (Newton-Raphson on `(beta, log sigma)`, Laplace inference, Gaussian
//! priors) lives in [`anofox_regression::solvers::aft`]. This module only
//! forwards the call and maps the error type onto [`StatsError`].

use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::aft;
use faer::{Col, Mat};

pub use aft::{AftDistribution, AftFitResult, AftInference, AftOptions, AftResult};

/// Fit an AFT model.
///
/// * `time` — strictly positive event or censoring times.
/// * `x` — column-major features.
/// * `event` — 1.0 when the event was observed, 0.0 when the row is right-censored.
pub fn fit_aft(
    time: &[f64],
    x: &[Vec<f64>],
    event: &[f64],
    options: &AftOptions,
) -> StatsResult<AftResult> {
    aft::fit_aft(time, x, event, options).map_err(StatsError::from)
}

/// Model-based survival `S(t | x)` at each of `times` for one covariate row of a
/// fitted AFT model given by its pieces (as the model STRUCT carries them), as R's
/// `psurvreg(t, eta, sigma, dist, lower.tail = FALSE)` with `eta = intercept + x'beta`
/// (NaN coefficients, i.e. aliased columns, are skipped).
///
/// `AftResult` is `#[non_exhaustive]` upstream and can't be rebuilt from stored
/// pieces, so this mirrors `AftResult::survival_curve` on the public
/// [`AftDistribution::log_survival`]: `S = 1` for `t <= 0`, `0` at `+Inf`, `NaN` for a
/// `NaN` time, a row of the wrong length or an invalid scale.
pub fn aft_survival_curve(
    coefficients: &[f64],
    intercept: Option<f64>,
    scale: f64,
    dist: AftDistribution,
    x_row: &[f64],
    times: &[f64],
) -> Vec<f64> {
    let eta = if x_row.len() == coefficients.len() {
        intercept.unwrap_or(0.0)
            + coefficients
                .iter()
                .zip(x_row)
                .filter(|(b, _)| !b.is_nan())
                .map(|(b, x)| b * x)
                .sum::<f64>()
    } else {
        f64::NAN
    };
    let valid = eta.is_finite() && scale.is_finite() && scale > 0.0;
    times
        .iter()
        .map(|&t| {
            if t.is_nan() || !valid {
                f64::NAN
            } else if t <= 0.0 {
                1.0
            } else if t == f64::INFINITY {
                0.0
            } else {
                dist.log_survival((t.ln() - eta) / scale).exp()
            }
        })
        .collect()
}

/// Per-row results of an AFT fit (`residuals.survreg`, RMS §18.3), one entry per
/// input row; NaN (and `is_training = false`) for the rows the fit left out.
#[derive(Debug, Clone, Default)]
pub struct AftAugment {
    pub is_training: Vec<bool>,
    /// `eta = intercept + x'beta`, the location of `log T`.
    pub linear_predictor: Vec<f64>,
    /// `(log t - eta) / sigma` ([`aft::FittedAft::standardized_residuals`]).
    pub standardized_residuals: Vec<f64>,
    /// `-log S(t | x)` ([`aft::FittedAft::cox_snell_residuals`]).
    pub cox_snell_residuals: Vec<f64>,
    /// Score contributions, `n_rows * n_params` row-major, in the parameter
    /// order intercept, coefficients, `log sigma` (no `log sigma` for the
    /// exponential) ([`aft::FittedAft::score_contributions`]).
    pub score: Vec<f64>,
    pub n_params: usize,
}

/// Fit the same model as [`fit_aft`] and return its per-row residuals.
///
/// The fit goes through [`fit_aft`], so that a degenerate input fails with
/// exactly the error (and SQL NULL / error policy) of `aft_fit_agg`; the
/// residuals come from upstream's fitted model of that result
/// (`FittedAft::from_result`) on the rows the fit used (finite time, event
/// and features).
pub fn aft_augment(
    time: &[f64],
    x: &[Vec<f64>],
    event: &[f64],
    options: &AftOptions,
) -> StatsResult<AftAugment> {
    let result = fit_aft(time, x, event, options)?;
    let n = time.len();
    let used: Vec<usize> = (0..n)
        .filter(|&i| {
            time[i].is_finite() && event[i].is_finite() && x.iter().all(|c| c[i].is_finite())
        })
        .collect();
    let xm = Mat::from_fn(used.len(), x.len(), |r, j| x[j][used[r]]);
    let t = Col::from_fn(used.len(), |r| time[used[r]]);
    // fit_aft accepted the rows, so every used event is 0 or 1.
    let ev: Vec<bool> = used.iter().map(|&i| event[i] == 1.0).collect();
    let fitted = aft::FittedAft::from_result(result)?;
    let lp = fitted.predict_linear(&xm);
    let z = fitted.standardized_residuals(&xm, &t);
    let cs = fitted.cox_snell_residuals(&xm, &t);
    let s = fitted
        .score_contributions(&xm, &t, &ev)
        .map_err(|e| StatsError::InvalidInput(e.to_string()))?;
    let k = s.ncols();
    let mut out = AftAugment {
        is_training: vec![false; n],
        linear_predictor: vec![f64::NAN; n],
        standardized_residuals: vec![f64::NAN; n],
        cox_snell_residuals: vec![f64::NAN; n],
        score: vec![f64::NAN; n * k],
        n_params: k,
    };
    for (r, &i) in used.iter().enumerate() {
        out.is_training[i] = true;
        out.linear_predictor[i] = lp[r];
        out.standardized_residuals[i] = z[r];
        out.cox_snell_residuals[i] = cs[r];
        for j in 0..k {
            out.score[i * k + j] = s[(r, j)];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_stats_errors() {
        let err = fit_aft(&[], &[], &[], &AftOptions::default()).unwrap_err();
        assert!(matches!(err, StatsError::EmptyInput { field: "time" }));
    }

    #[test]
    fn survival_curve_is_one_at_zero_and_decreasing() {
        let s = aft_survival_curve(
            &[0.5],
            Some(1.0),
            0.8,
            AftDistribution::Weibull,
            &[1.0],
            &[0.0, 1.0, 5.0, 20.0],
        );
        assert_eq!(s[0], 1.0);
        assert!(s[1] > s[2] && s[2] > s[3] && s[3] > 0.0);
    }

    #[test]
    fn augment_marks_left_out_rows_and_sums_scores_to_zero() {
        let time = [
            5.0,
            9.0,
            6.0,
            14.0,
            4.0,
            11.0,
            7.0,
            16.0,
            5.5,
            12.0,
            f64::NAN,
        ];
        let event = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let x = vec![(0..11).map(|i| (i % 2) as f64).collect::<Vec<f64>>()];
        let a = aft_augment(&time, &x, &event, &AftOptions::default()).unwrap();
        assert_eq!(a.n_params, 3);
        assert!(!a.is_training[10] && a.cox_snell_residuals[10].is_nan());
        assert!(a.is_training[..10].iter().all(|&b| b));
        // The score is zero at the MLE.
        for j in 0..3 {
            let total: f64 = (0..10).map(|i| a.score[i * 3 + j]).sum();
            assert!(total.abs() < 1e-4, "score {j} sums to {total}");
        }
        assert!(a.cox_snell_residuals[..10].iter().all(|&r| r > 0.0));
    }

    #[test]
    fn augment_fails_like_the_fit() {
        let err = aft_augment(
            &[1.0, 2.0, 3.0],
            &[vec![0.0, 1.0, 0.0]],
            &[0.0, 0.0, 0.0],
            &AftOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, StatsError::InsufficientDataMsg(_)));
    }
}
