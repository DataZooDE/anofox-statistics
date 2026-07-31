//! Curvature-based inference: observed information at the mode -> vcov -> SE / z / p / CI.
//!
//! This is the single primitive shared by every consumer of the engine. GLMs with
//! explicit priors, AFT survival fits and mixed-effects fits all arrive here with a
//! Hessian at the mode and leave with the same [`LaplaceInference`] shape.
//!
//! For a MAP estimate the correct curvature is `(X'WX + P)^-1`, not `(X'WX)^-1`.
//! Penalized GLM fits previously reported the latter, which understates uncertainty
//! for the penalty and is simply wrong; [`VcovType`] keeps the old behaviour
//! reachable but no longer default.

use crate::errors::StatsResult;
use crate::types::VcovType;
use faer::Mat;
use statrs::distribution::{ContinuousCDF, Normal};

use super::normal_eq::invert_spd;

/// Inference computed from the curvature at the mode.
#[derive(Debug, Clone)]
pub struct LaplaceInference {
    /// Standard errors, one per parameter (design order, intercept first if fitted).
    pub std_errors: Vec<f64>,
    /// Wald z statistics `beta / se`.
    pub z_values: Vec<f64>,
    /// Two-sided p-values from the standard normal.
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub confidence_level: f64,
    /// The full covariance matrix, retained for prediction intervals and for the
    /// conditional variances of random effects.
    pub vcov: Mat<f64>,
}

/// Compute inference from the penalized and unpenalized information matrices.
///
/// * `information` — `X'WX + P`, the negative Hessian of the log posterior at the mode.
/// * `unpenalized` — `X'WX` alone. Only needed for [`VcovType::Sandwich`] and
///   [`VcovType::Naive`]; pass `None` when no penalty is in play.
/// * `dispersion` — scale multiplier applied to the covariance (1.0 for families
///   with a fixed dispersion).
///
/// Parameters flagged in `inactive` (an L1 coefficient resting exactly at its prior
/// location) get `NaN` inference: the objective is not differentiable there, so a
/// curvature-based standard error does not exist. This mirrors how dropped constant
/// columns already surface as `NaN`.
pub fn inference(
    beta: &[f64],
    information: &Mat<f64>,
    unpenalized: Option<&Mat<f64>>,
    dispersion: f64,
    confidence_level: f64,
    vcov_type: VcovType,
    inactive: &[bool],
) -> StatsResult<LaplaceInference> {
    let p = beta.len();

    let inv = invert_spd(information)?;

    let vcov = match vcov_type {
        VcovType::Laplace => inv,
        VcovType::Naive => match unpenalized {
            Some(u) => invert_spd(u)?,
            None => inv,
        },
        VcovType::Sandwich => match unpenalized {
            // (X'WX + P)^-1 X'WX (X'WX + P)^-1
            Some(u) => &inv * u * &inv,
            None => inv,
        },
    };

    let z_crit = normal_quantile(0.5 + confidence_level / 2.0);

    let mut std_errors = vec![f64::NAN; p];
    let mut z_values = vec![f64::NAN; p];
    let mut p_values = vec![f64::NAN; p];
    let mut ci_lower = vec![f64::NAN; p];
    let mut ci_upper = vec![f64::NAN; p];

    let normal = Normal::new(0.0, 1.0).ok();

    for j in 0..p {
        if inactive.get(j).copied().unwrap_or(false) {
            continue;
        }
        let var = dispersion * vcov[(j, j)];
        if !var.is_finite() || var < 0.0 {
            continue;
        }
        let se = var.sqrt();
        std_errors[j] = se;

        if se > 0.0 {
            let z = beta[j] / se;
            z_values[j] = z;
            p_values[j] = match &normal {
                Some(d) => 2.0 * (1.0 - d.cdf(z.abs())),
                None => f64::NAN,
            };
            ci_lower[j] = beta[j] - z_crit * se;
            ci_upper[j] = beta[j] + z_crit * se;
        }
    }

    Ok(LaplaceInference {
        std_errors,
        z_values,
        p_values,
        ci_lower,
        ci_upper,
        confidence_level,
        vcov,
    })
}

/// Standard normal quantile. `statrs`' `inverse_cdf` panics on the open interval
/// endpoints, so the input is clamped first.
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    match Normal::new(0.0, 1.0) {
        Ok(d) => d.inverse_cdf(p),
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::glm_engine::penalty::{DiagonalPenalty, QuadraticPenalty};
    use crate::models::glm_engine::normal_eq::solve_qr;
    use crate::types::PriorSpec;
    use faer::Col;

    /// A Gaussian likelihood with a Gaussian prior is ridge regression, which has a
    /// closed form. This pins the engine's penalized solve and its Laplace covariance
    /// against exact algebra, independent of any IRLS iteration.
    #[test]
    fn gaussian_prior_reproduces_the_ridge_closed_form() {
        // y = 1 + 2*x1 - 0.5*x2 with an intercept column.
        let n = 20;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => (i as f64) / 4.0,
            _ => ((i % 5) as f64) - 2.0,
        });
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 2.0 * ((i as f64) / 4.0) - 0.5 * (((i % 5) as f64) - 2.0))
            .collect();

        // Flat on the intercept, N(0, 0.5) on x1, N(1.0, 2.0) on x2.
        let priors = [
            PriorSpec::flat(),
            PriorSpec::normal(0.0, 0.5),
            PriorSpec::normal(1.0, 2.0),
        ];
        let pen = DiagonalPenalty::from_priors(&priors);

        // Assemble X'X and X'y directly (Gaussian => W = I, z = y).
        let mut xtx: Mat<f64> = Mat::zeros(3, 3);
        let mut xty: Col<f64> = Col::zeros(3);
        for i in 0..n {
            for j in 0..3 {
                xty[j] += x[(i, j)] * y[i];
                for k in 0..3 {
                    xtx[(j, k)] += x[(i, j)] * x[(i, k)];
                }
            }
        }
        let unpenalized = xtx.clone();

        pen.accumulate(&mut xtx, &mut xty);
        let beta = solve_qr(&xtx, &xty, 1e-12).unwrap();

        // Closed form: beta = (X'X + P)^-1 (X'y + P mu0).
        let mut expect_a: Mat<f64> = Mat::zeros(3, 3);
        let mut expect_b: Col<f64> = Col::zeros(3);
        for i in 0..n {
            for j in 0..3 {
                expect_b[j] += x[(i, j)] * y[i];
                for k in 0..3 {
                    expect_a[(j, k)] += x[(i, j)] * x[(i, k)];
                }
            }
        }
        expect_a[(1, 1)] += 4.0; // 1 / 0.5^2
        expect_a[(2, 2)] += 0.25; // 1 / 2.0^2
        expect_b[2] += 0.25 * 1.0; // precision * loc
        let expect = solve_qr(&expect_a, &expect_b, 1e-12).unwrap();

        for j in 0..3 {
            assert!(
                (beta[j] - expect[j]).abs() < 1e-12,
                "coefficient {j}: {} vs {}",
                beta[j],
                expect[j]
            );
        }

        // The Laplace covariance must be (X'X + P)^-1, not (X'X)^-1.
        let beta_vec: Vec<f64> = beta.iter().copied().collect();
        let inactive = vec![false; 3];
        let lap = inference(
            &beta_vec,
            &xtx,
            Some(&unpenalized),
            1.0,
            0.95,
            VcovType::Laplace,
            &inactive,
        )
        .unwrap();
        let naive = inference(
            &beta_vec,
            &xtx,
            Some(&unpenalized),
            1.0,
            0.95,
            VcovType::Naive,
            &inactive,
        )
        .unwrap();

        // A prior adds information, so the Laplace SE must be strictly smaller than
        // the naive one for every penalized coefficient.
        for j in 1..3 {
            assert!(
                lap.std_errors[j] < naive.std_errors[j],
                "penalized SE should shrink at {j}: {} vs {}",
                lap.std_errors[j],
                naive.std_errors[j]
            );
        }
        // The unpenalized intercept is (near) unchanged.
        assert!((lap.std_errors[0] - naive.std_errors[0]).abs() < 0.5);
    }

    #[test]
    fn sandwich_sits_between_laplace_and_naive() {
        let mut info: Mat<f64> = Mat::zeros(2, 2);
        info[(0, 0)] = 10.0;
        info[(1, 1)] = 10.0;
        let mut unpen: Mat<f64> = Mat::zeros(2, 2);
        unpen[(0, 0)] = 6.0;
        unpen[(1, 1)] = 6.0;

        let beta = [1.0, 1.0];
        let inactive = [false, false];
        let lap = inference(&beta, &info, Some(&unpen), 1.0, 0.95, VcovType::Laplace, &inactive).unwrap();
        let sw = inference(&beta, &info, Some(&unpen), 1.0, 0.95, VcovType::Sandwich, &inactive).unwrap();
        let nv = inference(&beta, &info, Some(&unpen), 1.0, 0.95, VcovType::Naive, &inactive).unwrap();

        // 1/10 = 0.1 (laplace) ; 6/100 = 0.06 (sandwich) ; 1/6 = 0.167 (naive)
        assert!((lap.vcov[(0, 0)] - 0.1).abs() < 1e-12);
        assert!((sw.vcov[(0, 0)] - 0.06).abs() < 1e-12);
        assert!((nv.vcov[(0, 0)] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn inactive_parameters_get_nan_inference() {
        let mut info: Mat<f64> = Mat::zeros(2, 2);
        info[(0, 0)] = 4.0;
        info[(1, 1)] = 4.0;
        let beta = [0.5, 0.0];
        let inactive = [false, true];

        let inf = inference(&beta, &info, None, 1.0, 0.95, VcovType::Laplace, &inactive).unwrap();
        assert!(inf.std_errors[0].is_finite());
        assert!(inf.std_errors[1].is_nan());
        assert!(inf.p_values[1].is_nan());
    }

    #[test]
    fn confidence_interval_widens_with_the_level() {
        let mut info: Mat<f64> = Mat::zeros(1, 1);
        info[(0, 0)] = 1.0;
        let beta = [0.0];
        let inactive = [false];

        let narrow = inference(&beta, &info, None, 1.0, 0.90, VcovType::Laplace, &inactive).unwrap();
        let wide = inference(&beta, &info, None, 1.0, 0.99, VcovType::Laplace, &inactive).unwrap();
        assert!(wide.ci_upper[0] > narrow.ci_upper[0]);
        // z_{0.975} = 1.959964
        let mid = inference(&beta, &info, None, 1.0, 0.95, VcovType::Laplace, &inactive).unwrap();
        assert!((mid.ci_upper[0] - 1.959_963_984_540_054).abs() < 1e-9);
    }
}
