//! Family-generic penalized IRLS.
//!
//! One loop, driven by the upstream [`GlmFamily`] trait, replacing the six
//! hand-copied solver loops for the purposes of this extension. The convergence
//! policy deliberately mirrors R's `glm.fit` (and therefore upstream): dual
//! deviance/coefficient criteria plus step halving, so an unpenalized fit here is
//! numerically indistinguishable from the upstream one — see the parity tests.
//!
//! Penalties enter through [`Penalty`]. The Gaussian part is a precision block in
//! the normal equations; the L1 part switches the inner solve to proximal
//! coordinate descent (the glmnet algorithm) because a Laplace prior is not a
//! quadratic form.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::core::GlmFamily;
use faer::{Col, Mat};

use super::normal_eq::{solve_weighted_ls_qr, NormalEquations};
use super::penalty::{Penalty, QuadraticPenalty};

/// Tuning for the IRLS loop.
#[derive(Debug, Clone)]
pub struct IrlsConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub rank_tolerance: f64,
    /// Maximum step halvings per iteration, following R's `glm.control`.
    pub max_halvings: usize,
    /// Iterations for the inner coordinate-descent solve when an L1 prior is present.
    pub cd_max_iterations: usize,
    pub cd_tolerance: f64,
}

impl Default for IrlsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            rank_tolerance: 1e-10,
            max_halvings: 10,
            cd_max_iterations: 1000,
            cd_tolerance: 1e-10,
        }
    }
}

/// Outcome of a penalized IRLS fit, in design-matrix parameter order
/// (intercept first when one was fitted).
#[derive(Debug, Clone)]
pub struct IrlsFit {
    pub beta: Vec<f64>,
    pub mu: Vec<f64>,
    pub eta: Vec<f64>,
    pub weights: Vec<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: u32,
    pub converged: bool,
    /// `X'WX + P` at the mode — the observed information of the log posterior.
    pub information: Mat<f64>,
    /// `X'WX` alone, for the naive and sandwich covariance variants.
    pub unpenalized_information: Mat<f64>,
    /// Parameters resting exactly at an L1 prior location; their curvature-based
    /// inference does not exist.
    pub inactive: Vec<bool>,
}

/// Fit a GLM by penalized IRLS.
///
/// `x_design` already includes the intercept column (if any) and excludes any
/// dropped constant columns — [`super::design`] owns that contract. `offset` is
/// added to the linear predictor with an implicit coefficient of 1.
pub fn fit_irls<F: GlmFamily + ?Sized>(
    family: &F,
    x_design: &Mat<f64>,
    y: &[f64],
    offset: Option<&[f64]>,
    penalty: &Penalty,
    config: &IrlsConfig,
) -> StatsResult<IrlsFit> {
    let n = x_design.nrows();
    let p = x_design.ncols();

    if n == 0 || p == 0 {
        return Err(StatsError::InsufficientData { rows: n, cols: p });
    }
    if penalty.n_params() != p {
        return Err(StatsError::InvalidInput(format!(
            "penalty has {} entries but the design matrix has {p} columns",
            penalty.n_params()
        )));
    }

    let mut mu = family.initialize_mu(y);
    let mut eta: Vec<f64> = mu
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let base = family.link(m);
            match offset {
                Some(o) => base - o[i],
                None => base,
            }
        })
        .collect();

    let mut beta: Col<f64> = Col::zeros(p);
    let mut weights = vec![0.0; n];
    let mut z = vec![0.0; n];

    let mut dev = objective(family, y, &mu, &beta, penalty);
    let mut converged = false;
    let mut iterations = 0u32;

    // Problem scale, used to floor the step-halving test so a deviance that has
    // decayed to rounding noise cannot masquerade as divergence.
    let scale = family.null_deviance(y).abs();

    for iter in 0..config.max_iterations {
        iterations = iter as u32 + 1;
        let dev_old = dev;

        // Working weights and working response.
        for i in 0..n {
            weights[i] = family.irls_weight(mu[i]);
            let eta_no_offset = match offset {
                Some(o) => eta[i] - o[i],
                None => eta[i],
            };
            z[i] = eta_no_offset + (y[i] - mu[i]) * family.link_derivative(mu[i]);
        }

        let beta_old = beta.clone();
        let beta_new = solve_penalized_wls(x_design, &z, &weights, penalty, config, &beta_old)?;

        let max_change = beta_new
            .iter()
            .zip(beta_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        beta = beta_new;
        update_eta_mu(family, x_design, &beta, offset, &mut eta, &mut mu)?;
        dev = objective(family, y, &mu, &beta, penalty);

        // Two deliberate departures from the upstream loop, both no-ops for an
        // unpenalized, well-scaled fit (so parity holds) but necessary otherwise:
        //
        // 1. The monitored quantity is the *penalized* objective
        //    `deviance + 2 * penalty`, not the deviance. Under a strong prior the
        //    deviance legitimately rises as the mode is pulled toward the prior
        //    location, so a deviance-only criterion never settles and the fit
        //    reports ConvergenceFailed even though the MAP estimate converged.
        // 2. Convergence is tested before step halving, and the "got worse" test
        //    carries a scale-aware absolute floor. Once the objective decays to
        //    ~1e-14 on a well-fitting model, a purely relative threshold collapses
        //    to zero and floating-point jitter reads as divergence.
        let converged_now = |obj: f64, obj_old: f64, max_change: f64| {
            let obj_converged = (obj - obj_old).abs() / (0.1 + obj.abs()) < config.tolerance;
            let coef_converged = max_change < config.tolerance;
            obj.is_finite() && (obj_converged || coef_converged)
        };

        if converged_now(dev, dev_old, max_change) {
            converged = true;
            break;
        }

        if dev.is_finite() && dev_old.is_finite() {
            let floor = f64::EPSILON * scale.max(1.0);
            let mut halvings = 0;
            while dev > dev_old + 1e-7 * dev_old.abs() + floor && halvings < config.max_halvings {
                halvings += 1;
                for j in 0..p {
                    beta[j] = (beta[j] + beta_old[j]) / 2.0;
                }
                update_eta_mu(family, x_design, &beta, offset, &mut eta, &mut mu)?;
                dev = objective(family, y, &mu, &beta, penalty);
            }

            if converged_now(dev, dev_old, max_change) {
                converged = true;
                break;
            }
        }
    }

    // Report the plain deviance, not the penalized objective.
    let dev = family.deviance(y, &mu);

    if !converged {
        return Err(StatsError::ConvergenceFailure {
            iterations: config.max_iterations as u32,
            tolerance: config.tolerance,
        });
    }

    // Final weights at the mode, for the observed information.
    for i in 0..n {
        weights[i] = family.irls_weight(mu[i]);
    }

    let mut unpenalized = NormalEquations::zeros(p);
    unpenalized.accumulate_dense(x_design, &z, &weights);
    let unpenalized_information = unpenalized.xtwx.clone();

    let mut information = unpenalized_information.clone();
    let mut shift: Col<f64> = Col::zeros(p);
    penalty.quadratic.accumulate(&mut information, &mut shift);

    // An L1 coefficient sitting exactly at its prior location is not differentiable
    // there, so it carries no curvature-based inference.
    let inactive: Vec<bool> = (0..p)
        .map(|j| penalty.l1[j] != 0.0 && (beta[j] - penalty.l1_location[j]).abs() < 1e-12)
        .collect();

    Ok(IrlsFit {
        beta: beta.iter().copied().collect(),
        mu,
        eta,
        weights,
        deviance: dev,
        null_deviance: family.null_deviance(y),
        iterations,
        converged,
        information,
        unpenalized_information,
        inactive,
    })
}

/// Penalized objective: `deviance + 2 * penalty`.
///
/// The factor of two puts the penalty on the deviance scale (deviance is
/// `-2 * log-likelihood` up to a constant), so this is `-2 * log posterior` and its
/// stationary point is the MAP estimate. With no penalty it is exactly the deviance.
fn objective<F: GlmFamily + ?Sized>(
    family: &F,
    y: &[f64],
    mu: &[f64],
    beta: &Col<f64>,
    penalty: &Penalty,
) -> f64 {
    let mut total = family.deviance(y, mu);
    if penalty.is_zero() {
        return total;
    }

    let prec = penalty.quadratic.precisions();
    let loc = penalty.quadratic.locations();
    for j in 0..beta.nrows() {
        if prec[j] != 0.0 {
            let d = beta[j] - loc[j];
            total += prec[j] * d * d;
        }
        if penalty.l1[j] != 0.0 {
            total += 2.0 * penalty.l1[j] * (beta[j] - penalty.l1_location[j]).abs();
        }
    }
    total
}

fn update_eta_mu<F: GlmFamily + ?Sized>(
    family: &F,
    x_design: &Mat<f64>,
    beta: &Col<f64>,
    offset: Option<&[f64]>,
    eta: &mut [f64],
    mu: &mut [f64],
) -> StatsResult<()> {
    let n = x_design.nrows();
    let p = x_design.ncols();

    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += x_design[(i, j)] * beta[j];
        }
        if let Some(o) = offset {
            e += o[i];
        }
        eta[i] = e;

        if !family.valid_eta(e) {
            return Err(StatsError::RegressError(
                "invalid linear predictor (eta) during IRLS: non-finite value".to_string(),
            ));
        }

        let m = family.link_inverse(e);
        mu[i] = if family.valid_mu(m) {
            m
        } else {
            family.clamp_mu(m)
        };
    }
    Ok(())
}

/// One penalized weighted-least-squares solve.
///
/// Three routes, in increasing generality:
/// * no penalty at all -> column-pivoted QR on the `sqrt(W)`-scaled design, which
///   is what upstream does and what the parity gate compares against;
/// * Gaussian penalty only -> normal equations with the precision block added;
/// * any L1 term -> proximal coordinate descent.
fn solve_penalized_wls(
    x: &Mat<f64>,
    z: &[f64],
    weights: &[f64],
    penalty: &Penalty,
    config: &IrlsConfig,
    warm_start: &Col<f64>,
) -> StatsResult<Col<f64>> {
    if penalty.is_zero() {
        return solve_weighted_ls_qr(x, z, weights, config.rank_tolerance);
    }

    let p = x.ncols();
    let mut ne = NormalEquations::zeros(p);
    ne.accumulate_dense(x, z, weights);

    if !penalty.has_l1() {
        penalty.quadratic.accumulate(&mut ne.xtwx, &mut ne.xtwz);
        return ne.solve(config.rank_tolerance);
    }

    // Gaussian part folds into the quadratic form; L1 part is handled by the
    // soft-thresholding update below.
    let mut shift: Col<f64> = Col::zeros(p);
    penalty.quadratic.accumulate(&mut ne.xtwx, &mut shift);
    for j in 0..p {
        ne.xtwz[j] += shift[j];
    }

    coordinate_descent(&ne.xtwx, &ne.xtwz, penalty, config, warm_start)
}

/// Cyclic coordinate descent with soft thresholding on the quadratic form
/// `1/2 b' A b - b' c + sum_j w_j |b_j - loc_j|`.
fn coordinate_descent(
    a: &Mat<f64>,
    c: &Col<f64>,
    penalty: &Penalty,
    config: &IrlsConfig,
    warm_start: &Col<f64>,
) -> StatsResult<Col<f64>> {
    let p = a.nrows();
    let mut b: Col<f64> = warm_start.clone();
    if b.nrows() != p {
        b = Col::zeros(p);
    }

    for _ in 0..config.cd_max_iterations {
        let mut max_delta: f64 = 0.0;

        for j in 0..p {
            let ajj = a[(j, j)];
            if ajj.abs() < config.rank_tolerance {
                continue;
            }

            // Partial residual excluding coordinate j.
            let mut r = c[j];
            for k in 0..p {
                if k != j {
                    r -= a[(j, k)] * b[k];
                }
            }

            let old = b[j];
            let w = penalty.l1[j];
            let new = if w == 0.0 {
                r / ajj
            } else {
                // Shift so the penalty is centred on the prior location.
                let loc = penalty.l1_location[j];
                let r_shifted = r - ajj * loc;
                soft_threshold(r_shifted, w) / ajj + loc
            };

            b[j] = new;
            max_delta = max_delta.max((new - old).abs());
        }

        if max_delta < config.cd_tolerance {
            break;
        }
    }

    if b.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::SingularMatrix);
    }
    Ok(b)
}

#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PriorSpec;
    use anofox_regression::core::PoissonFamily;

    fn design_with_intercept(x: &[Vec<f64>], n: usize) -> Mat<f64> {
        let p = x.len();
        Mat::from_fn(n, p + 1, |i, j| if j == 0 { 1.0 } else { x[j - 1][i] })
    }

    #[test]
    fn poisson_recovers_a_known_log_linear_mean() {
        // log(mu) = 0.5 + 0.3 x, generated without noise so the MLE is exact.
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.3 * x).exp()).collect();
        let design = design_with_intercept(&[xs], n);

        let fit = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &Penalty::none(2),
            &IrlsConfig::default(),
        )
        .unwrap();

        assert!(fit.converged);
        assert!(
            (fit.beta[0] - 0.5).abs() < 1e-8,
            "intercept {}",
            fit.beta[0]
        );
        assert!((fit.beta[1] - 0.3).abs() < 1e-8, "slope {}", fit.beta[1]);
    }

    #[test]
    fn a_tighter_gaussian_prior_shrinks_harder() {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| (0.4 + 0.5 * x).exp().round()).collect();
        let design = design_with_intercept(&[xs], n);

        let mk =
            |scale: f64| Penalty::from_priors(&[PriorSpec::flat(), PriorSpec::normal(0.0, scale)]);

        let loose = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &mk(10.0),
            &IrlsConfig::default(),
        )
        .unwrap();
        let tight = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &mk(0.05),
            &IrlsConfig::default(),
        )
        .unwrap();

        assert!(
            tight.beta[1].abs() < loose.beta[1].abs(),
            "tight {} should shrink below loose {}",
            tight.beta[1],
            loose.beta[1]
        );
    }

    #[test]
    fn a_gaussian_prior_pulls_toward_its_location() {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| (0.4 + 0.5 * x).exp().round()).collect();
        let design = design_with_intercept(&[xs], n);

        // A very tight prior centred at 2.0 should drag the slope up toward 2.0.
        let pen = Penalty::from_priors(&[PriorSpec::flat(), PriorSpec::normal(2.0, 0.01)]);
        let fit = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &pen,
            &IrlsConfig::default(),
        )
        .unwrap();

        assert!(
            fit.beta[1] > 1.5,
            "slope {} should be pulled toward 2.0",
            fit.beta[1]
        );
    }

    #[test]
    fn an_l1_prior_can_zero_a_useless_coefficient() {
        let n = 60;
        let xs: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        // A pure noise-free signal in x1 and a column that carries no information.
        let noise: Vec<f64> = (0..n).map(|i| ((i * 7) % 3) as f64 * 1e-6).collect();
        let y: Vec<f64> = xs.iter().map(|&x| (0.4 + 0.5 * x).exp()).collect();
        let design = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => xs[i],
            _ => noise[i],
        });

        let pen = Penalty::from_priors(&[
            PriorSpec::flat(),
            PriorSpec::flat(),
            PriorSpec::laplace(0.0, 0.01),
        ]);
        let fit = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &pen,
            &IrlsConfig::default(),
        )
        .unwrap();

        assert_eq!(fit.beta[2], 0.0, "noise coefficient should be zeroed");
        assert!(
            fit.inactive[2],
            "zeroed L1 coefficient must be flagged inactive"
        );
        assert!(!fit.inactive[1]);
    }

    #[test]
    fn offset_shifts_the_linear_predictor() {
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| (i % 8) as f64).collect();
        let expo: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        let log_expo: Vec<f64> = expo.iter().map(|e| e.ln()).collect();
        // mu = exposure * exp(0.2 + 0.4 x)
        let y: Vec<f64> = xs
            .iter()
            .zip(expo.iter())
            .map(|(&x, &e)| e * (0.2 + 0.4 * x).exp())
            .collect();
        let design = design_with_intercept(&[xs], n);

        let fit = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            Some(&log_expo),
            &Penalty::none(2),
            &IrlsConfig::default(),
        )
        .unwrap();

        assert!(
            (fit.beta[0] - 0.2).abs() < 1e-8,
            "intercept {}",
            fit.beta[0]
        );
        assert!((fit.beta[1] - 0.4).abs() < 1e-8, "slope {}", fit.beta[1]);
    }

    #[test]
    fn penalty_length_mismatch_is_rejected() {
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![1.0; n];
        let design = design_with_intercept(&[xs], n);

        let err = fit_irls(
            &PoissonFamily::log(),
            &design,
            &y,
            None,
            &Penalty::none(5),
            &IrlsConfig::default(),
        );
        assert!(matches!(err, Err(StatsError::InvalidInput(_))));
    }
}
