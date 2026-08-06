//! Accelerated failure time (AFT) regression with right censoring.
//!
//! Models `log T = x'beta + sigma * W` for a fixed standard error distribution
//! `W` (see [`crate::models::aft_dist`]). An observation contributes its density
//! when the event was observed and its survival when the row is still censored:
//!
//! ```text
//!   loglik = sum_{event}    [ log f_W(z_i) - log sigma - log t_i ]
//!          + sum_{censored} [ log S_W(z_i) ]
//!   with   z_i = (log t_i - x_i'beta) / sigma
//! ```
//!
//! Estimation is Newton-Raphson on `(beta, log sigma)` with an analytic gradient
//! and Hessian, which is what `survival::survreg` does. `log sigma` rather than
//! `sigma` keeps the parameter unconstrained and makes the curvature better
//! behaved near zero.
//!
//! The observed information at the mode goes straight into
//! [`crate::models::glm_engine::laplace::inference`] — the same primitive the GLM
//! prior work uses — so standard errors, Wald tests and intervals are computed in
//! exactly one place for both model families. Gaussian priors on the coefficients
//! also work here, reusing [`crate::models::glm_engine::penalty`].

use crate::errors::{StatsError, StatsResult};
use crate::models::aft_dist::AftDistribution;
use crate::models::glm_engine::laplace::{self, LaplaceInference};
use crate::models::glm_engine::penalty::{Penalty, QuadraticPenalty};
use crate::types::{PriorSpec, VcovType};
use faer::{Col, Mat};

/// Options for an AFT fit.
#[derive(Debug, Clone)]
pub struct AftOptions {
    pub dist: AftDistribution,
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Gaussian/Laplace priors on the coefficients, in feature order with an
    /// optional leading entry for the intercept. Empty means none.
    pub priors: Vec<PriorSpec>,
    pub vcov: VcovType,
}

impl Default for AftOptions {
    fn default() -> Self {
        Self {
            dist: AftDistribution::Weibull,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-9,
            compute_inference: false,
            confidence_level: 0.95,
            priors: Vec::new(),
            vcov: VcovType::default(),
        }
    }
}

/// Core results of an AFT fit.
#[derive(Debug, Clone)]
pub struct AftFitResult {
    /// Coefficients on the log-time scale, excluding the intercept.
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    /// The scale parameter `sigma` (1.0 and not estimated for `exponential`).
    pub scale: f64,
    pub log_likelihood: f64,
    /// Log-likelihood of the intercept-only model, for a likelihood-ratio pseudo-R2.
    pub null_log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_observations: usize,
    pub n_events: usize,
    pub n_censored: usize,
    pub n_features: usize,
    pub iterations: u32,
    pub converged: bool,
}

/// An AFT fit plus optional inference.
#[derive(Debug, Clone)]
pub struct AftResult {
    pub core: AftFitResult,
    pub inference: Option<AftInference>,
}

/// Curvature-based inference for an AFT fit.
#[derive(Debug, Clone)]
pub struct AftInference {
    /// Per-coefficient values, excluding the intercept.
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub confidence_level: f64,
    pub intercept_std_error: Option<f64>,
    /// Standard error of `log sigma`; `None` when the scale is fixed.
    pub log_scale_std_error: Option<f64>,
    /// The full covariance of the fitted parameters at the mode, `None` when inference
    /// was not requested.
    ///
    /// Indexed in the order the fit parameterises: the intercept first when one is
    /// fitted, then the coefficients in feature order, then `log sigma` when the scale
    /// is estimated. So for a fit with an intercept and `p` features the matrix is
    /// `(1 + p + 1)` square, and `sqrt(vcov[(j, j)])` is the standard error this struct
    /// reports for parameter `j`.
    ///
    /// Reported because the diagonal is not the whole answer. Anything that needs the
    /// *joint* distribution -- a Laplace posterior to sample, a prediction interval on
    /// a linear combination, a delta-method standard error -- would otherwise have to
    /// rebuild this matrix from `AftDistribution`'s derivatives, re-differentiating a
    /// likelihood this function has already differentiated.
    pub vcov: Option<Mat<f64>>,
    /// The penalised observed information at the mode -- the inverse of [`Self::vcov`]
    /// -- in the same parameter order, `None` when inference was not requested.
    ///
    /// Both are reported because they answer different questions and each is awkward
    /// to get from the other. `vcov` is what a report reads. The information is what a
    /// *sampler* reads: drawing from the Laplace approximation factorises the
    /// curvature, and recovering it by inverting `vcov` is a round trip through a
    /// matrix this function already holds.
    ///
    /// "Penalised" is the operative word: it includes the contribution of any
    /// [`PriorSpec`] supplied, so it is the curvature of the log *posterior* at the
    /// mode rather than of the log likelihood.
    pub information: Option<Mat<f64>>,
}

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
    if time.is_empty() {
        return Err(StatsError::EmptyInput { field: "time" });
    }
    if event.len() != time.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: time.len(),
            x_rows: event.len(),
        });
    }
    for col in x.iter() {
        if col.len() != time.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: time.len(),
                x_rows: col.len(),
            });
        }
    }

    // Retain rows that are finite throughout, with a positive time and a 0/1 event.
    let mut rows: Vec<usize> = Vec::new();
    for i in 0..time.len() {
        if !time[i].is_finite() || !event[i].is_finite() {
            continue;
        }
        if x.iter().any(|c| !c[i].is_finite()) {
            continue;
        }
        if time[i] <= 0.0 {
            return Err(StatsError::InvalidValue {
                field: "time",
                message: "AFT regression requires strictly positive times".to_string(),
            });
        }
        if event[i] != 0.0 && event[i] != 1.0 {
            return Err(StatsError::InvalidValue {
                field: "event",
                message: "event indicator must be 0 (censored) or 1 (observed)".to_string(),
            });
        }
        rows.push(i);
    }
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n = rows.len();
    let n_features = x.len();
    let n_events = rows.iter().filter(|&&i| event[i] == 1.0).count();
    let n_censored = n - n_events;

    if n_events == 0 {
        return Err(StatsError::InvalidValue {
            field: "event",
            message: "every observation is censored, so the model is not identified".to_string(),
        });
    }

    let fit_scale = !options.dist.scale_is_fixed();
    let n_beta = n_features + usize::from(options.fit_intercept);
    let n_params = n_beta + usize::from(fit_scale);

    if n < n_params + 1 {
        return Err(StatsError::InsufficientData {
            rows: n,
            cols: n_params,
        });
    }

    let log_t: Vec<f64> = rows.iter().map(|&i| time[i].ln()).collect();
    let delta: Vec<f64> = rows.iter().map(|&i| event[i]).collect();
    let int_off = usize::from(options.fit_intercept);
    let design = Mat::from_fn(n, n_beta, |r, c| {
        if options.fit_intercept && c == 0 {
            1.0
        } else {
            x[c - int_off][rows[r]]
        }
    });

    let penalty = build_penalty(&options.priors, n_beta, options.fit_intercept, n_features)?;

    let state = newton(
        options.dist,
        &design,
        &log_t,
        &delta,
        fit_scale,
        &penalty,
        options.max_iterations,
        options.tolerance,
    )?;

    // Null model: intercept (and scale) only.
    let null_ll = if options.fit_intercept {
        let null_design = Mat::from_fn(n, 1, |_, _| 1.0);
        newton(
            options.dist,
            &null_design,
            &log_t,
            &delta,
            fit_scale,
            &Penalty::none(1),
            options.max_iterations,
            options.tolerance,
        )
        .map(|s| s.log_likelihood)
        .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    let (coefficients, intercept) = if options.fit_intercept {
        (state.beta[1..].to_vec(), Some(state.beta[0]))
    } else {
        (state.beta.clone(), None)
    };

    let k = n_params;
    let core = AftFitResult {
        coefficients,
        intercept,
        scale: state.sigma,
        log_likelihood: state.log_likelihood,
        null_log_likelihood: null_ll,
        aic: 2.0 * k as f64 - 2.0 * state.log_likelihood,
        bic: k as f64 * (n as f64).ln() - 2.0 * state.log_likelihood,
        n_observations: n,
        n_events,
        n_censored,
        n_features,
        iterations: state.iterations,
        converged: state.converged,
    };

    let inference = if options.compute_inference {
        let inactive = vec![false; n_params];
        let unpenalized = state.unpenalized_information.clone();
        let inf: LaplaceInference = laplace::inference(
            &state.params(),
            &state.information,
            Some(&unpenalized),
            1.0,
            options.confidence_level,
            options.vcov,
            &inactive,
        )?;

        Some(AftInference {
            std_errors: inf.std_errors[int_off..n_beta].to_vec(),
            z_values: inf.z_values[int_off..n_beta].to_vec(),
            p_values: inf.p_values[int_off..n_beta].to_vec(),
            ci_lower: inf.ci_lower[int_off..n_beta].to_vec(),
            ci_upper: inf.ci_upper[int_off..n_beta].to_vec(),
            confidence_level: options.confidence_level,
            intercept_std_error: if options.fit_intercept {
                Some(inf.std_errors[0])
            } else {
                None
            },
            log_scale_std_error: if fit_scale {
                Some(inf.std_errors[n_beta])
            } else {
                None
            },
            vcov: Some(inf.vcov.clone()),
            information: Some(state.information.clone()),
        })
    } else {
        None
    };

    Ok(AftResult { core, inference })
}

fn build_penalty(
    priors: &[PriorSpec],
    n_beta: usize,
    fit_intercept: bool,
    n_features: usize,
) -> StatsResult<Penalty> {
    if priors.is_empty() {
        return Ok(Penalty::none(n_beta));
    }
    let expected = n_features + usize::from(fit_intercept);
    if priors.len() != expected && priors.len() != n_features {
        return Err(StatsError::InvalidInput(format!(
            "expected {expected} priors, got {}",
            priors.len()
        )));
    }
    let has_intercept_entry = fit_intercept && priors.len() == expected;
    let mut aligned = vec![PriorSpec::flat(); n_beta];
    if has_intercept_entry {
        aligned[0] = priors[0];
    }
    let feature_priors = if has_intercept_entry {
        &priors[1..]
    } else {
        priors
    };
    let off = usize::from(fit_intercept);
    for (j, p) in feature_priors.iter().enumerate() {
        aligned[j + off] = *p;
    }
    Ok(Penalty::from_priors(&aligned))
}

/// State at the Newton optimum.
struct NewtonState {
    beta: Vec<f64>,
    log_sigma: f64,
    sigma: f64,
    log_likelihood: f64,
    iterations: u32,
    converged: bool,
    /// Negative Hessian of the penalized log-likelihood — the observed information.
    information: Mat<f64>,
    /// Negative Hessian of the unpenalized log-likelihood.
    unpenalized_information: Mat<f64>,
    fit_scale: bool,
}

impl NewtonState {
    fn params(&self) -> Vec<f64> {
        let mut v = self.beta.clone();
        if self.fit_scale {
            v.push(self.log_sigma);
        }
        v
    }
}

/// Newton-Raphson with step halving on `(beta, log sigma)`.
#[allow(clippy::too_many_arguments)]
fn newton(
    dist: AftDistribution,
    design: &Mat<f64>,
    log_t: &[f64],
    delta: &[f64],
    fit_scale: bool,
    penalty: &Penalty,
    max_iterations: u32,
    tolerance: f64,
) -> StatsResult<NewtonState> {
    let n = design.nrows();
    let n_beta = design.ncols();
    let n_params = n_beta + usize::from(fit_scale);

    // Start from a least-squares fit of log t, which is the exact MLE for the
    // lognormal case with no censoring and a good start for the others.
    let mut beta = initial_beta(design, log_t)?;
    let mut log_sigma = if fit_scale {
        initial_log_sigma(design, log_t, &beta)
    } else {
        0.0
    };

    let mut ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0u32;

    for iter in 0..max_iterations {
        iterations = iter + 1;
        let (cur_ll, grad, hess) = derivatives(
            dist, design, log_t, delta, &beta, log_sigma, fit_scale, penalty,
        );

        if !cur_ll.is_finite() {
            return Err(StatsError::RegressError(
                "non-finite AFT log-likelihood during Newton iteration".to_string(),
            ));
        }

        // Solve (-H) step = grad, i.e. the Newton direction.
        //
        // The AFT log-likelihood is concave in beta for a fixed sigma but the joint
        // objective in (beta, log sigma) is not globally concave, so far from the
        // optimum -H can fail to be positive definite and the "Newton direction"
        // then points downhill. A strong prior makes this easy to trigger: the
        // starting point sits far out on the penalized surface.
        //
        // The remedy is Levenberg-style damping — inflate the diagonal until the
        // direction is an ascent direction again, which interpolates between the
        // Newton step and steepest ascent.
        let g = Col::from_fn(n_params, |i| grad[i]);
        let build_info = |damping: f64| {
            Mat::from_fn(n_params, n_params, |r, c| {
                let base = -hess[(r, c)];
                if r == c {
                    base + damping * base.abs().max(1.0)
                } else {
                    base
                }
            })
        };

        let mut step = crate::models::glm_engine::normal_eq::solve_qr(&build_info(0.0), &g, 1e-12)?;
        let mut decrement: f64 = (0..n_params).map(|j| grad[j] * step[j]).sum();
        if !decrement.is_finite() || decrement <= 0.0 {
            let mut damping = 1e-3;
            for _ in 0..20 {
                step = crate::models::glm_engine::normal_eq::solve_qr(
                    &build_info(damping),
                    &g,
                    1e-12,
                )?;
                decrement = (0..n_params).map(|j| grad[j] * step[j]).sum();
                if decrement > 0.0 && decrement.is_finite() {
                    break;
                }
                damping *= 10.0;
            }
            if !decrement.is_finite() || decrement <= 0.0 {
                // Even heavy damping did not produce an ascent direction; fall back
                // to steepest ascent, normalised so the line search starts sanely.
                let gnorm = grad.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
                step = Col::from_fn(n_params, |i| grad[i] / gnorm);
                decrement = (0..n_params).map(|j| grad[j] * step[j]).sum();
            }
        }

        // The decrement approximates twice the remaining gap in the objective. Its
        // threshold is scaled by the magnitude of the objective rather than
        // absolute: a strong prior drives the penalized log-likelihood to ~1e5,
        // where double precision cannot resolve an absolute 1e-9 at all and the
        // iteration would spin to max_iterations despite having converged.
        let gmax = grad.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        if decrement.is_finite() && 0.5 * decrement.abs() < tolerance * (1.0 + cur_ll.abs()) {
            ll = cur_ll;
            converged = true;
            break;
        }

        // Step halving on the log-likelihood.
        let mut factor = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let mut trial_beta = beta.clone();
            for j in 0..n_beta {
                trial_beta[j] += factor * step[j];
            }
            let trial_log_sigma = if fit_scale {
                log_sigma + factor * step[n_beta]
            } else {
                0.0
            };

            let trial_ll = log_likelihood(
                dist,
                design,
                log_t,
                delta,
                &trial_beta,
                trial_log_sigma,
                penalty,
            );
            if trial_ll.is_finite() && trial_ll >= cur_ll - 1e-12 {
                let delta_ll = (trial_ll - cur_ll).abs();
                beta = trial_beta;
                log_sigma = trial_log_sigma;
                ll = trial_ll;
                accepted = true;
                // Only a *full* Newton step that barely moves the objective means
                // we are at the optimum. A heavily halved step also produces a
                // negligible change, but because the step is tiny, not because the
                // gradient is — treating that as convergence stops the iteration
                // far from the mode. The gradient test at the top of the loop is
                // the real criterion.
                if factor == 1.0 && delta_ll / (1.0 + trial_ll.abs()) < tolerance {
                    converged = true;
                }
                break;
            }
            factor *= 0.5;
        }

        if !accepted {
            // For a concave objective a sufficiently short Newton step always
            // improves unless the gradient is already ~0, so exhausting the line
            // search means we are at the optimum to numerical precision. Guard it
            // with a loose gradient check anyway, so a genuinely bad direction is
            // reported rather than silently accepted.
            ll = cur_ll;
            converged = gmax < 1e-4;
            break;
        }
        if converged {
            break;
        }
    }

    if !converged {
        return Err(StatsError::ConvergenceFailure {
            iterations: max_iterations,
            tolerance,
        });
    }

    let (final_ll, _, hess) = derivatives(
        dist, design, log_t, delta, &beta, log_sigma, fit_scale, penalty,
    );
    let (_, _, hess_unpen) = derivatives(
        dist,
        design,
        log_t,
        delta,
        &beta,
        log_sigma,
        fit_scale,
        &Penalty::none(n_beta),
    );

    let information = Mat::from_fn(n_params, n_params, |r, c| -hess[(r, c)]);
    let unpenalized_information = Mat::from_fn(n_params, n_params, |r, c| -hess_unpen[(r, c)]);

    let _ = n; // retained for clarity of the loops above
    Ok(NewtonState {
        beta,
        log_sigma,
        sigma: log_sigma.exp(),
        log_likelihood: if ll.is_finite() { ll } else { final_ll },
        iterations,
        converged,
        information,
        unpenalized_information,
        fit_scale,
    })
}

/// OLS of `log t` on the design, as a starting point.
fn initial_beta(design: &Mat<f64>, log_t: &[f64]) -> StatsResult<Vec<f64>> {
    let n = design.nrows();
    let p = design.ncols();
    let mut xtx: Mat<f64> = Mat::zeros(p, p);
    let mut xty: Col<f64> = Col::zeros(p);
    for i in 0..n {
        for j in 0..p {
            xty[j] += design[(i, j)] * log_t[i];
            for k in 0..p {
                xtx[(j, k)] += design[(i, j)] * design[(i, k)];
            }
        }
    }
    let beta = crate::models::glm_engine::normal_eq::solve_qr(&xtx, &xty, 1e-12)?;
    Ok(beta.iter().copied().collect())
}

fn initial_log_sigma(design: &Mat<f64>, log_t: &[f64], beta: &[f64]) -> f64 {
    let n = design.nrows();
    let p = design.ncols();
    let mut rss = 0.0;
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += design[(i, j)] * beta[j];
        }
        rss += (log_t[i] - eta).powi(2);
    }
    let df = (n.saturating_sub(p)).max(1) as f64;
    (rss / df).max(1e-8).sqrt().ln()
}

/// Penalized log-likelihood.
fn log_likelihood(
    dist: AftDistribution,
    design: &Mat<f64>,
    log_t: &[f64],
    delta: &[f64],
    beta: &[f64],
    log_sigma: f64,
    penalty: &Penalty,
) -> f64 {
    let n = design.nrows();
    let p = design.ncols();
    let sigma = log_sigma.exp();
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let mut ll = 0.0;
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += design[(i, j)] * beta[j];
        }
        let z = (log_t[i] - eta) / sigma;
        ll += if delta[i] == 1.0 {
            dist.log_density(z) - log_sigma - log_t[i]
        } else {
            dist.log_survival(z)
        };
    }

    // A Gaussian prior contributes -0.5 * precision * (beta - loc)^2 to the log
    // posterior; the L1 part contributes -weight * |beta - loc|.
    let prec = penalty.quadratic.precisions();
    let loc = penalty.quadratic.locations();
    for j in 0..p {
        if prec[j] != 0.0 {
            let d = beta[j] - loc[j];
            ll -= 0.5 * prec[j] * d * d;
        }
        if penalty.l1[j] != 0.0 {
            ll -= penalty.l1[j] * (beta[j] - penalty.l1_location[j]).abs();
        }
    }
    ll
}

/// Penalized log-likelihood together with its gradient and Hessian.
///
/// With `z = (log t - x'beta) / sigma`, `dz/dbeta_j = -x_j / sigma` and
/// `dz/dlog sigma = -z`. Writing `u` for the first and `v` for the second
/// derivative of the per-observation contribution with respect to `z`:
///
/// ```text
///   dl/dbeta_j        = sum -u * x_j / sigma
///   dl/dlog sigma     = sum (-u * z - delta)
///   d2l/dbeta_j dbeta_k = sum  v * x_j * x_k / sigma^2
///   d2l/dbeta_j dlogs   = sum  x_j * (v * z + u) / sigma
///   d2l/dlogs^2         = sum (v * z^2 + u * z)
/// ```
#[allow(clippy::too_many_arguments)]
fn derivatives(
    dist: AftDistribution,
    design: &Mat<f64>,
    log_t: &[f64],
    delta: &[f64],
    beta: &[f64],
    log_sigma: f64,
    fit_scale: bool,
    penalty: &Penalty,
) -> (f64, Vec<f64>, Mat<f64>) {
    let n = design.nrows();
    let p = design.ncols();
    let n_params = p + usize::from(fit_scale);
    let sigma = log_sigma.exp();

    let mut ll = 0.0;
    let mut grad = vec![0.0; n_params];
    let mut hess: Mat<f64> = Mat::zeros(n_params, n_params);

    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += design[(i, j)] * beta[j];
        }
        let z = (log_t[i] - eta) / sigma;

        let (contrib, u, v) = if delta[i] == 1.0 {
            (
                dist.log_density(z) - log_sigma - log_t[i],
                dist.d_log_density(z),
                dist.dd_log_density(z),
            )
        } else {
            (
                dist.log_survival(z),
                dist.d_log_survival(z),
                dist.dd_log_survival(z),
            )
        };
        ll += contrib;

        for j in 0..p {
            let xj = design[(i, j)];
            grad[j] += -u * xj / sigma;
            for k in j..p {
                let add = v * xj * design[(i, k)] / (sigma * sigma);
                hess[(j, k)] += add;
                if k != j {
                    hess[(k, j)] += add;
                }
            }
            if fit_scale {
                let cross = xj * (v * z + u) / sigma;
                hess[(j, p)] += cross;
                hess[(p, j)] += cross;
            }
        }

        if fit_scale {
            grad[p] += -u * z - delta[i];
            hess[(p, p)] += v * z * z + u * z;
        }
    }

    // Prior contributions.
    let prec = penalty.quadratic.precisions();
    let loc = penalty.quadratic.locations();
    for j in 0..p {
        if prec[j] != 0.0 {
            let d = beta[j] - loc[j];
            ll -= 0.5 * prec[j] * d * d;
            grad[j] -= prec[j] * d;
            hess[(j, j)] -= prec[j];
        }
        if penalty.l1[j] != 0.0 {
            let d = beta[j] - penalty.l1_location[j];
            ll -= penalty.l1[j] * d.abs();
            // Subgradient; the Hessian of |.| is zero away from the kink.
            grad[j] -= penalty.l1[j] * d.signum();
        }
    }

    (ll, grad, hess)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Weibull-ish survival fixture with a controllable censoring
    /// fraction. Times are generated by inverting the AFT quantile function at
    /// spread-out probabilities, so the data really does come from the model.
    fn survival_fixture(
        dist: AftDistribution,
        beta0: f64,
        beta1: f64,
        sigma: f64,
        n: usize,
        censor_at: Option<f64>,
    ) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        // Censoring thresholds are staggered per row rather than shared. A single
        // shared threshold piles every censored row onto one identical time, which
        // is both unrealistic (real censoring comes from staggered entry) and
        // degenerate to fit.
        let mut time = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 10) as f64 / 3.0;
            let p = (i as f64 + 0.5) / n as f64;
            let eta = beta0 + beta1 * x;
            let t = dist.quantile_time(p, eta, sigma);
            let threshold = censor_at.map(|c| c + (i % 7) as f64 * 0.9);
            match threshold {
                Some(c) if t > c => {
                    time.push(c);
                    event.push(0.0);
                }
                _ => {
                    time.push(t);
                    event.push(1.0);
                }
            }
            xs.push(x);
        }
        (time, vec![xs], event)
    }

    #[test]
    fn uncensored_lognormal_matches_ols_on_log_time() {
        // With no censoring the lognormal AFT is exactly OLS of log t.
        let (time, x, event) =
            survival_fixture(AftDistribution::LogNormal, 1.0, 0.4, 0.5, 200, None);

        let opts = AftOptions {
            dist: AftDistribution::LogNormal,
            ..Default::default()
        };
        let fit = fit_aft(&time, &x, &event, &opts).unwrap();

        // OLS reference.
        let n = time.len();
        let log_t: Vec<f64> = time.iter().map(|t| t.ln()).collect();
        let xs = &x[0];
        let mx = xs.iter().sum::<f64>() / n as f64;
        let my = log_t.iter().sum::<f64>() / n as f64;
        let sxy: f64 = xs
            .iter()
            .zip(log_t.iter())
            .map(|(a, b)| (a - mx) * (b - my))
            .sum();
        let sxx: f64 = xs.iter().map(|a| (a - mx).powi(2)).sum();
        let slope = sxy / sxx;
        let intercept = my - slope * mx;

        assert!(
            (fit.core.coefficients[0] - slope).abs() < 1e-6,
            "slope {} vs OLS {slope}",
            fit.core.coefficients[0]
        );
        assert!(
            (fit.core.intercept.unwrap() - intercept).abs() < 1e-6,
            "intercept {} vs OLS {intercept}",
            fit.core.intercept.unwrap()
        );
    }

    /// **The curvature is reported, not just its diagonal.**
    ///
    /// `fit_aft` computes the full observed information at the mode and hands it to
    /// `laplace::inference`, which returns the whole covariance matrix. Until now only
    /// slices of the *diagonal* survived into `AftInference` -- `std_errors`,
    /// `z_values`, the interval bounds -- and the off-diagonal was discarded on the way
    /// out of the crate.
    ///
    /// That is a real loss rather than a tidy-up. Any consumer that needs the joint
    /// distribution of the coefficients -- a Laplace posterior to sample from, a
    /// prediction interval on a linear combination, a delta-method standard error --
    /// has to reconstruct the matrix from `AftDistribution`'s public derivatives, which
    /// means re-deriving the likelihood this function has already differentiated.
    ///
    /// So the matrix is part of the result. Its diagonal must reproduce the standard
    /// errors this crate already reports, which is the assertion that keeps the two
    /// from drifting apart.
    /// **The curvature itself, not only its inverse.**
    ///
    /// `vcov` answers "how uncertain", which is what a report wants. A consumer that
    /// *samples* the Laplace approximation wants the other one: the observed
    /// information is what a Cholesky factorises to draw from the Gaussian, and
    /// recovering it by inverting `vcov` is a round trip through a matrix this
    /// function already has in hand.
    ///
    /// `anofox-bayes` is that consumer, and until this existed it rebuilt the matrix
    /// from `AftDistribution`'s derivatives per observation -- re-differentiating a
    /// likelihood already differentiated here.
    #[test]
    fn the_reported_information_is_the_inverse_of_the_reported_covariance() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            let (time, x, event) = survival_fixture(dist, 1.5, 0.35, 0.6, 400, Some(0.3));
            let opts = AftOptions {
                dist,
                compute_inference: true,
                ..Default::default()
            };
            let fit = fit_aft(&time, &x, &event, &opts).unwrap();
            let inf = fit.inference.as_ref().expect("inference was requested");
            let info = inf
                .information
                .as_ref()
                .expect("the observed information must be reported");
            let vcov = inf.vcov.as_ref().expect("the covariance must be reported");

            assert_eq!(info.nrows(), vcov.nrows(), "{dist:?} same shape");
            // information * vcov = I, which is the whole claim about what it is.
            let n = info.nrows();
            for r in 0..n {
                for c in 0..n {
                    let entry: f64 = (0..n).map(|k| info[(r, k)] * vcov[(k, c)]).sum();
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (entry - want).abs() < 1e-8,
                        "{dist:?}: (information * vcov)[{r},{c}] = {entry}, expected {want}"
                    );
                }
            }
            // Penalised, not naive: the diagonal must exceed nothing in particular,
            // but it must be symmetric and positive on the diagonal to be a curvature.
            for j in 0..n {
                assert!(info[(j, j)] > 0.0, "{dist:?}: information diagonal {j}");
            }
        }
    }

    #[test]
    fn the_reported_curvature_agrees_with_the_standard_errors_it_summarises() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            let (time, x, event) = survival_fixture(dist, 1.5, 0.35, 0.6, 400, Some(0.3));
            let opts = AftOptions {
                dist,
                compute_inference: true,
                ..Default::default()
            };
            let fit = fit_aft(&time, &x, &event, &opts).unwrap();
            let inf = fit.inference.as_ref().expect("inference was requested");
            let vcov = inf.vcov.as_ref().expect("the covariance must be reported");

            // Intercept first, then the coefficients, then log-scale: the order
            // `fit_aft` fits in, and the order the matrix is indexed by.
            let n_beta = 1 + 1; // intercept + one feature
            assert_eq!(vcov.nrows(), n_beta + 1, "{dist:?} covariance is p x p");
            assert_eq!(vcov.ncols(), vcov.nrows(), "{dist:?} covariance is square");

            let se_of = |j: usize| vcov[(j, j)].sqrt();
            assert!(
                (se_of(0) - inf.intercept_std_error.unwrap()).abs() < 1e-9,
                "{dist:?} intercept SE {} vs sqrt(vcov[0,0]) {}",
                inf.intercept_std_error.unwrap(),
                se_of(0)
            );
            assert!(
                (se_of(1) - inf.std_errors[0]).abs() < 1e-9,
                "{dist:?} slope SE {} vs sqrt(vcov[1,1]) {}",
                inf.std_errors[0],
                se_of(1)
            );
            assert!(
                (se_of(2) - inf.log_scale_std_error.unwrap()).abs() < 1e-9,
                "{dist:?} log-scale SE {} vs sqrt(vcov[2,2]) {}",
                inf.log_scale_std_error.unwrap(),
                se_of(2)
            );

            // Symmetric, and not diagonal: the off-diagonal is the part that was being
            // thrown away, so a matrix that happened to be diagonal would prove nothing.
            for r in 0..vcov.nrows() {
                for c in 0..vcov.ncols() {
                    assert!(
                        (vcov[(r, c)] - vcov[(c, r)]).abs() < 1e-12,
                        "{dist:?} covariance must be symmetric at ({r},{c})"
                    );
                }
            }
            assert!(
                vcov[(0, 1)].abs() > 1e-12,
                "{dist:?} intercept and slope must covary on a fixture whose \
                 covariate is not centred; got {}",
                vcov[(0, 1)]
            );
        }
    }

    #[test]
    fn recovers_the_generating_parameters_without_censoring() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            let (time, x, event) = survival_fixture(dist, 1.5, 0.35, 0.6, 400, None);
            let opts = AftOptions {
                dist,
                ..Default::default()
            };
            let fit = fit_aft(&time, &x, &event, &opts).unwrap();

            assert!(fit.core.converged, "{dist:?} did not converge");
            assert!(
                (fit.core.coefficients[0] - 0.35).abs() < 0.05,
                "{dist:?} slope {} should be near 0.35",
                fit.core.coefficients[0]
            );
            assert!(
                (fit.core.intercept.unwrap() - 1.5).abs() < 0.1,
                "{dist:?} intercept {} should be near 1.5",
                fit.core.intercept.unwrap()
            );
            assert!(
                (fit.core.scale - 0.6).abs() < 0.1,
                "{dist:?} scale {} should be near 0.6",
                fit.core.scale
            );
        }
    }

    /// This is the bias the issue is about: fitting a duration model on observed
    /// times while ignoring which rows are still open.
    ///
    /// The same data is fitted twice, once honouring the event indicator and once
    /// pretending every censored time was an observed event. The censoring-aware
    /// fit recovers all three parameters; the naive one attenuates the covariate
    /// effect by more than half and underestimates the scale by ~30%, because
    /// replacing a large unobserved time with its (smaller) censoring time both
    /// compresses the spread and flattens the relationship with x.
    ///
    /// Note the *intercept* is nearly unaffected — the downward pull of the
    /// truncated times and the upward pull of the shrunken scale largely cancel.
    /// It is the slope and the scale that give the problem away.
    #[test]
    fn censoring_is_accounted_for_rather_than_ignored() {
        let dist = AftDistribution::Weibull;
        let (time, x, event) = survival_fixture(dist, 2.0, 0.3, 0.5, 300, Some(9.0));
        let n_censored = event.iter().filter(|&&e| e == 0.0).count();
        assert!(
            n_censored > 50,
            "fixture must actually censor: {n_censored}"
        );

        let opts = AftOptions {
            dist,
            ..Default::default()
        };
        let honest = fit_aft(&time, &x, &event, &opts).unwrap();
        let naive = fit_aft(&time, &x, &vec![1.0; time.len()], &opts).unwrap();

        assert_eq!(honest.core.n_censored, n_censored);
        assert_eq!(naive.core.n_censored, 0);

        // The censoring-aware fit recovers the truth.
        assert!(
            (honest.core.intercept.unwrap() - 2.0).abs() < 0.1,
            "honest intercept {}",
            honest.core.intercept.unwrap()
        );
        assert!(
            (honest.core.coefficients[0] - 0.3).abs() < 0.05,
            "honest slope {}",
            honest.core.coefficients[0]
        );
        assert!(
            (honest.core.scale - 0.5).abs() < 0.05,
            "honest scale {}",
            honest.core.scale
        );

        // The naive fit does not.
        assert!(
            naive.core.coefficients[0] < 0.5 * honest.core.coefficients[0],
            "ignoring censoring should attenuate the slope: naive {} vs honest {}",
            naive.core.coefficients[0],
            honest.core.coefficients[0]
        );
        assert!(
            naive.core.scale < 0.8 * honest.core.scale,
            "ignoring censoring should compress the scale: naive {} vs honest {}",
            naive.core.scale,
            honest.core.scale
        );
    }

    #[test]
    fn exponential_holds_the_scale_at_one() {
        let (time, x, event) =
            survival_fixture(AftDistribution::Exponential, 1.0, 0.2, 1.0, 200, None);
        let opts = AftOptions {
            dist: AftDistribution::Exponential,
            compute_inference: true,
            ..Default::default()
        };
        let fit = fit_aft(&time, &x, &event, &opts).unwrap();
        assert_eq!(fit.core.scale, 1.0);
        assert!(fit.inference.unwrap().log_scale_std_error.is_none());
    }

    #[test]
    fn inference_is_populated_and_finite() {
        let (time, x, event) =
            survival_fixture(AftDistribution::Weibull, 1.5, 0.35, 0.6, 300, Some(12.0));
        let opts = AftOptions {
            dist: AftDistribution::Weibull,
            compute_inference: true,
            ..Default::default()
        };
        let fit = fit_aft(&time, &x, &event, &opts).unwrap();
        let inf = fit.inference.unwrap();

        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        assert!(inf.p_values[0].is_finite());
        assert!(inf.ci_lower[0] < fit.core.coefficients[0]);
        assert!(inf.ci_upper[0] > fit.core.coefficients[0]);
        assert!(inf.intercept_std_error.unwrap() > 0.0);
        assert!(inf.log_scale_std_error.unwrap() > 0.0);
    }

    #[test]
    fn a_prior_shrinks_an_aft_coefficient() {
        let (time, x, event) =
            survival_fixture(AftDistribution::Weibull, 1.5, 0.35, 0.6, 300, None);

        let flat = fit_aft(
            &time,
            &x,
            &event,
            &AftOptions {
                dist: AftDistribution::Weibull,
                ..Default::default()
            },
        )
        .unwrap();

        let shrunk = fit_aft(
            &time,
            &x,
            &event,
            &AftOptions {
                dist: AftDistribution::Weibull,
                priors: vec![PriorSpec::normal(0.0, 0.01)],
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            shrunk.core.coefficients[0].abs() < flat.core.coefficients[0].abs(),
            "prior should shrink: {} vs {}",
            shrunk.core.coefficients[0],
            flat.core.coefficients[0]
        );
    }

    /// A very tight prior puts the starting point far out on the penalized
    /// surface, where the joint Hessian in (beta, log sigma) need not be negative
    /// definite. Without Levenberg damping the "Newton" direction points downhill,
    /// the intercept runs away and the fit reports ConvergenceFailure.
    #[test]
    fn a_very_tight_prior_still_converges() {
        let (time, x, event) =
            survival_fixture(AftDistribution::Weibull, 2.0, 0.3, 0.5, 300, Some(9.0));

        let mut previous = f64::INFINITY;
        for scale in [0.01, 0.005, 0.002, 0.001] {
            let fit = fit_aft(
                &time,
                &x,
                &event,
                &AftOptions {
                    dist: AftDistribution::Weibull,
                    priors: vec![PriorSpec::normal(0.0, scale)],
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("prior scale {scale} failed: {e:?}"));

            let magnitude = fit.core.coefficients[0].abs();
            assert!(
                magnitude < previous,
                "a tighter prior must shrink harder: scale {scale} gave {magnitude}, \
                 previous was {previous}"
            );
            previous = magnitude;
        }
        assert!(
            previous < 1e-3,
            "the tightest prior should pin the coefficient near 0"
        );
    }

    #[test]
    fn all_censored_is_rejected_rather_than_returning_nan() {
        let (time, x, _) = survival_fixture(AftDistribution::Weibull, 1.0, 0.2, 0.5, 50, None);
        let event = vec![0.0; time.len()];
        let err = fit_aft(&time, &x, &event, &AftOptions::default());
        assert!(matches!(
            err,
            Err(StatsError::InvalidValue { field: "event", .. })
        ));
    }

    #[test]
    fn non_positive_times_are_rejected() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let time = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        let event = vec![1.0; 5];
        assert!(matches!(
            fit_aft(&time, &x, &event, &AftOptions::default()),
            Err(StatsError::InvalidValue { field: "time", .. })
        ));
    }

    #[test]
    fn a_non_binary_event_indicator_is_rejected() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1.0, 0.0, 2.0, 1.0, 0.0];
        assert!(matches!(
            fit_aft(&time, &x, &event, &AftOptions::default()),
            Err(StatsError::InvalidValue { field: "event", .. })
        ));
    }

    #[test]
    fn too_few_observations_is_rejected() {
        let x = vec![vec![1.0, 2.0]];
        let time = vec![1.0, 2.0];
        let event = vec![1.0, 1.0];
        assert!(matches!(
            fit_aft(&time, &x, &event, &AftOptions::default()),
            Err(StatsError::InsufficientData { .. })
        ));
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1.0, 1.0];
        assert!(matches!(
            fit_aft(&time, &x, &event, &AftOptions::default()),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_rows_are_dropped() {
        let (mut time, mut x, mut event) =
            survival_fixture(AftDistribution::Weibull, 1.5, 0.35, 0.6, 100, None);
        let clean = time.len();
        time.push(f64::NAN);
        x[0].push(1.0);
        event.push(1.0);

        let fit = fit_aft(
            &time,
            &x,
            &event,
            &AftOptions {
                dist: AftDistribution::Weibull,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(fit.core.n_observations, clean);
    }

    #[test]
    fn the_analytic_gradient_matches_finite_differences() {
        let (time, x, event) =
            survival_fixture(AftDistribution::Weibull, 1.2, 0.4, 0.7, 60, Some(6.0));
        let n = time.len();
        let log_t: Vec<f64> = time.iter().map(|t| t.ln()).collect();
        let design = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x[0][i] });
        let beta = vec![1.1, 0.35];
        let log_sigma = -0.3;
        let pen = Penalty::none(2);
        let dist = AftDistribution::Weibull;

        let (_, grad, hess) =
            derivatives(dist, &design, &log_t, &event, &beta, log_sigma, true, &pen);

        let h = 1e-6;
        // Gradient with respect to each beta.
        for j in 0..2 {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[j] += h;
            bm[j] -= h;
            let num = (log_likelihood(dist, &design, &log_t, &event, &bp, log_sigma, &pen)
                - log_likelihood(dist, &design, &log_t, &event, &bm, log_sigma, &pen))
                / (2.0 * h);
            assert!(
                (grad[j] - num).abs() < 1e-4,
                "grad[{j}] analytic {} vs numeric {num}",
                grad[j]
            );
        }
        // Gradient with respect to log sigma.
        let num_s = (log_likelihood(dist, &design, &log_t, &event, &beta, log_sigma + h, &pen)
            - log_likelihood(dist, &design, &log_t, &event, &beta, log_sigma - h, &pen))
            / (2.0 * h);
        assert!(
            (grad[2] - num_s).abs() < 1e-4,
            "grad[logsigma] analytic {} vs numeric {num_s}",
            grad[2]
        );

        // Hessian diagonal.
        for j in 0..2 {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[j] += h;
            bm[j] -= h;
            let num = (log_likelihood(dist, &design, &log_t, &event, &bp, log_sigma, &pen)
                - 2.0 * log_likelihood(dist, &design, &log_t, &event, &beta, log_sigma, &pen)
                + log_likelihood(dist, &design, &log_t, &event, &bm, log_sigma, &pen))
                / (h * h);
            assert!(
                (hess[(j, j)] - num).abs() < 1e-2 * hess[(j, j)].abs().max(1.0),
                "hess[{j},{j}] analytic {} vs numeric {num}",
                hess[(j, j)]
            );
        }
    }
}
