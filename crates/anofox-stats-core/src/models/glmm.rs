//! Mixed-effects GLMs: a random intercept over one grouping factor.
//!
//! ```text
//!   g(mu_ij) = x_ij' beta + b_j ,   b_j ~ N(0, sigma_b^2)
//! ```
//!
//! This is where the prior work in [`crate::models::glm_engine`] pays off. A
//! Gaussian random effect and a Gaussian prior are the same object — a quadratic
//! precision block in the penalized normal equations — so the inner loop here is
//! penalized IRLS over `[X | Z]` with a zero block on `X` and
//! `sigma_e^2 / sigma_b^2` on `Z`. Those are Henderson's mixed-model equations.
//!
//! Two things stop this from being a literal call into the dense GLM path:
//!
//! * `Z` is a group-indicator matrix with one column per group. For thousands of
//!   SKUs a dense `[X | Z]` is hopeless, so the normal equations are accumulated
//!   block-wise: the `Z'WZ` block is *diagonal*, and `X'WZ` is one column per
//!   group. `Z` is never materialized. This is the reason
//!   [`crate::models::glm_engine::normal_eq`] exists as its own seam.
//! * `sigma_b^2` has to be estimated. The outer loop maximizes the
//!   Laplace-approximated log marginal likelihood by bounded Brent search, over
//!   the *ratio* `theta = sigma_b / sigma_e` rather than `sigma_b` itself. The
//!   mixed-model penalty is `sigma_e^2 / sigma_b^2`, so it is scale-free;
//!   searching over `sigma_b` alone makes the objective depend on the units of
//!   the response. `lme4` profiles over the same ratio.
//!
//! Scope is deliberately one random intercept over one grouping factor. Random
//! slopes need an unstructured `Lambda(theta)` and a multi-dimensional outer
//! optimization; they are a follow-up.

use crate::errors::{StatsError, StatsResult};
use crate::models::glm_engine::loglik::{self, LogLikKind};
use anofox_regression::core::{
    BinomialFamily, GlmFamily, NegativeBinomialFamily, PoissonFamily, TweedieFamily,
};
use faer::{Col, Mat};

/// Which response family the mixed model uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlmmFamily {
    /// Identity link, Gaussian errors — an `lmer`-style linear mixed model.
    Gaussian,
    Poisson,
    Binomial,
    NegativeBinomial {
        theta: f64,
    },
    Gamma,
    Tweedie {
        power: f64,
    },
}

impl GlmmFamily {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "gaussian" | "normal" | "lmm" => Some(GlmmFamily::Gaussian),
            "poisson" => Some(GlmmFamily::Poisson),
            "binomial" | "logistic" => Some(GlmmFamily::Binomial),
            "negbinomial" | "negative_binomial" | "negbin" => {
                Some(GlmmFamily::NegativeBinomial { theta: 1.0 })
            }
            "gamma" => Some(GlmmFamily::Gamma),
            "tweedie" => Some(GlmmFamily::Tweedie { power: 1.5 }),
            _ => None,
        }
    }

    fn is_gaussian(&self) -> bool {
        matches!(self, GlmmFamily::Gaussian)
    }

    /// The upstream family object driving the IRLS weights, or `None` for
    /// Gaussian, which needs no iteration.
    fn upstream(&self) -> Option<Box<dyn GlmFamily>> {
        match *self {
            GlmmFamily::Gaussian => None,
            GlmmFamily::Poisson => Some(Box::new(PoissonFamily::log())),
            GlmmFamily::Binomial => Some(Box::new(BinomialFamily::logistic())),
            GlmmFamily::NegativeBinomial { theta } => {
                Some(Box::new(NegativeBinomialFamily::new(theta)))
            }
            GlmmFamily::Gamma => Some(Box::new(TweedieFamily::new(2.0, 0.0))),
            GlmmFamily::Tweedie { power } => Some(Box::new(TweedieFamily::new(power, 0.0))),
        }
    }

    fn loglik_kind(&self, dispersion: f64) -> LogLikKind {
        match *self {
            // Handled separately; never reached for Gaussian.
            GlmmFamily::Gaussian => LogLikKind::Poisson,
            GlmmFamily::Poisson => LogLikKind::Poisson,
            GlmmFamily::Binomial => LogLikKind::Binomial,
            GlmmFamily::NegativeBinomial { theta } => LogLikKind::NegativeBinomial { theta },
            GlmmFamily::Gamma => LogLikKind::Gamma { dispersion },
            GlmmFamily::Tweedie { power } => LogLikKind::Tweedie { power, dispersion },
        }
    }
}

/// Options for a mixed-effects fit.
#[derive(Debug, Clone)]
pub struct GlmmOptions {
    pub family: GlmmFamily,
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Use REML rather than ML for the Gaussian variance components.
    pub reml: bool,
    /// 1-based index into `x` of an offset column, added to the linear predictor
    /// with coefficient 1 and removed from the design.
    pub offset_column: Option<usize>,
}

impl Default for GlmmOptions {
    fn default() -> Self {
        Self {
            family: GlmmFamily::Gaussian,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            reml: true,
            offset_column: None,
        }
    }
}

/// One group's random effect.
#[derive(Debug, Clone, Copy)]
pub struct RandomEffect {
    /// Dense group id, matching the order groups were first seen.
    pub group: i32,
    /// Conditional mode of the random intercept (the BLUP).
    pub value: f64,
    /// Conditional standard deviation.
    pub se: f64,
    /// Number of observations in the group.
    pub n: usize,
}

/// Result of a mixed-effects fit.
#[derive(Debug, Clone)]
pub struct GlmmResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub z_values: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
    pub ci_lower: Option<Vec<f64>>,
    pub ci_upper: Option<Vec<f64>>,
    pub intercept_std_error: Option<f64>,
    pub confidence_level: f64,
    /// Between-group variance `sigma_b^2`.
    pub var_group: f64,
    /// Residual variance; 1.0 for families with a fixed dispersion.
    pub var_residual: f64,
    /// Intraclass correlation `var_group / (var_group + var_residual)`.
    pub icc: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub deviance: f64,
    pub n_observations: usize,
    pub n_groups: usize,
    pub n_features: usize,
    pub iterations: u32,
    pub converged: bool,
    pub ranef: Vec<RandomEffect>,
}

/// Fit a mixed-effects GLM with a random intercept.
///
/// `group_ids` must be dense group indices in `0..n_groups`; the caller (the C++
/// aggregate) dictionary-encodes whatever key type SQL supplied.
pub fn fit_glmm(
    y: &[f64],
    x: &[Vec<f64>],
    group_ids: &[i32],
    options: &GlmmOptions,
) -> StatsResult<GlmmResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if group_ids.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: y.len(),
            x_rows: group_ids.len(),
        });
    }
    for col in x.iter() {
        if col.len() != y.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: y.len(),
                x_rows: col.len(),
            });
        }
    }

    // Split off the offset column, if any.
    let offset_idx = match options.offset_column {
        Some(one_based) => {
            if one_based == 0 || one_based > x.len() {
                return Err(StatsError::InvalidValue {
                    field: "offset",
                    message: format!(
                        "offset must be a 1-based index into x (1..={}), got {one_based}",
                        x.len()
                    ),
                });
            }
            Some(one_based - 1)
        }
        None => None,
    };
    let feature_cols: Vec<usize> = (0..x.len()).filter(|i| Some(*i) != offset_idx).collect();

    // Keep rows that are finite throughout and carry a valid group.
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && group_ids[i] >= 0 && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n = rows.len();
    let n_features = feature_cols.len();
    let n_fixed = n_features + usize::from(options.fit_intercept);

    let n_groups = rows
        .iter()
        .map(|&i| group_ids[i] as usize + 1)
        .max()
        .unwrap_or(0);
    if n_groups < 2 {
        return Err(StatsError::InvalidValue {
            field: "group",
            message: "a mixed-effects model needs at least two groups; with one group the \
                      random intercept is not separable from the fixed intercept"
                .to_string(),
        });
    }
    if n <= n_fixed + 1 {
        return Err(StatsError::InsufficientData {
            rows: n,
            cols: n_fixed,
        });
    }

    let yv: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let groups: Vec<usize> = rows.iter().map(|&i| group_ids[i] as usize).collect();
    let offset: Option<Vec<f64>> = offset_idx.map(|oi| rows.iter().map(|&i| x[oi][i]).collect());

    let int_off = usize::from(options.fit_intercept);
    let xd = Mat::from_fn(n, n_fixed, |r, c| {
        if options.fit_intercept && c == 0 {
            1.0
        } else {
            x[feature_cols[c - int_off]][rows[r]]
        }
    });

    let mut group_sizes = vec![0usize; n_groups];
    for &g in &groups {
        group_sizes[g] += 1;
    }
    if group_sizes.iter().all(|&s| s <= 1) {
        return Err(StatsError::InvalidValue {
            field: "group",
            message: "every group has a single observation, so the between-group variance is \
                      not identified"
                .to_string(),
        });
    }

    // Outer loop: profile over `theta = sigma_b / sigma_e`, the *ratio* of the
    // two scales, not sigma_b itself.
    //
    // This matters. Henderson's mixed-model equations penalise the random block
    // by sigma_e^2 / sigma_b^2, so the penalty is scale-free; parameterising by
    // sigma_b alone makes the objective depend on the units of y, and doubling y
    // then fails to quadruple the fitted variance component. lme4 profiles over
    // the same ratio for exactly this reason. sigma_e^2 is recovered afterwards
    // from the fit, and sigma_b^2 = theta^2 * sigma_e^2.
    //
    // Because theta is dimensionless, a fixed bracket works for any response.
    let lo = 1e-4_f64.ln();
    let hi = 1e3_f64.ln();

    let objective = |log_theta: f64| -> f64 {
        match inner_fit(
            options,
            &xd,
            &yv,
            &groups,
            n_groups,
            offset.as_deref(),
            log_theta.exp(),
        ) {
            Ok(f) => f.marginal_loglik,
            Err(_) => f64::NEG_INFINITY,
        }
    };

    let (best_log_theta, iterations) = brent_maximize(objective, lo, hi, options.tolerance, 200);
    let theta = best_log_theta.exp();

    let fit = inner_fit(
        options,
        &xd,
        &yv,
        &groups,
        n_groups,
        offset.as_deref(),
        theta,
    )?;

    // Fixed-effect inference from the top-left block of the joint covariance.
    let (std_errors, z_values, p_values, ci_lower, ci_upper, intercept_std_error) =
        if options.compute_inference {
            let inf = crate::models::glm_engine::laplace::inference(
                &fit.beta,
                &fit.fixed_information,
                None,
                fit.dispersion,
                options.confidence_level,
                crate::types::VcovType::Laplace,
                &vec![false; n_fixed],
            )?;
            (
                Some(inf.std_errors[int_off..].to_vec()),
                Some(inf.z_values[int_off..].to_vec()),
                Some(inf.p_values[int_off..].to_vec()),
                Some(inf.ci_lower[int_off..].to_vec()),
                Some(inf.ci_upper[int_off..].to_vec()),
                if options.fit_intercept {
                    Some(inf.std_errors[0])
                } else {
                    None
                },
            )
        } else {
            (None, None, None, None, None, None)
        };

    let ranef: Vec<RandomEffect> = (0..n_groups)
        .map(|g| RandomEffect {
            group: g as i32,
            value: fit.b[g],
            se: fit.b_se[g],
            n: group_sizes[g],
        })
        .collect();

    let var_residual = fit.dispersion;
    let var_group = theta * theta * var_residual;
    let icc = if var_group + var_residual > 0.0 {
        var_group / (var_group + var_residual)
    } else {
        0.0
    };

    // Variance components count toward k: sigma_b always, the residual variance
    // only for families that estimate one.
    let k = n_fixed
        + 1
        + usize::from(!matches!(
            options.family,
            GlmmFamily::Poisson | GlmmFamily::Binomial
        ));

    let (coefficients, intercept) = if options.fit_intercept {
        (fit.beta[1..].to_vec(), Some(fit.beta[0]))
    } else {
        (fit.beta.clone(), None)
    };

    Ok(GlmmResult {
        coefficients,
        intercept,
        std_errors,
        z_values,
        p_values,
        ci_lower,
        ci_upper,
        intercept_std_error,
        confidence_level: options.confidence_level,
        var_group,
        var_residual,
        icc,
        log_likelihood: fit.marginal_loglik,
        aic: 2.0 * k as f64 - 2.0 * fit.marginal_loglik,
        bic: k as f64 * (n as f64).ln() - 2.0 * fit.marginal_loglik,
        deviance: fit.deviance,
        n_observations: n,
        n_groups,
        n_features,
        iterations,
        converged: true,
        ranef,
    })
}

/// The inner fit at a fixed `sigma_b`.
struct InnerFit {
    beta: Vec<f64>,
    b: Vec<f64>,
    b_se: Vec<f64>,
    /// Covariance-ready information matrix for the fixed effects, after
    /// eliminating the random effects.
    fixed_information: Mat<f64>,
    dispersion: f64,
    deviance: f64,
    /// Laplace-approximated log marginal likelihood at this `sigma_b`.
    marginal_loglik: f64,
}

/// Penalized IRLS over `[X | Z]` with `Z` left implicit.
///
/// The mixed-model equations at fixed weights `W` are
///
/// ```text
///   [ X'WX      X'WZ         ] [beta]   [X'Wz]
///   [ Z'WX      Z'WZ + I/s^2 ] [ b  ] = [Z'Wz]
/// ```
///
/// For a random intercept `Z` is a group indicator, so `Z'WZ` is diagonal with
/// entry `sum_{i in g} w_i`, and `X'WZ` has one column per group. That lets the
/// `b` block be eliminated by a rank-one-per-group update rather than by forming
/// or factorising anything of size `n_groups`.
#[allow(clippy::too_many_arguments)]
fn inner_fit(
    options: &GlmmOptions,
    xd: &Mat<f64>,
    y: &[f64],
    groups: &[usize],
    n_groups: usize,
    offset: Option<&[f64]>,
    theta: f64,
) -> StatsResult<InnerFit> {
    let n = xd.nrows();
    let p = xd.ncols();
    // The penalty on the random block is sigma_e^2 / sigma_b^2 = 1 / theta^2.
    // Working weights already carry the variance function, so this is the ratio
    // in every family, with sigma_e^2 == 1 where the dispersion is fixed.
    let lambda = 1.0 / (theta * theta);

    let family = options.family.upstream();
    let gaussian = options.family.is_gaussian();

    // Working response and weights. Gaussian needs no iteration: z = y, w = 1.
    let mut mu: Vec<f64> = match &family {
        Some(f) => f.initialize_mu(y),
        None => y.to_vec(),
    };
    let mut eta: Vec<f64> = match &family {
        Some(f) => mu
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let base = f.link(m);
                match offset {
                    Some(o) => base - o[i],
                    None => base,
                }
            })
            .collect(),
        None => mu.clone(),
    };

    let mut beta = vec![0.0; p];
    let mut b = vec![0.0; n_groups];
    let mut w = vec![1.0; n];
    let mut z = y.to_vec();
    let mut fixed_information = Mat::zeros(p, p);
    let mut zwz = vec![0.0; n_groups];

    let iters = if gaussian { 1 } else { options.max_iterations };

    for _ in 0..iters {
        if let Some(f) = &family {
            for i in 0..n {
                w[i] = f.irls_weight(mu[i]);
                let eta_no_offset = match offset {
                    Some(o) => eta[i] - o[i],
                    None => eta[i],
                };
                z[i] = eta_no_offset + (y[i] - mu[i]) * f.link_derivative(mu[i]);
            }
        }

        // Accumulate the blocks. Z is never built.
        let mut xwx: Mat<f64> = Mat::zeros(p, p);
        let mut xwz_t: Vec<Vec<f64>> = vec![vec![0.0; p]; n_groups]; // X'WZ, one column per group
        let mut xwy: Col<f64> = Col::zeros(p);
        let mut zwy = vec![0.0; n_groups];
        zwz.iter_mut().for_each(|v| *v = 0.0);

        for i in 0..n {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            let g = groups[i];
            zwz[g] += wi;
            zwy[g] += wi * z[i];
            for j in 0..p {
                let xij = xd[(i, j)];
                xwy[j] += wi * xij * z[i];
                xwz_t[g][j] += wi * xij;
                for k in j..p {
                    xwx[(j, k)] += wi * xij * xd[(i, k)];
                }
            }
        }
        for j in 0..p {
            for k in (j + 1)..p {
                xwx[(k, j)] = xwx[(j, k)];
            }
        }

        // Eliminate b: the Schur complement of the (diagonal) random block.
        //   A = X'WX - sum_g (X'WZ)_g (X'WZ)_g' / d_g
        //   c = X'Wz - sum_g (X'WZ)_g * (Z'Wz)_g / d_g          with d_g = zwz_g + lambda
        let mut a = xwx.clone();
        let mut c = xwy.clone();
        for g in 0..n_groups {
            let d = zwz[g] + lambda;
            if d <= 0.0 {
                continue;
            }
            let col = &xwz_t[g];
            let s = zwy[g] / d;
            for j in 0..p {
                c[j] -= col[j] * s;
                for k in 0..p {
                    a[(j, k)] -= col[j] * col[k] / d;
                }
            }
        }

        let beta_col = crate::models::glm_engine::normal_eq::solve_qr(&a, &c, 1e-12)?;
        for j in 0..p {
            beta[j] = beta_col[j];
        }

        // Back-substitute the random effects, one scalar per group.
        for g in 0..n_groups {
            let d = zwz[g] + lambda;
            if d <= 0.0 {
                b[g] = 0.0;
                continue;
            }
            let mut dot = 0.0;
            for j in 0..p {
                dot += xwz_t[g][j] * beta[j];
            }
            b[g] = (zwy[g] - dot) / d;
        }

        fixed_information = a.clone();

        // Update eta and mu.
        if let Some(f) = &family {
            for i in 0..n {
                let mut e = b[groups[i]];
                for j in 0..p {
                    e += xd[(i, j)] * beta[j];
                }
                if let Some(o) = offset {
                    e += o[i];
                }
                eta[i] = e;
                let m = f.link_inverse(e);
                mu[i] = if f.valid_mu(m) { m } else { f.clamp_mu(m) };
            }
        } else {
            for i in 0..n {
                let mut e = b[groups[i]];
                for j in 0..p {
                    e += xd[(i, j)] * beta[j];
                }
                if let Some(o) = offset {
                    e += o[i];
                }
                eta[i] = e;
                mu[i] = e;
            }
        }
    }

    // Residual variance / dispersion.
    let df_resid = (n.saturating_sub(p) as f64).max(1.0);
    let dispersion = match options.family {
        GlmmFamily::Poisson | GlmmFamily::Binomial => 1.0,
        GlmmFamily::Gaussian => {
            let rss: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
            let denom = if options.reml {
                (n as f64 - p as f64).max(1.0)
            } else {
                n as f64
            };
            rss / denom
        }
        _ => {
            let f = family.as_ref().expect("non-gaussian family");
            let chi2: f64 = (0..n)
                .map(|i| {
                    let v = f.variance(mu[i]);
                    if v > 0.0 {
                        (y[i] - mu[i]).powi(2) / v
                    } else {
                        0.0
                    }
                })
                .sum();
            chi2 / df_resid
        }
    };

    // Conditional standard deviations of the random effects. Ignoring the (small)
    // covariance with the fixed effects, the conditional precision of b_g is
    // d_g = sum_g w_i + lambda, scaled by the dispersion.
    let b_se: Vec<f64> = (0..n_groups)
        .map(|g| {
            let d = zwz[g] + lambda;
            if d > 0.0 {
                (dispersion / d).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();

    let deviance = match &family {
        Some(f) => f.deviance(y, &mu),
        None => (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum(),
    };

    // Laplace approximation to the log marginal likelihood:
    //   log L ~ log f(y | b_hat) + log N(b_hat; 0, sigma_b^2) - 0.5 log|H/(2 pi)|
    // with H the joint posterior precision of b. For a random intercept H is
    // diagonal, so the determinant is a plain sum of logs.
    let conditional_ll = match &family {
        Some(_) => {
            let kind = options.family.loglik_kind(dispersion);
            loglik::log_likelihood(kind, y, &mu)
        }
        None => {
            let s2 = dispersion.max(1e-300);
            -0.5 * (n as f64) * (2.0 * std::f64::consts::PI * s2).ln() - deviance / (2.0 * s2)
        }
    };

    // b ~ N(0, sigma_b^2) with sigma_b = theta * sigma_e.
    let phi = dispersion.max(1e-300);
    let sigma_b = theta * phi.sqrt();
    let prior_ll: f64 = -0.5 * (n_groups as f64) * (2.0 * std::f64::consts::PI).ln()
        - (n_groups as f64) * sigma_b.ln()
        - 0.5 * b.iter().map(|v| v * v).sum::<f64>() / (sigma_b * sigma_b);

    // Joint posterior precision of b_g is (Z'WZ)_g / phi + 1 / sigma_b^2, which is
    // (zwz_g + lambda) / phi. Diagonal, so the determinant is a sum of logs.
    let mut log_det = 0.0;
    for &z in zwz.iter().take(n_groups) {
        let d = (z + lambda) / phi;
        if d > 0.0 {
            log_det += d.ln();
        }
    }

    let marginal_loglik = conditional_ll + prior_ll - 0.5 * log_det;

    Ok(InnerFit {
        beta,
        b,
        b_se,
        fixed_information,
        dispersion,
        deviance,
        marginal_loglik,
    })
}

/// Golden-section / Brent-style maximization of a unimodal scalar function on
/// `[lo, hi]`. Returns the maximizer and the number of iterations used.
///
/// Deliberately derivative-free: the Laplace objective is only piecewise smooth
/// once the inner IRLS stopping rule is taken into account, and the search space
/// is one-dimensional, so robustness beats speed here.
fn brent_maximize<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    tolerance: f64,
    max_iter: u32,
) -> (f64, u32) {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;

    let mut c = hi - INV_PHI * (hi - lo);
    let mut d = lo + INV_PHI * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);
    let mut iterations = 0u32;

    for i in 0..max_iter {
        iterations = i + 1;
        if (hi - lo).abs() < tolerance.max(1e-10) {
            break;
        }
        if fc > fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - INV_PHI * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + INV_PHI * (hi - lo);
            fd = f(d);
        }
    }

    ((lo + hi) / 2.0, iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Balanced Gaussian panel: `y = 1 + 0.5x + b_g + e`, with the group effects
    /// and residuals laid out deterministically so the test is reproducible.
    fn gaussian_panel(
        n_groups: usize,
        per_group: usize,
        sigma_b: f64,
        sigma_e: f64,
    ) -> (Vec<f64>, Vec<Vec<f64>>, Vec<i32>) {
        let mut y = Vec::new();
        let mut xs = Vec::new();
        let mut g = Vec::new();
        for gi in 0..n_groups {
            // Spread group effects symmetrically about zero.
            let b = sigma_b * ((gi as f64 + 0.5) / n_groups as f64 * 2.0 - 1.0) * 1.732;
            for j in 0..per_group {
                let x = (j % 5) as f64;
                let e = sigma_e * (((gi * 7 + j * 3) % 11) as f64 / 5.0 - 1.0);
                y.push(1.0 + 0.5 * x + b + e);
                xs.push(x);
                g.push(gi as i32);
            }
        }
        (y, vec![xs], g)
    }

    #[test]
    fn gaussian_recovers_the_fixed_effects() {
        let (y, x, g) = gaussian_panel(20, 15, 0.8, 0.3);
        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();

        assert!(
            (fit.coefficients[0] - 0.5).abs() < 0.05,
            "slope {}",
            fit.coefficients[0]
        );
        assert!(
            (fit.intercept.unwrap() - 1.0).abs() < 0.2,
            "intercept {}",
            fit.intercept.unwrap()
        );
        assert_eq!(fit.n_groups, 20);
        assert_eq!(fit.n_observations, 300);
        assert_eq!(fit.ranef.len(), 20);
    }

    #[test]
    fn the_between_group_variance_is_recovered() {
        let (y, x, g) = gaussian_panel(30, 20, 1.0, 0.25);
        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();
        // Group effects were spread over +-1.732*sigma_b uniformly, whose variance
        // is sigma_b^2. Recovery need only be in the right ballpark.
        assert!(
            fit.var_group > 0.3 && fit.var_group < 3.0,
            "var_group {}",
            fit.var_group
        );
        assert!(fit.icc > 0.5, "icc {} should be high here", fit.icc);
    }

    #[test]
    fn blups_track_the_true_group_effects() {
        let n_groups = 24;
        let (y, x, g) = gaussian_panel(n_groups, 20, 1.0, 0.2);
        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();

        // The true effects are monotone in the group index by construction, so the
        // BLUPs must be too.
        let mut increasing = 0;
        for w in fit.ranef.windows(2) {
            if w[1].value > w[0].value {
                increasing += 1;
            }
        }
        assert!(
            increasing >= n_groups - 3,
            "BLUPs should follow the group effects: {increasing} of {} increasing",
            n_groups - 1
        );
        // And they are centred.
        let mean: f64 = fit.ranef.iter().map(|r| r.value).sum::<f64>() / n_groups as f64;
        assert!(mean.abs() < 0.3, "BLUP mean {mean}");
    }

    #[test]
    fn blups_shrink_toward_zero_relative_to_group_means() {
        // The defining property of partial pooling: a BLUP is smaller in magnitude
        // than the raw group deviation it is estimating.
        let (y, x, g) = gaussian_panel(15, 6, 1.0, 1.5);
        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();

        let grand: f64 = y.iter().sum::<f64>() / y.len() as f64;
        for r in &fit.ranef {
            let members: Vec<f64> = (0..y.len())
                .filter(|&i| g[i] == r.group)
                .map(|i| y[i])
                .collect();
            let raw = members.iter().sum::<f64>() / members.len() as f64 - grand;
            assert!(
                r.value.abs() <= raw.abs() + 1e-9,
                "group {}: BLUP {} should not exceed raw deviation {raw}",
                r.group,
                r.value
            );
        }
    }

    #[test]
    fn a_tiny_between_group_variance_approaches_the_pooled_fit() {
        // With no real group structure the mixed fit should land on the pooled one.
        let (y, x, g) = gaussian_panel(12, 20, 0.0, 0.5);
        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();
        assert!(fit.var_group < 0.05, "var_group {}", fit.var_group);
        for r in &fit.ranef {
            assert!(r.value.abs() < 0.2, "BLUP {} should be near zero", r.value);
        }
    }

    #[test]
    fn poisson_mixed_model_fits() {
        let n_groups = 15;
        let mut y = Vec::new();
        let mut xs = Vec::new();
        let mut g = Vec::new();
        for gi in 0..n_groups {
            let b = 0.6 * ((gi as f64 + 0.5) / n_groups as f64 * 2.0 - 1.0);
            for j in 0..20 {
                let x = (j % 4) as f64;
                y.push((0.5 + 0.3 * x + b).exp().round());
                xs.push(x);
                g.push(gi);
            }
        }
        let opts = GlmmOptions {
            family: GlmmFamily::Poisson,
            compute_inference: true,
            ..Default::default()
        };
        let fit = fit_glmm(&y, &[xs], &g, &opts).unwrap();

        assert!(
            (fit.coefficients[0] - 0.3).abs() < 0.1,
            "slope {}",
            fit.coefficients[0]
        );
        assert!(fit.var_group > 0.0);
        assert!(fit.std_errors.unwrap()[0] > 0.0);
    }

    #[test]
    fn inference_is_populated() {
        let (y, x, g) = gaussian_panel(20, 15, 0.8, 0.3);
        let opts = GlmmOptions {
            compute_inference: true,
            ..Default::default()
        };
        let fit = fit_glmm(&y, &x, &g, &opts).unwrap();

        let se = fit.std_errors.as_ref().unwrap();
        assert!(se[0].is_finite() && se[0] > 0.0);
        assert!(fit.ci_lower.as_ref().unwrap()[0] < fit.coefficients[0]);
        assert!(fit.ci_upper.as_ref().unwrap()[0] > fit.coefficients[0]);
        assert!(fit.intercept_std_error.unwrap() > 0.0);
        for r in &fit.ranef {
            assert!(r.se.is_finite() && r.se > 0.0, "ranef se {}", r.se);
        }
    }

    #[test]
    fn a_single_group_is_rejected() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let g = vec![0i32; 6];
        assert!(matches!(
            fit_glmm(&y, &x, &g, &GlmmOptions::default()),
            Err(StatsError::InvalidValue { field: "group", .. })
        ));
    }

    #[test]
    fn all_singleton_groups_are_rejected() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let g: Vec<i32> = (0..6).collect();
        assert!(matches!(
            fit_glmm(&y, &x, &g, &GlmmOptions::default()),
            Err(StatsError::InvalidValue { field: "group", .. })
        ));
    }

    #[test]
    fn non_finite_rows_are_dropped() {
        let (mut y, mut x, mut g) = gaussian_panel(10, 10, 0.5, 0.2);
        let clean = y.len();
        y.push(f64::NAN);
        x[0].push(1.0);
        g.push(0);

        let fit = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();
        assert_eq!(fit.n_observations, clean);
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![vec![1.0, 2.0, 3.0]];
        let g = vec![0i32, 1];
        assert!(matches!(
            fit_glmm(&y, &x, &g, &GlmmOptions::default()),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn family_names_parse() {
        assert_eq!(
            GlmmFamily::from_name("gaussian"),
            Some(GlmmFamily::Gaussian)
        );
        assert_eq!(GlmmFamily::from_name("Poisson"), Some(GlmmFamily::Poisson));
        assert!(GlmmFamily::from_name("negbinomial").is_some());
        assert_eq!(GlmmFamily::from_name("nope"), None);
    }

    #[test]
    fn scaling_the_response_scales_the_variance_component() {
        // A pure sanity check on the outer search: doubling y should roughly
        // quadruple the between-group variance.
        let (y, x, g) = gaussian_panel(20, 12, 1.0, 0.3);
        let base = fit_glmm(&y, &x, &g, &GlmmOptions::default()).unwrap();

        let y2: Vec<f64> = y.iter().map(|v| v * 2.0).collect();
        let scaled = fit_glmm(&y2, &x, &g, &GlmmOptions::default()).unwrap();

        let ratio = scaled.var_group / base.var_group;
        assert!(
            ratio > 3.0 && ratio < 5.0,
            "var_group ratio {ratio} should be near 4"
        );
    }
}
