//! Per-family log-likelihoods.
//!
//! The upstream solvers substitute `log L = -deviance / 2`, which is correct up to
//! an additive constant only for the Gaussian family. For Poisson, Binomial,
//! Negative Binomial, Gamma and Tweedie it is wrong, and the `aic` / `bic` fields
//! derived from it are not comparable with R's. This module computes the actual
//! log-likelihood so those fields mean what users expect.

use statrs::function::gamma::ln_gamma;

/// Which likelihood to evaluate, with any nuisance parameters attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLikKind {
    /// Poisson with mean `mu`.
    Poisson,
    /// Bernoulli / binomial with a single trial per row.
    Binomial,
    /// Negative binomial with dispersion `theta` (the `size` parameter; variance
    /// is `mu + mu^2 / theta`).
    NegativeBinomial { theta: f64 },
    /// Gamma with dispersion `phi` (shape `1 / phi`).
    Gamma { dispersion: f64 },
    /// Tweedie compound Poisson-Gamma with `1 < p < 2` and dispersion `phi`.
    Tweedie { power: f64, dispersion: f64 },
}

impl LogLikKind {
    /// Number of nuisance parameters estimated alongside the coefficients. These
    /// count toward `k` in AIC/BIC, matching R.
    pub fn n_nuisance(&self) -> usize {
        match self {
            LogLikKind::Poisson | LogLikKind::Binomial => 0,
            LogLikKind::NegativeBinomial { .. }
            | LogLikKind::Gamma { .. }
            | LogLikKind::Tweedie { .. } => 1,
        }
    }
}

/// Total log-likelihood over all observations.
///
/// Returns `NaN` if any observation produces a non-finite contribution, so a bad
/// value surfaces rather than silently poisoning the sum.
pub fn log_likelihood(kind: LogLikKind, y: &[f64], mu: &[f64]) -> f64 {
    let mut total = 0.0;
    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let li = unit_log_likelihood(kind, yi, mui);
        if !li.is_finite() {
            return f64::NAN;
        }
        total += li;
    }
    total
}

/// Log-likelihood contribution of a single observation.
pub fn unit_log_likelihood(kind: LogLikKind, y: f64, mu: f64) -> f64 {
    match kind {
        LogLikKind::Poisson => {
            let mu = mu.max(1e-300);
            y * mu.ln() - mu - ln_gamma(y + 1.0)
        }
        LogLikKind::Binomial => {
            let eps = 1e-15;
            let mu = mu.clamp(eps, 1.0 - eps);
            y * mu.ln() + (1.0 - y) * (1.0 - mu).ln()
        }
        LogLikKind::NegativeBinomial { theta } => {
            let mu = mu.max(1e-300);
            let theta = theta.max(1e-300);
            ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
                + theta * (theta / (theta + mu)).ln()
                + y * (mu / (theta + mu)).ln()
        }
        LogLikKind::Gamma { dispersion } => {
            // shape a = 1/phi, rate = a/mu
            let phi = dispersion.max(1e-300);
            let a = 1.0 / phi;
            let mu = mu.max(1e-300);
            let y = y.max(1e-300);
            a * (a / mu).ln() + (a - 1.0) * y.ln() - a * y / mu - ln_gamma(a)
        }
        LogLikKind::Tweedie { power, dispersion } => {
            tweedie_log_density(y, mu, power, dispersion)
        }
    }
}

/// Log-density of the compound Poisson-Gamma (Tweedie, `1 < p < 2`) distribution.
///
/// Evaluated with the Dunn & Smyth series: the density is a Poisson mixture of
/// Gamma densities, `f(y) = sum_j W_j * exp(...)`. The series is summed in log
/// space around its largest term so it stays stable for small `phi`.
///
/// Reference: Dunn, P.K. and Smyth, G.K. (2005), "Series evaluation of Tweedie
/// exponential dispersion model densities", Statistics and Computing 15, 267-280.
fn tweedie_log_density(y: f64, mu: f64, p: f64, phi: f64) -> f64 {
    if !(1.0..2.0).contains(&p) {
        // p == 1 is Poisson, p == 2 is Gamma; both handled by their own variants.
        if (p - 1.0).abs() < 1e-12 {
            return unit_log_likelihood(LogLikKind::Poisson, y, mu);
        }
        if (p - 2.0).abs() < 1e-12 {
            return unit_log_likelihood(LogLikKind::Gamma { dispersion: phi }, y, mu);
        }
        return f64::NAN;
    }

    let phi = phi.max(1e-300);
    let mu = mu.max(1e-300);

    // The exponential-dispersion kernel, shared by the point mass at zero and the
    // continuous part: exp((y*theta - kappa(theta)) / phi).
    let theta = mu.powf(1.0 - p) / (1.0 - p);
    let kappa = mu.powf(2.0 - p) / (2.0 - p);
    let kernel = (y * theta - kappa) / phi;

    if y == 0.0 {
        return kernel;
    }
    if y < 0.0 {
        return f64::NAN;
    }

    // Series: log W(y, phi, p) = log sum_j exp(w_j), with
    // w_j = j*alpha_term - ln_gamma(j+1) - ln_gamma(-j*alpha) ...
    let alpha = (2.0 - p) / (1.0 - p); // negative for 1 < p < 2
    let log_z = -alpha * (y / (p - 1.0)).ln() - (1.0 - alpha) * phi.ln() - (2.0 - p).ln();

    // Index of the largest term (Dunn & Smyth eq. 4).
    let j_max = (y.powf(2.0 - p) / (phi * (2.0 - p))).max(1.0);
    let j_center = j_max.round().max(1.0) as usize;

    let term = |j: usize| -> f64 {
        let jf = j as f64;
        jf * log_z - ln_gamma(jf + 1.0) - ln_gamma(-alpha * jf)
    };

    // Walk outward from the peak until terms are negligible.
    let peak = term(j_center);
    let mut max_w = peak;
    let mut indices: Vec<usize> = vec![j_center];

    let mut j = j_center + 1;
    loop {
        let w = term(j);
        if w > max_w {
            max_w = w;
        }
        indices.push(j);
        if w < max_w - 40.0 || j > j_center + 100_000 {
            break;
        }
        j += 1;
    }
    if j_center > 1 {
        let mut j = j_center - 1;
        loop {
            let w = term(j);
            if w > max_w {
                max_w = w;
            }
            indices.push(j);
            if w < max_w - 40.0 || j == 1 {
                break;
            }
            j -= 1;
        }
    }

    let sum: f64 = indices.iter().map(|&j| (term(j) - max_w).exp()).sum();
    let log_w = max_w + sum.ln();

    log_w - y.ln() + kernel
}

/// AIC from a log-likelihood and a parameter count.
pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    2.0 * n_params as f64 - 2.0 * log_likelihood
}

/// BIC from a log-likelihood, a parameter count and a sample size.
pub fn bic(log_likelihood: f64, n_params: usize, n_obs: usize) -> f64 {
    n_params as f64 * (n_obs as f64).ln() - 2.0 * log_likelihood
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_matches_dpois() {
        // R: dpois(3, lambda = 2.5, log = TRUE) = -1.542887273605590
        let ll = unit_log_likelihood(LogLikKind::Poisson, 3.0, 2.5);
        assert!((ll - (-1.542_887_273_605_590)).abs() < 1e-12, "got {ll}");
    }

    #[test]
    fn binomial_matches_dbinom() {
        // R: dbinom(1, 1, 0.3, log = TRUE) = log(0.3) = -1.203973
        let ll = unit_log_likelihood(LogLikKind::Binomial, 1.0, 0.3);
        assert!((ll - (0.3f64).ln()).abs() < 1e-12, "got {ll}");
        // R: dbinom(0, 1, 0.3, log = TRUE) = log(0.7)
        let ll0 = unit_log_likelihood(LogLikKind::Binomial, 0.0, 0.3);
        assert!((ll0 - (0.7f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn negbinomial_matches_dnbinom() {
        // R: dnbinom(4, size = 2, mu = 3, log = TRUE) = -2.266446046378171
        let ll = unit_log_likelihood(LogLikKind::NegativeBinomial { theta: 2.0 }, 4.0, 3.0);
        assert!((ll - (-2.266_446_046_378_171)).abs() < 1e-12, "got {ll}");
    }

    #[test]
    fn gamma_matches_dgamma() {
        // Gamma with dispersion phi = 0.5 => shape 2, mean 3 => rate = 2/3.
        // R: dgamma(1.5, shape = 2, rate = 2/3, log = TRUE) = -1.302585
        let ll = unit_log_likelihood(LogLikKind::Gamma { dispersion: 0.5 }, 1.5, 3.0);
        let expect = {
            let a: f64 = 2.0;
            let rate: f64 = 2.0 / 3.0;
            a * rate.ln() + (a - 1.0) * 1.5f64.ln() - rate * 1.5 - ln_gamma(a)
        };
        assert!((ll - expect).abs() < 1e-10, "got {ll}, want {expect}");
    }

    #[test]
    fn tweedie_reduces_to_the_kernel_at_zero() {
        let p = 1.5;
        let phi = 1.0;
        let mu = 2.0;
        let ll = unit_log_likelihood(LogLikKind::Tweedie { power: p, dispersion: phi }, 0.0, mu);
        let expect = -mu.powf(2.0 - p) / (2.0 - p);
        assert!((ll - expect).abs() < 1e-12, "got {ll}, want {expect}");
    }

    #[test]
    fn tweedie_density_integrates_to_about_one() {
        // Crude trapezoid over the continuous part plus the atom at zero.
        let (p, phi, mu) = (1.5, 0.8, 2.0);
        let atom = unit_log_likelihood(
            LogLikKind::Tweedie { power: p, dispersion: phi },
            0.0,
            mu,
        )
        .exp();

        let n = 40_000;
        let hi = 30.0;
        let h = hi / n as f64;
        let mut integral = 0.0;
        for i in 0..=n {
            let y = i as f64 * h;
            if y == 0.0 {
                continue;
            }
            let d = unit_log_likelihood(
                LogLikKind::Tweedie { power: p, dispersion: phi },
                y,
                mu,
            )
            .exp();
            let w = if i == n { 0.5 } else { 1.0 };
            integral += w * d * h;
        }

        let total = atom + integral;
        assert!(
            (total - 1.0).abs() < 5e-3,
            "tweedie density mass = {total} (atom {atom}, cont {integral})"
        );
    }

    #[test]
    fn aic_and_bic_use_the_real_log_likelihood() {
        let y = [1.0, 2.0, 3.0];
        let mu = [1.2, 2.1, 2.8];
        let ll = log_likelihood(LogLikKind::Poisson, &y, &mu);
        assert!(ll.is_finite());
        assert!((aic(ll, 2) - (4.0 - 2.0 * ll)).abs() < 1e-12);
        assert!((bic(ll, 2, 3) - (2.0 * 3f64.ln() - 2.0 * ll)).abs() < 1e-12);
    }

    #[test]
    fn non_finite_contribution_poisons_the_total() {
        let y = [1.0, -1.0];
        let mu = [1.0, 1.0];
        // lgamma(0) is +inf, so y = -1 gives -inf.
        assert!(log_likelihood(LogLikKind::Poisson, &y, &mu).is_nan());
    }
}
