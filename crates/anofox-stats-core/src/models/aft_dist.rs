//! Error distributions for accelerated failure time (AFT) models.
//!
//! An AFT model writes `log T = x'beta + sigma * W`, where `W` follows a fixed
//! standard distribution. Everything a censored likelihood needs is therefore the
//! log-density and log-survival of `W`, plus their first two derivatives.
//!
//! Nothing comparable exists in the ALM code this crate already wraps: those
//! likelihoods are uncensored and none of them expose a survival function, which
//! is exactly the term a right-censored observation contributes.

use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;

/// The error distribution of an AFT model, named after the distribution induced
/// on `T` rather than on `W`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AftDistribution {
    /// `T` is Weibull; `W` is standard extreme value (Gumbel minimum).
    Weibull,
    /// `T` is lognormal; `W` is standard normal.
    LogNormal,
    /// `T` is log-logistic; `W` is standard logistic.
    LogLogistic,
    /// `T` is exponential — Weibull with `sigma` fixed at 1.
    Exponential,
}

impl AftDistribution {
    /// Parse a distribution name as it appears in the options MAP.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "weibull" => Some(AftDistribution::Weibull),
            "lognormal" | "log_normal" | "log-normal" => Some(AftDistribution::LogNormal),
            "loglogistic" | "log_logistic" | "log-logistic" => Some(AftDistribution::LogLogistic),
            "exponential" | "exp" => Some(AftDistribution::Exponential),
            _ => None,
        }
    }

    /// True when `sigma` is fixed rather than estimated.
    pub fn scale_is_fixed(&self) -> bool {
        matches!(self, AftDistribution::Exponential)
    }

    /// The underlying standard distribution of `W`.
    fn kernel(&self) -> Kernel {
        match self {
            AftDistribution::Weibull | AftDistribution::Exponential => Kernel::ExtremeValue,
            AftDistribution::LogNormal => Kernel::Normal,
            AftDistribution::LogLogistic => Kernel::Logistic,
        }
    }

    /// `log f_W(z)`.
    pub fn log_density(&self, z: f64) -> f64 {
        self.kernel().log_density(z)
    }

    /// `log S_W(z)`, the log survival.
    pub fn log_survival(&self, z: f64) -> f64 {
        self.kernel().log_survival(z)
    }

    /// `d/dz log f_W(z)`.
    pub fn d_log_density(&self, z: f64) -> f64 {
        self.kernel().d_log_density(z)
    }

    /// `d2/dz2 log f_W(z)`.
    pub fn dd_log_density(&self, z: f64) -> f64 {
        self.kernel().dd_log_density(z)
    }

    /// `d/dz log S_W(z)` — the negative hazard.
    pub fn d_log_survival(&self, z: f64) -> f64 {
        self.kernel().d_log_survival(z)
    }

    /// `d2/dz2 log S_W(z)`.
    pub fn dd_log_survival(&self, z: f64) -> f64 {
        self.kernel().dd_log_survival(z)
    }

    /// `P(W <= z)`.
    pub fn cdf(&self, z: f64) -> f64 {
        self.kernel().cdf(z)
    }

    /// The `p`-quantile of `W`.
    pub fn quantile(&self, p: f64) -> f64 {
        self.kernel().quantile(p)
    }

    /// `P(T <= t)` for a fit with linear predictor `eta` and scale `sigma`.
    pub fn cdf_time(&self, t: f64, eta: f64, sigma: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        self.cdf((t.ln() - eta) / sigma)
    }

    /// `P(T > t)`.
    pub fn survival_time(&self, t: f64, eta: f64, sigma: f64) -> f64 {
        1.0 - self.cdf_time(t, eta, sigma)
    }

    /// The `p`-quantile of `T`.
    pub fn quantile_time(&self, p: f64, eta: f64, sigma: f64) -> f64 {
        (eta + sigma * self.quantile(p)).exp()
    }
}

#[derive(Debug, Clone, Copy)]
enum Kernel {
    /// Standard extreme value (minimum): `f(z) = exp(z - e^z)`.
    ExtremeValue,
    Normal,
    Logistic,
}

/// Clamp exponent arguments so a far-tail observation saturates instead of
/// producing an infinity that poisons the whole Hessian.
fn safe_exp(z: f64) -> f64 {
    z.clamp(-700.0, 700.0).exp()
}

impl Kernel {
    fn log_density(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => z - safe_exp(z),
            Kernel::Normal => -0.5 * z * z - 0.5 * (2.0 * PI).ln(),
            Kernel::Logistic => {
                // z - 2*log(1 + e^z), written to stay stable for large |z|.
                let s = log1p_exp(z);
                z - 2.0 * s
            }
        }
    }

    fn log_survival(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => -safe_exp(z),
            Kernel::Normal => {
                let s = normal_sf(z);
                if s <= 0.0 {
                    // Far right tail: log(1 - Phi(z)) ~ -z^2/2 - log(z sqrt(2 pi)).
                    -0.5 * z * z - (z * (2.0 * PI).sqrt()).ln()
                } else {
                    s.ln()
                }
            }
            Kernel::Logistic => -log1p_exp(z),
        }
    }

    fn d_log_density(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => 1.0 - safe_exp(z),
            Kernel::Normal => -z,
            Kernel::Logistic => 1.0 - 2.0 * logistic(z),
        }
    }

    fn dd_log_density(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => -safe_exp(z),
            Kernel::Normal => -1.0,
            Kernel::Logistic => {
                let p = logistic(z);
                -2.0 * p * (1.0 - p)
            }
        }
    }

    fn d_log_survival(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => -safe_exp(z),
            Kernel::Normal => -normal_hazard(z),
            Kernel::Logistic => -logistic(z),
        }
    }

    fn dd_log_survival(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => -safe_exp(z),
            Kernel::Normal => {
                // d/dz of -r(z) where r = phi/S; dr/dz = -z r + r^2.
                let r = normal_hazard(z);
                z * r - r * r
            }
            Kernel::Logistic => {
                let p = logistic(z);
                -p * (1.0 - p)
            }
        }
    }

    fn cdf(&self, z: f64) -> f64 {
        match self {
            Kernel::ExtremeValue => 1.0 - (-safe_exp(z)).exp(),
            Kernel::Normal => normal_cdf(z),
            Kernel::Logistic => logistic(z),
        }
    }

    fn quantile(&self, p: f64) -> f64 {
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        match self {
            Kernel::ExtremeValue => (-(1.0 - p).ln()).ln(),
            Kernel::Normal => match Normal::new(0.0, 1.0) {
                Ok(d) => d.inverse_cdf(p),
                Err(_) => f64::NAN,
            },
            Kernel::Logistic => (p / (1.0 - p)).ln(),
        }
    }
}

#[inline]
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + safe_exp(-z))
    } else {
        let e = safe_exp(z);
        e / (1.0 + e)
    }
}

/// `log(1 + e^z)`, stable for large `|z|`.
#[inline]
fn log1p_exp(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

fn normal_cdf(z: f64) -> f64 {
    match Normal::new(0.0, 1.0) {
        Ok(d) => d.cdf(z),
        Err(_) => f64::NAN,
    }
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

fn normal_sf(z: f64) -> f64 {
    1.0 - normal_cdf(z)
}

/// `phi(z) / (1 - Phi(z))`, with an asymptotic expansion in the far right tail
/// where the naive ratio is 0/0 in floating point.
fn normal_hazard(z: f64) -> f64 {
    let s = normal_sf(z);
    if s > 1e-300 {
        normal_pdf(z) / s
    } else {
        // Mills ratio asymptotics: phi/S ~ z + 1/z - 2/z^3.
        z + 1.0 / z - 2.0 / (z * z * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference check that the analytic derivatives are consistent
    /// with the functions they claim to differentiate.
    fn check_derivatives(dist: AftDistribution, z: f64) {
        let h = 1e-5;

        let num_d = (dist.log_density(z + h) - dist.log_density(z - h)) / (2.0 * h);
        assert!(
            (dist.d_log_density(z) - num_d).abs() < 1e-6,
            "{dist:?} d_log_density at {z}: analytic {} vs numeric {num_d}",
            dist.d_log_density(z)
        );

        let num_dd = (dist.log_density(z + h) - 2.0 * dist.log_density(z)
            + dist.log_density(z - h))
            / (h * h);
        assert!(
            (dist.dd_log_density(z) - num_dd).abs() < 1e-4,
            "{dist:?} dd_log_density at {z}: analytic {} vs numeric {num_dd}",
            dist.dd_log_density(z)
        );

        let num_s = (dist.log_survival(z + h) - dist.log_survival(z - h)) / (2.0 * h);
        assert!(
            (dist.d_log_survival(z) - num_s).abs() < 1e-6,
            "{dist:?} d_log_survival at {z}: analytic {} vs numeric {num_s}",
            dist.d_log_survival(z)
        );

        let num_ss = (dist.log_survival(z + h) - 2.0 * dist.log_survival(z)
            + dist.log_survival(z - h))
            / (h * h);
        assert!(
            (dist.dd_log_survival(z) - num_ss).abs() < 1e-4,
            "{dist:?} dd_log_survival at {z}: analytic {} vs numeric {num_ss}",
            dist.dd_log_survival(z)
        );
    }

    #[test]
    fn analytic_derivatives_match_finite_differences() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            for z in [-2.0, -0.5, 0.0, 0.5, 1.5] {
                check_derivatives(dist, z);
            }
        }
    }

    #[test]
    fn survival_and_cdf_are_complementary() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            for z in [-1.0, 0.0, 1.0] {
                let s = dist.log_survival(z).exp();
                assert!(
                    ((1.0 - dist.cdf(z)) - s).abs() < 1e-10,
                    "{dist:?} at {z}: 1-cdf {} vs exp(log S) {s}",
                    1.0 - dist.cdf(z)
                );
            }
        }
    }

    #[test]
    fn quantile_inverts_the_cdf() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
                let z = dist.quantile(p);
                assert!(
                    (dist.cdf(z) - p).abs() < 1e-9,
                    "{dist:?} quantile({p}) = {z}, cdf = {}",
                    dist.cdf(z)
                );
            }
        }
    }

    #[test]
    fn weibull_kernel_matches_the_closed_form() {
        // f(z) = exp(z - e^z), S(z) = exp(-e^z)
        let z = 0.3;
        assert!((AftDistribution::Weibull.log_density(z) - (z - z.exp())).abs() < 1e-14);
        assert!((AftDistribution::Weibull.log_survival(z) - (-z.exp())).abs() < 1e-14);
    }

    #[test]
    fn lognormal_survival_stays_finite_in_the_far_tail() {
        // 1 - Phi(z) underflows past about z = 38; the asymptotic branch must
        // keep log S finite so a far-tail censored row does not poison the fit.
        let ls = AftDistribution::LogNormal.log_survival(50.0);
        assert!(ls.is_finite(), "log S(50) = {ls}");
        assert!(ls < -1000.0);
        let h = AftDistribution::LogNormal.d_log_survival(50.0);
        assert!(h.is_finite() && h < 0.0, "hazard at 50 = {h}");
    }

    #[test]
    fn quantile_time_is_monotone_in_p() {
        let d = AftDistribution::Weibull;
        let (eta, sigma) = (1.0, 0.7);
        let q25 = d.quantile_time(0.25, eta, sigma);
        let q50 = d.quantile_time(0.50, eta, sigma);
        let q75 = d.quantile_time(0.75, eta, sigma);
        assert!(q25 < q50 && q50 < q75);
        // And cdf_time inverts it.
        assert!((d.cdf_time(q50, eta, sigma) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn distribution_names_parse() {
        assert_eq!(
            AftDistribution::from_name("Weibull"),
            Some(AftDistribution::Weibull)
        );
        assert_eq!(
            AftDistribution::from_name("log_normal"),
            Some(AftDistribution::LogNormal)
        );
        assert_eq!(
            AftDistribution::from_name("loglogistic"),
            Some(AftDistribution::LogLogistic)
        );
        assert_eq!(AftDistribution::from_name("cauchy"), None);
        assert!(AftDistribution::Exponential.scale_is_fixed());
        assert!(!AftDistribution::Weibull.scale_is_fixed());
    }
}
