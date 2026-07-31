//! Family-generic penalized GLM engine.
//!
//! The upstream `anofox-regression` crate carries six hand-copied IRLS loops, one
//! per family, none of which supports explicit priors, a corrected penalized
//! covariance, or a real per-family log-likelihood. This module implements the loop
//! once against upstream's [`GlmFamily`] trait and adds those three things.
//!
//! It is also the foundation for the rest of issue #107. A Gaussian prior on a
//! coefficient and a Gaussian random effect are the same object — a quadratic
//! precision block in the normal equations — so the mixed-effects work reuses
//! [`irls`] and [`laplace`] rather than forking them, and the survival work reuses
//! [`laplace`] alone.
//!
//! ```text
//!   design::build ──▶ Design ──▶ build_penalty ──▶ Penalty
//!                                     │
//!                                     ▼
//!                          irls::fit_irls (generic over GlmFamily)
//!                                     │
//!                       information at the mode (X'WX + P)
//!                                     │
//!                                     ▼
//!                          laplace::inference ──▶ SE / z / p / CI
//! ```

pub mod design;
pub mod irls;
pub mod laplace;
pub mod loglik;
pub mod normal_eq;
pub mod penalty;

#[cfg(test)]
mod parity;

use crate::errors::StatsResult;
use crate::types::{
    GlmFitResult, GlmInferenceResult, PriorSpec, VcovType,
};
use anofox_regression::core::GlmFamily;

pub use design::{ConstantColumnPolicy, Design, DesignSpec};
pub use irls::{IrlsConfig, IrlsFit};
pub use laplace::LaplaceInference;
pub use loglik::LogLikKind;
pub use penalty::{DiagonalPenalty, Penalty, QuadraticPenalty};

/// Everything the engine needs for one fit, independent of family.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Legacy uniform ridge (`glm_lambda`), composed additively with `priors`.
    pub lambda: f64,
    /// Per-feature priors in the caller's feature order, optionally prefixed with
    /// an entry for the intercept. Empty means no explicit priors.
    pub priors: Vec<PriorSpec>,
    pub vcov: VcovType,
    /// 1-based index into `x` of an offset column, if any.
    pub offset_column: Option<usize>,
    pub constant_policy: ConstantColumnPolicy,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: Vec::new(),
            vcov: VcovType::default(),
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        }
    }
}

/// How the dispersion parameter is obtained for a family.
///
/// Note that the value a family *reports* as its dispersion and the factor that
/// scales its coefficient covariance are not always the same number. For Negative
/// Binomial the reported "dispersion" is `theta`, a shape parameter that already
/// enters the IRLS weights through the variance function `mu + mu^2 / theta`;
/// multiplying the covariance by it again would inflate every standard error by
/// `sqrt(theta)`. [`DispersionRule::covariance_scale`] keeps the two separate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispersionRule {
    /// Fixed at 1.0 (Poisson, Binomial).
    Fixed,
    /// Pearson chi-squared over residual degrees of freedom, floored at 1.0.
    /// This is what the upstream Poisson solver does for quasi-Poisson behaviour.
    PearsonFlooredAtOne,
    /// Pearson chi-squared over residual degrees of freedom, unfloored
    /// (Gamma, Tweedie).
    Pearson,
    /// A shape parameter supplied by the caller (Negative Binomial `theta`). It is
    /// reported as the dispersion but does **not** scale the covariance.
    Given(f64),
}

impl DispersionRule {
    /// The multiplier applied to `(X'WX + P)^-1` when forming the covariance.
    fn covariance_scale(&self, estimated: f64) -> f64 {
        match self {
            DispersionRule::Fixed | DispersionRule::Given(_) => 1.0,
            DispersionRule::Pearson | DispersionRule::PearsonFlooredAtOne => estimated,
        }
    }
}

/// A completed engine fit, ready to be mapped onto the crate's result types.
#[derive(Debug, Clone)]
pub struct EngineFit {
    pub design: Design,
    pub irls: IrlsFit,
    pub dispersion: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub inference: Option<LaplaceInference>,
}

impl EngineFit {
    /// Map onto the crate-wide [`GlmFitResult`] shape.
    pub fn to_glm_fit_result(&self) -> GlmFitResult {
        let (coefficients, intercept) = self.design.expand(&self.irls.beta);
        let pseudo_r_squared = if self.irls.null_deviance > 0.0 {
            1.0 - self.irls.deviance / self.irls.null_deviance
        } else {
            0.0
        };

        GlmFitResult {
            coefficients,
            intercept,
            null_deviance: self.irls.null_deviance,
            residual_deviance: self.irls.deviance,
            pseudo_r_squared,
            aic: self.aic,
            n_observations: self.design.n_observations(),
            n_features: self.design.n_features_original,
            iterations: self.irls.iterations,
            converged: self.irls.converged,
            dispersion: Some(self.dispersion),
        }
    }

    /// Map the inference onto [`GlmInferenceResult`], expanding dropped columns
    /// back to `NaN`. Intercept entries are stripped, matching the existing shape.
    pub fn to_glm_inference(&self) -> Option<GlmInferenceResult> {
        let inf = self.inference.as_ref()?;
        let expand = |v: &[f64]| self.design.expand(v).0;

        Some(GlmInferenceResult {
            std_errors: expand(&inf.std_errors),
            z_values: expand(&inf.z_values),
            p_values: expand(&inf.p_values),
            ci_lower: expand(&inf.ci_lower),
            ci_upper: expand(&inf.ci_upper),
            confidence_level: inf.confidence_level,
        })
    }
}

/// Fit a GLM with the given family, priors and covariance policy.
///
/// `x` is column-major (one inner vector per feature), matching the rest of the
/// crate. The `loglik` argument carries any nuisance parameter the family needs
/// (Negative Binomial `theta`, Gamma/Tweedie dispersion); pass the value estimated
/// for this fit.
pub fn fit<F: GlmFamily + ?Sized>(
    family: &F,
    y: &[f64],
    x: &[Vec<f64>],
    options: &EngineOptions,
    dispersion_rule: DispersionRule,
    loglik_for: impl Fn(f64) -> LogLikKind,
) -> StatsResult<EngineFit> {
    let design = design::build(&DesignSpec {
        y,
        x,
        fit_intercept: options.fit_intercept,
        offset_column: options.offset_column,
        constant_policy: options.constant_policy,
    })?;

    let penalty = design.build_penalty(&options.priors, options.lambda)?;

    let config = IrlsConfig {
        max_iterations: options.max_iterations as usize,
        tolerance: options.tolerance,
        ..IrlsConfig::default()
    };

    let irls = irls::fit_irls(
        family,
        &design.matrix,
        &design.y,
        design.offset.as_deref(),
        &penalty,
        &config,
    )?;

    let n = design.n_observations();
    let p = design.n_params();
    let df_resid = n.saturating_sub(p) as f64;

    let dispersion = match dispersion_rule {
        DispersionRule::Fixed => 1.0,
        DispersionRule::Given(v) => v,
        DispersionRule::Pearson | DispersionRule::PearsonFlooredAtOne => {
            let d = if df_resid > 0.0 {
                let chi2: f64 = design
                    .y
                    .iter()
                    .zip(irls.mu.iter())
                    .map(|(&yi, &mui)| {
                        let v = family.variance(mui);
                        if v > 0.0 {
                            (yi - mui).powi(2) / v
                        } else {
                            0.0
                        }
                    })
                    .sum();
                chi2 / df_resid
            } else {
                1.0
            };
            if dispersion_rule == DispersionRule::PearsonFlooredAtOne {
                d.max(1.0)
            } else {
                d
            }
        }
    };

    let kind = loglik_for(dispersion);
    let log_likelihood = loglik::log_likelihood(kind, &design.y, &irls.mu);
    let k = p + kind.n_nuisance();
    let aic = loglik::aic(log_likelihood, k);
    let bic = loglik::bic(log_likelihood, k, n);

    let inference = if options.compute_inference {
        Some(laplace::inference(
            &irls.beta,
            &irls.information,
            Some(&irls.unpenalized_information),
            dispersion_rule.covariance_scale(dispersion),
            options.confidence_level,
            options.vcov,
            &irls.inactive,
        )?)
    } else {
        None
    };

    Ok(EngineFit {
        design,
        irls,
        dispersion,
        log_likelihood,
        aic,
        bic,
        inference,
    })
}
