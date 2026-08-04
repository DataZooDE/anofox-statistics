//! Generalized Linear Models (GLM) — Poisson, Binomial, Negative Binomial,
//! Tweedie, Gamma, Logistic.
//!
//! Every family routes through [`crate::models::glm_engine`], a single
//! family-generic penalized IRLS loop. Each `fit_*` below is therefore a thin
//! adapter: validate the family's response domain, pick the family object, the
//! dispersion rule and the log-likelihood, and hand over.
//!
//! Two user-visible consequences of the switch, both intentional:
//!
//! * **AIC changed for Gamma, Negative Binomial and Tweedie.** The engine computes
//!   a real per-family log-likelihood; upstream substituted `-deviance / 2`, which
//!   is only correct up to a constant for the Gaussian family. The new values are
//!   comparable with R's.
//! * **Standard errors changed for penalized fits.** `glm_lambda > 0` previously
//!   reported standard errors computed from the *unpenalized* `X'WX`. The default
//!   is now the Laplace curvature `(X'WX + P)^-1`; `vcov := 'naive'` restores the
//!   old numbers.

use crate::errors::{StatsError, StatsResult};
use crate::models::glm_engine::{
    self, ConstantColumnPolicy, DispersionRule, EngineFit, EngineOptions, LogLikKind,
};
use crate::types::{
    BinomialLink, BinomialOptions, GammaOptions, GlmFitResult, GlmInferenceResult, LogisticOptions,
    NegBinomialOptions, PoissonLink, PoissonOptions, TweedieOptions,
};
use anofox_regression::core::{
    BinomialFamily, NegativeBinomialFamily, PoissonFamily, TweedieFamily,
};
use anofox_regression::prelude::*;

/// Combined GLM result with optional inference
#[derive(Debug, Clone)]
pub struct GlmResult {
    pub core: GlmFitResult,
    pub inference: Option<GlmInferenceResult>,
}

impl From<EngineFit> for GlmResult {
    fn from(fit: EngineFit) -> Self {
        GlmResult {
            core: fit.to_glm_fit_result(),
            inference: fit.to_glm_inference(),
        }
    }
}

/// Reject responses outside a family's support.
fn require<F: Fn(f64) -> bool>(
    y: &[f64],
    field: &'static str,
    message: &str,
    ok: F,
) -> StatsResult<()> {
    for &v in y.iter() {
        if !v.is_nan() && !ok(v) {
            return Err(StatsError::InvalidValue {
                field,
                message: message.to_string(),
            });
        }
    }
    Ok(())
}

/// Fit a Poisson regression model (for count data)
///
/// # Arguments
/// * `y` - Response variable (counts, must be non-negative integers)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_poisson(y: &[f64], x: &[Vec<f64>], options: &PoissonOptions) -> StatsResult<GlmResult> {
    require(
        y,
        "y",
        "Poisson regression requires non-negative response values",
        |v| v >= 0.0,
    )?;

    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        // Poisson has always dropped constant columns and reported NaN for them.
        constant_policy: ConstantColumnPolicy::Drop,
    };

    let family: Box<dyn GlmFamily> = match options.link {
        PoissonLink::Log => Box::new(PoissonFamily::log()),
        PoissonLink::Identity => Box::new(PoissonFamily::identity()),
        PoissonLink::Sqrt => Box::new(PoissonFamily::sqrt()),
    };

    let fit = glm_engine::fit(
        family.as_ref(),
        y,
        x,
        &engine_opts,
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )?;
    Ok(fit.into())
}

/// Fit a Binomial (Logistic) regression model (for binary outcomes)
///
/// # Arguments
/// * `y` - Response variable (0 or 1 for binary, or proportion in [0,1])
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_binomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &BinomialOptions,
) -> StatsResult<GlmResult> {
    require(
        y,
        "y",
        "Binomial regression requires y values in [0, 1]",
        |v| (0.0..=1.0).contains(&v),
    )?;

    let family = binomial_family(options.link)?;
    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        constant_policy: ConstantColumnPolicy::Keep,
    };

    let fit = glm_engine::fit(&family, y, x, &engine_opts, DispersionRule::Fixed, |_| {
        LogLikKind::Binomial
    })?;
    Ok(fit.into())
}

fn binomial_family(link: BinomialLink) -> StatsResult<BinomialFamily> {
    match link {
        BinomialLink::Logit => Ok(BinomialFamily::logistic()),
        BinomialLink::Probit => Ok(BinomialFamily::probit()),
        BinomialLink::Cloglog => Ok(BinomialFamily::cloglog()),
        BinomialLink::Cauchit | BinomialLink::Log => Err(StatsError::InvalidValue {
            field: "link",
            message: "Cauchit and Log links are not supported. Use Logit, Probit, or Cloglog."
                .to_string(),
        }),
    }
}

/// Fit a Negative Binomial regression model (for overdispersed count data)
///
/// # Arguments
/// * `y` - Response variable (counts, must be non-negative)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
///
/// When `options.alpha` is `None` the dispersion `theta` is estimated from the data
/// by alternating between an IRLS fit at the current `theta` and a method-of-moments
/// update, which is how `MASS::glm.nb` proceeds.
pub fn fit_negbinomial(
    y: &[f64],
    x: &[Vec<f64>],
    options: &NegBinomialOptions,
) -> StatsResult<GlmResult> {
    require(
        y,
        "y",
        "Negative Binomial regression requires non-negative response values",
        |v| v >= 0.0,
    )?;

    if let Some(alpha) = options.alpha {
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(StatsError::InvalidValue {
                field: "alpha",
                message: "Negative Binomial alpha (theta) must be finite and positive".to_string(),
            });
        }
    }

    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        constant_policy: ConstantColumnPolicy::Keep,
    };

    let run = |theta: f64, opts: &EngineOptions| {
        glm_engine::fit(
            &NegativeBinomialFamily::new(theta),
            y,
            x,
            opts,
            DispersionRule::Given(theta),
            |_| LogLikKind::NegativeBinomial { theta },
        )
    };

    // `alpha` given: a single fit at that theta.
    if let Some(theta) = options.alpha {
        return Ok(run(theta, &engine_opts)?.into());
    }

    // Otherwise alternate IRLS and a moment update for theta.
    let mut theta = 1.0_f64;
    let mut fit = {
        let mut probe = engine_opts.clone();
        probe.compute_inference = false;
        run(theta, &probe)?
    };

    for _ in 0..25 {
        let next = estimate_theta_moments(&fit.design.y, &fit.irls.mu);
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        if (next - theta).abs() / theta.max(1e-8) < 1e-6 {
            theta = next;
            break;
        }
        theta = next;
        let mut probe = engine_opts.clone();
        probe.compute_inference = false;
        fit = run(theta, &probe)?;
    }

    Ok(run(theta, &engine_opts)?.into())
}

/// Method-of-moments estimate of the Negative Binomial `theta`.
///
/// Solves `sum (y - mu)^2 / (mu + mu^2/theta) = n - p` approximately by matching
/// the Pearson statistic, clamped to a sane range so a near-Poisson sample cannot
/// drive `theta` to infinity.
fn estimate_theta_moments(y: &[f64], mu: &[f64]) -> f64 {
    let n = y.len() as f64;
    let num: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| (yi - mui).powi(2) - mui)
        .sum();
    let den: f64 = mu.iter().map(|&m| m * m).sum();
    if den <= 0.0 || num <= 0.0 {
        // No detectable overdispersion — a large theta approaches Poisson.
        return 1e6;
    }
    let alpha = (num / den).max(1e-12) * n / n;
    (1.0 / alpha).clamp(1e-6, 1e6)
}

/// Fit a Tweedie regression model (for zero-inflated continuous data)
///
/// # Arguments
/// * `y` - Response variable (non-negative, can include zeros)
/// * `x` - Feature matrix (n observations x p features, column-major)
/// * `options` - Fitting options
pub fn fit_tweedie(y: &[f64], x: &[Vec<f64>], options: &TweedieOptions) -> StatsResult<GlmResult> {
    require(
        y,
        "y",
        "Tweedie regression requires non-negative response values",
        |v| v >= 0.0,
    )?;

    if !(1.0..=2.0).contains(&options.power) {
        return Err(StatsError::InvalidValue {
            field: "power",
            message: "Tweedie power parameter must be in [1, 2]".to_string(),
        });
    }

    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        constant_policy: ConstantColumnPolicy::Keep,
    };

    // link_power 0.0 pins the log link. The upstream builder otherwise defaults to
    // the canonical link `1 - var_power` (mu^-0.5 for p = 1.5), which is neither
    // what the docs advertise nor what sklearn / statsmodels users expect.
    let power = options.power;
    let fit = glm_engine::fit(
        &TweedieFamily::new(power, 0.0),
        y,
        x,
        &engine_opts,
        DispersionRule::Pearson,
        move |phi| LogLikKind::Tweedie {
            power,
            dispersion: phi,
        },
    )?;
    Ok(fit.into())
}

/// Fit a Gamma GLM. Equivalent to Tweedie with `var_power = 2.0` baked in;
/// log link (the upstream solver's default for Gamma).
pub fn fit_gamma(y: &[f64], x: &[Vec<f64>], options: &GammaOptions) -> StatsResult<GlmResult> {
    require(
        y,
        "y",
        "Gamma regression requires strictly positive response values",
        |v| v > 0.0,
    )?;

    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        constant_policy: ConstantColumnPolicy::Keep,
    };

    let fit = glm_engine::fit(
        &TweedieFamily::new(2.0, 0.0),
        y,
        x,
        &engine_opts,
        DispersionRule::Pearson,
        |phi| LogLikKind::Gamma { dispersion: phi },
    )?;
    Ok(fit.into())
}

/// Result from Logistic regression fit. Bundles the standard GLM result
/// with the classification-specific diagnostics (accuracy on the training
/// set and the classification threshold that was used).
#[derive(Debug, Clone)]
pub struct LogisticResult {
    pub fit: GlmResult,
    /// Classification accuracy on the training data using the configured
    /// threshold.
    pub accuracy: f64,
    /// Classification threshold used (echoed from options).
    pub threshold: f64,
}

/// Fit a binary Logistic regression — a binomial GLM with logit link, plus the
/// training-set accuracy at the configured threshold.
pub fn fit_logistic(
    y: &[f64],
    x: &[Vec<f64>],
    options: &LogisticOptions,
) -> StatsResult<LogisticResult> {
    require(
        y,
        "y",
        "Logistic regression requires binary response values (0.0 or 1.0)",
        |v| v == 0.0 || v == 1.0,
    )?;

    if !(0.0..=1.0).contains(&options.threshold) {
        return Err(StatsError::InvalidInput(format!(
            "threshold must be in [0, 1], got {}",
            options.threshold
        )));
    }

    let engine_opts = EngineOptions {
        fit_intercept: options.fit_intercept,
        max_iterations: options.max_iterations,
        tolerance: options.tolerance,
        compute_inference: options.compute_inference,
        confidence_level: options.confidence_level,
        lambda: options.lambda,
        priors: options.prior_opts.priors.clone(),
        vcov: options.prior_opts.vcov,
        offset_column: options.offset_column,
        constant_policy: ConstantColumnPolicy::Keep,
    };

    let fit = glm_engine::fit(
        &BinomialFamily::logistic(),
        y,
        x,
        &engine_opts,
        DispersionRule::Fixed,
        |_| LogLikKind::Binomial,
    )?;

    // Training accuracy over the rows that were actually fitted.
    let correct = fit
        .design
        .y
        .iter()
        .zip(fit.irls.mu.iter())
        .filter(|(&yi, &mui)| {
            let predicted = f64::from(u8::from(mui >= options.threshold));
            (predicted - yi).abs() < f64::EPSILON
        })
        .count();
    let accuracy = correct as f64 / fit.design.n_observations().max(1) as f64;

    Ok(LogisticResult {
        fit: fit.into(),
        accuracy,
        threshold: options.threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PriorSpec, VcovType};

    #[test]
    fn test_poisson_basic() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0];

        let options = PoissonOptions::default();
        let result = fit_poisson(&y, &x, &options);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.core.pseudo_r_squared > 0.0);
    }

    #[test]
    fn test_binomial_basic() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let options = BinomialOptions::default();
        assert!(fit_binomial(&y, &x, &options).is_ok());
    }

    #[test]
    fn test_poisson_negative_y_error() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![-1.0, 2.0, 3.0, 4.0, 5.0];

        let options = PoissonOptions::default();
        let result = fit_poisson(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn test_binomial_invalid_y_error() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![0.5, 1.5, 0.0, 0.5, 0.5];

        let options = BinomialOptions::default();
        let result = fit_binomial(&y, &x, &options);

        assert!(matches!(result, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn test_poisson_with_lambda() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0];

        let no_penalty = PoissonOptions {
            lambda: 0.0,
            ..Default::default()
        };
        let with_penalty = PoissonOptions {
            lambda: 1.0,
            ..Default::default()
        };

        let result_no = fit_poisson(&y, &x, &no_penalty).unwrap();
        let result_pen = fit_poisson(&y, &x, &with_penalty).unwrap();

        assert!(
            result_pen.core.coefficients[0].abs() <= result_no.core.coefficients[0].abs() + 0.01
        );
    }

    #[test]
    fn test_binomial_with_lambda() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let options = BinomialOptions {
            lambda: 0.5,
            ..Default::default()
        };

        assert!(fit_binomial(&y, &x, &options).is_ok());
    }

    // --- new surface -------------------------------------------------------

    fn count_data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 60;
        let x1: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 3.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                ((0.6 + 0.25 * x1[i] - 0.15 * x2[i]).exp() + ((i * 13) % 4) as f64 * 0.3).round()
            })
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn a_prior_shrinks_the_coefficient_toward_its_location() {
        let (y, x) = count_data();

        let flat = fit_poisson(&y, &x, &PoissonOptions::default()).unwrap();

        let mut opts = PoissonOptions::default();
        opts.prior_opts.priors = vec![PriorSpec::normal(0.0, 0.02), PriorSpec::flat()];
        let shrunk = fit_poisson(&y, &x, &opts).unwrap();

        assert!(
            shrunk.core.coefficients[0].abs() < flat.core.coefficients[0].abs(),
            "prior should shrink x1: {} vs {}",
            shrunk.core.coefficients[0],
            flat.core.coefficients[0]
        );
        // The unpenalized coefficient is essentially untouched.
        assert!((shrunk.core.coefficients[1] - flat.core.coefficients[1]).abs() < 0.2);
    }

    #[test]
    fn vcov_choice_changes_only_the_standard_errors() {
        let (y, x) = count_data();

        let mk = |vcov: VcovType| {
            let mut o = PoissonOptions {
                compute_inference: true,
                lambda: 5.0,
                ..Default::default()
            };
            o.prior_opts.vcov = vcov;
            o
        };

        let lap = fit_poisson(&y, &x, &mk(VcovType::Laplace)).unwrap();
        let naive = fit_poisson(&y, &x, &mk(VcovType::Naive)).unwrap();

        for j in 0..2 {
            assert!((lap.core.coefficients[j] - naive.core.coefficients[j]).abs() < 1e-12);
        }
        let li = lap.inference.unwrap();
        let ni = naive.inference.unwrap();
        for j in 0..2 {
            assert!(
                li.std_errors[j] < ni.std_errors[j],
                "laplace SE should be tighter at {j}"
            );
        }
    }

    #[test]
    fn negbinomial_estimates_theta_when_alpha_is_absent() {
        let (y, x) = count_data();
        let fit = fit_negbinomial(&y, &x, &NegBinomialOptions::default()).unwrap();
        let theta = fit.core.dispersion.unwrap();
        assert!(theta > 0.0 && theta.is_finite(), "theta = {theta}");
    }

    #[test]
    fn negbinomial_honours_a_supplied_alpha() {
        let (y, x) = count_data();
        let opts = NegBinomialOptions {
            alpha: Some(2.5),
            ..Default::default()
        };
        let fit = fit_negbinomial(&y, &x, &opts).unwrap();
        assert!((fit.core.dispersion.unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn negbinomial_rejects_a_non_positive_alpha() {
        let (y, x) = count_data();
        let opts = NegBinomialOptions {
            alpha: Some(-1.0),
            ..Default::default()
        };
        assert!(matches!(
            fit_negbinomial(&y, &x, &opts),
            Err(StatsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn multi_feature_fits_now_succeed() {
        // Three design columns; upstream cannot fit this at all (see the parity
        // module for the pivot back-permutation defect).
        let (y, x) = count_data();
        let fit = fit_poisson(&y, &x, &PoissonOptions::default()).unwrap();
        assert!((fit.core.intercept.unwrap() - 0.783_761_952_889_341_5).abs() < 1e-7);
        assert!((fit.core.coefficients[0] - 0.241_563_412_876_373_3).abs() < 1e-7);
        assert!((fit.core.coefficients[1] + 0.128_771_260_171_794).abs() < 1e-7);
    }

    #[test]
    fn logistic_reports_training_accuracy() {
        let n = 50;
        let xs: Vec<f64> = (0..n).map(|i| (i % 12) as f64 / 4.0 - 1.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| f64::from(u8::from(0.8 * xs[i] + ((i % 3) as f64 - 1.0) * 0.5 > 0.0)))
            .collect();
        let fit = fit_logistic(&y, &[xs], &LogisticOptions::default()).unwrap();
        assert!((0.0..=1.0).contains(&fit.accuracy));
        assert!(fit.accuracy > 0.5, "accuracy {}", fit.accuracy);
    }

    #[test]
    fn gamma_rejects_non_positive_response() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        assert!(matches!(
            fit_gamma(&y, &x, &GammaOptions::default()),
            Err(StatsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn tweedie_rejects_a_power_outside_one_to_two() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let opts = TweedieOptions {
            power: 2.5,
            ..Default::default()
        };
        assert!(matches!(
            fit_tweedie(&y, &x, &opts),
            Err(StatsError::InvalidValue { .. })
        ));
    }
}
