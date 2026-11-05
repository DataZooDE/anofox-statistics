//! Likelihood-ratio and score tests and profile-likelihood confidence intervals
//! (#310): thin delegation to `anofox_regression::inference::likelihood`
//! (`lr_test`, `model_lr_test`, `glm_score_test`, `profile_confint`).
//!
//! This module only selects the upstream quantities a fitted model STRUCT
//! corresponds to, drops incomplete rows, builds the designs and maps errors.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::core::{
    BinomialFamily, BinomialLink, GlmFamily, NegativeBinomialFamily, PoissonFamily, PoissonLink,
    TweedieFamily,
};
use anofox_regression::inference::{
    fit_glm, glm_score_test as upstream_score_test, lr_test as upstream_lr_test,
    model_lr_test as upstream_model_lr_test, profile_confint as upstream_profile_confint,
    ChiSqTest, Dispersion, GlmData, LikelihoodError, LikelihoodFit,
};
use faer::Mat;

impl From<LikelihoodError> for StatsError {
    fn from(err: LikelihoodError) -> Self {
        match err {
            LikelihoodError::Data(msg) => StatsError::InvalidInput(msg),
            LikelihoodError::Fit(msg) => StatsError::NumericalFailure(msg.to_string()),
            LikelihoodError::NotConverged => StatsError::NumericalFailure(
                "profiling found a better solution; the original fit had not converged".into(),
            ),
            // `LikelihoodError` is `#[non_exhaustive]`: surface a new variant.
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// The likelihood summary of a fitted model, as its result STRUCT carries it.
/// Unknown numbers are NaN, unknown names empty.
#[derive(Debug, Clone, Default)]
pub struct ModelLikelihood {
    pub model_type: String,
    pub family: String,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub dispersion: f64,
    /// Estimated mean parameters (intercept included).
    pub n_parameters: usize,
    pub n_observations: Option<usize>,
}

/// Which upstream `LikelihoodFit` implementation a model corresponds to.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LikelihoodKind {
    /// Binomial / Poisson GLM: the deviance, dispersion 1.
    FixedDeviance,
    /// GLM with an estimated dispersion (Gamma, Tweedie, ...): the deviance
    /// divided by the dispersion of the larger model (`anova(test = "LRT")`).
    ScaledDeviance,
    /// Everything else (OLS, Negative Binomial, AFT, GLMM, ...): `-2 log L`.
    LogLik,
}

impl ModelLikelihood {
    fn kind(&self) -> LikelihoodKind {
        if self.model_type != "glm" || !self.deviance.is_finite() {
            return LikelihoodKind::LogLik;
        }
        match self.family.as_str() {
            "binomial" | "poisson" => LikelihoodKind::FixedDeviance,
            // theta differs between fits, so the deviances are not comparable.
            "negative_binomial" => LikelihoodKind::LogLik,
            _ => LikelihoodKind::ScaledDeviance,
        }
    }

    fn to_fit(&self, name: &str) -> StatsResult<Fit> {
        let kind = self.kind();
        let (m2ll, null) = match kind {
            LikelihoodKind::FixedDeviance | LikelihoodKind::ScaledDeviance => {
                (self.deviance, self.null_deviance)
            }
            LikelihoodKind::LogLik => {
                let m2ll = -2.0 * self.log_likelihood;
                let null = if self.null_log_likelihood.is_finite() {
                    -2.0 * self.null_log_likelihood
                } else if self.family == "negative_binomial" {
                    // As upstream's FittedNegativeBinomial: -2 log L shifted by
                    // the deviance gap to the null model.
                    m2ll + self.null_deviance - self.deviance
                } else {
                    f64::NAN
                };
                (m2ll, null)
            }
        };
        if !m2ll.is_finite() {
            return Err(StatsError::InvalidInput(format!(
                "{name} has no log-likelihood (the model STRUCT needs a log_likelihood or, for a GLM, a deviance)"
            )));
        }
        let dispersion = match kind {
            LikelihoodKind::ScaledDeviance => self.dispersion,
            _ => 1.0,
        };
        Ok(Fit {
            m2ll,
            null_m2ll: null.is_finite().then_some(null),
            dispersion,
            k: self.n_parameters,
        })
    }
}

/// A model STRUCT seen as an upstream [`LikelihoodFit`].
struct Fit {
    m2ll: f64,
    null_m2ll: Option<f64>,
    dispersion: f64,
    k: usize,
}

impl LikelihoodFit for Fit {
    fn minus_two_loglik(&self) -> f64 {
        self.m2ll
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        self.null_m2ll
    }
    fn lr_dispersion(&self) -> f64 {
        self.dispersion
    }
    fn n_parameters(&self) -> usize {
        self.k
    }
}

fn check_dispersion(fit: &Fit, name: &str) -> StatsResult<()> {
    if !(fit.dispersion.is_finite() && fit.dispersion > 0.0) {
        return Err(StatsError::InvalidInput(format!(
            "{name} has no finite positive dispersion"
        )));
    }
    Ok(())
}

/// A likelihood-ratio test and the dispersion its statistic was divided by.
#[derive(Debug, Clone, Copy)]
pub struct LrTestResult {
    pub test: ChiSqTest,
    pub dispersion: f64,
}

/// Likelihood-ratio test of the nested fits `reduced ⊂ full` (R's
/// `anova(reduced, full, test = "LRT")` / `lmtest::lrtest`). Both must be the
/// same kind of model and family, fitted on the same number of rows, and
/// `full` must have more parameters. Nesting of the predictors themselves
/// cannot be checked from the fits and is the caller's responsibility.
pub fn lr_test(full: &ModelLikelihood, reduced: &ModelLikelihood) -> StatsResult<LrTestResult> {
    if full.model_type != reduced.model_type {
        return Err(StatsError::InvalidInput(format!(
            "models are not comparable: model_type '{}' vs '{}'",
            full.model_type, reduced.model_type
        )));
    }
    if full.family != reduced.family {
        return Err(StatsError::InvalidInput(format!(
            "models are not comparable: family '{}' vs '{}'",
            full.family, reduced.family
        )));
    }
    if let (Some(a), Some(b)) = (full.n_observations, reduced.n_observations) {
        if a != b {
            return Err(StatsError::InvalidInput(format!(
                "models are not nested: fitted on {a} and {b} observations"
            )));
        }
    }
    if full.n_parameters <= reduced.n_parameters {
        return Err(StatsError::InvalidInput(format!(
            "model_full must have more parameters than model_reduced ({} vs {})",
            full.n_parameters, reduced.n_parameters
        )));
    }
    let f = full.to_fit("model_full")?;
    let r = reduced.to_fit("model_reduced")?;
    check_dispersion(&f, "model_full")?;
    Ok(LrTestResult {
        test: upstream_lr_test(&r, &f),
        dispersion: f.dispersion,
    })
}

/// Likelihood-ratio test of a fit against its null (intercept-only) model, on
/// `k - 1` degrees of freedom (the "Model L.R." of `rms`).
pub fn model_lr_test(fit: &ModelLikelihood) -> StatsResult<LrTestResult> {
    let f = fit.to_fit("model")?;
    check_dispersion(&f, "model")?;
    if f.k < 2 {
        return Err(StatsError::InvalidInput(
            "model must have a parameter besides the intercept".into(),
        ));
    }
    let test = upstream_model_lr_test(&f).ok_or_else(|| {
        StatsError::InvalidInput(
            "model has no null likelihood (null_deviance or null_log_likelihood)".into(),
        )
    })?;
    Ok(LrTestResult {
        test,
        dispersion: f.dispersion,
    })
}

/// A GLM family (and link) for the score test and profile intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlmFamilySpec {
    Poisson(PoissonLink),
    Binomial(BinomialLink),
    /// Negative Binomial (log link) with the given shape `theta`, or `None` to
    /// take the estimate of the (reduced) model's `MASS::glm.nb` fit.
    NegativeBinomial(Option<f64>),
    /// Gamma with log link.
    Gamma,
    /// Tweedie with log link and the given variance power.
    Tweedie(f64),
    /// Gaussian with identity link.
    Gaussian,
}

impl GlmFamilySpec {
    fn family(&self, theta: f64) -> Box<dyn GlmFamily> {
        match *self {
            Self::Poisson(link) => Box::new(PoissonFamily::new(link)),
            Self::Binomial(link) => Box::new(BinomialFamily::new(link)),
            Self::NegativeBinomial(_) => Box::new(NegativeBinomialFamily::new(theta)),
            Self::Gamma => Box::new(TweedieFamily::new(2.0, 0.0)),
            Self::Tweedie(power) => Box::new(TweedieFamily::new(power, 0.0)),
            Self::Gaussian => Box::new(TweedieFamily::gaussian()),
        }
    }

    fn dispersion(&self) -> Dispersion {
        match self {
            Self::Poisson(_) | Self::Binomial(_) | Self::NegativeBinomial(_) => {
                Dispersion::Fixed(1.0)
            }
            Self::Gamma | Self::Tweedie(_) | Self::Gaussian => Dispersion::Estimated,
        }
    }
}

/// Rows where `y` and every column are finite.
fn complete_rows(y: &[f64], columns: &[&[Vec<f64>]]) -> Vec<usize> {
    (0..y.len())
        .filter(|&i| {
            y[i].is_finite() && columns.iter().all(|cs| cs.iter().all(|c| c[i].is_finite()))
        })
        .collect()
}

fn design(rows: &[usize], cols: &[&Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), cols.len(), |i, j| cols[j][rows[i]])
}

/// The feature columns of `x` and its offset column.
type FeaturesAndOffset<'a> = (Vec<&'a Vec<f64>>, Option<&'a Vec<f64>>);

/// Profile interval of the intercept (if fitted) and of each feature.
pub type ProfileIntervals = (Option<(f64, f64)>, Vec<(f64, f64)>);

/// Split a 1-based offset column off `x`.
fn split_offset(
    x: &[Vec<f64>],
    offset_column: Option<usize>,
) -> StatsResult<FeaturesAndOffset<'_>> {
    match offset_column {
        None => Ok((x.iter().collect(), None)),
        Some(c) if c >= 1 && c <= x.len() => Ok((
            x.iter()
                .enumerate()
                .filter(|(j, _)| *j != c - 1)
                .map(|(_, v)| v)
                .collect(),
            Some(&x[c - 1]),
        )),
        Some(c) => Err(StatsError::InvalidInput(format!(
            "offset column {c} is out of range (x has {} columns)",
            x.len()
        ))),
    }
}

/// Result of [`glm_score_test`].
#[derive(Debug, Clone)]
pub struct GlmScoreTestResult {
    pub test: ChiSqTest,
    /// Per-column z statistics of the added columns.
    pub z: Vec<f64>,
    /// Dispersion the statistic is divided by.
    pub dispersion: f64,
    /// Negative Binomial shape used (NaN for other families).
    pub theta: f64,
    pub n_observations: usize,
}

/// Rao score test for adding the `x_add` columns to the GLM of `y` on
/// `x_reduced` (R's `anova(reduced, full, test = "Rao")`), on the rows where
/// every value is finite. A fixed-dispersion family uses dispersion 1, the
/// others the Pearson estimate of the larger model (as `anova`). The Negative
/// Binomial uses `theta`, or the estimate of the reduced `glm.nb` fit.
pub fn glm_score_test(
    family: GlmFamilySpec,
    y: &[f64],
    x_reduced: &[Vec<f64>],
    x_add: &[Vec<f64>],
    with_intercept: bool,
    offset_column: Option<usize>,
) -> StatsResult<GlmScoreTestResult> {
    if x_add.is_empty() {
        return Err(StatsError::InvalidInput(
            "x_add must have at least one column".into(),
        ));
    }
    crate::validation::validate_x_columns(y.len(), x_reduced)?;
    crate::validation::validate_x_columns(y.len(), x_add)?;
    let (reduced_cols, offset) = split_offset(x_reduced, offset_column)?;
    if !with_intercept && reduced_cols.is_empty() {
        return Err(StatsError::InvalidInput(
            "the reduced model is empty (no intercept and no x_reduced column)".into(),
        ));
    }
    let rows = complete_rows(y, &[x_reduced, x_add]);
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n = rows.len();
    let yv: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let off: Option<Vec<f64>> = offset.map(|o| rows.iter().map(|&i| o[i]).collect());
    let x1 = design(&rows, &reduced_cols);
    let x2 = design(&rows, &x_add.iter().collect::<Vec<_>>());

    let theta = match family {
        GlmFamilySpec::NegativeBinomial(Some(t)) => {
            if !(t.is_finite() && t > 0.0) {
                return Err(StatsError::InvalidInput(
                    "theta must be finite and positive".into(),
                ));
            }
            t
        }
        GlmFamilySpec::NegativeBinomial(None) => {
            let cols: Vec<Vec<f64>> = x_reduced
                .iter()
                .map(|c| rows.iter().map(|&i| c[i]).collect())
                .collect();
            let opts = crate::types::NegBinomialOptions::default()
                .with_intercept(with_intercept)
                .offset_column(offset_column);
            let fit = crate::models::fit_negbinomial(&yv, &cols, &opts)?;
            fit.core.dispersion.unwrap_or(f64::NAN)
        }
        _ => f64::NAN,
    };
    let fam = family.family(theta);
    let data = GlmData::new(&x1, &yv)
        .with_intercept(with_intercept)
        .offset(off.as_deref());
    let dispersion = match family.dispersion() {
        Dispersion::Fixed(v) => v,
        _ => {
            let full = Mat::from_fn(n, x1.ncols() + x2.ncols(), |i, j| {
                if j < x1.ncols() {
                    x1[(i, j)]
                } else {
                    x2[(i, j - x1.ncols())]
                }
            });
            fit_glm(fam.as_ref(), &data.covariates(&full), Dispersion::Estimated)?.dispersion
        }
    };
    let st = upstream_score_test(fam.as_ref(), &data, &x2, dispersion)?;
    Ok(GlmScoreTestResult {
        test: st.test,
        z: st.z,
        dispersion,
        theta,
        n_observations: n,
    })
}

/// Profile-likelihood confidence intervals of a GLM fit (R's `confint(glm)`,
/// `MASS::confint.glm`): the intercept's (if fitted) and one per feature, in
/// the feature order of `x` without the offset column. `coefficients` are the
/// fit's feature coefficients: a NaN marks a column the fit dropped, which is
/// left out of the profiled design and gets a NaN interval. A bound the
/// profile does not reach is NaN (R's `NA`).
pub fn glm_profile_confint(
    family: GlmFamilySpec,
    y: &[f64],
    x: &[Vec<f64>],
    coefficients: &[f64],
    with_intercept: bool,
    offset_column: Option<usize>,
    level: f64,
) -> StatsResult<ProfileIntervals> {
    crate::validation::validate_x_columns(y.len(), x)?;
    let (features, offset) = split_offset(x, offset_column)?;
    if features.len() != coefficients.len() {
        return Err(StatsError::InvalidInput(format!(
            "{} coefficients for {} feature columns",
            coefficients.len(),
            features.len()
        )));
    }
    let theta = match family {
        GlmFamilySpec::NegativeBinomial(Some(t)) if t.is_finite() && t > 0.0 => t,
        GlmFamilySpec::NegativeBinomial(_) => {
            return Err(StatsError::InvalidInput(
                "the Negative Binomial profile needs the fitted theta".into(),
            ))
        }
        _ => f64::NAN,
    };
    let rows = complete_rows(y, &[x]);
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let kept: Vec<usize> = (0..features.len())
        .filter(|&j| coefficients[j].is_finite())
        .collect();
    let xm = design(
        &rows,
        &kept.iter().map(|&j| features[j]).collect::<Vec<_>>(),
    );
    let yv: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let off: Option<Vec<f64>> = offset.map(|o| rows.iter().map(|&i| o[i]).collect());
    let data = GlmData::new(&xm, &yv)
        .with_intercept(with_intercept)
        .offset(off.as_deref());
    let fam = family.family(theta);
    let ci = upstream_profile_confint(fam.as_ref(), &data, family.dispersion(), level)?;
    let shift = usize::from(with_intercept);
    let intercept = with_intercept.then(|| ci[0]);
    let mut slopes = vec![(f64::NAN, f64::NAN); features.len()];
    for (pos, &j) in kept.iter().enumerate() {
        slopes[j] = ci[pos + shift];
    }
    Ok((intercept, slopes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm(family: &str, dev: f64, null_dev: f64, disp: f64, k: usize) -> ModelLikelihood {
        ModelLikelihood {
            model_type: "glm".into(),
            family: family.into(),
            log_likelihood: f64::NAN,
            null_log_likelihood: f64::NAN,
            deviance: dev,
            null_deviance: null_dev,
            dispersion: disp,
            n_parameters: k,
            n_observations: Some(20),
        }
    }

    #[test]
    fn lr_of_fixed_dispersion_glms_is_the_deviance_gap() {
        let t = lr_test(
            &glm("poisson", 10.0, 30.0, 1.0, 3),
            &glm("poisson", 16.0, 30.0, 1.0, 2),
        )
        .unwrap()
        .test;
        assert!((t.statistic - 6.0).abs() < 1e-12);
        assert_eq!(t.df, 1);
    }

    #[test]
    fn lr_scales_by_the_full_dispersion() {
        let t = lr_test(
            &glm("gamma", 2.0, 5.0, 0.5, 3),
            &glm("gamma", 3.0, 5.0, 0.7, 1),
        )
        .unwrap()
        .test;
        assert!((t.statistic - 2.0).abs() < 1e-12);
        assert_eq!(t.df, 2);
    }

    #[test]
    fn lr_rejects_incompatible_models() {
        let full = glm("poisson", 10.0, 30.0, 1.0, 3);
        assert!(lr_test(&full, &glm("binomial", 16.0, 30.0, 1.0, 2)).is_err());
        assert!(lr_test(&full, &glm("poisson", 16.0, 30.0, 1.0, 3)).is_err());
        let mut other_n = glm("poisson", 16.0, 30.0, 1.0, 2);
        other_n.n_observations = Some(19);
        assert!(lr_test(&full, &other_n).is_err());
    }

    #[test]
    fn model_lr_uses_the_null_deviance() {
        let t = model_lr_test(&glm("binomial", 10.0, 30.0, 1.0, 3))
            .unwrap()
            .test;
        assert!((t.statistic - 20.0).abs() < 1e-12);
        assert_eq!(t.df, 2);
    }

    #[test]
    fn profile_and_score_run_on_poisson_counts() {
        let x = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
        let y = vec![1.0, 1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 11.0];
        let fam = GlmFamilySpec::Poisson(PoissonLink::Log);
        let (b0, b) = glm_profile_confint(fam, &y, &x, &[0.3], true, None, 0.95).unwrap();
        let (lo, hi) = b0.unwrap();
        assert!(lo < hi && b[0].0 < b[0].1);
        let st = glm_score_test(fam, &y, &[], &x, true, None).unwrap();
        assert_eq!(st.test.df, 1);
        assert!(st.test.statistic > 0.0);
    }
}
