//! Mixed-effects GLMs (GLMM):
//!
//! ```text
//!   g(mu_ij) = x_ij' beta + z_ij' b_j ,   b_j ~ N(0, Sigma)
//! ```
//!
//! The solver itself lives upstream in [`anofox_regression::solvers::GlmmRegressor`]
//! (Henderson's mixed-model equations with a block-per-group elimination, the
//! variance components profiled by golden-section for a random intercept and by
//! Nelder–Mead once slopes give an unstructured `Sigma`). This module is the thin
//! marshalling layer: it validates and filters rows, compacts the grouping factors,
//! hands dense matrices to the solver, and reshapes the fit into [`GlmmResult`].
//!
//! Supported here:
//!
//! * A random intercept over one grouping factor ([`fit_glmm`]).
//! * Random slopes on named feature columns, with an unstructured covariance
//!   ([`GlmmOptions::random_slopes`]).
//! * Several crossed / nested random-intercept factors ([`fit_glmm_crossed`]).
//!
//! Families are limited to gaussian, poisson and binomial; NegBinomial/Gamma/
//! Tweedie mixed-effects, an offset, per-group BLUP standard errors, and random
//! slopes combined with multiple factors are tracked upstream (anofox-regression#29)
//! and return a clear error until then.

use crate::errors::{StatsError, StatsResult};
use anofox_regression::solvers::GlmmRegressor;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal};

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
    /// 0-based indices into `x` of feature columns that additionally carry a
    /// random slope (alongside the random intercept), with an unstructured
    /// covariance shared across groups. Empty = random intercept only.
    pub random_slopes: Vec<usize>,
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
            random_slopes: Vec::new(),
        }
    }
}

/// One group's random effect.
#[derive(Debug, Clone)]
pub struct RandomEffect {
    /// Dense group id, matching the order groups were first seen.
    pub group: i32,
    /// Conditional mode of the random intercept (the BLUP) — `effects[0]`.
    pub value: f64,
    /// Conditional standard deviation (NaN: not exposed by the upstream solver).
    pub se: f64,
    /// Full random-effect vector for this group: `[intercept, slope_1, …]` in the
    /// order the slopes were requested. Length `q`.
    pub effects: Vec<f64>,
    /// Number of observations in the group.
    pub n: usize,
}

/// One grouping factor's random-intercept variance (crossed / nested fits).
#[derive(Debug, Clone, Copy)]
pub struct FactorVariance {
    /// Number of levels of this grouping factor.
    pub n_levels: usize,
    /// Random-intercept variance σ²_f for this factor.
    pub var: f64,
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
    /// Random-effects covariance matrix Σ (`q × q`, row-major). `q = 1 + #slopes`;
    /// `random_cov[0][0] == var_group`.
    pub random_cov: Vec<Vec<f64>>,
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
    /// Per-factor variance components for crossed/nested fits. Empty for the
    /// single-factor path (whose variance is `var_group` / `random_cov`).
    pub factors: Vec<FactorVariance>,
}

/// Fit a mixed-effects GLM with a random intercept over one grouping factor.
///
/// The solver itself lives upstream in [`anofox_regression::solvers::GlmmRegressor`];
/// this function is the thin marshalling layer that filters rows, hands dense
/// matrices to the solver, and reshapes the fit into a [`GlmmResult`].
///
/// `group_ids` are the caller's dense group indices; the C++ aggregate
/// dictionary-encodes whatever key type SQL supplied. Non-finite rows and rows
/// with a negative group id are dropped, and the surviving groups are re-compacted
/// so the returned [`RandomEffect::group`] ids still index the caller's labels.
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

    // Offset and the NegBinomial/Gamma/Tweedie families are not yet in the upstream
    // mixed-effects solver (tracked upstream: anofox-regression#29). Reject them
    // explicitly rather than silently ignoring the request.
    if options.offset_column.is_some() {
        return Err(StatsError::InvalidValue {
            field: "offset",
            message: "offset is not yet supported for mixed-effects models \
                      (tracked upstream: anofox-regression#29)"
                .to_string(),
        });
    }
    let builder = match options.family {
        GlmmFamily::Gaussian => GlmmRegressor::gaussian(),
        GlmmFamily::Poisson => GlmmRegressor::poisson(),
        GlmmFamily::Binomial => GlmmRegressor::binomial(),
        _ => {
            return Err(StatsError::InvalidValue {
                field: "family",
                message: "mixed-effects models currently support gaussian, poisson and binomial \
                          only (NegBinomial/Gamma/Tweedie tracked upstream: anofox-regression#29)"
                    .to_string(),
            })
        }
    };

    // Keep rows finite throughout and carrying a valid group.
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && group_ids[i] >= 0 && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n = rows.len();
    let n_features = x.len();
    let n_fixed = n_features + usize::from(options.fit_intercept);

    // Re-compact group ids to a dense 0..k over the surviving rows, keeping the map
    // back to the caller's id for labelling the random effects.
    let mut compact: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut orig: Vec<i32> = Vec::new();
    let mut groups: Vec<usize> = Vec::with_capacity(n);
    for &i in &rows {
        let g = group_ids[i];
        let id = *compact.entry(g).or_insert_with(|| {
            orig.push(g);
            orig.len() - 1
        });
        groups.push(id);
    }
    let n_groups = orig.len();
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

    // Dense design (no intercept column; the builder adds one) and response.
    let xm = Mat::from_fn(n, n_features, |r, c| x[c][rows[r]]);
    let yv = Col::from_fn(n, |r| y[rows[r]]);

    for &c in &options.random_slopes {
        if c >= n_features {
            return Err(StatsError::InvalidValue {
                field: "random",
                message: format!(
                    "random-slope column {} is out of range for {} feature(s)",
                    c + 1,
                    n_features
                ),
            });
        }
    }

    let reg = builder
        .with_intercept(options.fit_intercept)
        .random_slopes(options.random_slopes.clone())
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .reml(options.reml)
        .build();

    let fit = reg
        .fit(&xm, &yv, &groups)
        .map_err(|e| StatsError::RegressError(format!("{e:?}")))?;

    // Fixed effects: element 0 is the intercept when one is fitted.
    let (coefficients, intercept) = if options.fit_intercept {
        (fit.slopes().to_vec(), fit.intercept())
    } else {
        (fit.fixed_effects().to_vec(), None)
    };

    // Inference derived from the reported fixed-effect standard errors.
    let (std_errors, z_values, p_values, ci_lower, ci_upper, intercept_std_error) =
        if options.compute_inference {
            let se = fit.std_errors();
            let normal = Normal::new(0.0, 1.0).ok();
            let z_crit = normal
                .as_ref()
                .map(|nrm| nrm.inverse_cdf(0.5 + options.confidence_level / 2.0))
                .unwrap_or(1.959_963_984_540_054);
            let int_off = usize::from(options.fit_intercept);
            let mut se_v = Vec::with_capacity(coefficients.len());
            let mut z_v = Vec::with_capacity(coefficients.len());
            let mut p_v = Vec::with_capacity(coefficients.len());
            let mut lo_v = Vec::with_capacity(coefficients.len());
            let mut hi_v = Vec::with_capacity(coefficients.len());
            for (k, &b) in coefficients.iter().enumerate() {
                let s = se.get(int_off + k).copied().unwrap_or(f64::NAN);
                let z = if s > 0.0 { b / s } else { f64::NAN };
                let p = match &normal {
                    Some(nrm) if z.is_finite() => 2.0 * (1.0 - nrm.cdf(z.abs())),
                    _ => f64::NAN,
                };
                se_v.push(s);
                z_v.push(z);
                p_v.push(p);
                lo_v.push(b - z_crit * s);
                hi_v.push(b + z_crit * s);
            }
            let icpt_se = if options.fit_intercept {
                Some(se.first().copied().unwrap_or(f64::NAN))
            } else {
                None
            };
            (
                Some(se_v),
                Some(z_v),
                Some(p_v),
                Some(lo_v),
                Some(hi_v),
                icpt_se,
            )
        } else {
            (None, None, None, None, None, None)
        };

    // Per-group random-effect vectors `[intercept, slope_1, …]`. Upstream does not
    // expose the conditional SE of the BLUPs (tracked upstream:
    // anofox-regression#29), so `se` is NaN.
    let re_matrix = fit.random_effects_matrix();
    let ranef: Vec<RandomEffect> = (0..n_groups)
        .map(|g| {
            let effects = re_matrix.get(g).cloned().unwrap_or_default();
            RandomEffect {
                group: orig[g],
                value: effects.first().copied().unwrap_or(f64::NAN),
                se: f64::NAN,
                effects,
                n: group_sizes[g],
            }
        })
        .collect();

    let random_cov: Vec<Vec<f64>> = fit.random_cov().to_vec();

    let var_group = fit.var_random();
    let var_residual = fit.sigma() * fit.sigma();
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
    let ll = fit.log_likelihood();

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
        random_cov,
        log_likelihood: ll,
        aic: 2.0 * k as f64 - 2.0 * ll,
        bic: k as f64 * (n as f64).ln() - 2.0 * ll,
        deviance: fit.deviance(),
        n_observations: n,
        n_groups,
        n_features,
        iterations: fit.iterations() as u32,
        converged: fit.converged(),
        ranef,
        factors: Vec::new(),
    })
}

/// Fit a mixed-effects GLM with several crossed / nested random-**intercept**
/// factors. `group_factors[f]` holds the caller's dense ids for factor `f`, one
/// per observation. A single factor is exactly [`fit_glmm`]. Random slopes
/// combined with multiple factors are not yet supported upstream
/// (tracked: anofox-regression#29).
pub fn fit_glmm_crossed(
    y: &[f64],
    x: &[Vec<f64>],
    group_factors: &[&[i32]],
    options: &GlmmOptions,
) -> StatsResult<GlmmResult> {
    if group_factors.is_empty() {
        return Err(StatsError::InvalidValue {
            field: "group",
            message: "at least one grouping factor is required".to_string(),
        });
    }
    if group_factors.len() == 1 {
        return fit_glmm(y, x, group_factors[0], options);
    }

    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    for g in group_factors {
        if g.len() != y.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: y.len(),
                x_rows: g.len(),
            });
        }
    }
    for col in x.iter() {
        if col.len() != y.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: y.len(),
                x_rows: col.len(),
            });
        }
    }
    if options.offset_column.is_some() {
        return Err(StatsError::InvalidValue {
            field: "offset",
            message: "offset is not yet supported for mixed-effects models \
                      (tracked upstream: anofox-regression#29)"
                .to_string(),
        });
    }
    if !options.random_slopes.is_empty() {
        return Err(StatsError::InvalidValue {
            field: "random",
            message: "random slopes combined with multiple grouping factors are not yet \
                      supported (tracked upstream: anofox-regression#29)"
                .to_string(),
        });
    }
    let builder = match options.family {
        GlmmFamily::Gaussian => GlmmRegressor::gaussian(),
        GlmmFamily::Poisson => GlmmRegressor::poisson(),
        GlmmFamily::Binomial => GlmmRegressor::binomial(),
        _ => {
            return Err(StatsError::InvalidValue {
                field: "family",
                message: "mixed-effects models currently support gaussian, poisson and binomial \
                          only (tracked upstream: anofox-regression#29)"
                    .to_string(),
            })
        }
    };

    // Rows finite throughout and with a valid level in every factor.
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| {
            y[i].is_finite()
                && x.iter().all(|c| c[i].is_finite())
                && group_factors.iter().all(|g| g[i] >= 0)
        })
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n = rows.len();
    let n_features = x.len();
    let n_fixed = n_features + usize::from(options.fit_intercept);
    if n <= n_fixed + 1 {
        return Err(StatsError::InsufficientData {
            rows: n,
            cols: n_fixed,
        });
    }

    // Compact each factor independently over the surviving rows.
    let mut factor_ids: Vec<Vec<usize>> = Vec::with_capacity(group_factors.len());
    for g in group_factors {
        let mut compact: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        let mut ids = Vec::with_capacity(n);
        let mut next = 0usize;
        for &i in &rows {
            let id = *compact.entry(g[i]).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            });
            ids.push(id);
        }
        if next < 2 {
            return Err(StatsError::InvalidValue {
                field: "group",
                message: "each grouping factor needs at least two levels".to_string(),
            });
        }
        factor_ids.push(ids);
    }

    let xm = Mat::from_fn(n, n_features, |r, c| x[c][rows[r]]);
    let yv = Col::from_fn(n, |r| y[rows[r]]);

    let reg = builder
        .with_intercept(options.fit_intercept)
        .max_iterations(options.max_iterations as usize)
        .tolerance(options.tolerance)
        .reml(options.reml)
        .build();

    let group_refs: Vec<&[usize]> = factor_ids.iter().map(|v| v.as_slice()).collect();
    let fit = reg
        .fit_crossed(&xm, &yv, &group_refs)
        .map_err(|e| StatsError::RegressError(format!("{e:?}")))?;

    let (coefficients, intercept) = if options.fit_intercept {
        (fit.slopes().to_vec(), fit.intercept())
    } else {
        (fit.fixed_effects().to_vec(), None)
    };

    let (std_errors, z_values, p_values, ci_lower, ci_upper, intercept_std_error) =
        if options.compute_inference {
            let se = fit.std_errors();
            let normal = Normal::new(0.0, 1.0).ok();
            let z_crit = normal
                .as_ref()
                .map(|nrm| nrm.inverse_cdf(0.5 + options.confidence_level / 2.0))
                .unwrap_or(1.959_963_984_540_054);
            let int_off = usize::from(options.fit_intercept);
            let (mut se_v, mut z_v, mut p_v, mut lo_v, mut hi_v) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (k, &b) in coefficients.iter().enumerate() {
                let s = se.get(int_off + k).copied().unwrap_or(f64::NAN);
                let z = if s > 0.0 { b / s } else { f64::NAN };
                let p = match &normal {
                    Some(nrm) if z.is_finite() => 2.0 * (1.0 - nrm.cdf(z.abs())),
                    _ => f64::NAN,
                };
                se_v.push(s);
                z_v.push(z);
                p_v.push(p);
                lo_v.push(b - z_crit * s);
                hi_v.push(b + z_crit * s);
            }
            let icpt_se = if options.fit_intercept {
                Some(se.first().copied().unwrap_or(f64::NAN))
            } else {
                None
            };
            (
                Some(se_v),
                Some(z_v),
                Some(p_v),
                Some(lo_v),
                Some(hi_v),
                icpt_se,
            )
        } else {
            (None, None, None, None, None, None)
        };

    let factors: Vec<FactorVariance> = fit
        .factors()
        .iter()
        .map(|f| FactorVariance {
            n_levels: f.n_levels,
            var: f.sd * f.sd,
        })
        .collect();
    let var_group = factors.first().map(|f| f.var).unwrap_or(f64::NAN);
    let var_residual = fit.sigma() * fit.sigma();
    let total_random: f64 = factors.iter().map(|f| f.var).sum();
    let icc = if total_random + var_residual > 0.0 {
        total_random / (total_random + var_residual)
    } else {
        0.0
    };

    // k: fixed effects + one variance per factor + residual variance (Gaussian).
    let k = n_fixed
        + factors.len()
        + usize::from(!matches!(
            options.family,
            GlmmFamily::Poisson | GlmmFamily::Binomial
        ));
    let ll = fit.log_likelihood();

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
        random_cov: Vec::new(),
        log_likelihood: ll,
        aic: 2.0 * k as f64 - 2.0 * ll,
        bic: k as f64 * (n as f64).ln() - 2.0 * ll,
        deviance: fit.deviance(),
        n_observations: n,
        n_groups: fit.n_groups(),
        n_features,
        iterations: fit.iterations() as u32,
        converged: fit.converged(),
        ranef: Vec::new(),
        factors,
    })
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
                g.push(gi as i32);
            }
        }
        let opts = GlmmOptions {
            family: GlmmFamily::Poisson,
            compute_inference: true,
            ..Default::default()
        };
        let fit = fit_glmm(&y, &vec![xs], &g, &opts).unwrap();

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
        // BLUPs are populated; their conditional SE is not yet exposed by the
        // upstream solver (tracked: anofox-regression#29), so `se` is NaN for now.
        for r in &fit.ranef {
            assert!(r.value.is_finite(), "ranef value {}", r.value);
            assert!(r.se.is_nan(), "ranef se currently NaN pending upstream #29");
        }
    }

    #[test]
    fn random_slopes_report_a_full_covariance_and_per_term_blups() {
        // y = 1 + 0.5 x + b0_g + b1_g * x + small noise, with a group-specific
        // intercept and slope, so both random variances are identified.
        let n_groups = 24usize;
        let per = 12usize;
        let (mut y, mut xcol, mut g) = (Vec::new(), Vec::new(), Vec::new());
        for gi in 0..n_groups {
            let b0 = (gi as f64 - n_groups as f64 / 2.0) * 0.4;
            let b1 = (((gi * 7) % 5) as f64 - 2.0) * 0.3;
            for k in 0..per {
                let xv = k as f64 / 3.0 - 2.0;
                let noise = (((gi * per + k) % 7) as f64 - 3.0) * 0.05;
                y.push(1.0 + 0.5 * xv + b0 + b1 * xv + noise);
                xcol.push(xv);
                g.push(gi as i32);
            }
        }
        let x = vec![xcol];
        let opts = GlmmOptions {
            random_slopes: vec![0],
            ..Default::default()
        };
        let fit = fit_glmm(&y, &x, &g, &opts).unwrap();

        // q = 2 (intercept + one slope): a 2x2 covariance and length-2 BLUPs.
        assert_eq!(fit.random_cov.len(), 2);
        assert_eq!(fit.random_cov[0].len(), 2);
        assert!(fit.random_cov[0][0] > 0.0, "intercept variance positive");
        assert!(fit.random_cov[1][1] > 0.0, "slope variance positive");
        assert_eq!(fit.var_group, fit.random_cov[0][0]);
        for r in &fit.ranef {
            assert_eq!(r.effects.len(), 2, "per-group [intercept, slope]");
        }

        // An out-of-range slope column is rejected.
        let bad = GlmmOptions {
            random_slopes: vec![5],
            ..Default::default()
        };
        assert!(fit_glmm(&y, &x, &g, &bad).is_err());
    }

    #[test]
    fn crossed_factors_report_per_factor_variances() {
        // Two crossed factors: region (3 levels) and store (5 levels), each with
        // its own random intercept on top of a fixed slope.
        let (mut y, mut xcol, mut region, mut store) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..150usize {
            let r = (i % 3) as i32;
            let s = (i % 5) as i32;
            let xv = (i % 7) as f64 / 3.0;
            y.push(
                1.0 + 0.5 * xv
                    + (r as f64 - 1.0) * 0.8
                    + (s as f64 - 2.0) * 0.5
                    + (((i * 13) % 5) as f64 - 2.0) * 0.05,
            );
            xcol.push(xv);
            region.push(r);
            store.push(s);
        }
        let x = vec![xcol];
        let opts = GlmmOptions::default();

        let fit = fit_glmm_crossed(&y, &x, &[&region, &store], &opts).unwrap();
        assert_eq!(fit.factors.len(), 2, "one variance component per factor");
        assert_eq!(fit.factors[0].n_levels, 3);
        assert_eq!(fit.factors[1].n_levels, 5);
        assert!(fit.factors[0].var >= 0.0 && fit.factors[1].var >= 0.0);
        assert!(fit.converged);

        // A single factor delegates to the single-factor path (ranef/random_cov).
        let single = fit_glmm_crossed(&y, &x, &[&region], &opts).unwrap();
        assert!(single.factors.is_empty());
        assert_eq!(single.n_groups, 3);

        // Random slopes combined with multiple factors is rejected.
        let bad = GlmmOptions {
            random_slopes: vec![0],
            ..Default::default()
        };
        assert!(fit_glmm_crossed(&y, &x, &[&region, &store], &bad).is_err());
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
