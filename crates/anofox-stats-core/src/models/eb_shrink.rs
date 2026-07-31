//! Empirical-Bayes shrinkage of per-group estimates toward a common mean.
//!
//! This is the cheap version of partial pooling described in issue #107: rather
//! than fitting a hierarchical model, take estimates that already exist — one per
//! group, each with a standard error — and shrink them toward their precision-
//! weighted mean by an amount the data itself determines.
//!
//! The model is the familiar random-effects meta-analysis one:
//!
//! ```text
//!   theta_g ~ N(mu, tau^2)        (between-group variation)
//!   est_g   ~ N(theta_g, se_g^2)  (within-group sampling error)
//! ```
//!
//! `tau^2` is estimated by the DerSimonian-Laird moment estimator, after which
//! each group's posterior mean is the precision-weighted blend of its own estimate
//! and the pooled mean. A group measured precisely (small `se_g`) barely moves; a
//! group measured poorly is pulled most of the way to `mu`. That is exactly the
//! behaviour sparse groups need.
//!
//! Deliberately independent of the GLM engine: the inputs are estimates, not data,
//! so this composes with *any* per-group fit — `GROUP BY ... poisson_fit_agg(...)`
//! or anything else that yields an estimate and a standard error.

use crate::errors::{StatsError, StatsResult};

/// How `tau^2` (the between-group variance) is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TauMethod {
    /// DerSimonian-Laird moment estimator. Closed form, no iteration, and what
    /// `metafor::rma(method = "DL")` computes.
    #[default]
    DerSimonianLaird,
    /// Treat `tau^2` as zero, i.e. complete pooling. Every group collapses onto
    /// the precision-weighted mean.
    None,
}

/// Options for empirical-Bayes shrinkage.
#[derive(Debug, Clone)]
pub struct EbShrinkOptions {
    pub method: TauMethod,
    /// A fixed `tau^2` instead of estimating one. Overrides `method` when set.
    pub tau_squared: Option<f64>,
}

impl Default for EbShrinkOptions {
    fn default() -> Self {
        Self {
            method: TauMethod::default(),
            tau_squared: None,
        }
    }
}

/// One group's shrunken estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShrunkenGroup {
    /// The estimate as supplied.
    pub estimate: f64,
    /// The standard error as supplied.
    pub se: f64,
    /// Posterior mean: the shrunken estimate.
    pub shrunken: f64,
    /// Posterior standard deviation.
    pub shrunken_se: f64,
    /// Shrinkage weight in `[0, 1]`: the share of the group's own estimate that
    /// survives. 1 means untouched, 0 means fully pooled.
    pub weight: f64,
}

/// Result of an empirical-Bayes shrinkage pass.
#[derive(Debug, Clone)]
pub struct EbShrinkResult {
    /// Precision-weighted pooled mean.
    pub mu: f64,
    /// Standard error of the pooled mean.
    pub mu_se: f64,
    /// Estimated between-group variance.
    pub tau_squared: f64,
    /// Share of total variance attributable to between-group variation, the
    /// familiar `I^2`. Zero when the groups are indistinguishable.
    pub i_squared: f64,
    /// Cochran's Q heterogeneity statistic.
    pub q: f64,
    pub n_groups: usize,
    /// One entry per input row, in input order.
    pub groups: Vec<ShrunkenGroup>,
}

/// Shrink per-group estimates toward their common mean.
///
/// `estimates` and `standard_errors` are parallel arrays, one entry per group.
/// Rows with a non-finite estimate, or a non-positive or non-finite standard
/// error, are dropped from the `tau^2` and `mu` calculations but still appear in
/// `groups` with `NaN` outputs, so the result stays aligned with the input.
pub fn eb_shrink(
    estimates: &[f64],
    standard_errors: &[f64],
    options: &EbShrinkOptions,
) -> StatsResult<EbShrinkResult> {
    if estimates.is_empty() {
        return Err(StatsError::EmptyInput { field: "estimate" });
    }
    if estimates.len() != standard_errors.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: estimates.len(),
            x_rows: standard_errors.len(),
        });
    }

    let usable: Vec<usize> = (0..estimates.len())
        .filter(|&i| {
            estimates[i].is_finite() && standard_errors[i].is_finite() && standard_errors[i] > 0.0
        })
        .collect();

    if usable.len() < 2 {
        return Err(StatsError::InsufficientData {
            rows: usable.len(),
            cols: 2,
        });
    }

    // Fixed-effect (inverse-variance) weights.
    let w: Vec<f64> = usable
        .iter()
        .map(|&i| 1.0 / (standard_errors[i] * standard_errors[i]))
        .collect();
    let sum_w: f64 = w.iter().sum();
    let sum_wy: f64 = usable
        .iter()
        .zip(w.iter())
        .map(|(&i, &wi)| wi * estimates[i])
        .sum();
    let fixed_mean = sum_wy / sum_w;

    // Cochran's Q about the fixed-effect mean.
    let q: f64 = usable
        .iter()
        .zip(w.iter())
        .map(|(&i, &wi)| wi * (estimates[i] - fixed_mean).powi(2))
        .sum();

    let k = usable.len() as f64;
    let df = k - 1.0;

    let tau_squared = match options.tau_squared {
        Some(v) => {
            if !v.is_finite() || v < 0.0 {
                return Err(StatsError::InvalidValue {
                    field: "tau_squared",
                    message: "tau_squared must be finite and non-negative".to_string(),
                });
            }
            v
        }
        None => match options.method {
            TauMethod::None => 0.0,
            TauMethod::DerSimonianLaird => {
                // tau^2 = max(0, (Q - df) / C), C = sum(w) - sum(w^2)/sum(w).
                let sum_w2: f64 = w.iter().map(|x| x * x).sum();
                let c = sum_w - sum_w2 / sum_w;
                if c > 0.0 {
                    ((q - df) / c).max(0.0)
                } else {
                    0.0
                }
            }
        },
    };

    // Random-effects weights and pooled mean.
    let wr: Vec<f64> = usable
        .iter()
        .map(|&i| 1.0 / (standard_errors[i] * standard_errors[i] + tau_squared))
        .collect();
    let sum_wr: f64 = wr.iter().sum();
    let mu: f64 = usable
        .iter()
        .zip(wr.iter())
        .map(|(&i, &wi)| wi * estimates[i])
        .sum::<f64>()
        / sum_wr;
    let mu_se = (1.0 / sum_wr).sqrt();

    let i_squared = if q > df && q > 0.0 {
        ((q - df) / q).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Per-group posteriors, over every input row so the output stays aligned.
    let mut groups = Vec::with_capacity(estimates.len());
    for i in 0..estimates.len() {
        let est = estimates[i];
        let se = standard_errors[i];
        if !est.is_finite() || !se.is_finite() || se <= 0.0 {
            groups.push(ShrunkenGroup {
                estimate: est,
                se,
                shrunken: f64::NAN,
                shrunken_se: f64::NAN,
                weight: f64::NAN,
            });
            continue;
        }

        let v = se * se;
        // Posterior precision is the sum of the two precisions. With tau^2 = 0 the
        // group's own estimate carries no weight at all and everything collapses to
        // the pooled mean.
        let (shrunken, shrunken_se, weight) = if tau_squared > 0.0 {
            let prec_within = 1.0 / v;
            let prec_between = 1.0 / tau_squared;
            let post_prec = prec_within + prec_between;
            let weight = prec_within / post_prec;
            (
                weight * est + (1.0 - weight) * mu,
                (1.0 / post_prec).sqrt(),
                weight,
            )
        } else {
            (mu, mu_se, 0.0)
        };

        groups.push(ShrunkenGroup {
            estimate: est,
            se,
            shrunken,
            shrunken_se,
            weight,
        });
    }

    Ok(EbShrinkResult {
        mu,
        mu_se,
        tau_squared,
        i_squared,
        q,
        n_groups: usable.len(),
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from a direct implementation of the DerSimonian-Laird
    /// formulas, which is what `metafor::rma(yi, sei, method = "DL")` computes.
    fn fixture() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.10, 0.30, 0.35, 0.65, 1.00],
            vec![0.30, 0.10, 0.50, 0.20, 0.40],
        )
    }

    #[test]
    fn matches_the_dersimonian_laird_formulas() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();

        // Recompute independently.
        let w: Vec<f64> = se.iter().map(|s| 1.0 / (s * s)).collect();
        let sw: f64 = w.iter().sum();
        let fixed: f64 = w.iter().zip(est.iter()).map(|(a, b)| a * b).sum::<f64>() / sw;
        let q: f64 = w
            .iter()
            .zip(est.iter())
            .map(|(a, b)| a * (b - fixed).powi(2))
            .sum();
        let sw2: f64 = w.iter().map(|a| a * a).sum();
        let c = sw - sw2 / sw;
        let tau2 = ((q - 4.0) / c).max(0.0);

        assert!((r.q - q).abs() < 1e-12, "Q {} vs {q}", r.q);
        assert!(
            (r.tau_squared - tau2).abs() < 1e-12,
            "tau2 {} vs {tau2}",
            r.tau_squared
        );

        let wr: Vec<f64> = se.iter().map(|s| 1.0 / (s * s + tau2)).collect();
        let swr: f64 = wr.iter().sum();
        let mu: f64 = wr.iter().zip(est.iter()).map(|(a, b)| a * b).sum::<f64>() / swr;
        assert!((r.mu - mu).abs() < 1e-12, "mu {} vs {mu}", r.mu);
        assert!((r.mu_se - (1.0 / swr).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn every_shrunken_value_lies_between_its_estimate_and_the_mean() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        for g in &r.groups {
            let lo = g.estimate.min(r.mu);
            let hi = g.estimate.max(r.mu);
            assert!(
                g.shrunken >= lo - 1e-12 && g.shrunken <= hi + 1e-12,
                "shrunken {} outside [{lo}, {hi}]",
                g.shrunken
            );
        }
    }

    #[test]
    fn noisier_groups_are_pulled_harder() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();

        // The weight retained by a group must decrease as its standard error grows.
        let mut pairs: Vec<(f64, f64)> = r.groups.iter().map(|g| (g.se, g.weight)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for pair in pairs.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "weight should fall as se rises: se {} -> w {}, se {} -> w {}",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1
            );
        }
    }

    #[test]
    fn shrinkage_always_reduces_the_standard_error() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        for g in &r.groups {
            assert!(
                g.shrunken_se <= g.se + 1e-12,
                "posterior SE {} should not exceed the input SE {}",
                g.shrunken_se,
                g.se
            );
        }
    }

    #[test]
    fn homogeneous_groups_collapse_to_complete_pooling() {
        // Identical estimates carry no between-group signal, so DL returns tau2 = 0
        // and every group is pulled all the way onto the mean.
        let est = vec![0.5; 6];
        let se = vec![0.2, 0.3, 0.1, 0.4, 0.25, 0.35];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();

        assert_eq!(r.tau_squared, 0.0);
        assert_eq!(r.i_squared, 0.0);
        for g in &r.groups {
            assert!((g.shrunken - r.mu).abs() < 1e-12);
            assert_eq!(g.weight, 0.0);
        }
    }

    #[test]
    fn a_large_tau_leaves_the_estimates_essentially_alone() {
        let (est, se) = fixture();
        let opts = EbShrinkOptions {
            tau_squared: Some(1e6),
            ..Default::default()
        };
        let r = eb_shrink(&est, &se, &opts).unwrap();
        for g in &r.groups {
            assert!(
                (g.shrunken - g.estimate).abs() < 1e-3,
                "with tau2 huge, {} should stay at {}",
                g.shrunken,
                g.estimate
            );
            assert!(g.weight > 0.999);
        }
    }

    #[test]
    fn heterogeneity_shows_up_in_i_squared() {
        // Widely separated estimates with tight standard errors.
        let est = vec![0.0, 1.0, 2.0, 3.0];
        let se = vec![0.05; 4];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        assert!(r.i_squared > 0.95, "I^2 = {}", r.i_squared);
        assert!(r.tau_squared > 0.5, "tau2 = {}", r.tau_squared);
    }

    #[test]
    fn the_none_method_pools_completely() {
        let (est, se) = fixture();
        let opts = EbShrinkOptions {
            method: TauMethod::None,
            ..Default::default()
        };
        let r = eb_shrink(&est, &se, &opts).unwrap();
        assert_eq!(r.tau_squared, 0.0);
        for g in &r.groups {
            assert!((g.shrunken - r.mu).abs() < 1e-12);
        }
    }

    #[test]
    fn unusable_rows_stay_in_place_as_nan() {
        let est = vec![0.1, f64::NAN, 0.3, 0.5];
        let se = vec![0.2, 0.2, -1.0, 0.3];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();

        assert_eq!(r.n_groups, 2, "only two rows are usable");
        assert_eq!(r.groups.len(), 4, "output must stay aligned with input");
        assert!(r.groups[1].shrunken.is_nan());
        assert!(r.groups[2].shrunken.is_nan());
        assert!(r.groups[0].shrunken.is_finite());
        assert!(r.groups[3].shrunken.is_finite());
    }

    #[test]
    fn a_single_group_is_rejected() {
        let err = eb_shrink(&[0.5], &[0.2], &EbShrinkOptions::default());
        assert!(matches!(err, Err(StatsError::InsufficientData { .. })));
    }

    #[test]
    fn empty_input_is_rejected() {
        let err = eb_shrink(&[], &[], &EbShrinkOptions::default());
        assert!(matches!(err, Err(StatsError::EmptyInput { .. })));
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let err = eb_shrink(&[0.1, 0.2], &[0.1], &EbShrinkOptions::default());
        assert!(matches!(err, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn a_negative_fixed_tau_is_rejected() {
        let (est, se) = fixture();
        let opts = EbShrinkOptions {
            tau_squared: Some(-1.0),
            ..Default::default()
        };
        assert!(matches!(
            eb_shrink(&est, &se, &opts),
            Err(StatsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn the_pooled_mean_sits_inside_the_range_of_the_estimates() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        let lo = est.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = est.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(r.mu > lo && r.mu < hi, "mu {} outside [{lo}, {hi}]", r.mu);
    }
}
