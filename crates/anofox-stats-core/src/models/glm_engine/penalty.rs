//! Penalty blocks derived from explicit priors.
//!
//! A Gaussian prior on a coefficient and a Gaussian random effect are the same
//! mathematical object: a quadratic precision block added to the penalized normal
//! equations. [`QuadraticPenalty`] is the seam both go through, so the mixed-effects
//! work in a later phase reuses this code rather than forking it.
//!
//! Laplace/L1 priors are deliberately *not* expressible as a [`QuadraticPenalty`] —
//! the objective is non-differentiable at the prior location. They are carried
//! separately in [`Penalty::l1`] and handled by the proximal path in [`super::irls`].

use crate::types::{PriorKind, PriorSpec};
use faer::{Col, Mat};

/// A quadratic (Gaussian) penalty contribution to the normal equations.
///
/// Implementors add `P` to `X'WX` and `P·mu0` to `X'Wz`, where `P` is the prior
/// precision matrix and `mu0` the prior mean. The dense diagonal case is
/// [`DiagonalPenalty`]; the mixed-effects phase adds a block-structured
/// implementation that never materializes `Z`.
pub trait QuadraticPenalty {
    /// Number of parameters this penalty applies to.
    fn n_params(&self) -> usize;

    /// Add the precision block `P` into `xtwx` and the shift `P·mu0` into `xtwz`.
    fn accumulate(&self, xtwx: &mut Mat<f64>, xtwz: &mut Col<f64>);

    /// True when the penalty contributes nothing (all precisions zero).
    fn is_zero(&self) -> bool;

    /// Per-parameter precision, used to build the augmented-design rows.
    fn precisions(&self) -> &[f64];

    /// Per-parameter prior mean.
    fn locations(&self) -> &[f64];
}

/// Diagonal Gaussian penalty: independent priors, one per coefficient.
///
/// This covers everything the SQL surface can express today (priors are given
/// per named feature, never as a covariance matrix).
#[derive(Debug, Clone, Default)]
pub struct DiagonalPenalty {
    precision: Vec<f64>,
    location: Vec<f64>,
}

impl DiagonalPenalty {
    /// Build a penalty of `n_params` entries with no contribution.
    pub fn zeros(n_params: usize) -> Self {
        Self {
            precision: vec![0.0; n_params],
            location: vec![0.0; n_params],
        }
    }

    /// Build from per-parameter priors, already positionally aligned with the
    /// design matrix columns (intercept first when one is fitted).
    pub fn from_priors(priors: &[PriorSpec]) -> Self {
        Self {
            precision: priors.iter().map(PriorSpec::precision).collect(),
            location: priors
                .iter()
                .map(|p| if p.is_flat() { 0.0 } else { p.loc })
                .collect(),
        }
    }

    /// Uniform ridge penalty `lambda` on every parameter except an optional
    /// leading intercept. This is the shape `glm_lambda` has always had, kept so
    /// the legacy option maps onto the same machinery as an explicit prior.
    pub fn ridge(n_params: usize, lambda: f64, skip_intercept: bool) -> Self {
        let start = usize::from(skip_intercept);
        let mut precision = vec![0.0; n_params];
        for p in precision.iter_mut().skip(start) {
            *p = lambda;
        }
        Self {
            precision,
            location: vec![0.0; n_params],
        }
    }

    /// Set one parameter's precision and location directly.
    pub fn set(&mut self, idx: usize, precision: f64, location: f64) {
        self.precision[idx] = precision;
        self.location[idx] = location;
    }
}

impl QuadraticPenalty for DiagonalPenalty {
    fn n_params(&self) -> usize {
        self.precision.len()
    }

    fn accumulate(&self, xtwx: &mut Mat<f64>, xtwz: &mut Col<f64>) {
        for j in 0..self.precision.len() {
            let p = self.precision[j];
            if p == 0.0 {
                continue;
            }
            xtwx[(j, j)] += p;
            xtwz[j] += p * self.location[j];
        }
    }

    fn is_zero(&self) -> bool {
        self.precision.iter().all(|&p| p == 0.0)
    }

    fn precisions(&self) -> &[f64] {
        &self.precision
    }

    fn locations(&self) -> &[f64] {
        &self.location
    }
}

/// The full penalty attached to a fit: a quadratic part plus an optional L1 part.
#[derive(Debug, Clone, Default)]
pub struct Penalty {
    /// Gaussian precisions and locations.
    pub quadratic: DiagonalPenalty,
    /// L1 weight per parameter (`1 / scale` for a Laplace prior, else zero).
    pub l1: Vec<f64>,
    /// Location of the L1 penalty per parameter (the Laplace prior's `loc`).
    pub l1_location: Vec<f64>,
}

impl Penalty {
    /// No penalty at all.
    pub fn none(n_params: usize) -> Self {
        Self {
            quadratic: DiagonalPenalty::zeros(n_params),
            l1: vec![0.0; n_params],
            l1_location: vec![0.0; n_params],
        }
    }

    /// Build from per-parameter priors aligned with the design columns.
    pub fn from_priors(priors: &[PriorSpec]) -> Self {
        let l1 = priors
            .iter()
            .map(|p| {
                if p.kind == PriorKind::Laplace && !p.is_flat() {
                    1.0 / p.scale
                } else {
                    0.0
                }
            })
            .collect();
        let l1_location = priors
            .iter()
            .map(|p| {
                if p.kind == PriorKind::Laplace && !p.is_flat() {
                    p.loc
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            quadratic: DiagonalPenalty::from_priors(priors),
            l1,
            l1_location,
        }
    }

    /// Legacy uniform ridge, equivalent to a `N(0, 1/sqrt(lambda))` prior on every
    /// non-intercept coefficient.
    pub fn ridge(n_params: usize, lambda: f64, skip_intercept: bool) -> Self {
        Self {
            quadratic: DiagonalPenalty::ridge(n_params, lambda, skip_intercept),
            l1: vec![0.0; n_params],
            l1_location: vec![0.0; n_params],
        }
    }

    pub fn n_params(&self) -> usize {
        self.quadratic.n_params()
    }

    /// True when neither the quadratic nor the L1 part contributes.
    pub fn is_zero(&self) -> bool {
        self.quadratic.is_zero() && self.l1.iter().all(|&w| w == 0.0)
    }

    /// True when an L1 term is present, requiring the proximal solver.
    pub fn has_l1(&self) -> bool {
        self.l1.iter().any(|&w| w != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_prior_precision_is_inverse_variance() {
        let p = PriorSpec::normal(0.0, 2.0);
        assert!((p.precision() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn flat_and_laplace_priors_have_no_quadratic_part() {
        assert_eq!(PriorSpec::flat().precision(), 0.0);
        assert_eq!(PriorSpec::laplace(0.0, 0.5).precision(), 0.0);
    }

    #[test]
    fn diagonal_penalty_accumulates_precision_and_shift() {
        let priors = vec![PriorSpec::flat(), PriorSpec::normal(1.0, 0.5)];
        let pen = DiagonalPenalty::from_priors(&priors);

        let mut xtwx: Mat<f64> = Mat::zeros(2, 2);
        let mut xtwz: Col<f64> = Col::zeros(2);
        pen.accumulate(&mut xtwx, &mut xtwz);

        // Flat prior touches nothing.
        assert_eq!(xtwx[(0, 0)], 0.0);
        assert_eq!(xtwz[0], 0.0);
        // N(1.0, 0.5) => precision 4, shift 4 * 1.0.
        assert!((xtwx[(1, 1)] - 4.0).abs() < 1e-12);
        assert!((xtwz[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn ridge_skips_the_intercept() {
        let pen = DiagonalPenalty::ridge(3, 2.0, true);
        assert_eq!(pen.precisions(), &[0.0, 2.0, 2.0]);
    }

    #[test]
    fn laplace_prior_populates_the_l1_part_only() {
        let pen = Penalty::from_priors(&[PriorSpec::laplace(0.0, 0.25)]);
        assert!(pen.quadratic.is_zero());
        assert!(pen.has_l1());
        assert!((pen.l1[0] - 4.0).abs() < 1e-12);
    }
}
