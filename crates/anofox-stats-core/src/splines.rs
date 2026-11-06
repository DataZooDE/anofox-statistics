//! Regression-spline bases (#317).
//!
//! Thin wrappers around upstream `anofox_regression::splines`: restricted cubic
//! splines (`Rcs`, `Hmisc::rcspline.eval` / `rms::rcs`), linear splines (`Lsp`,
//! `rms::lsp`), B-splines (`BSpline`, scikit-learn `SplineTransformer`), the
//! default knots of `rcspline.eval` and the restricted interaction `rms::%ia%`.
//! The wrapper only translates the extension's spec (kind, knots and options)
//! into the upstream types and the upstream errors into [`StatsError`].

use crate::{StatsError, StatsResult};
use anofox_regression::splines::{
    self as up, BSpline, Basis, KnotPlacement, Lsp, Rcs, RcsNorm, SplineError, SplineExtrapolation,
};
use faer::Mat;

/// Kind of spline basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineKind {
    /// Restricted cubic spline (`rms::rcs`).
    Rcs,
    /// Linear spline (`rms::lsp`).
    Lsp,
    /// B-spline (sklearn `SplineTransformer`).
    BSpline,
}

/// Extrapolation of a B-spline outside its base knots (sklearn names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrapolation {
    Constant,
    Linear,
    Continue,
    Periodic,
    Error,
}

/// Everything that defines a spline basis.
#[derive(Debug, Clone, PartialEq)]
pub struct SplineSpec {
    pub kind: SplineKind,
    /// Knots (rcs, lsp) or base knots (bspline).
    pub knots: Vec<f64>,
    /// `norm` of `rcspline.eval` (0, 1, 2); rcs only.
    pub norm: i32,
    /// Degree; bspline only.
    pub degree: usize,
    /// Extrapolation; bspline only.
    pub extrapolation: Extrapolation,
    /// Keep the last B-spline (sklearn `include_bias`); bspline only.
    pub include_bias: bool,
}

/// A validated upstream spline basis.
#[derive(Debug, Clone)]
pub enum SplineBasis {
    Rcs(Rcs),
    Lsp(Lsp),
    BSpline(BSpline),
}

/// Upstream spline errors: data that cannot support default knots is a
/// degenerate outcome (NULL in SQL), anything else is an input error.
fn map_err(e: SplineError) -> StatsError {
    match e {
        SplineError::TooFewObservations | SplineError::TooFewUniqueKnots => {
            StatsError::InsufficientDataMsg(e.to_string())
        }
        other => StatsError::InvalidInput(other.to_string()),
    }
}

fn rcs_norm(norm: i32) -> StatsResult<RcsNorm> {
    match norm {
        0 => Ok(RcsNorm::None),
        1 => Ok(RcsNorm::LastInterval),
        2 => Ok(RcsNorm::Range),
        other => Err(StatsError::InvalidInput(format!(
            "norm must be 0, 1 or 2, got {}",
            other
        ))),
    }
}

fn extrapolation(e: Extrapolation) -> SplineExtrapolation {
    match e {
        Extrapolation::Constant => SplineExtrapolation::Constant,
        Extrapolation::Linear => SplineExtrapolation::Linear,
        Extrapolation::Continue => SplineExtrapolation::Continue,
        Extrapolation::Periodic => SplineExtrapolation::Periodic,
        Extrapolation::Error => SplineExtrapolation::Error,
    }
}

impl SplineBasis {
    /// Build the upstream basis of a spec (upstream validates the knots).
    pub fn new(spec: &SplineSpec) -> StatsResult<Self> {
        let basis = match spec.kind {
            SplineKind::Rcs => SplineBasis::Rcs(
                Rcs::new(&spec.knots)
                    .map_err(map_err)?
                    .with_norm(rcs_norm(spec.norm)?),
            ),
            SplineKind::Lsp => SplineBasis::Lsp(Lsp::new(&spec.knots).map_err(map_err)?),
            SplineKind::BSpline => SplineBasis::BSpline(
                BSpline::new(&spec.knots, spec.degree)
                    .map_err(map_err)?
                    .with_extrapolation(extrapolation(spec.extrapolation))
                    .with_bias(spec.include_bias),
            ),
        };
        // Surface configuration errors (periodic degree) at construction.
        basis.basis(&[])?;
        Ok(basis)
    }

    /// The knots as upstream keeps them (rcs / lsp: sorted, de-duplicated;
    /// bspline: the base knots).
    pub fn knots(&self) -> Vec<f64> {
        match self {
            SplineBasis::Rcs(r) => r.knots().to_vec(),
            SplineBasis::Lsp(l) => l.knots().to_vec(),
            SplineBasis::BSpline(b) => b.base_knots().to_vec(),
        }
    }

    fn basis(&self, x: &[f64]) -> StatsResult<Basis> {
        Ok(match self {
            SplineBasis::Rcs(r) => r.basis(x, "x"),
            SplineBasis::Lsp(l) => l.basis(x, "x"),
            SplineBasis::BSpline(b) => b.basis(x, "x").map_err(map_err)?,
        })
    }

    /// Number of basis columns.
    pub fn n_columns(&self) -> usize {
        match self {
            SplineBasis::Rcs(r) => r.n_columns(),
            SplineBasis::Lsp(l) => l.basis(&[], "x").n_columns(),
            SplineBasis::BSpline(b) => b.n_columns(),
        }
    }

    /// Nonlinearity flag of every column (upstream `Basis::nonlinear`).
    pub fn nonlinear(&self) -> StatsResult<Vec<bool>> {
        Ok(self.basis(&[])?.nonlinear)
    }

    /// The basis row at one value (NaN row for a non-finite `x`, as upstream).
    pub fn row(&self, x: f64) -> StatsResult<Vec<f64>> {
        let b = self.basis(&[x])?;
        Ok((0..b.n_columns()).map(|j| b.columns[(0, j)]).collect())
    }
}

/// Default knots of a restricted cubic spline with `nk` knots
/// (`Hmisc::rcspline.eval(x, nk, knots.only = TRUE, fractied)`); non-finite
/// values are ignored.
pub fn rcs_default_knots(x: &[f64], nk: usize, fractied: f64) -> StatsResult<Vec<f64>> {
    // anofox-regression 0.11.0 panics on a single distinct value (it slices
    // `xu[1..0]`); report it as the "fewer than 3 unique knots" outcome.
    let mut finite = x.iter().copied().filter(|v| v.is_finite());
    if let Some(first) = finite.next() {
        if nk >= 3 && x.len() >= 6 && finite.all(|v| v == first) {
            return Err(map_err(SplineError::TooFewUniqueKnots));
        }
    }
    up::rcs_default_knots_with(x, nk, fractied).map_err(map_err)
}

/// Base knots of a B-spline placed on the finite values of `x` (sklearn
/// `n_knots` with `knots = 'uniform' | 'quantile'`).
pub fn bspline_data_knots(x: &[f64], n_knots: usize, quantile: bool) -> StatsResult<Vec<f64>> {
    let placement = if quantile {
        KnotPlacement::Quantile
    } else {
        KnotPlacement::Uniform
    };
    match BSpline::from_data(x, n_knots, 3, placement) {
        Ok(b) => Ok(b.base_knots().to_vec()),
        // Constant data (or heavy ties at quantile knots) cannot place distinct
        // knots: a degenerate outcome, not an input error.
        Err(SplineError::UnsortedKnots) => Err(StatsError::InsufficientDataMsg(
            "the data do not give distinct knots".to_string(),
        )),
        Err(e) => Err(map_err(e)),
    }
}

/// Knot quantiles of RMS Table 2.3 for `nk` knots.
pub fn default_knot_quantiles(nk: usize) -> StatsResult<Vec<f64>> {
    if nk < 3 {
        return Err(StatsError::InvalidInput(format!(
            "need at least 3 knots, got {}",
            nk
        )));
    }
    Ok(up::default_knot_quantiles(nk))
}

/// Restricted interaction (`rms::%ia%`) of two basis rows with their
/// nonlinearity flags. Returns the product row and its flags.
pub fn restricted_interaction(
    a: &[f64],
    a_nonlinear: &[bool],
    b: &[f64],
    b_nonlinear: &[bool],
) -> StatsResult<(Vec<f64>, Vec<bool>)> {
    let basis = |v: &[f64], nl: &[bool]| {
        Basis::new(
            Mat::from_fn(1, v.len(), |_, j| v[j]),
            (0..v.len()).map(|j| format!("c{}", j)).collect(),
            nl.to_vec(),
        )
        .map_err(map_err)
    };
    let p = up::restricted_interaction(&basis(a, a_nonlinear)?, &basis(b, b_nonlinear)?)
        .map_err(map_err)?;
    let row = (0..p.n_columns()).map(|j| p.columns[(0, j)]).collect();
    Ok((row, p.nonlinear))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn spec(kind: SplineKind, knots: &[f64]) -> SplineSpec {
        SplineSpec {
            kind,
            knots: knots.to_vec(),
            norm: 2,
            degree: 3,
            extrapolation: Extrapolation::Constant,
            include_bias: true,
        }
    }

    #[test]
    fn rcs_matches_hmisc() {
        // Hmisc::rcspline.eval(2.5, knots = c(0, 1, 3, 4, 6), inclx = TRUE)
        let b = SplineBasis::new(&spec(SplineKind::Rcs, &[0.0, 1.0, 3.0, 4.0, 6.0])).unwrap();
        assert_eq!(b.n_columns(), 4);
        assert_eq!(b.nonlinear().unwrap(), vec![false, true, true, true]);
        let row = b.row(2.5).unwrap();
        assert_relative_eq!(row[0], 2.5);
        // (2.5/36^(1/3))^3 = 15.625 / 36 (only the first knot is passed)
        assert_relative_eq!(row[1], 15.625 / 36.0, epsilon = 1e-12);
        assert_relative_eq!(row[2], 1.5_f64.powi(3) / 36.0, epsilon = 1e-12);
        assert_relative_eq!(row[3], 0.0);
    }

    #[test]
    fn lsp_and_bspline_shapes() {
        let l = SplineBasis::new(&spec(SplineKind::Lsp, &[2.0, 1.0])).unwrap();
        assert_eq!(l.knots(), vec![1.0, 2.0]);
        assert_eq!(l.row(3.0).unwrap(), vec![3.0, 2.0, 1.0]);
        let b = SplineBasis::new(&spec(SplineKind::BSpline, &[0.0, 1.0, 2.0])).unwrap();
        assert_eq!(b.n_columns(), 5);
        let s: f64 = b.row(0.7).unwrap().iter().sum();
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn errors_map_to_input_or_degenerate() {
        assert!(matches!(
            SplineBasis::new(&spec(SplineKind::Rcs, &[1.0, 2.0])),
            Err(StatsError::InvalidInput(_))
        ));
        let mut s = spec(SplineKind::BSpline, &[0.0, 1.0]);
        s.extrapolation = Extrapolation::Periodic;
        assert!(matches!(
            SplineBasis::new(&s),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            rcs_default_knots(&[1.0, 2.0, 3.0], 3, 0.05),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            bspline_data_knots(&[1.0; 10], 4, false),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(default_knot_quantiles(2).is_err());
        // A constant column (upstream 0.11.0 panics here).
        assert!(matches!(
            rcs_default_knots(&[1.0; 20], 3, 0.05),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }

    #[test]
    fn interaction_drops_doubly_nonlinear_terms() {
        let (row, nl) = restricted_interaction(
            &[2.0, 0.5, 0.25],
            &[false, true, true],
            &[3.0, 0.1],
            &[false, true],
        )
        .unwrap();
        assert_eq!(row.len(), 4);
        assert_relative_eq!(row[0], 6.0);
        assert_relative_eq!(row[1], 0.2);
        assert_relative_eq!(row[2], 1.5);
        assert_relative_eq!(row[3], 0.75);
        assert_eq!(nl, vec![false, true, true, true]);
    }
}
