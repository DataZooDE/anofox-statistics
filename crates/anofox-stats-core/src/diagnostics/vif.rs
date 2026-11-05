//! Variance inflation factors: the ordinary VIF and the generalized VIF (GVIF)
//! of Fox & Monette (1992). The math is anofox-regression's
//! (`diagnostics::variance_inflation_factor` / `diagnostics::generalized_vif_terms`);
//! this module only validates the feature columns, drops incomplete rows and
//! builds the design matrix.

use crate::errors::{StatsError, StatsResult};
pub use anofox_regression::diagnostics::GvifTerm;
use anofox_regression::diagnostics::{generalized_vif_terms, variance_inflation_factor};
use faer::Mat;

/// Validate the feature columns and return the design matrix of the complete
/// rows (every column finite).
fn complete_design(x: &[Vec<f64>]) -> StatsResult<Mat<f64>> {
    if x.is_empty() {
        return Err(StatsError::InvalidInput("No features provided".into()));
    }
    let n_obs = x[0].len();
    if n_obs == 0 {
        return Err(StatsError::InsufficientDataMsg(
            "Empty feature arrays".into(),
        ));
    }
    for (i, col) in x.iter().enumerate() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Feature {} has {} observations, expected {}",
                i,
                col.len(),
                n_obs
            )));
        }
    }
    let rows: Vec<usize> = (0..n_obs)
        .filter(|&i| x.iter().all(|col| col[i].is_finite()))
        .collect();
    // Without a single complete row the result would be meaningless.
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    Ok(Mat::from_fn(rows.len(), x.len(), |i, j| x[j][rows[i]]))
}

/// Compute VIF (Variance Inflation Factor) for each feature.
///
/// For each feature j, `VIF_j = 1 / (1 - R²_j)` with `R²_j` from regressing
/// feature j on all other features (with an intercept), as `car::vif` reports
/// it for an `lm` fit. Rows with a non-finite value in any feature are left
/// out. Degenerate designs follow upstream: `inf` for a column that is
/// aliased with the intercept and the other columns (a constant column, every
/// column when there are no more rows than features), a finite VIF on the
/// span of the remaining columns for the others.
///
/// # Arguments
/// * `x` - Feature arrays, each Vec<f64> is one feature column
///
/// # Returns
/// * Vector of VIF values, one per feature
pub fn compute_vif(x: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let design = complete_design(x)?;
    Ok(variance_inflation_factor(&design).iter().copied().collect())
}

/// Result of [`compute_gvif`].
#[derive(Debug, Clone, PartialEq)]
pub struct GvifResult {
    /// One entry per term, in term-index order: GVIF, `Df` and
    /// `GVIF^(1/(2 Df))` as `car::vif` reports them.
    pub terms: Vec<GvifTerm>,
    /// Number of complete rows used.
    pub n_observations: usize,
}

/// Compute the generalized VIF (Fox & Monette 1992) per term, as `car::vif`
/// reports it for a model with multi-column terms (e.g. factor dummies).
///
/// `term_of_column[j]` is the term index (`0..n_terms`) of feature column j;
/// the columns of a term need not be adjacent. Every term must have at least
/// one column. Rows with a non-finite value in any feature are left out.
pub fn compute_gvif(
    x: &[Vec<f64>],
    term_of_column: &[usize],
    n_terms: usize,
) -> StatsResult<GvifResult> {
    if term_of_column.len() != x.len() {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "{} term indices for {} feature columns",
            term_of_column.len(),
            x.len()
        )));
    }
    if n_terms == 0 {
        return Err(StatsError::InvalidInput("No terms provided".into()));
    }
    let mut sizes = vec![0usize; n_terms];
    for &t in term_of_column {
        if t >= n_terms {
            return Err(StatsError::InvalidInput(format!(
                "term index {} out of range for {} terms",
                t, n_terms
            )));
        }
        sizes[t] += 1;
    }
    if let Some(t) = sizes.iter().position(|&s| s == 0) {
        return Err(StatsError::InvalidInput(format!(
            "term {} has no columns",
            t
        )));
    }
    let design = complete_design(x)?;
    // Upstream takes the groups as runs of consecutive columns: order the
    // columns by term (stable, so each term keeps its column order).
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by_key(|&j| term_of_column[j]);
    let grouped = Mat::from_fn(design.nrows(), design.ncols(), |i, j| design[(i, order[j])]);
    Ok(GvifResult {
        terms: generalized_vif_terms(&grouped, &sizes),
        n_observations: design.nrows(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vif_single_feature() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_vif_uncorrelated_features() {
        // Two uncorrelated features should have VIF ≈ 1
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![5.0, 3.0, 1.0, 4.0, 2.0]; // Random order
        let x = vec![x1, x2];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 2);
        // VIF should be close to 1 for uncorrelated features
        assert!(result[0] < 2.0);
        assert!(result[1] < 2.0);
    }

    #[test]
    fn test_vif_perfectly_correlated() {
        // x2 = 2 * x1, perfect collinearity
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x = vec![x1, x2];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 2);
        // VIF should be very high (or infinite) for perfectly correlated features
        assert!(result[0] > 1000.0 || result[0].is_infinite());
        assert!(result[1] > 1000.0 || result[1].is_infinite());
    }

    #[test]
    fn test_vif_moderate_correlation() {
        // Features with moderate correlation
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![1.1, 2.2, 2.8, 4.1, 5.0, 5.9, 7.2, 8.0]; // Correlated with noise
        let x3 = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]; // Negatively correlated
        let x = vec![x1, x2, x3];

        let result = compute_vif(&x).unwrap();
        assert_eq!(result.len(), 3);
        // All VIFs should be > 1 due to correlations
        for vif in &result {
            assert!(*vif >= 1.0);
        }
    }

    #[test]
    fn test_vif_saturated_is_infinite() {
        // n <= p: every auxiliary regression fits exactly.
        let x = vec![vec![1.0, 2.0], vec![3.0, 1.0]];
        assert!(compute_vif(&x).unwrap().iter().all(|v| v.is_infinite()));
    }

    #[test]
    fn test_vif_constant_column_only_that_one_infinite() {
        let x = gvif_data();
        let mut with_const = x[..3].to_vec();
        with_const.push(vec![2.0; 30]);
        let vif = compute_vif(&with_const).unwrap();
        let reference = compute_vif(&x[..3]).unwrap();
        assert!(vif[3].is_infinite());
        for (v, r) in vif[..3].iter().zip(&reference) {
            assert!((v - r).abs() < 1e-9 * r);
        }
        let gvif = compute_gvif(&with_const, &[0, 0, 1, 2], 3).unwrap();
        assert!(gvif.terms[0].gvif.is_finite() && gvif.terms[1].gvif.is_finite());
        assert!(gvif.terms[2].gvif.is_infinite() && gvif.terms[2].adjusted_gvif.is_infinite());
    }

    #[test]
    fn test_vif_drops_incomplete_rows() {
        let x = vec![
            vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0],
            vec![5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 1.0],
            vec![2.0, 1.0, 3.0, 1.0, 5.0, 2.0, 0.5],
        ];
        let clean: Vec<Vec<f64>> = x
            .iter()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 3)
                    .map(|(_, v)| *v)
                    .collect()
            })
            .collect();
        assert_eq!(compute_vif(&x).unwrap(), compute_vif(&clean).unwrap());
    }

    fn gvif_data() -> Vec<Vec<f64>> {
        let n = 30;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 1.3).cos() + 0.5 * a[i])
            .collect();
        let c: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 + b[i]).collect();
        let d: Vec<f64> = (0..n).map(|i| ((i * 3) % 4) as f64).collect();
        vec![a, b, c, d]
    }

    #[test]
    fn test_gvif_single_columns_match_vif() {
        let x = gvif_data();
        let vif = compute_vif(&x).unwrap();
        let gvif = compute_gvif(&x, &[0, 1, 2, 3], 4).unwrap();
        assert_eq!(gvif.n_observations, 30);
        for (g, v) in gvif.terms.iter().zip(&vif) {
            assert_eq!(g.df, 1);
            assert!((g.gvif - v).abs() < 1e-9 * v);
            assert!((g.adjusted_gvif - v.sqrt()).abs() < 1e-9 * v);
        }
    }

    #[test]
    fn test_gvif_non_adjacent_terms() {
        // Term 1 = columns 1 and 3; the same as passing them adjacent.
        let x = gvif_data();
        let a = compute_gvif(&x, &[0, 1, 2, 1], 3).unwrap();
        let swapped = vec![x[0].clone(), x[1].clone(), x[3].clone(), x[2].clone()];
        let b = compute_gvif(&swapped, &[0, 1, 1, 2], 3).unwrap();
        assert_eq!(a.terms[1].df, 2);
        for (s, t) in a.terms.iter().zip(&b.terms) {
            assert!((s.gvif - t.gvif).abs() < 1e-12 * t.gvif);
        }
    }

    #[test]
    fn test_gvif_invalid_terms() {
        let x = gvif_data();
        assert!(compute_gvif(&x, &[0, 1, 2], 3).is_err());
        assert!(compute_gvif(&x, &[0, 1, 2, 5], 3).is_err());
        assert!(compute_gvif(&x, &[0, 0, 2, 2], 3).is_err());
        assert!(compute_gvif(&x, &[0, 0, 0, 0], 0).is_err());
    }

    #[test]
    fn test_vif_empty_input() {
        let x: Vec<Vec<f64>> = vec![];
        assert!(compute_vif(&x).is_err());
    }
}
