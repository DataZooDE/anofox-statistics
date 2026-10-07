//! Shared input-validation helpers for the model fit entry points.

use crate::errors::{StatsError, StatsResult};
use crate::types::{FitResultCore, FitResultInference};

/// Validate that every feature column has exactly `y_len` observations.
///
/// Every fit entry point must call this (or an equivalent check) before
/// indexing `x[j][i]` for `i < y_len`; otherwise a short column panics.
pub(crate) fn validate_x_columns(y_len: usize, x: &[Vec<f64>]) -> StatsResult<()> {
    for col in x.iter() {
        if col.len() != y_len {
            return Err(StatsError::DimensionMismatch {
                y_len,
                x_rows: col.len(),
            });
        }
    }
    Ok(())
}

/// Relative tolerance for treating a column as constant.
///
/// A column is constant when its range over the retained rows is at most
/// `CONSTANT_RANGE_RTOL * max|value|`. Being *relative* to the column's own
/// magnitude makes the test unit-free: a column measured in tiny units (values
/// around 1e-8) is NOT dropped, while a column like `1e6 + 1e-9 * noise` (whose
/// variation is at the level of floating-point round-off after centering) is.
/// 1e-12 is roughly 4500 ulps of relative precision — far above round-off from
/// arithmetic producing the column, far below any genuine variation.
pub(crate) const CONSTANT_RANGE_RTOL: f64 = 1e-12;

/// Returns `true` when `col` (restricted to `rows`) is numerically constant.
pub(crate) fn is_constant_over(col: &[f64], rows: &[usize]) -> bool {
    if rows.is_empty() {
        return true;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut max_abs = 0.0_f64;
    for &i in rows {
        let v = col[i];
        lo = lo.min(v);
        hi = hi.max(v);
        max_abs = max_abs.max(v.abs());
    }
    (hi - lo) <= CONSTANT_RANGE_RTOL * max_abs
}

/// Returns `true` when `col` (restricted to `rows`) is identically zero.
fn is_all_zero_over(col: &[f64], rows: &[usize]) -> bool {
    rows.iter().all(|&i| col[i] == 0.0)
}

/// Decide, per column, whether it must be dropped from the design.
///
/// * With an intercept, a constant column is perfectly collinear with the
///   intercept and is dropped (its coefficient is reported as NaN).
/// * WITHOUT an intercept a constant non-zero column (e.g. all ones) *is* the
///   intercept and must be kept. Only identically-zero columns, which carry no
///   information under any scaling, are dropped.
pub(crate) fn droppable_columns(x: &[Vec<f64>], rows: &[usize], fit_intercept: bool) -> Vec<bool> {
    x.iter()
        .map(|col| is_droppable_column(col, rows, fit_intercept))
        .collect()
}

/// Single-column form of [`droppable_columns`].
pub(crate) fn is_droppable_column(col: &[f64], rows: &[usize], fit_intercept: bool) -> bool {
    if fit_intercept {
        is_constant_over(col, rows)
    } else {
        is_all_zero_over(col, rows)
    }
}

/// Inference statistics for an intercept term (NaN when unavailable).
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterceptStats {
    pub estimate: f64,
    pub std_error: f64,
    pub t_value: f64,
    pub p_value: f64,
    pub ci: (f64, f64),
}

/// For a no-intercept fit, find the first retained column that is constant
/// over `rows`. Returns `(column index, constant value)`.
///
/// The upstream solvers drop constant columns even without an intercept. A
/// no-intercept model containing a constant column `c·1` spans exactly the same
/// column space as an intercept model without that column, so the wrappers fit
/// the latter and fold the intercept back into the column's coefficient
/// (`beta_c = intercept / c`) via [`fold_pseudo_intercept`]. Any further
/// constant columns are perfectly collinear with the first and are reported as
/// aliased (NaN), as R's `lm` does.
pub(crate) fn find_pseudo_intercept(
    x: &[Vec<f64>],
    rows: &[usize],
    retained: &[usize],
) -> Option<(usize, f64)> {
    retained
        .iter()
        .copied()
        .find(|&j| is_constant_over(&x[j], rows))
        .map(|j| (j, x[j][rows[0]]))
}

/// Fold an intercept fitted in place of constant column `col` (value `c`) back
/// into that column's coefficient and inference entries. `core.intercept` must
/// already be `None`.
pub(crate) fn fold_pseudo_intercept(
    core: &mut FitResultCore,
    inference: Option<&mut FitResultInference>,
    col: usize,
    c: f64,
    stats: InterceptStats,
) {
    core.coefficients[col] = stats.estimate / c;
    if let Some(inf) = inference {
        inf.std_errors[col] = stats.std_error / c.abs();
        inf.t_values[col] = stats.t_value * c.signum();
        inf.p_values[col] = stats.p_value;
        let (a, b) = (stats.ci.0 / c, stats.ci.1 / c);
        inf.ci_lower[col] = a.min(b);
        inf.ci_upper[col] = a.max(b);
    }
}

/// Replace R², adjusted R² and the F test of a model fitted WITHOUT an intercept
/// by R's `summary.lm` convention for such models: the total sum of squares is
/// uncentered,
///
/// ```text
/// mss = sum(w * f^2), rss = sum(w * (y - f)^2), R² = mss / (mss + rss)
/// adj R² = 1 - (1 - R²) * n / (n - rank),  F = (mss / rank) / (rss / (n - rank))
/// ```
///
/// where `f` are the fitted values, `rank` the number of estimated (non-NaN)
/// coefficients and `w` the weights (1 for OLS). The centered TSS used before
/// compares a no-intercept model with an intercept-only model it does not nest.
pub(crate) fn apply_no_intercept_fit_stats(
    core: &mut FitResultCore,
    inference: Option<&mut FitResultInference>,
    y: &[f64],
    x: &[Vec<f64>],
    weights: Option<&[f64]>,
    rows: &[usize],
) {
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};
    let n = rows.len();
    let active: Vec<(usize, f64)> = core
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_finite())
        .map(|(j, &c)| (j, c))
        .collect();
    let rank = active.len();
    let (mut mss, mut rss) = (0.0, 0.0);
    for &i in rows {
        let f: f64 = active.iter().map(|&(j, c)| c * x[j][i]).sum();
        let w = weights.map_or(1.0, |w| w[i]);
        mss += w * f * f;
        rss += w * (y[i] - f).powi(2);
    }
    let rdf = n.saturating_sub(rank);
    let r2 = if mss + rss > 0.0 {
        mss / (mss + rss)
    } else {
        f64::NAN
    };
    core.r_squared = r2;
    core.adj_r_squared = if rdf > 0 {
        1.0 - (1.0 - r2) * n as f64 / rdf as f64
    } else {
        f64::NAN
    };
    if let Some(inf) = inference {
        let (f_stat, f_p) = if rank > 0 && rdf > 0 && rss > 0.0 {
            let f = (mss / rank as f64) / (rss / rdf as f64);
            let p = FisherSnedecor::new(rank as f64, rdf as f64)
                .map(|d| d.sf(f))
                .unwrap_or(f64::NAN);
            (f, p)
        } else {
            (f64::NAN, f64::NAN)
        };
        inf.f_statistic = Some(f_stat);
        inf.f_pvalue = Some(f_p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_unit_column_is_not_constant() {
        let col = vec![1e-8, 2e-8, 3e-8, 4e-8];
        assert!(!is_constant_over(&col, &[0, 1, 2, 3]));
    }

    #[test]
    fn exactly_constant_column_is_constant() {
        let col = vec![5.0; 4];
        assert!(is_constant_over(&col, &[0, 1, 2, 3]));
        let zeros = vec![0.0; 4];
        assert!(is_constant_over(&zeros, &[0, 1, 2, 3]));
    }

    #[test]
    fn no_intercept_keeps_constant_nonzero_columns() {
        let x = vec![vec![1.0; 4], vec![0.0; 4], vec![1.0, 2.0, 3.0, 4.0]];
        let rows = [0, 1, 2, 3];
        assert_eq!(
            droppable_columns(&x, &rows, false),
            vec![false, true, false]
        );
        assert_eq!(droppable_columns(&x, &rows, true), vec![true, true, false]);
    }

    #[test]
    fn mismatched_columns_error() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert!(matches!(
            validate_x_columns(3, &x),
            Err(StatsError::DimensionMismatch {
                y_len: 3,
                x_rows: 2
            })
        ));
    }
}
