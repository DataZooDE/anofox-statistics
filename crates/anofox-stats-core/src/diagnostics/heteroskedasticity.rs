//! Breusch-Pagan and White tests for heteroskedasticity (#155)
//!
//! Delegates to `anofox_regression::diagnostics::{breusch_pagan, white_test}`
//! (R `lmtest::bptest`).

use crate::errors::{StatsError, StatsResult};
use crate::tests::UnifiedTestResult;
use anofox_regression::diagnostics as reg;
use faer::{Col, Mat};

fn design(residuals: &[f64], x: &[Vec<f64>]) -> StatsResult<(Col<f64>, Mat<f64>)> {
    let n = residuals.len();
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if let Some(col) = x.iter().find(|c| c.len() != n) {
        return Err(StatsError::DimensionMismatch {
            y_len: n,
            x_rows: col.len(),
        });
    }
    Ok((
        Col::from_fn(n, |i| residuals[i]),
        Mat::from_fn(n, x.len(), |i, j| x[j][i]),
    ))
}

fn to_unified(t: reg::HeteroskedasticityTest, n: usize) -> UnifiedTestResult {
    // R's method strings (lmtest::bptest).
    let method = match t.method {
        "breusch_pagan_koenker" => "studentized Breusch-Pagan test",
        "breusch_pagan" => "Breusch-Pagan test",
        _ => "White test",
    };
    let mut u = UnifiedTestResult::new(method);
    u.statistic = Some(t.statistic);
    u.df1 = Some(t.df);
    u.p_value = Some(t.p_value);
    u.n = Some(n);
    u
}

/// Breusch-Pagan test of `residuals` against the regressors `x` (column-major,
/// no intercept column). `studentize = true` is Koenker's variant, the
/// `bptest` default.
pub fn breusch_pagan(
    residuals: &[f64],
    x: &[Vec<f64>],
    studentize: bool,
) -> StatsResult<UnifiedTestResult> {
    let (e, z) = design(residuals, x)?;
    let t = reg::breusch_pagan(&e, &z, studentize).map_err(StatsError::from)?;
    Ok(to_unified(t, residuals.len()))
}

/// White's test: the studentized Breusch-Pagan test on `x`, its squares and
/// cross-products.
pub fn white_test(residuals: &[f64], x: &[Vec<f64>]) -> StatsResult<UnifiedTestResult> {
    let (e, z) = design(residuals, x)?;
    let t = reg::white_test(&e, &z).map_err(StatsError::from)?;
    Ok(to_unified(t, residuals.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_has_more_df_than_breusch_pagan() {
        let x1: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x2: Vec<f64> = (0..30).map(|i| ((i * 7) % 11) as f64).collect();
        let e: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.37).sin() * (1.0 + i as f64 / 10.0))
            .collect();
        let bp = breusch_pagan(&e, &[x1.clone(), x2.clone()], true).unwrap();
        let w = white_test(&e, &[x1, x2]).unwrap();
        assert_eq!(bp.df1, Some(2.0));
        assert_eq!(w.df1, Some(5.0));
        assert_eq!(bp.method, "studentized Breusch-Pagan test");
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(matches!(
            breusch_pagan(&[1.0, 2.0, 3.0], &[vec![1.0, 2.0]], true),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }
}
