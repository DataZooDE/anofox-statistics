//! Partial and semi-partial correlation (issue #296).
//!
//! Thin wrappers around upstream `anofox_statistics::correlation::{partial_cor,
//! semi_partial_cor}` (R `ppcor::pcor.test` / `ppcor::spcor.test`). The wrapper
//! drops incomplete rows, rank-transforms every column with the upstream `rank`
//! (average ties) for `method = spearman` exactly as ppcor does, and reports
//! degenerate samples (a constant column) as insufficient data.

use super::{convert_error, Alternative, UnifiedTestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{partial_cor as lib_partial_cor, rank as lib_rank, semi_partial_cor};

/// Which coefficient to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialCorKind {
    /// Controls removed from both x and y (`ppcor::pcor.test`).
    Partial,
    /// Controls removed from y only (`ppcor::spcor.test`).
    SemiPartial,
}

/// Correlation flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialCorMethod {
    /// Product-moment correlation of the raw values.
    Pearson,
    /// Product-moment correlation of the ranks (average ties).
    Spearman,
}

/// Result of a partial / semi-partial correlation test.
#[derive(Debug, Clone)]
pub struct PartialCorrelationResult {
    /// Partial (or semi-partial) correlation coefficient
    pub r: f64,
    /// t statistic, `r * sqrt(df / (1 - r^2))`
    pub statistic: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Degrees of freedom, `n - 2 - n_controls`
    pub df: f64,
    /// Complete rows used
    pub n: usize,
    /// Number of control variables
    pub n_controls: usize,
    /// Method name
    pub method: String,
    /// Unified result (contract `test` row), from the upstream result
    pub unified: UnifiedTestResult,
}

fn is_constant(v: &[f64]) -> bool {
    v.iter().all(|&a| a == v[0])
}

/// Partial (`kind = Partial`) or semi-partial (`kind = SemiPartial`)
/// correlation of `x` and `y` controlling for the columns of `z`. Rows with a
/// NaN in any column are dropped.
pub fn partial_correlation(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
    kind: PartialCorKind,
    method: PartialCorMethod,
) -> StatsResult<PartialCorrelationResult> {
    if x.len() != y.len() || z.iter().any(|zi| zi.len() != x.len()) {
        return Err(StatsError::DimensionMismatchMsg(
            "Partial correlation requires x, y and every control of equal length".into(),
        ));
    }
    let k = z.len();
    let rows: Vec<usize> = (0..x.len())
        .filter(|&i| !x[i].is_nan() && !y[i].is_nan() && z.iter().all(|zi| !zi[i].is_nan()))
        .collect();
    if rows.len() < k + 3 {
        return Err(StatsError::InsufficientDataMsg(format!(
            "Partial correlation with {k} control(s) needs at least {} complete rows, got {}",
            k + 3,
            rows.len()
        )));
    }
    let pick = |v: &[f64]| -> Vec<f64> { rows.iter().map(|&i| v[i]).collect() };
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(k + 2);
    columns.push(pick(x));
    columns.push(pick(y));
    columns.extend(z.iter().map(|zi| pick(zi)));

    if columns.iter().any(|c| is_constant(c)) {
        return Err(StatsError::InsufficientDataMsg(
            "Partial correlation is undefined for a constant variable (zero variance)".into(),
        ));
    }
    if method == PartialCorMethod::Spearman {
        columns = columns
            .iter()
            .map(|c| lib_rank(c))
            .collect::<Result<_, _>>()
            .map_err(convert_error)?;
    }

    let z_refs: Vec<&[f64]> = columns[2..].iter().map(|c| c.as_slice()).collect();
    let result = match kind {
        PartialCorKind::Partial => lib_partial_cor(&columns[0], &columns[1], &z_refs),
        PartialCorKind::SemiPartial => semi_partial_cor(&columns[0], &columns[1], &z_refs),
    }
    .map_err(convert_error)?;

    let method_name = match method {
        PartialCorMethod::Pearson => result.method.clone(),
        PartialCorMethod::Spearman => format!("Spearman {}", lowercase_first(&result.method)),
    };
    let mut unified = UnifiedTestResult::from(&result).with_method(method_name.clone());
    unified.alternative = Some(Alternative::TwoSided);

    Ok(PartialCorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        n: result.n,
        n_controls: result.n_controls,
        method: method_name,
        unified,
    })
}

fn lowercase_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().chain(chars).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    // Dataset and values: validation/generators/make_reference_partial_cor.R (ppcor 1.1).
    const X: [f64; 15] = [
        3.1, 4.7, 2.2, 6.8, 5.5, 7.9, 4.1, 8.6, 6.0, 9.4, 5.2, 7.3, 3.8, 8.1, 6.6,
    ];
    const Y: [f64; 15] = [
        4.2, 3.9, 2.0, 4.9, 4.2, 6.4, 3.6, 6.9, 6.6, 10.2, 4.5, 8.0, 5.8, 8.6, 6.3,
    ];
    const Z1: [f64; 15] = [
        0.6, 2.6, -0.2, 1.3, 3.8, 2.8, 2.9, 4.3, 3.4, 3.4, 1.4, 2.7, 0.6, 4.2, 1.4,
    ];
    const Z2: [f64; 15] = [
        0.9, 2.8, 1.6, 2.0, 2.5, 3.6, 1.1, 3.0, 3.9, 4.2, 1.8, 2.4, 2.3, 4.8, 1.7,
    ];

    fn run(
        z: &[&[f64]],
        kind: PartialCorKind,
        method: PartialCorMethod,
    ) -> PartialCorrelationResult {
        let z: Vec<Vec<f64>> = z.iter().map(|c| c.to_vec()).collect();
        partial_correlation(&X, &Y, &z, kind, method).unwrap()
    }

    #[test]
    fn pcor_matches_ppcor() {
        let r = run(
            &[&Z1, &Z2],
            PartialCorKind::Partial,
            PartialCorMethod::Pearson,
        );
        assert!((r.r - 0.72398176186543961).abs() < 1e-12);
        assert!((r.statistic - 3.4808775436276367).abs() < 1e-10);
        assert!((r.p_value - 0.0051404940770096033).abs() < 1e-12);
        assert_eq!(r.df, 11.0);
        assert_eq!((r.n, r.n_controls), (15, 2));
        let r = run(
            &[&Z1, &Z2],
            PartialCorKind::Partial,
            PartialCorMethod::Spearman,
        );
        assert!((r.r - 0.73564804042047482).abs() < 1e-12);
        assert!(r.method.starts_with("Spearman partial"));
    }

    #[test]
    fn spcor_matches_ppcor() {
        let r = run(
            &[&Z1],
            PartialCorKind::SemiPartial,
            PartialCorMethod::Pearson,
        );
        assert!((r.r - 0.55234422212853096).abs() < 1e-12);
        assert!((r.p_value - 0.040542944165616455).abs() < 1e-12);
        // Two correlated controls: y residualized on both jointly (spcor.test)
        let r = run(
            &[&Z1, &Z2],
            PartialCorKind::SemiPartial,
            PartialCorMethod::Pearson,
        );
        assert!((r.r - 0.4565251230535648).abs() < 1e-12);
        assert!((r.p_value - 0.1168476978748559).abs() < 1e-12);
        assert_eq!(r.df, 11.0);
    }

    #[test]
    fn degenerate_inputs() {
        let c = vec![1.0; 15];
        let e = partial_correlation(
            &X,
            &Y,
            &[c],
            PartialCorKind::Partial,
            PartialCorMethod::Pearson,
        );
        assert!(matches!(e, Err(StatsError::InsufficientDataMsg(_))));
        let e = partial_correlation(
            &X[..3],
            &Y[..3],
            &[Z1[..3].to_vec()],
            PartialCorKind::Partial,
            PartialCorMethod::Pearson,
        );
        assert!(matches!(e, Err(StatsError::InsufficientDataMsg(_))));
    }
}
