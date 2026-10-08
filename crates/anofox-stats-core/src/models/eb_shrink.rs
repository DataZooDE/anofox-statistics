//! Empirical-Bayes shrinkage of per-group estimates toward their
//! precision-weighted mean: thin delegation to
//! `anofox_regression::solvers::eb_shrink` (DerSimonian-Laird `tau^2`, matching
//! `metafor::rma(method = "DL")` / `blup()`). This module only maps errors.

pub use anofox_regression::solvers::eb_shrink::{
    EbShrinkOptions, EbShrinkResult, ShrunkenGroup, TauMethod,
};

use crate::errors::StatsResult;

/// Shrink per-group estimates toward their common mean. Rows with a non-finite
/// estimate or a non-positive/non-finite standard error stay aligned as `NaN`.
pub fn eb_shrink(
    estimates: &[f64],
    standard_errors: &[f64],
    options: &EbShrinkOptions,
) -> StatsResult<EbShrinkResult> {
    Ok(anofox_regression::solvers::eb_shrink::eb_shrink(
        estimates,
        standard_errors,
        options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::StatsError;

    #[test]
    fn delegates_shrinkage() {
        let est = [0.10, 0.30, 0.35, 0.65, 1.00];
        let se = [0.30, 0.10, 0.50, 0.20, 0.40];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        assert_eq!(r.n_groups, 5);
        assert_eq!(r.groups.len(), 5);
        assert!(r.tau_squared >= 0.0);
    }

    #[test]
    fn maps_errors() {
        let o = EbShrinkOptions::default();
        assert!(matches!(
            eb_shrink(&[], &[], &o),
            Err(StatsError::EmptyInput { .. })
        ));
        assert!(matches!(
            eb_shrink(&[1.0, 2.0], &[1.0], &o),
            Err(StatsError::DimensionMismatch {
                y_len: 2,
                x_rows: 1
            })
        ));
        assert!(matches!(
            eb_shrink(&[1.0], &[1.0], &o),
            Err(StatsError::InsufficientData { rows: 1, cols: 2 })
        ));
        let fixed = EbShrinkOptions {
            tau_squared: Some(-1.0),
            ..Default::default()
        };
        assert!(matches!(
            eb_shrink(&[1.0, 2.0], &[1.0, 1.0], &fixed),
            Err(StatsError::InvalidValue { .. })
        ));
    }
}
