//! Nested-model F test of two OLS fits (reduced vs full, R's
//! `anova(lm(reduced), lm(full))`): thin delegation to
//! `anofox_regression::inference::nested_f_test`. This module only drops
//! incomplete rows and maps errors.

pub use anofox_regression::inference::{NestedFTest, NestedFTestOptions};

use crate::errors::{StatsError, StatsResult};
use anofox_regression::inference::NestedFTestError;
use faer::{Col, Mat};

impl From<NestedFTestError> for StatsError {
    fn from(err: NestedFTestError) -> Self {
        match err {
            NestedFTestError::DimensionMismatch {
                full_rows, y_len, ..
            } => StatsError::DimensionMismatch {
                y_len,
                x_rows: full_rows,
            },
            NestedFTestError::NotNested { column } => StatsError::InvalidInput(format!(
                "x_reduced is not nested in x_full: x_reduced[{}] is not a column of x_full",
                column + 1
            )),
            NestedFTestError::Fit(e) => e.into(),
            // `NestedFTestError` is `#[non_exhaustive]`; a variant added upstream
            // is surfaced as an error rather than silently degraded to NULL.
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// F test of the OLS fit of `y` on `x_reduced` against the fit on `x_full`,
/// both on the rows where `y` and every column are finite. `x_reduced` may be
/// empty (the intercept-only model); its columns must lie in the span of
/// `x_full`.
pub fn ols_anova(
    y: &[f64],
    x_reduced: &[Vec<f64>],
    x_full: &[Vec<f64>],
    options: &NestedFTestOptions,
) -> StatsResult<NestedFTest> {
    if x_full.is_empty() {
        return Err(StatsError::InvalidInput(
            "x_full must have at least one column".to_string(),
        ));
    }
    crate::validation::validate_x_columns(y.len(), x_reduced)?;
    crate::validation::validate_x_columns(y.len(), x_full)?;

    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| {
            y[i].is_finite()
                && x_reduced.iter().all(|c| c[i].is_finite())
                && x_full.iter().all(|c| c[i].is_finite())
        })
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n = rows.len();
    let design = |cols: &[Vec<f64>]| Mat::from_fn(n, cols.len(), |i, j| cols[j][rows[i]]);
    let y = Col::from_fn(n, |i| y[rows[i]]);
    Ok(anofox_regression::inference::nested_f_test(
        &design(x_reduced),
        &design(x_full),
        &y,
        options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x1 = vec![
            1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9,
        ];
        let x2 = vec![
            0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5,
        ];
        let y = vec![
            2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4,
        ];
        (y, x1, x2)
    }

    #[test]
    fn delegates_the_test() {
        let (y, x1, x2) = data();
        let r = ols_anova(&y, &[x1.clone()], &[x1, x2], &NestedFTestOptions::default()).unwrap();
        assert_eq!((r.df1, r.df2, r.n_observations), (1, 9, 12));
        assert!(r.f_statistic > 0.0 && r.p_value < 1.0);
    }

    #[test]
    fn drops_non_finite_rows() {
        let (mut y, x1, x2) = data();
        y[0] = f64::NAN;
        let r = ols_anova(&y, &[], &[x1, x2], &NestedFTestOptions::default()).unwrap();
        assert_eq!(r.n_observations, 11);
    }

    #[test]
    fn maps_errors() {
        let (y, x1, x2) = data();
        let o = NestedFTestOptions::default();
        assert!(matches!(
            ols_anova(&y, &[x2], &[x1.clone()], &o),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            ols_anova(&y, &[], &[], &o),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            ols_anova(&y[..1], &[], &[x1[..1].to_vec()], &o),
            Err(StatsError::InsufficientData { .. })
        ));
    }
}
