//! Per-term ANOVA table of an OLS fit (R's `anova(lm(...))`, type I, and
//! `car::Anova(fit, type = 2)`, type II): thin delegation to
//! `anofox_regression::inference::anova_table`. This module only drops
//! incomplete rows, groups the columns into terms and maps errors.

pub use anofox_regression::inference::{AnovaTable, AnovaTableOptions, AnovaType};

use crate::errors::{StatsError, StatsResult};
use anofox_regression::inference::AnovaTableError;
use faer::{Col, Mat};

impl From<AnovaTableError> for StatsError {
    fn from(err: AnovaTableError) -> Self {
        match err {
            AnovaTableError::DimensionMismatch { x_rows, y_len } => {
                StatsError::DimensionMismatch { y_len, x_rows }
            }
            AnovaTableError::InvalidTerms(msg) => StatsError::InvalidInput(msg),
            AnovaTableError::Fit(e) => e.into(),
            // `AnovaTableError` is `#[non_exhaustive]`; a variant added upstream
            // is surfaced as an error rather than silently degraded to NULL.
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// ANOVA table of the OLS fit of `y` on the columns of `x`, on the rows where
/// `y` and every column are finite. `term_of_column[j]` is the (0-based) term
/// of column `j`; terms are numbered `0..n_terms` in table order and each must
/// have at least one column.
pub fn ols_anova_table(
    y: &[f64],
    x: &[Vec<f64>],
    term_of_column: &[usize],
    n_terms: usize,
    options: &AnovaTableOptions,
) -> StatsResult<AnovaTable> {
    if x.is_empty() {
        return Err(StatsError::InvalidInput(
            "x must have at least one column".to_string(),
        ));
    }
    crate::validation::validate_x_columns(y.len(), x)?;
    if term_of_column.len() != x.len() {
        return Err(StatsError::InvalidInput(format!(
            "terms has {} entries for {} x columns",
            term_of_column.len(),
            x.len()
        )));
    }
    let mut terms = vec![Vec::new(); n_terms];
    for (j, &t) in term_of_column.iter().enumerate() {
        terms
            .get_mut(t)
            .ok_or_else(|| StatsError::InvalidInput(format!("term {t} out of range")))?
            .push(j);
    }

    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n = rows.len();
    let design = Mat::from_fn(n, x.len(), |i, j| x[j][rows[i]]);
    let y = Col::from_fn(n, |i| y[rows[i]]);
    Ok(anofox_regression::inference::anova_table(
        &design,
        &y,
        Some(&terms),
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
    fn delegates_the_table() {
        let (y, x1, x2) = data();
        let t = ols_anova_table(&y, &[x1, x2], &[0, 1], 2, &AnovaTableOptions::default()).unwrap();
        assert_eq!(t.terms.len(), 2);
        assert_eq!((t.residual_df, t.n_observations), (9, 12));
    }

    #[test]
    fn groups_columns_and_drops_non_finite_rows() {
        let (mut y, x1, x2) = data();
        y[0] = f64::NAN;
        let t = ols_anova_table(&y, &[x1, x2], &[0, 0], 1, &AnovaTableOptions::default()).unwrap();
        assert_eq!(t.terms.len(), 1);
        assert_eq!(t.terms[0].df, 2);
        assert_eq!(t.n_observations, 11);
    }

    #[test]
    fn maps_errors() {
        let (y, x1, x2) = data();
        let o = AnovaTableOptions::default();
        assert!(matches!(
            ols_anova_table(&y, &[], &[], 0, &o),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            ols_anova_table(&y, &[x1.clone(), x2.clone()], &[0, 2], 2, &o),
            Err(StatsError::InvalidInput(_))
        ));
        // Term 1 has no column.
        assert!(matches!(
            ols_anova_table(&y, &[x1.clone(), x2], &[0, 0], 2, &o),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            ols_anova_table(&y[..1], &[x1[..1].to_vec()], &[0], 1, &o),
            Err(StatsError::InsufficientData { .. })
        ));
    }
}
