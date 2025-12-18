//! Categorical data tests and association measures.
//!
//! This module provides tests for categorical and contingency table data:
//! - Chi-square tests (independence, goodness-of-fit, G-test)
//! - Fisher's exact test
//! - Association measures (Cramér's V, phi, contingency coefficient)
//! - Agreement measures (Cohen's kappa, weighted kappa)
//! - Proportion tests and exact binomial test
//! - McNemar's test for paired proportions

mod association;
mod chisq;
mod fisher;
mod mcnemar;
mod proportions;

pub use association::{
    cohen_kappa, contingency_coef, cramers_v, phi_coefficient, AssociationResult, KappaResult,
};
pub use chisq::{chisq_goodness_of_fit, chisq_test, g_test, ChiSquareResult};
pub use fisher::{fisher_exact, FisherResult};
pub use mcnemar::{mcnemar_exact, mcnemar_test, McNemarkExactResult, McNemarkResult};
pub use proportions::{binom_test, prop_test_one, prop_test_two, BinomTestResult, PropTestResult};

// Re-export Alternative from parametric module for consistency
pub use crate::parametric::Alternative;

use crate::error::{Result, StatError};

/// Validate a 2D contingency table (non-empty, rectangular, non-negative)
pub(crate) fn validate_contingency_table(observed: &[Vec<usize>]) -> Result<(usize, usize)> {
    if observed.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n_rows = observed.len();
    let n_cols = observed[0].len();

    if n_cols == 0 {
        return Err(StatError::EmptyData);
    }

    // Check all rows have the same length
    for (i, row) in observed.iter().enumerate() {
        if row.len() != n_cols {
            return Err(StatError::InvalidParameter(format!(
                "Row {} has {} columns, expected {}",
                i,
                row.len(),
                n_cols
            )));
        }
    }

    Ok((n_rows, n_cols))
}

/// Validate a 2x2 table
pub(crate) fn validate_2x2_table(table: &[[usize; 2]; 2]) -> Result<()> {
    // 2x2 tables are always valid structurally
    // Just check that there's at least some data
    let total: usize = table.iter().map(|row| row.iter().sum::<usize>()).sum();
    if total == 0 {
        return Err(StatError::EmptyData);
    }
    Ok(())
}

/// Compute row and column marginals
pub(crate) fn marginals(observed: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, usize) {
    let n_cols = observed[0].len();

    let row_totals: Vec<usize> = observed.iter().map(|row| row.iter().sum()).collect();

    let col_totals: Vec<usize> = (0..n_cols)
        .map(|j| observed.iter().map(|row| row[j]).sum())
        .collect();

    let total: usize = row_totals.iter().sum();

    (row_totals, col_totals, total)
}

/// Compute expected frequencies under independence assumption
pub(crate) fn expected_frequencies(observed: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let (row_totals, col_totals, total) = marginals(observed);
    let n_rows = observed.len();
    let n_cols = observed[0].len();
    let total_f = total as f64;

    let mut expected = vec![vec![0.0; n_cols]; n_rows];
    for i in 0..n_rows {
        for j in 0..n_cols {
            expected[i][j] = (row_totals[i] as f64 * col_totals[j] as f64) / total_f;
        }
    }

    expected
}
