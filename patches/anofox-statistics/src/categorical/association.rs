//! Association measures for categorical data.

use crate::categorical::{expected_frequencies, validate_2x2_table, validate_contingency_table};
use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of an association measure calculation
#[derive(Debug, Clone)]
pub struct AssociationResult {
    /// The association measure value
    pub estimate: f64,
    /// Standard error (if available)
    pub se: Option<f64>,
    /// 95% confidence interval lower bound (if available)
    pub conf_int_lower: Option<f64>,
    /// 95% confidence interval upper bound (if available)
    pub conf_int_upper: Option<f64>,
    /// Name of the method
    pub method: String,
}

/// Result of a kappa calculation
#[derive(Debug, Clone)]
pub struct KappaResult {
    /// Kappa coefficient
    pub kappa: f64,
    /// Standard error
    pub se: f64,
    /// Z-statistic
    pub z: f64,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound
    pub conf_int_upper: f64,
    /// Whether weighted kappa was used
    pub weighted: bool,
    /// Name of the method
    pub method: String,
}

/// Cramér's V - effect size measure for chi-square test.
///
/// Ranges from 0 (no association) to 1 (perfect association).
/// It's normalized by the smaller of (rows-1) or (cols-1).
///
/// # Arguments
/// * `observed` - Contingency table as a 2D vector of observed counts
///
/// # Returns
/// * `AssociationResult` containing Cramér's V
///
/// # Formula
/// V = sqrt(χ² / (n * min(r-1, c-1)))
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::cramers_v;
///
/// let observed = vec![
///     vec![10, 20],
///     vec![30, 40],
/// ];
///
/// let result = cramers_v(&observed).unwrap();
/// println!("Cramér's V = {:.4}", result.estimate);
/// ```
///
/// # R equivalent
/// `DescTools::CramerV(matrix)`
pub fn cramers_v(observed: &[Vec<usize>]) -> Result<AssociationResult> {
    let (n_rows, n_cols) = validate_contingency_table(observed)?;

    if n_rows < 2 || n_cols < 2 {
        return Err(StatError::InvalidParameter(
            "Table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    // Compute chi-square statistic
    let expected = expected_frequencies(observed);
    let total: usize = observed.iter().map(|row| row.iter().sum::<usize>()).sum();

    let mut chi_sq = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let o = observed[i][j] as f64;
            let e = expected[i][j];
            if e > 0.0 {
                chi_sq += (o - e).powi(2) / e;
            }
        }
    }

    // Cramér's V
    let k = (n_rows - 1).min(n_cols - 1) as f64;
    let v = if total > 0 && k > 0.0 {
        (chi_sq / (total as f64 * k)).sqrt()
    } else {
        0.0
    };

    Ok(AssociationResult {
        estimate: v,
        se: None,
        conf_int_lower: None,
        conf_int_upper: None,
        method: "Cramer's V".to_string(),
    })
}

/// Phi coefficient for 2x2 contingency tables.
///
/// Equivalent to Cramér's V for 2x2 tables and the Pearson correlation
/// of two binary variables.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
///
/// # Returns
/// * `AssociationResult` containing phi
///
/// # Formula
/// φ = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::phi_coefficient;
///
/// let table = [[10, 20], [30, 40]];
///
/// let result = phi_coefficient(&table).unwrap();
/// println!("Phi = {:.4}", result.estimate);
/// ```
///
/// # R equivalent
/// `psych::phi(matrix)`
pub fn phi_coefficient(table: &[[usize; 2]; 2]) -> Result<AssociationResult> {
    validate_2x2_table(table)?;

    let a = table[0][0] as f64;
    let b = table[0][1] as f64;
    let c = table[1][0] as f64;
    let d = table[1][1] as f64;

    let numerator = a * d - b * c;
    let denominator = ((a + b) * (c + d) * (a + c) * (b + d)).sqrt();

    let phi = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    Ok(AssociationResult {
        estimate: phi,
        se: None,
        conf_int_lower: None,
        conf_int_upper: None,
        method: "Phi coefficient".to_string(),
    })
}

/// Contingency coefficient (Pearson's C).
///
/// Related to chi-square but bounded by sqrt((k-1)/k) where k = min(rows, cols).
///
/// # Arguments
/// * `observed` - Contingency table as a 2D vector of observed counts
///
/// # Returns
/// * `AssociationResult` containing C
///
/// # Formula
/// C = sqrt(χ² / (χ² + n))
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::contingency_coef;
///
/// let observed = vec![
///     vec![10, 20],
///     vec![30, 40],
/// ];
///
/// let result = contingency_coef(&observed).unwrap();
/// println!("C = {:.4}", result.estimate);
/// ```
///
/// # R equivalent
/// `sqrt(chisq.test(x)$statistic / (chisq.test(x)$statistic + sum(x)))`
pub fn contingency_coef(observed: &[Vec<usize>]) -> Result<AssociationResult> {
    let (n_rows, n_cols) = validate_contingency_table(observed)?;

    if n_rows < 2 || n_cols < 2 {
        return Err(StatError::InvalidParameter(
            "Table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    // Compute chi-square statistic
    let expected = expected_frequencies(observed);
    let total: usize = observed.iter().map(|row| row.iter().sum::<usize>()).sum();

    let mut chi_sq = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let o = observed[i][j] as f64;
            let e = expected[i][j];
            if e > 0.0 {
                chi_sq += (o - e).powi(2) / e;
            }
        }
    }

    // Contingency coefficient
    let c = if chi_sq + total as f64 > 0.0 {
        (chi_sq / (chi_sq + total as f64)).sqrt()
    } else {
        0.0
    };

    Ok(AssociationResult {
        estimate: c,
        se: None,
        conf_int_lower: None,
        conf_int_upper: None,
        method: "Contingency coefficient".to_string(),
    })
}

/// Cohen's kappa for inter-rater agreement.
///
/// Measures agreement between two raters beyond chance agreement.
/// Ranges from -1 (complete disagreement) through 0 (chance agreement)
/// to 1 (perfect agreement).
///
/// # Arguments
/// * `table` - Square confusion matrix where rows = rater 1, cols = rater 2
/// * `weighted` - If true, use weighted kappa with linear weights
///
/// # Returns
/// * `KappaResult` containing kappa, SE, z-statistic, and p-value
///
/// # Formula
/// κ = (Po - Pe) / (1 - Pe)
/// where Po = observed agreement, Pe = expected agreement by chance
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::cohen_kappa;
///
/// // Two raters, 3 categories
/// let table = vec![
///     vec![20, 5, 0],
///     vec![10, 30, 5],
///     vec![0, 5, 25],
/// ];
///
/// let result = cohen_kappa(&table, false).unwrap();
/// println!("Kappa = {:.4}", result.kappa);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `psych::cohen.kappa(matrix)`
pub fn cohen_kappa(table: &[Vec<usize>], weighted: bool) -> Result<KappaResult> {
    let (n_rows, n_cols) = validate_contingency_table(table)?;

    if n_rows != n_cols {
        return Err(StatError::InvalidParameter(
            "Kappa requires a square table (same number of rows and columns)".to_string(),
        ));
    }

    let k = n_rows; // Number of categories
    let n: usize = table.iter().map(|row| row.iter().sum::<usize>()).sum();

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    let n_f = n as f64;

    // Compute row and column marginals
    let row_totals: Vec<f64> = table
        .iter()
        .map(|row| row.iter().sum::<usize>() as f64)
        .collect();
    let col_totals: Vec<f64> = (0..k)
        .map(|j| table.iter().map(|row| row[j]).sum::<usize>() as f64)
        .collect();

    let (kappa, se) = if weighted && k > 2 {
        // Weighted kappa with linear weights
        let w: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| 1.0 - (i as f64 - j as f64).abs() / (k - 1) as f64)
                    .collect()
            })
            .collect();

        // Observed weighted agreement
        let mut po_w = 0.0;
        for i in 0..k {
            for j in 0..k {
                po_w += w[i][j] * table[i][j] as f64 / n_f;
            }
        }

        // Expected weighted agreement
        let mut pe_w = 0.0;
        for i in 0..k {
            for j in 0..k {
                pe_w += w[i][j] * row_totals[i] * col_totals[j] / (n_f * n_f);
            }
        }

        let kappa_w = if (1.0 - pe_w).abs() > 1e-10 {
            (po_w - pe_w) / (1.0 - pe_w)
        } else {
            1.0
        };

        // Approximate SE for weighted kappa
        let se_w = weighted_kappa_se(&w, table, n, &row_totals, &col_totals, kappa_w);

        (kappa_w, se_w)
    } else {
        // Unweighted (simple) kappa
        // Po = proportion of observed agreement (diagonal sum)
        let po: f64 = (0..k).map(|i| table[i][i] as f64).sum::<f64>() / n_f;

        // Pe = proportion of expected agreement by chance
        let pe: f64 = (0..k).map(|i| row_totals[i] * col_totals[i]).sum::<f64>() / (n_f * n_f);

        let kappa = if (1.0 - pe).abs() > 1e-10 {
            (po - pe) / (1.0 - pe)
        } else {
            1.0
        };

        // Standard error using Fleiss et al. formula
        let se = kappa_se(table, n, &row_totals, &col_totals, po, pe);

        (kappa, se)
    };

    // Z-statistic
    let z = if se > 0.0 { kappa / se } else { 0.0 };

    // Two-sided p-value
    let p_value = if z.is_finite() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        2.0 * (1.0 - normal.cdf(z.abs()))
    } else {
        0.0
    };

    // 95% CI
    let z_crit = 1.96;
    let conf_int_lower = kappa - z_crit * se;
    let conf_int_upper = kappa + z_crit * se;

    let method = if weighted {
        "Weighted Cohen's Kappa"
    } else {
        "Cohen's Kappa"
    };

    Ok(KappaResult {
        kappa,
        se,
        z,
        p_value,
        conf_int_lower,
        conf_int_upper,
        weighted,
        method: method.to_string(),
    })
}

/// Compute standard error for unweighted kappa
fn kappa_se(
    table: &[Vec<usize>],
    n: usize,
    row_totals: &[f64],
    col_totals: &[f64],
    po: f64,
    pe: f64,
) -> f64 {
    let n_f = n as f64;
    let k = table.len();

    // Variance formula from Fleiss et al.
    let mut sum1 = 0.0;
    for i in 0..k {
        let p_ii = table[i][i] as f64 / n_f;
        let p_i_plus = row_totals[i] / n_f;
        let p_plus_i = col_totals[i] / n_f;
        sum1 += p_ii * (1.0 - (p_i_plus + p_plus_i) * (1.0 - po)).powi(2);
    }

    let term1 = sum1 - (po * pe - 2.0 * pe + po).powi(2);
    let term2 = (1.0 - pe).powi(4);

    if term2 > 0.0 {
        (term1 / (n_f * term2)).sqrt()
    } else {
        0.0
    }
}

/// Compute standard error for weighted kappa
fn weighted_kappa_se(
    w: &[Vec<f64>],
    table: &[Vec<usize>],
    n: usize,
    row_totals: &[f64],
    col_totals: &[f64],
    _kappa: f64,
) -> f64 {
    let n_f = n as f64;
    let k = table.len();

    // Approximate SE using formula from Fleiss, Cohen, Everitt (1969)
    let mut po = 0.0;
    let mut pe = 0.0;
    for i in 0..k {
        for j in 0..k {
            po += w[i][j] * table[i][j] as f64 / n_f;
            pe += w[i][j] * row_totals[i] * col_totals[j] / (n_f * n_f);
        }
    }

    // Simplified SE approximation
    let var_kappa = (po * (1.0 - po)) / (n_f * (1.0 - pe).powi(2));

    var_kappa.sqrt().max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cramers_v_2x2() {
        let observed = vec![vec![10, 20], vec![30, 40]];

        let result = cramers_v(&observed).unwrap();

        assert!(result.estimate >= 0.0 && result.estimate <= 1.0);
    }

    #[test]
    fn test_cramers_v_perfect_association() {
        // Perfect association
        let observed = vec![vec![50, 0], vec![0, 50]];

        let result = cramers_v(&observed).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cramers_v_no_association() {
        // Independence
        let observed = vec![vec![25, 25], vec![25, 25]];

        let result = cramers_v(&observed).unwrap();

        assert!((result.estimate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_phi_coefficient() {
        let table = [[10, 20], [30, 40]];

        let result = phi_coefficient(&table).unwrap();

        // phi is bounded by [-1, 1]
        assert!(result.estimate >= -1.0 && result.estimate <= 1.0);
    }

    #[test]
    fn test_phi_perfect_positive() {
        let table = [[50, 0], [0, 50]];

        let result = phi_coefficient(&table).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contingency_coef() {
        let observed = vec![vec![10, 20], vec![30, 40]];

        let result = contingency_coef(&observed).unwrap();

        // C is bounded by [0, max_C]
        assert!(result.estimate >= 0.0 && result.estimate <= 1.0);
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        // Perfect agreement
        let table = vec![vec![30, 0, 0], vec![0, 30, 0], vec![0, 0, 40]];

        let result = cohen_kappa(&table, false).unwrap();

        assert!((result.kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa_no_agreement() {
        // Complete off-diagonal
        let table = vec![vec![0, 50, 0], vec![50, 0, 0], vec![0, 0, 0]];

        let result = cohen_kappa(&table, false).unwrap();

        // Should be negative (worse than chance)
        assert!(result.kappa < 0.0);
    }

    #[test]
    fn test_cohen_kappa_moderate() {
        // Some agreement
        let table = vec![vec![20, 5, 0], vec![10, 30, 5], vec![0, 5, 25]];

        let result = cohen_kappa(&table, false).unwrap();

        // Should be positive
        assert!(result.kappa > 0.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_weighted_kappa() {
        let table = vec![vec![20, 5, 0], vec![10, 30, 5], vec![0, 5, 25]];

        let unweighted = cohen_kappa(&table, false).unwrap();
        let weighted = cohen_kappa(&table, true).unwrap();

        // Weighted kappa typically differs from unweighted
        assert!((unweighted.kappa - weighted.kappa).abs() > 0.001);
    }
}
