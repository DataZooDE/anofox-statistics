//! ANOVA (Analysis of Variance) implementations.
//!
//! This module provides one-way, two-way, and repeated measures ANOVA tests
//! with full R output parity.

// Allow index-based loops for matrix operations where they are clearer than iterators
#![allow(clippy::needless_range_loop)]

use crate::error::{Result, StatError};
use crate::utils::math::{mean, variance};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

/// Type alias for validated ANOVA group statistics: (sizes, means, variances, total_n).
type GroupStats = (Vec<usize>, Vec<f64>, Vec<f64>, usize);

// =============================================================================
// One-Way ANOVA
// =============================================================================

/// The kind of one-way ANOVA to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnovaKind {
    /// Fisher's ANOVA - assumes equal variances across groups.
    /// Matches R's `aov()` or `oneway.test(..., var.equal=TRUE)`.
    Fisher,
    /// Welch's ANOVA - robust to unequal variances.
    /// Matches R's `oneway.test(..., var.equal=FALSE)`.
    Welch,
}

/// Result of a one-way ANOVA test.
#[derive(Debug, Clone)]
pub struct OneWayAnovaResult {
    /// The F-statistic.
    pub statistic: f64,
    /// Numerator degrees of freedom (between groups: k-1).
    pub df_between: f64,
    /// Denominator degrees of freedom (within groups: N-k, or Welch-adjusted).
    pub df_within: f64,
    /// The p-value.
    pub p_value: f64,
    /// Sum of squares between groups (None for Welch's ANOVA).
    pub ss_between: Option<f64>,
    /// Sum of squares within groups (None for Welch's ANOVA).
    pub ss_within: Option<f64>,
    /// Total sum of squares (None for Welch's ANOVA).
    pub ss_total: Option<f64>,
    /// Mean square between groups (None for Welch's ANOVA).
    pub ms_between: Option<f64>,
    /// Mean square within groups (None for Welch's ANOVA).
    pub ms_within: Option<f64>,
    /// Number of groups.
    pub n_groups: usize,
    /// Sample size of each group.
    pub group_sizes: Vec<usize>,
    /// Mean of each group.
    pub group_means: Vec<f64>,
    /// Grand mean of all observations (None for Welch's ANOVA).
    pub grand_mean: Option<f64>,
}

/// Validate groups and return their sizes, means, variances, and total count.
fn validate_anova_groups(groups: &[&[f64]]) -> Result<GroupStats> {
    if groups.len() < 2 {
        return Err(StatError::InvalidParameter(
            "ANOVA requires at least 2 groups".to_string(),
        ));
    }

    let mut group_sizes: Vec<usize> = Vec::with_capacity(groups.len());
    let mut group_means: Vec<f64> = Vec::with_capacity(groups.len());
    let mut group_vars: Vec<f64> = Vec::with_capacity(groups.len());
    let mut n_total = 0usize;

    for (i, group) in groups.iter().enumerate() {
        if group.len() < 2 {
            return Err(StatError::InsufficientData {
                needed: 2,
                got: group.len(),
            });
        }

        // Check for NaN values
        if group.iter().any(|x| x.is_nan()) {
            return Err(StatError::InvalidParameter(format!(
                "Group {} contains NaN values",
                i + 1
            )));
        }

        group_sizes.push(group.len());
        group_means.push(mean(group)?);
        group_vars.push(variance(group)?);
        n_total += group.len();
    }

    Ok((group_sizes, group_means, group_vars, n_total))
}

/// Perform Fisher's one-way ANOVA (assumes equal variances).
fn fisher_anova(
    groups: &[&[f64]],
    group_sizes: &[usize],
    group_means: &[f64],
    n_total: usize,
) -> Result<OneWayAnovaResult> {
    let k = groups.len();
    let n_f = n_total as f64;

    // Compute grand mean
    let grand_sum: f64 = groups.iter().flat_map(|g| g.iter()).sum();
    let grand_mean = grand_sum / n_f;

    // Compute SS_between: sum(n_i * (x_bar_i - x_bar)^2)
    let ss_between: f64 = group_sizes
        .iter()
        .zip(group_means.iter())
        .map(|(&n_i, &mean_i)| n_i as f64 * (mean_i - grand_mean).powi(2))
        .sum();

    // Compute SS_within: sum of within-group sum of squares
    let ss_within: f64 = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &group_mean)| group.iter().map(|&x| (x - group_mean).powi(2)).sum::<f64>())
        .sum();

    // SS_total = SS_between + SS_within
    let ss_total = ss_between + ss_within;

    // Degrees of freedom
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;

    // Mean squares
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    // F-statistic
    let f_stat = ms_between / ms_within;

    // P-value from F-distribution
    let f_dist = FisherSnedecor::new(df_between, df_within).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_value = 1.0 - f_dist.cdf(f_stat);

    Ok(OneWayAnovaResult {
        statistic: f_stat,
        df_between,
        df_within,
        p_value,
        ss_between: Some(ss_between),
        ss_within: Some(ss_within),
        ss_total: Some(ss_total),
        ms_between: Some(ms_between),
        ms_within: Some(ms_within),
        n_groups: k,
        group_sizes: group_sizes.to_vec(),
        group_means: group_means.to_vec(),
        grand_mean: Some(grand_mean),
    })
}

/// Perform Welch's one-way ANOVA (robust to unequal variances).
///
/// This implements the Welch-Satterthwaite approximation for F-test
/// with heterogeneous variances, matching R's `oneway.test(var.equal=FALSE)`.
fn welch_anova(
    group_sizes: &[usize],
    group_means: &[f64],
    group_vars: &[f64],
) -> Result<OneWayAnovaResult> {
    let k = group_sizes.len();

    // Compute weights: w_i = n_i / s_i^2
    let weights: Vec<f64> = group_sizes
        .iter()
        .zip(group_vars.iter())
        .map(|(&n_i, &var_i)| {
            if var_i < 1e-15 {
                // Handle near-zero variance (constant group)
                f64::MAX / (k as f64)
            } else {
                n_i as f64 / var_i
            }
        })
        .collect();

    let sum_weights: f64 = weights.iter().sum();

    // Weighted grand mean
    let weighted_grand_mean: f64 = weights
        .iter()
        .zip(group_means.iter())
        .map(|(&w, &mean)| w * mean)
        .sum::<f64>()
        / sum_weights;

    // Numerator of F-statistic
    let f_numerator: f64 = weights
        .iter()
        .zip(group_means.iter())
        .map(|(&w, &mean)| w * (mean - weighted_grand_mean).powi(2))
        .sum::<f64>()
        / (k - 1) as f64;

    // Lambda term for denominator adjustment
    let lambda: f64 = group_sizes
        .iter()
        .zip(weights.iter())
        .map(|(&n_i, &w_i)| {
            let term = 1.0 - w_i / sum_weights;
            term * term / (n_i - 1) as f64
        })
        .sum::<f64>()
        * 3.0
        / ((k * k - 1) as f64);

    // F-statistic (Welch's)
    let f_stat = f_numerator / (1.0 + 2.0 * lambda * (k - 2) as f64 / 3.0);

    // Degrees of freedom
    let df_between = (k - 1) as f64;

    // Welch-Satterthwaite approximation for denominator df
    let df_within = ((k * k - 1) as f64)
        / (3.0
            * group_sizes
                .iter()
                .zip(weights.iter())
                .map(|(&n_i, &w_i)| {
                    let term = 1.0 - w_i / sum_weights;
                    term * term / (n_i - 1) as f64
                })
                .sum::<f64>());

    // P-value from F-distribution
    let f_dist = FisherSnedecor::new(df_between, df_within).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_value = 1.0 - f_dist.cdf(f_stat);

    Ok(OneWayAnovaResult {
        statistic: f_stat,
        df_between,
        df_within,
        p_value,
        // Welch's ANOVA doesn't produce traditional SS/MS values
        ss_between: None,
        ss_within: None,
        ss_total: None,
        ms_between: None,
        ms_within: None,
        n_groups: k,
        group_sizes: group_sizes.to_vec(),
        group_means: group_means.to_vec(),
        grand_mean: None,
    })
}

/// Perform a one-way ANOVA comparing multiple independent groups.
///
/// # Arguments
/// * `groups` - Slice of slices, each containing one group's data
/// * `kind` - Type of ANOVA: `Fisher` (equal variances) or `Welch` (unequal variances)
///
/// # Returns
/// * `OneWayAnovaResult` containing F-statistic, degrees of freedom, p-value,
///   and ANOVA table components (SS, MS for Fisher's ANOVA)
///
/// # Examples
/// ```
/// use anofox_statistics::{one_way_anova, AnovaKind};
///
/// let group1 = vec![23.0, 25.0, 21.0, 24.0, 22.0];
/// let group2 = vec![30.0, 32.0, 28.0, 31.0, 29.0];
/// let group3 = vec![27.0, 29.0, 26.0, 28.0, 27.0];
///
/// // Fisher's ANOVA (assumes equal variances)
/// let result = one_way_anova(&[&group1, &group2, &group3], AnovaKind::Fisher)
///     .expect("ANOVA should succeed");
/// println!("F({}, {}) = {:.4}, p = {:.6}",
///     result.df_between, result.df_within, result.statistic, result.p_value);
///
/// // Welch's ANOVA (robust to unequal variances)
/// let result = one_way_anova(&[&group1, &group2, &group3], AnovaKind::Welch)
///     .expect("ANOVA should succeed");
/// println!("F({}, {:.2}) = {:.4}, p = {:.6}",
///     result.df_between, result.df_within, result.statistic, result.p_value);
/// ```
///
/// # Notes
/// - For 2 groups, Fisher's ANOVA F-statistic equals the squared t-statistic from Student's t-test
/// - Welch's ANOVA is recommended when group variances are unequal (heteroscedasticity)
/// - Each group must have at least 2 observations
pub fn one_way_anova(groups: &[&[f64]], kind: AnovaKind) -> Result<OneWayAnovaResult> {
    let (group_sizes, group_means, group_vars, n_total) = validate_anova_groups(groups)?;

    match kind {
        AnovaKind::Fisher => fisher_anova(groups, &group_sizes, &group_means, n_total),
        AnovaKind::Welch => welch_anova(&group_sizes, &group_means, &group_vars),
    }
}

// =============================================================================
// Two-Way ANOVA
// =============================================================================

/// A single row in an ANOVA table.
#[derive(Debug, Clone)]
pub struct AnovaTableRow {
    /// Sum of Squares.
    pub ss: f64,
    /// Degrees of freedom.
    pub df: f64,
    /// Mean Square (SS / df).
    pub ms: f64,
    /// F-statistic (None for residual/total rows).
    pub f_statistic: Option<f64>,
    /// P-value (None for residual/total rows).
    pub p_value: Option<f64>,
}

/// Result of a two-way ANOVA test.
#[derive(Debug, Clone)]
pub struct TwoWayAnovaResult {
    /// Main effect of Factor A.
    pub factor_a: AnovaTableRow,
    /// Main effect of Factor B.
    pub factor_b: AnovaTableRow,
    /// Interaction effect (A × B).
    pub interaction: AnovaTableRow,
    /// Residual (error) term.
    pub residual: AnovaTableRow,
    /// Total variation.
    pub total: AnovaTableRow,
    /// Number of levels in Factor A.
    pub levels_a: usize,
    /// Number of levels in Factor B.
    pub levels_b: usize,
    /// Total number of observations.
    pub n: usize,
    /// Grand mean of all observations.
    pub grand_mean: f64,
    /// Cell means indexed by `cell_means[a][b]`.
    pub cell_means: Vec<Vec<f64>>,
    /// Marginal means for Factor A.
    pub marginal_means_a: Vec<f64>,
    /// Marginal means for Factor B.
    pub marginal_means_b: Vec<f64>,
}

/// Organized data for two-way ANOVA.
struct TwoWayData {
    /// Observations organized by cell: cells[a][b] = Vec<f64>
    cells: Vec<Vec<Vec<f64>>>,
    /// Number of levels in Factor A
    levels_a: usize,
    /// Number of levels in Factor B
    levels_b: usize,
    /// Total number of observations
    n_total: usize,
    /// Cell counts: cell_n[a][b]
    cell_n: Vec<Vec<usize>>,
}

/// Validate and organize two-way ANOVA data.
fn organize_two_way_data(
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> Result<TwoWayData> {
    let n = values.len();

    // Validate lengths match
    if factor_a.len() != n || factor_b.len() != n {
        return Err(StatError::InvalidParameter(format!(
            "Mismatched lengths: values ({}), factor_a ({}), factor_b ({})",
            n,
            factor_a.len(),
            factor_b.len()
        )));
    }

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    // Check for NaN values
    if values.iter().any(|x| x.is_nan()) {
        return Err(StatError::InvalidParameter(
            "Values contain NaN".to_string(),
        ));
    }

    // Determine number of levels
    let levels_a = factor_a.iter().max().map(|m| m + 1).unwrap_or(0);
    let levels_b = factor_b.iter().max().map(|m| m + 1).unwrap_or(0);

    if levels_a < 2 {
        return Err(StatError::InvalidParameter(
            "Factor A must have at least 2 levels".to_string(),
        ));
    }
    if levels_b < 2 {
        return Err(StatError::InvalidParameter(
            "Factor B must have at least 2 levels".to_string(),
        ));
    }

    // Organize observations into cells
    let mut cells: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); levels_b]; levels_a];

    for i in 0..n {
        let a = factor_a[i];
        let b = factor_b[i];
        if a >= levels_a || b >= levels_b {
            return Err(StatError::InvalidParameter(format!(
                "Factor index out of bounds at position {}: a={}, b={}",
                i, a, b
            )));
        }
        cells[a][b].push(values[i]);
    }

    // Check for empty cells
    let mut cell_n: Vec<Vec<usize>> = vec![vec![0; levels_b]; levels_a];
    for a in 0..levels_a {
        for b in 0..levels_b {
            if cells[a][b].is_empty() {
                return Err(StatError::InvalidParameter(format!(
                    "Empty cell: no observations for factor combination (A={}, B={})",
                    a, b
                )));
            }
            cell_n[a][b] = cells[a][b].len();
        }
    }

    Ok(TwoWayData {
        cells,
        levels_a,
        levels_b,
        n_total: n,
        cell_n,
    })
}

/// Compute Type III Sum of Squares for two-way ANOVA.
///
/// This implements the Type III SS approach which tests each effect as if entered last,
/// controlling for all other effects. This matches R's `car::Anova(model, type = "III")`.
///
/// For balanced designs, Type III SS equals Type I SS.
/// For unbalanced designs, we use model comparison via design matrices.
fn compute_two_way_type3_ss(
    data: &TwoWayData,
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> (f64, f64, f64, f64) {
    let n = data.n_total;
    let a = data.levels_a;
    let b = data.levels_b;

    // Compute cell means
    let mut cell_means: Vec<Vec<f64>> = vec![vec![0.0; b]; a];
    for ai in 0..a {
        for bi in 0..b {
            cell_means[ai][bi] =
                data.cells[ai][bi].iter().sum::<f64>() / data.cells[ai][bi].len() as f64;
        }
    }

    // SS_error (within cells) = sum of squared deviations from cell means
    let mut ss_error = 0.0;
    for ai in 0..a {
        for bi in 0..b {
            let cm = cell_means[ai][bi];
            for &y in &data.cells[ai][bi] {
                ss_error += (y - cm).powi(2);
            }
        }
    }

    // Check if design is balanced
    let first_n = data.cell_n[0][0];
    let is_balanced = data
        .cell_n
        .iter()
        .all(|row| row.iter().all(|&n| n == first_n));

    if is_balanced {
        // For balanced designs, use simple formulas (Type I = Type III)
        let n_per_cell = first_n;

        // Compute weighted marginal means (for balanced data, equals unweighted)
        let mut marginal_means_a: Vec<f64> = vec![0.0; a];
        for ai in 0..a {
            let mut sum = 0.0;
            for bi in 0..b {
                sum += cell_means[ai][bi];
            }
            marginal_means_a[ai] = sum / b as f64;
        }

        let mut marginal_means_b: Vec<f64> = vec![0.0; b];
        for bi in 0..b {
            let mut sum = 0.0;
            for ai in 0..a {
                sum += cell_means[ai][bi];
            }
            marginal_means_b[bi] = sum / a as f64;
        }

        // Grand mean
        let grand_mean: f64 = cell_means.iter().flatten().sum::<f64>() / (a * b) as f64;

        // SS_A = n * b * sum((marginal_a - grand)^2)
        let ss_a: f64 = (n_per_cell * b) as f64
            * marginal_means_a
                .iter()
                .map(|m| (m - grand_mean).powi(2))
                .sum::<f64>();

        // SS_B = n * a * sum((marginal_b - grand)^2)
        let ss_b: f64 = (n_per_cell * a) as f64
            * marginal_means_b
                .iter()
                .map(|m| (m - grand_mean).powi(2))
                .sum::<f64>();

        // SS_AB = n * sum((cell - marginal_a - marginal_b + grand)^2)
        let mut ss_ab = 0.0;
        for ai in 0..a {
            for bi in 0..b {
                let interaction_effect =
                    cell_means[ai][bi] - marginal_means_a[ai] - marginal_means_b[bi] + grand_mean;
                ss_ab += n_per_cell as f64 * interaction_effect.powi(2);
            }
        }

        (ss_a, ss_b, ss_ab, ss_error)
    } else {
        // For unbalanced designs, use hypothesis testing with contrast matrices
        // Type III SS = (L * beta)' * [L * (X'X)^-1 * L']^-1 * (L * beta)
        // where L selects the coefficients for each effect

        // Build design matrix with effect coding (sum-to-zero constraint)
        let n_params = 1 + (a - 1) + (b - 1) + (a - 1) * (b - 1);
        let mut x: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n];

        for i in 0..n {
            let ai = factor_a[i];
            let bi = factor_b[i];

            // Intercept
            x[i][0] = 1.0;

            // Effect coding for factor A
            for k in 0..(a - 1) {
                if ai == k {
                    x[i][1 + k] = 1.0;
                } else if ai == a - 1 {
                    x[i][1 + k] = -1.0;
                }
            }

            // Effect coding for factor B
            for k in 0..(b - 1) {
                if bi == k {
                    x[i][1 + (a - 1) + k] = 1.0;
                } else if bi == b - 1 {
                    x[i][1 + (a - 1) + k] = -1.0;
                }
            }

            // Interaction terms
            let mut idx = 1 + (a - 1) + (b - 1);
            for k_a in 0..(a - 1) {
                for k_b in 0..(b - 1) {
                    x[i][idx] = x[i][1 + k_a] * x[i][1 + (a - 1) + k_b];
                    idx += 1;
                }
            }
        }

        // Compute X'X and X'y
        let mut xtx = vec![vec![0.0; n_params]; n_params];
        for j in 0..n_params {
            for k in 0..n_params {
                for i in 0..n {
                    xtx[j][k] += x[i][j] * x[i][k];
                }
            }
        }

        let mut xty = vec![0.0; n_params];
        for j in 0..n_params {
            for i in 0..n {
                xty[j] += x[i][j] * values[i];
            }
        }

        // Solve for coefficients beta = (X'X)^-1 X'y
        let beta = solve_linear_system(&xtx, &xty);

        // Compute (X'X)^-1
        let xtx_inv = matrix_inverse(&xtx);

        // Indices for effects
        let a_start = 1;
        let a_end = 1 + (a - 1);
        let b_start = 1 + (a - 1);
        let b_end = 1 + (a - 1) + (b - 1);
        let ab_start = 1 + (a - 1) + (b - 1);

        // Compute Type III SS using Wald-type test for each effect
        // SS(effect) = (L * beta)' * [L * (X'X)^-1 * L']^-1 * (L * beta)
        let ss_a = compute_wald_ss(&beta, &xtx_inv, a_start, a_end, n_params);
        let ss_b = compute_wald_ss(&beta, &xtx_inv, b_start, b_end, n_params);
        let ss_ab = compute_wald_ss(&beta, &xtx_inv, ab_start, n_params, n_params);

        (ss_a, ss_b, ss_ab, ss_error)
    }
}

/// Compute Wald-type SS for an effect using the formula:
/// SS(effect) = (L * beta)' * [L * (X'X)^-1 * L']^-1 * (L * beta)
/// where L is an identity-like matrix selecting the effect's coefficients.
fn compute_wald_ss(
    beta: &[f64],
    xtx_inv: &[Vec<f64>],
    effect_start: usize,
    effect_end: usize,
    _n_params: usize,
) -> f64 {
    let effect_size = effect_end - effect_start;

    // Extract beta for this effect
    let beta_effect: Vec<f64> = beta[effect_start..effect_end].to_vec();

    // Extract (X'X)^-1 submatrix for this effect
    let mut cov_effect = vec![vec![0.0; effect_size]; effect_size];
    for i in 0..effect_size {
        for j in 0..effect_size {
            cov_effect[i][j] = xtx_inv[effect_start + i][effect_start + j];
        }
    }

    // Compute inverse of the covariance submatrix
    let cov_inv = matrix_inverse(&cov_effect);

    // Compute SS = beta' * cov_inv * beta
    let mut ss = 0.0;
    for i in 0..effect_size {
        for j in 0..effect_size {
            ss += beta_effect[i] * cov_inv[i][j] * beta_effect[j];
        }
    }

    ss
}

/// Compute the inverse of a matrix using Gaussian elimination.
fn matrix_inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();

    // Create augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-14 {
            continue;
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    inv
}

/// Fit a model and return the residual SS.
#[allow(dead_code)]
fn fit_model_ss_error(x: &[Vec<f64>], y: &[f64], n_params: usize) -> f64 {
    let n = x.len();

    // Compute X'X
    let mut xtx = vec![vec![0.0; n_params]; n_params];
    for j in 0..n_params {
        for k in 0..n_params {
            for i in 0..n {
                xtx[j][k] += x[i][j] * x[i][k];
            }
        }
    }

    // Compute X'y
    let mut xty = vec![0.0; n_params];
    for j in 0..n_params {
        for i in 0..n {
            xty[j] += x[i][j] * y[i];
        }
    }

    // Solve for coefficients
    let beta = solve_linear_system(&xtx, &xty);

    // Compute SS_error
    let mut ss_error = 0.0;
    for i in 0..n {
        let mut y_hat_i = 0.0;
        for j in 0..n_params {
            y_hat_i += x[i][j] * beta[j];
        }
        ss_error += (y[i] - y_hat_i).powi(2);
    }

    ss_error
}

/// Fit a reduced model (without specified columns) and return SS difference.
#[allow(dead_code)]
fn fit_reduced_model_ss_diff(
    x: &[Vec<f64>],
    y: &[f64],
    n_params: usize,
    exclude_start: usize,
    exclude_end: usize,
    ss_error_full: f64,
) -> f64 {
    let n = x.len();
    let reduced_params = n_params - (exclude_end - exclude_start);

    // Build reduced design matrix
    let mut x_reduced: Vec<Vec<f64>> = vec![vec![0.0; reduced_params]; n];
    for i in 0..n {
        let mut j_reduced = 0;
        for j in 0..n_params {
            if j < exclude_start || j >= exclude_end {
                x_reduced[i][j_reduced] = x[i][j];
                j_reduced += 1;
            }
        }
    }

    let ss_error_reduced = fit_model_ss_error(&x_reduced, y, reduced_params);

    // Type III SS = SS_error(reduced) - SS_error(full)
    ss_error_reduced - ss_error_full
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = a.to_vec();
    for i in 0..n {
        aug[i].push(b[i]);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        aug.swap(col, max_row);

        // Check for singular matrix
        if aug[col][col].abs() < 1e-14 {
            continue;
        }

        // Eliminate column
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-14 {
            x[i] = 0.0;
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    x
}

/// Perform a two-way ANOVA (factorial ANOVA) with Type III Sum of Squares.
///
/// This tests:
/// - Main effect of Factor A
/// - Main effect of Factor B
/// - Interaction effect (A × B)
///
/// # Arguments
/// * `values` - The dependent variable values
/// * `factor_a` - Factor A level for each observation (0-indexed)
/// * `factor_b` - Factor B level for each observation (0-indexed)
///
/// # Returns
/// * `TwoWayAnovaResult` containing ANOVA table rows for each effect
///
/// # Examples
/// ```
/// use anofox_statistics::two_way_anova;
///
/// // 2x2 factorial design: Drug (0=Placebo, 1=Treatment) x Gender (0=F, 1=M)
/// let values = vec![
///     5.2, 5.8, 5.5, 5.3,  // Placebo-Female
///     6.1, 6.3, 5.9, 6.0,  // Placebo-Male
///     7.2, 7.5, 7.1, 7.4,  // Treatment-Female
///     8.5, 8.8, 8.2, 8.6,  // Treatment-Male
/// ];
/// let factor_a = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
/// let factor_b = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
///
/// let result = two_way_anova(&values, &factor_a, &factor_b)
///     .expect("ANOVA should succeed");
///
/// println!("Factor A: F = {:.4}, p = {:.4}",
///     result.factor_a.f_statistic.unwrap(),
///     result.factor_a.p_value.unwrap());
/// println!("Factor B: F = {:.4}, p = {:.4}",
///     result.factor_b.f_statistic.unwrap(),
///     result.factor_b.p_value.unwrap());
/// println!("Interaction: F = {:.4}, p = {:.4}",
///     result.interaction.f_statistic.unwrap(),
///     result.interaction.p_value.unwrap());
/// ```
///
/// # Notes
/// - Uses Type III Sum of Squares (each effect adjusted for all others)
/// - Handles both balanced and unbalanced designs
/// - Matches R's `car::Anova(model, type = "III")`
pub fn two_way_anova(
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> Result<TwoWayAnovaResult> {
    let data = organize_two_way_data(values, factor_a, factor_b)?;

    let a = data.levels_a;
    let b = data.levels_b;
    let n = data.n_total;

    // Compute grand mean
    let grand_mean = values.iter().sum::<f64>() / n as f64;

    // Compute cell means
    let mut cell_means: Vec<Vec<f64>> = vec![vec![0.0; b]; a];
    for ai in 0..a {
        for bi in 0..b {
            if !data.cells[ai][bi].is_empty() {
                cell_means[ai][bi] =
                    data.cells[ai][bi].iter().sum::<f64>() / data.cells[ai][bi].len() as f64;
            }
        }
    }

    // Compute marginal means for Factor A (mean across all levels of B)
    let mut marginal_means_a: Vec<f64> = vec![0.0; a];
    for ai in 0..a {
        let mut sum = 0.0;
        let mut count = 0.0;
        for bi in 0..b {
            for &y in &data.cells[ai][bi] {
                sum += y;
                count += 1.0;
            }
        }
        marginal_means_a[ai] = sum / count;
    }

    // Compute marginal means for Factor B (mean across all levels of A)
    let mut marginal_means_b: Vec<f64> = vec![0.0; b];
    for bi in 0..b {
        let mut sum = 0.0;
        let mut count = 0.0;
        for ai in 0..a {
            for &y in &data.cells[ai][bi] {
                sum += y;
                count += 1.0;
            }
        }
        marginal_means_b[bi] = sum / count;
    }

    // Compute Type III Sum of Squares
    let (ss_a, ss_b, ss_ab, ss_error) = compute_two_way_type3_ss(&data, values, factor_a, factor_b);

    // Compute total SS
    let ss_total: f64 = values.iter().map(|y| (y - grand_mean).powi(2)).sum();

    // Degrees of freedom
    let df_a = (a - 1) as f64;
    let df_b = (b - 1) as f64;
    let df_ab = ((a - 1) * (b - 1)) as f64;
    let df_error = (n - a * b) as f64;
    let df_total = (n - 1) as f64;

    // Check for sufficient error df
    if df_error <= 0.0 {
        return Err(StatError::InsufficientData {
            needed: a * b + 1,
            got: n,
        });
    }

    // Mean squares
    let ms_a = ss_a / df_a;
    let ms_b = ss_b / df_b;
    let ms_ab = ss_ab / df_ab;
    let ms_error = ss_error / df_error;

    // F-statistics
    let f_a = ms_a / ms_error;
    let f_b = ms_b / ms_error;
    let f_ab = ms_ab / ms_error;

    // P-values
    let f_dist_a = FisherSnedecor::new(df_a, df_error).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_a = 1.0 - f_dist_a.cdf(f_a);

    let f_dist_b = FisherSnedecor::new(df_b, df_error).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_b = 1.0 - f_dist_b.cdf(f_b);

    let f_dist_ab = FisherSnedecor::new(df_ab, df_error).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_ab = 1.0 - f_dist_ab.cdf(f_ab);

    Ok(TwoWayAnovaResult {
        factor_a: AnovaTableRow {
            ss: ss_a,
            df: df_a,
            ms: ms_a,
            f_statistic: Some(f_a),
            p_value: Some(p_a),
        },
        factor_b: AnovaTableRow {
            ss: ss_b,
            df: df_b,
            ms: ms_b,
            f_statistic: Some(f_b),
            p_value: Some(p_b),
        },
        interaction: AnovaTableRow {
            ss: ss_ab,
            df: df_ab,
            ms: ms_ab,
            f_statistic: Some(f_ab),
            p_value: Some(p_ab),
        },
        residual: AnovaTableRow {
            ss: ss_error,
            df: df_error,
            ms: ms_error,
            f_statistic: None,
            p_value: None,
        },
        total: AnovaTableRow {
            ss: ss_total,
            df: df_total,
            ms: ss_total / df_total,
            f_statistic: None,
            p_value: None,
        },
        levels_a: a,
        levels_b: b,
        n,
        grand_mean,
        cell_means,
        marginal_means_a,
        marginal_means_b,
    })
}

// =============================================================================
// Repeated Measures ANOVA
// =============================================================================

/// Result of Mauchly's test of sphericity.
///
/// Sphericity is the assumption that the variances of the differences between
/// all combinations of related groups (conditions) are equal.
#[derive(Debug, Clone)]
pub struct SphericityResult {
    /// Mauchly's W statistic.
    pub w: f64,
    /// Chi-square approximation.
    pub chi_square: f64,
    /// Degrees of freedom.
    pub df: f64,
    /// P-value (small p = sphericity violated).
    pub p_value: f64,
}

/// Epsilon-corrected result for repeated measures ANOVA.
///
/// When sphericity is violated, these corrections adjust the degrees of freedom
/// to produce more accurate p-values.
#[derive(Debug, Clone)]
pub struct CorrectedResult {
    /// Epsilon correction factor (1.0 = perfect sphericity).
    pub epsilon: f64,
    /// Corrected numerator degrees of freedom.
    pub df_num_corrected: f64,
    /// Corrected denominator degrees of freedom.
    pub df_den_corrected: f64,
    /// F-statistic (same as uncorrected).
    pub f_statistic: f64,
    /// Corrected p-value.
    pub p_value: f64,
}

/// Result of a one-way repeated measures ANOVA test.
#[derive(Debug, Clone)]
pub struct RmAnovaResult {
    /// Within-subjects (treatment/condition) effect.
    pub within_subjects: AnovaTableRow,
    /// Between-subjects (individual differences) effect.
    pub subjects: AnovaTableRow,
    /// Error term (subjects × condition interaction).
    pub error: AnovaTableRow,
    /// Total variation.
    pub total: AnovaTableRow,
    /// Mauchly's sphericity test (None if k < 3).
    pub sphericity: Option<SphericityResult>,
    /// Greenhouse-Geisser correction (None if k < 3).
    pub greenhouse_geisser: Option<CorrectedResult>,
    /// Huynh-Feldt correction (None if k < 3).
    pub huynh_feldt: Option<CorrectedResult>,
    /// Grand mean of all observations.
    pub grand_mean: f64,
    /// Mean of each condition.
    pub condition_means: Vec<f64>,
    /// Mean of each subject (across conditions).
    pub subject_means: Vec<f64>,
}

/// Perform a one-way repeated measures ANOVA.
///
/// Tests whether there are significant differences between repeated measurements
/// taken from the same subjects under different conditions.
///
/// # Arguments
/// * `data` - Matrix where rows = subjects, columns = conditions. Each slice is one subject's data.
/// * `compute_sphericity` - Whether to compute Mauchly's test and epsilon corrections.
///
/// # Returns
/// * `RmAnovaResult` containing ANOVA table, sphericity test, and corrections.
///
/// # Examples
/// ```
/// use anofox_statistics::repeated_measures_anova;
///
/// // 6 subjects measured under 3 conditions
/// let data: Vec<&[f64]> = vec![
///     &[10.0, 12.0, 14.0],  // Subject 1
///     &[9.0, 11.0, 13.0],   // Subject 2
///     &[11.0, 13.0, 15.0],  // Subject 3
///     &[8.0, 10.0, 12.0],   // Subject 4
///     &[10.0, 11.0, 14.0],  // Subject 5
///     &[12.0, 14.0, 16.0],  // Subject 6
/// ];
///
/// let result = repeated_measures_anova(&data, true)
///     .expect("RM ANOVA should succeed");
///
/// println!("F({}, {}) = {:.4}, p = {:.6}",
///     result.within_subjects.df,
///     result.error.df,
///     result.within_subjects.f_statistic.unwrap(),
///     result.within_subjects.p_value.unwrap());
///
/// if let Some(ref sph) = result.sphericity {
///     println!("Mauchly's W = {:.4}, p = {:.4}", sph.w, sph.p_value);
/// }
///
/// if let Some(ref gg) = result.greenhouse_geisser {
///     println!("GG epsilon = {:.4}, corrected p = {:.4}", gg.epsilon, gg.p_value);
/// }
/// ```
///
/// # Notes
/// - All subjects must have the same number of conditions (measurements)
/// - Requires at least 2 subjects and 2 conditions
/// - Sphericity test and corrections are only computed for k >= 3 conditions
/// - Matches R's `ez::ezANOVA()` output
pub fn repeated_measures_anova(data: &[&[f64]], compute_sphericity: bool) -> Result<RmAnovaResult> {
    // Validate input
    let n_subjects = data.len();
    if n_subjects < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: n_subjects,
        });
    }

    let n_conditions = data[0].len();
    if n_conditions < 2 {
        return Err(StatError::InvalidParameter(
            "Repeated measures ANOVA requires at least 2 conditions".to_string(),
        ));
    }

    // Check all subjects have same number of conditions
    for (i, subject) in data.iter().enumerate() {
        if subject.len() != n_conditions {
            return Err(StatError::InvalidParameter(format!(
                "Subject {} has {} conditions, expected {}",
                i + 1,
                subject.len(),
                n_conditions
            )));
        }
        // Check for NaN
        if subject.iter().any(|x| x.is_nan()) {
            return Err(StatError::InvalidParameter(format!(
                "Subject {} contains NaN values",
                i + 1
            )));
        }
    }

    let n = n_subjects as f64;
    let k = n_conditions as f64;

    // Compute grand mean
    let grand_sum: f64 = data.iter().flat_map(|s| s.iter()).sum();
    let grand_mean = grand_sum / (n * k);

    // Compute subject means (row means)
    let subject_means: Vec<f64> = data.iter().map(|s| s.iter().sum::<f64>() / k).collect();

    // Compute condition means (column means)
    let mut condition_means: Vec<f64> = vec![0.0; n_conditions];
    for j in 0..n_conditions {
        let sum: f64 = data.iter().map(|s| s[j]).sum();
        condition_means[j] = sum / n;
    }

    // Compute SS components
    // SS_total = Σ_ij (y_ij - grand_mean)^2
    let ss_total: f64 = data
        .iter()
        .flat_map(|s| s.iter())
        .map(|&y| (y - grand_mean).powi(2))
        .sum();

    // SS_subjects = k * Σ_i (subject_mean_i - grand_mean)^2
    let ss_subjects: f64 = k * subject_means
        .iter()
        .map(|m| (m - grand_mean).powi(2))
        .sum::<f64>();

    // SS_conditions (within-subjects effect) = n * Σ_j (condition_mean_j - grand_mean)^2
    let ss_conditions: f64 = n * condition_means
        .iter()
        .map(|m| (m - grand_mean).powi(2))
        .sum::<f64>();

    // SS_error = SS_total - SS_subjects - SS_conditions
    let ss_error = ss_total - ss_subjects - ss_conditions;

    // Degrees of freedom
    let df_subjects = n - 1.0;
    let df_conditions = k - 1.0;
    let df_error = (n - 1.0) * (k - 1.0);
    let df_total = n * k - 1.0;

    // Mean squares
    let ms_subjects = ss_subjects / df_subjects;
    let ms_conditions = ss_conditions / df_conditions;
    let ms_error = ss_error / df_error;

    // F-statistic for condition effect
    let f_stat = ms_conditions / ms_error;

    // P-value
    let f_dist = FisherSnedecor::new(df_conditions, df_error).map_err(|e| {
        StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
    })?;
    let p_value = 1.0 - f_dist.cdf(f_stat);

    // Sphericity test and corrections (only for k >= 3)
    let (sphericity, greenhouse_geisser, huynh_feldt) = if compute_sphericity && n_conditions >= 3 {
        compute_sphericity_corrections(data, f_stat, df_conditions, df_error)?
    } else {
        (None, None, None)
    };

    Ok(RmAnovaResult {
        within_subjects: AnovaTableRow {
            ss: ss_conditions,
            df: df_conditions,
            ms: ms_conditions,
            f_statistic: Some(f_stat),
            p_value: Some(p_value),
        },
        subjects: AnovaTableRow {
            ss: ss_subjects,
            df: df_subjects,
            ms: ms_subjects,
            f_statistic: None,
            p_value: None,
        },
        error: AnovaTableRow {
            ss: ss_error,
            df: df_error,
            ms: ms_error,
            f_statistic: None,
            p_value: None,
        },
        total: AnovaTableRow {
            ss: ss_total,
            df: df_total,
            ms: ss_total / df_total,
            f_statistic: None,
            p_value: None,
        },
        sphericity,
        greenhouse_geisser,
        huynh_feldt,
        grand_mean,
        condition_means,
        subject_means,
    })
}

/// Compute Mauchly's sphericity test and epsilon corrections.
fn compute_sphericity_corrections(
    data: &[&[f64]],
    f_stat: f64,
    df_num: f64,
    df_den: f64,
) -> Result<(
    Option<SphericityResult>,
    Option<CorrectedResult>,
    Option<CorrectedResult>,
)> {
    let n = data.len();
    let k = data[0].len();

    // Compute the covariance matrix of conditions
    // S[i][j] = Cov(condition_i, condition_j)
    let mut cov_matrix: Vec<Vec<f64>> = vec![vec![0.0; k]; k];

    // First compute means of each condition
    let mut condition_means: Vec<f64> = vec![0.0; k];
    for j in 0..k {
        condition_means[j] = data.iter().map(|s| s[j]).sum::<f64>() / n as f64;
    }

    // Compute covariance matrix
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for subject in data.iter() {
                sum += (subject[i] - condition_means[i]) * (subject[j] - condition_means[j]);
            }
            cov_matrix[i][j] = sum / (n - 1) as f64;
        }
    }

    // Compute orthonormalized contrasts (difference contrasts)
    // Transform to k-1 x k-1 matrix
    let p = k - 1;

    // Create contrast matrix C (k-1 x k)
    // Simple difference contrasts: C[i] = condition[i] - condition[i+1]
    let mut c: Vec<Vec<f64>> = vec![vec![0.0; k]; p];
    for i in 0..p {
        c[i][i] = 1.0;
        c[i][i + 1] = -1.0;
    }

    // Orthonormalize using Gram-Schmidt
    let c_orth = orthonormalize(&c);

    // Compute transformed covariance matrix: C * S * C'
    let csc = matrix_multiply_csc(&c_orth, &cov_matrix);

    // Compute eigenvalues of transformed matrix
    let eigenvalues = compute_eigenvalues(&csc);

    // Greenhouse-Geisser epsilon
    let sum_eigenvalues: f64 = eigenvalues.iter().sum();
    let sum_eigenvalues_sq: f64 = eigenvalues.iter().map(|e| e * e).sum();

    let epsilon_gg = if sum_eigenvalues_sq > 1e-15 {
        (sum_eigenvalues * sum_eigenvalues) / (p as f64 * sum_eigenvalues_sq)
    } else {
        1.0
    };

    // Bound GG epsilon between 1/(k-1) and 1.0
    let epsilon_gg = epsilon_gg.max(1.0 / (k - 1) as f64).min(1.0);

    // Huynh-Feldt epsilon (correction for bias in GG)
    let n_f = n as f64;
    let k_f = k as f64;
    let epsilon_hf = if (n_f - 1.0) * (k_f - 1.0) * epsilon_gg - 2.0 > 1e-15 {
        ((n_f - 1.0) * (k_f - 1.0) * epsilon_gg - 2.0)
            / ((k_f - 1.0) * (n_f - 1.0 - (k_f - 1.0) * epsilon_gg))
    } else {
        1.0
    };
    let epsilon_hf = epsilon_hf.max(epsilon_gg).min(1.0);

    // Mauchly's W
    // W = det(C * S * C') / (trace(C * S * C') / p)^p
    let det_csc = matrix_determinant(&csc);
    let trace_csc: f64 = (0..p).map(|i| csc[i][i]).sum();
    let mauchly_w = if (trace_csc / p as f64).abs() > 1e-15 {
        det_csc / (trace_csc / p as f64).powi(p as i32)
    } else {
        1.0
    };
    let mauchly_w = mauchly_w.clamp(0.0, 1.0);

    // Chi-square statistic for Mauchly's test
    // chi^2 = -[n - 1 - (2k^2 - 3k + 3)/(6(k-1))] * ln(W)
    let correction = (2.0 * k_f * k_f - 3.0 * k_f + 3.0) / (6.0 * (k_f - 1.0));
    let chi_sq = if mauchly_w > 1e-15 {
        -(n_f - 1.0 - correction) * mauchly_w.ln()
    } else {
        f64::INFINITY
    };

    // Degrees of freedom for chi-square
    let df_chi = (p * (p + 1)) as f64 / 2.0 - 1.0;

    // P-value for Mauchly's test
    let p_mauchly = if chi_sq.is_finite() && df_chi > 0.0 {
        let chi_dist = ChiSquared::new(df_chi).map_err(|e| {
            StatError::InvalidParameter(format!("Failed to create chi-square distribution: {}", e))
        })?;
        1.0 - chi_dist.cdf(chi_sq)
    } else {
        0.0
    };

    let sphericity = SphericityResult {
        w: mauchly_w,
        chi_square: chi_sq,
        df: df_chi,
        p_value: p_mauchly,
    };

    // Greenhouse-Geisser corrected results
    let df_num_gg = df_num * epsilon_gg;
    let df_den_gg = df_den * epsilon_gg;
    let p_gg = if df_num_gg > 0.0 && df_den_gg > 0.0 {
        let f_dist = FisherSnedecor::new(df_num_gg, df_den_gg).map_err(|e| {
            StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
        })?;
        1.0 - f_dist.cdf(f_stat)
    } else {
        1.0
    };

    let gg = CorrectedResult {
        epsilon: epsilon_gg,
        df_num_corrected: df_num_gg,
        df_den_corrected: df_den_gg,
        f_statistic: f_stat,
        p_value: p_gg,
    };

    // Huynh-Feldt corrected results
    let df_num_hf = df_num * epsilon_hf;
    let df_den_hf = df_den * epsilon_hf;
    let p_hf = if df_num_hf > 0.0 && df_den_hf > 0.0 {
        let f_dist = FisherSnedecor::new(df_num_hf, df_den_hf).map_err(|e| {
            StatError::InvalidParameter(format!("Failed to create F-distribution: {}", e))
        })?;
        1.0 - f_dist.cdf(f_stat)
    } else {
        1.0
    };

    let hf = CorrectedResult {
        epsilon: epsilon_hf,
        df_num_corrected: df_num_hf,
        df_den_corrected: df_den_hf,
        f_statistic: f_stat,
        p_value: p_hf,
    };

    Ok((Some(sphericity), Some(gg), Some(hf)))
}

/// Orthonormalize a matrix of row vectors using Gram-Schmidt.
fn orthonormalize(c: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = c.len();
    let k = c[0].len();
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; k]; p];

    for i in 0..p {
        // Start with original vector
        result[i] = c[i].clone();

        // Subtract projections onto previous vectors
        for j in 0..i {
            let dot_product: f64 = (0..k).map(|m| result[i][m] * result[j][m]).sum();
            for m in 0..k {
                result[i][m] -= dot_product * result[j][m];
            }
        }

        // Normalize
        let norm: f64 = result[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for m in 0..k {
                result[i][m] /= norm;
            }
        }
    }

    result
}

/// Compute C * S * C' where C is p×k and S is k×k.
fn matrix_multiply_csc(c: &[Vec<f64>], s: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = c.len();
    let k = c[0].len();

    // First compute C * S (p×k)
    let mut cs: Vec<Vec<f64>> = vec![vec![0.0; k]; p];
    for i in 0..p {
        for j in 0..k {
            for m in 0..k {
                cs[i][j] += c[i][m] * s[m][j];
            }
        }
    }

    // Then compute (C * S) * C' (p×p)
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            for m in 0..k {
                result[i][j] += cs[i][m] * c[j][m];
            }
        }
    }

    result
}

/// Compute eigenvalues of a symmetric matrix using power iteration.
fn compute_eigenvalues(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // For small matrices, use QR iteration
    let mut matrix: Vec<Vec<f64>> = a.to_vec();
    let mut eigenvalues = vec![0.0; n];

    // Simple QR iteration (sufficient for small covariance matrices)
    for _ in 0..100 {
        // QR decomposition
        let (q, r) = qr_decomposition(&matrix);
        // A = R * Q
        matrix = matrix_multiply(&r, &q);
    }

    // Eigenvalues are on diagonal
    for i in 0..n {
        eigenvalues[i] = matrix[i][i];
    }

    eigenvalues
}

/// QR decomposition using Gram-Schmidt.
fn qr_decomposition(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut r: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for j in 0..n {
        // Start with column j of A
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();

        // Subtract projections
        for i in 0..j {
            r[i][j] = (0..n).map(|k| q[k][i] * a[k][j]).sum();
            for k in 0..n {
                v[k] -= r[i][j] * q[k][i];
            }
        }

        // Normalize
        r[j][j] = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r[j][j] > 1e-15 {
            for k in 0..n {
                q[k][j] = v[k] / r[j][j];
            }
        }
    }

    (q, r)
}

/// Multiply two matrices.
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();

    let mut result: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Compute determinant of a matrix using LU decomposition.
fn matrix_determinant(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return a[0][0];
    }
    if n == 2 {
        return a[0][0] * a[1][1] - a[0][1] * a[1][0];
    }

    // LU decomposition with partial pivoting
    let mut lu: Vec<Vec<f64>> = a.to_vec();
    let mut sign = 1.0;

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = lu[col][col].abs();
        for row in (col + 1)..n {
            if lu[row][col].abs() > max_val {
                max_val = lu[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return 0.0;
        }

        if max_row != col {
            lu.swap(col, max_row);
            sign = -sign;
        }

        for row in (col + 1)..n {
            let factor = lu[row][col] / lu[col][col];
            for j in col..n {
                lu[row][j] -= factor * lu[col][j];
            }
        }
    }

    // Determinant is product of diagonal
    let mut det = sign;
    for i in 0..n {
        det *= lu[i][i];
    }

    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_way_anova_fisher_basic() {
        // Simple 3-group test
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let g3 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = one_way_anova(&[&g1, &g2, &g3], AnovaKind::Fisher).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.group_sizes, vec![5, 5, 5]);
        assert_eq!(result.df_between, 2.0);
        assert_eq!(result.df_within, 12.0);
        assert!(result.statistic > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.ss_between.is_some());
        assert!(result.ss_within.is_some());
    }

    #[test]
    fn test_one_way_anova_welch_basic() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let g3 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = one_way_anova(&[&g1, &g2, &g3], AnovaKind::Welch).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df_between, 2.0);
        assert!(result.df_within > 0.0);
        assert!(result.statistic > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // Welch doesn't produce SS/MS
        assert!(result.ss_between.is_none());
        assert!(result.ss_within.is_none());
    }

    #[test]
    fn test_one_way_anova_two_groups() {
        // 2-group case
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = one_way_anova(&[&g1, &g2], AnovaKind::Fisher).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df_between, 1.0);
        assert_eq!(result.df_within, 8.0);
    }

    #[test]
    fn test_one_way_anova_single_group_error() {
        let g1 = vec![1.0, 2.0, 3.0];
        let result = one_way_anova(&[&g1], AnovaKind::Fisher);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_way_anova_empty_group_error() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2: Vec<f64> = vec![];
        let result = one_way_anova(&[&g1, &g2[..]], AnovaKind::Fisher);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_way_anova_single_observation_error() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![4.0]; // Only 1 observation
        let result = one_way_anova(&[&g1, &g2[..]], AnovaKind::Fisher);
        assert!(result.is_err());
    }

    #[test]
    fn test_ss_total_equals_ss_between_plus_ss_within() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let g3 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = one_way_anova(&[&g1, &g2, &g3], AnovaKind::Fisher).unwrap();

        let ss_total = result.ss_total.unwrap();
        let ss_between = result.ss_between.unwrap();
        let ss_within = result.ss_within.unwrap();

        assert!((ss_total - (ss_between + ss_within)).abs() < 1e-10);
    }
}
