use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank_with_ties;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of Kruskal-Wallis test
#[derive(Debug, Clone)]
pub struct KruskalResult {
    /// The H statistic (chi-squared approximation)
    pub statistic: f64,
    /// Degrees of freedom (k-1)
    pub df: f64,
    /// The p-value
    pub p_value: f64,
}

/// Validate groups and return their sizes and total count.
fn validate_groups(groups: &[&[f64]]) -> Result<(Vec<usize>, usize)> {
    if groups.len() < 2 {
        return Err(StatError::InvalidParameter(
            "Kruskal-Wallis test requires at least 2 groups".to_string(),
        ));
    }

    let mut group_sizes: Vec<usize> = Vec::with_capacity(groups.len());
    let mut n_total = 0usize;

    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(StatError::InvalidParameter(format!(
                "Group {} is empty",
                i + 1
            )));
        }
        group_sizes.push(group.len());
        n_total += group.len();
    }

    Ok((group_sizes, n_total))
}

/// Compute rank sums for each group.
fn compute_rank_sums(ranks: &[f64], group_sizes: &[usize]) -> Vec<f64> {
    let mut rank_sums: Vec<f64> = Vec::with_capacity(group_sizes.len());
    let mut idx = 0;
    for &size in group_sizes {
        let sum: f64 = ranks[idx..idx + size].iter().sum();
        rank_sums.push(sum);
        idx += size;
    }
    rank_sums
}

/// Compute tie correction factor: C = 1 - sum(t^3 - t) / (N^3 - N).
fn compute_tie_correction(tie_sizes: &[usize], n: f64) -> f64 {
    let tie_sum: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    1.0 - tie_sum / (n * n * n - n)
}

/// Perform Kruskal-Wallis H test for comparing multiple independent groups.
///
/// This is the nonparametric equivalent of one-way ANOVA.
///
/// # Arguments
/// * `groups` - Slice of slices, each containing one group's data
///
/// # Returns
/// * `KruskalResult` containing H statistic, df, and p-value
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalResult> {
    let (group_sizes, n_total) = validate_groups(groups)?;
    let k = groups.len();
    let n_f = n_total as f64;

    // Combine all data and rank
    let combined: Vec<f64> = groups.iter().flat_map(|g| g.iter().cloned()).collect();
    let (ranks, tie_sizes) = rank_with_ties(&combined)?;

    // Compute rank sums
    let rank_sums = compute_rank_sums(&ranks, &group_sizes);

    // H statistic (uncorrected): H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1)
    let sum_term: f64 = rank_sums
        .iter()
        .zip(group_sizes.iter())
        .map(|(&r_sum, &n_i)| r_sum * r_sum / n_i as f64)
        .sum();
    let h_uncorrected = (12.0 / (n_f * (n_f + 1.0))) * sum_term - 3.0 * (n_f + 1.0);

    // Apply tie correction
    let c = compute_tie_correction(&tie_sizes, n_f);
    let h = h_uncorrected / c;

    // Degrees of freedom and p-value
    let df = (k - 1) as f64;
    let chi_sq = ChiSquared::new(df).unwrap();
    let p_value = 1.0 - chi_sq.cdf(h);

    Ok(KruskalResult {
        statistic: h,
        df,
        p_value,
    })
}
