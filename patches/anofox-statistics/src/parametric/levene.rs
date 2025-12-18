use crate::error::{Result, StatError};
use crate::utils::math::median;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Result of Brown-Forsythe/Levene test
#[derive(Debug, Clone)]
pub struct LeveneResult {
    /// The F-statistic
    pub statistic: f64,
    /// Numerator degrees of freedom (k-1)
    pub df1: f64,
    /// Denominator degrees of freedom (N-k)
    pub df2: f64,
    /// The p-value
    pub p_value: f64,
}

/// Validate groups and return total count.
fn validate_bf_groups(groups: &[&[f64]]) -> Result<usize> {
    if groups.len() < 2 {
        return Err(StatError::InvalidParameter(
            "Brown-Forsythe test requires at least 2 groups".to_string(),
        ));
    }

    let mut n_total = 0usize;
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(StatError::InvalidParameter(format!(
                "Group {} is empty",
                i + 1
            )));
        }
        n_total += group.len();
    }
    Ok(n_total)
}

/// Compute absolute deviations from group medians (z-values).
fn compute_z_values(groups: &[&[f64]]) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut z_values: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
    let mut group_sizes: Vec<usize> = Vec::with_capacity(groups.len());

    for group in groups {
        let group_median = median(group)?;
        let z: Vec<f64> = group.iter().map(|x| (x - group_median).abs()).collect();
        group_sizes.push(group.len());
        z_values.push(z);
    }

    Ok((z_values, group_sizes))
}

/// Compute between-group sum of squares.
fn compute_ss_between(group_sizes: &[usize], z_means: &[f64], z_grand_mean: f64) -> f64 {
    group_sizes
        .iter()
        .zip(z_means.iter())
        .map(|(&n_i, &z_mean_i)| n_i as f64 * (z_mean_i - z_grand_mean).powi(2))
        .sum()
}

/// Compute within-group sum of squares.
fn compute_ss_within(z_values: &[Vec<f64>], z_means: &[f64]) -> f64 {
    z_values
        .iter()
        .zip(z_means.iter())
        .map(|(z_group, &z_mean)| {
            z_group
                .iter()
                .map(|&z_ij| (z_ij - z_mean).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// Perform the Brown-Forsythe test for homogeneity of variances.
///
/// This is Levene's test using the median instead of the mean,
/// making it more robust to non-normality.
///
/// # Arguments
/// * `groups` - Slice of slices, each containing one group's data
///
/// # Returns
/// * `LeveneResult` containing F-statistic, degrees of freedom, and p-value
pub fn brown_forsythe(groups: &[&[f64]]) -> Result<LeveneResult> {
    let n_total = validate_bf_groups(groups)?;
    let k = groups.len();

    // Compute z-values (absolute deviations from group medians)
    let (z_values, group_sizes) = compute_z_values(groups)?;

    // Compute group means and grand mean of z values
    let z_means: Vec<f64> = z_values
        .iter()
        .map(|z| z.iter().sum::<f64>() / z.len() as f64)
        .collect();
    let z_grand_mean: f64 = z_values.iter().flatten().sum::<f64>() / n_total as f64;

    // Sum of squares
    let ss_between = compute_ss_between(&group_sizes, &z_means, z_grand_mean);
    let ss_within = compute_ss_within(&z_values, &z_means);

    // Degrees of freedom and mean squares
    let df1 = (k - 1) as f64;
    let df2 = (n_total - k) as f64;
    let ms_between = ss_between / df1;
    let ms_within = ss_within / df2;

    // F-statistic and p-value
    let f_stat = ms_between / ms_within;
    let f_dist = FisherSnedecor::new(df1, df2).unwrap();
    let p_value = 1.0 - f_dist.cdf(f_stat);

    Ok(LeveneResult {
        statistic: f_stat,
        df1,
        df2,
        p_value,
    })
}
