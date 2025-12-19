//! Shared utilities for SPA-style bootstrap tests.
//!
//! This module provides common functionality used by both `spa_test` and
//! `mspe_adjusted_spa` to reduce code duplication and complexity.

use crate::error::{Result, StatError};
use crate::resampling::bootstrap::StationaryBootstrap;

/// Validate that model data has consistent dimensions.
pub fn validate_model_data(
    benchmark_len: usize,
    model_data: &[Vec<f64>],
    data_name: &str,
) -> Result<()> {
    if benchmark_len == 0 {
        return Err(StatError::EmptyData);
    }

    if model_data.is_empty() {
        return Err(StatError::InvalidParameter(format!(
            "At least one {} required",
            data_name
        )));
    }

    for (i, data) in model_data.iter().enumerate() {
        if data.len() != benchmark_len {
            return Err(StatError::InvalidParameter(format!(
                "Model {} has {} observations, expected {}",
                i,
                data.len(),
                benchmark_len
            )));
        }
    }

    Ok(())
}

/// Compute sample means for each model's differentials.
pub fn compute_means(differentials: &[Vec<f64>]) -> Vec<f64> {
    let t = differentials[0].len() as f64;
    differentials
        .iter()
        .map(|d| d.iter().sum::<f64>() / t)
        .collect()
}

/// Compute sample variances of the means for each model.
pub fn compute_variances(differentials: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    let t = differentials[0].len() as f64;
    differentials
        .iter()
        .zip(means.iter())
        .map(|(d, &mean)| {
            let var: f64 = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (t - 1.0);
            var / t // variance of the mean
        })
        .collect()
}

/// Compute standardized statistics for each model.
/// Handles near-zero variance by scaling the mean directly.
pub fn compute_standardized(means: &[f64], variances: &[f64], t: usize) -> Vec<f64> {
    let t_f = t as f64;
    means
        .iter()
        .zip(variances.iter())
        .map(|(&mean, &var)| {
            if var > 1e-14 {
                mean * t_f.sqrt() / var.sqrt()
            } else {
                mean * t_f.sqrt() * 1e6
            }
        })
        .collect()
}

/// Find the index of the best performing model (highest standardized statistic).
pub fn find_best_model(standardized: &[f64]) -> Option<usize> {
    standardized
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

/// Compute bootstrap p-values (consistent and upper) for SPA-style tests.
///
/// This is the core bootstrap loop shared by `spa_test` and `mspe_adjusted_spa`.
///
/// # Arguments
/// * `differentials` - Loss differentials for each model (K x T)
/// * `d_bar` - Sample means of differentials
/// * `observed` - Observed test statistic (max of standardized stats)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_length` - Expected block length for stationary bootstrap
/// * `seed` - Optional random seed
///
/// # Returns
/// Tuple of (p_value_consistent, p_value_upper)
pub fn compute_spa_pvalues(
    differentials: &[Vec<f64>],
    d_bar: &[f64],
    observed: f64,
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> (f64, f64) {
    let t = differentials[0].len();
    let t_f = t as f64;

    let mut bootstrap = StationaryBootstrap::new(block_length, seed);
    let mut count_consistent = 0usize;
    let mut count_upper = 0usize;

    for _ in 0..n_bootstrap {
        let mut boot_max_consistent = f64::NEG_INFINITY;
        let mut boot_max_upper = f64::NEG_INFINITY;

        for (i, di) in differentials.iter().enumerate() {
            let boot_sample = bootstrap.sample(di, t);
            let boot_mean: f64 = boot_sample.iter().sum::<f64>() / t_f;

            // Variance of bootstrap sample
            let boot_var: f64 = boot_sample
                .iter()
                .map(|x| (x - boot_mean).powi(2))
                .sum::<f64>()
                / (t_f - 1.0)
                / t_f;

            // Consistent p-value: centered bootstrap statistic
            let centered_mean = boot_mean - d_bar[i];
            let boot_stat_consistent = standardize_single(centered_mean, boot_var, t_f);

            // Upper p-value: use g(d_bar) = max(d_bar, 0)
            let g_d_bar = d_bar[i].max(0.0);
            let upper_centered = boot_mean - g_d_bar;
            let boot_stat_upper = standardize_single(upper_centered, boot_var, t_f);

            boot_max_consistent = boot_max_consistent.max(boot_stat_consistent);
            boot_max_upper = boot_max_upper.max(boot_stat_upper);
        }

        if boot_max_consistent >= observed {
            count_consistent += 1;
        }
        if boot_max_upper >= observed {
            count_upper += 1;
        }
    }

    let p_consistent = (count_consistent as f64 + 1.0) / (n_bootstrap as f64 + 1.0);
    let p_upper = (count_upper as f64 + 1.0) / (n_bootstrap as f64 + 1.0);

    (p_consistent, p_upper)
}

/// Standardize a single mean value, handling near-zero variance.
#[inline]
fn standardize_single(mean: f64, var: f64, t_f: f64) -> f64 {
    if var > 1e-14 {
        mean * t_f.sqrt() / var.sqrt()
    } else {
        mean * t_f.sqrt() * 1e6
    }
}
