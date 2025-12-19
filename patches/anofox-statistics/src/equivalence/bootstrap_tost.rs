//! Bootstrap TOST equivalence test.
//!
//! Uses resampling to construct confidence intervals and test equivalence
//! without distributional assumptions.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Perform bootstrap TOST for two independent samples.
///
/// Uses the percentile bootstrap method to construct confidence intervals
/// and test whether the difference in means is practically equivalent to zero.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `bounds` - Equivalence bounds
/// * `alpha` - Significance level (default: 0.05)
/// * `n_bootstrap` - Number of bootstrap samples (default: 1000)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `TostResult` containing bootstrap-based inference
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_bootstrap, EquivalenceBounds};
///
/// let x = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0];
/// let y = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_bootstrap(&x, &y, &bounds, 0.05, 1000, Some(42)).unwrap();
/// println!("Groups equivalent: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::boot_t_TOST(x, y, low_eqbound, high_eqbound, R = n_bootstrap)`
pub fn tost_bootstrap(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Result<TostResult> {
    validate_inputs(x, y, alpha, n_bootstrap)?;

    let nx = x.len();
    let ny = y.len();

    // Compute observed statistic (difference in means)
    let mean_x: f64 = x.iter().sum::<f64>() / nx as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / ny as f64;
    let estimate = mean_x - mean_y;

    // Compute pooled SD for Cohen's d conversion
    let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / (nx - 1) as f64;
    let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / (ny - 1) as f64;
    let pooled_var = ((nx - 1) as f64 * var_x + (ny - 1) as f64 * var_y) / (nx + ny - 2) as f64;
    let pooled_sd = pooled_var.sqrt();

    // Convert bounds
    let (lower_bound, upper_bound) = bounds.to_raw(Some(pooled_sd))?;

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Generate bootstrap distribution of mean difference
    let mut boot_diffs: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let boot_x: Vec<f64> = (0..nx).map(|_| *x.choose(&mut rng).unwrap()).collect();
        let boot_y: Vec<f64> = (0..ny).map(|_| *y.choose(&mut rng).unwrap()).collect();

        let boot_mean_x: f64 = boot_x.iter().sum::<f64>() / nx as f64;
        let boot_mean_y: f64 = boot_y.iter().sum::<f64>() / ny as f64;
        boot_diffs.push(boot_mean_x - boot_mean_y);
    }

    // Sort for percentile calculation
    boot_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Percentile confidence interval at (1 - 2*alpha) level
    let lower_idx = ((n_bootstrap as f64 * alpha).floor() as usize).max(0);
    let upper_idx = ((n_bootstrap as f64 * (1.0 - alpha)).ceil() as usize).min(n_bootstrap - 1);
    let ci = (boot_diffs[lower_idx], boot_diffs[upper_idx]);

    // Bootstrap p-values for TOST
    // Lower test: proportion of bootstrap samples <= lower_bound
    let p_lower = count_proportion(&boot_diffs, lower_bound, true);

    // Upper test: proportion of bootstrap samples >= upper_bound
    let p_upper = count_proportion(&boot_diffs, upper_bound, false);

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: effect <= {:.4}", lower_bound),
            statistic: estimate, // Use point estimate as "statistic"
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: effect >= {:.4}", upper_bound),
            statistic: estimate,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n: nx + ny,
        df: None,
        method: format!("Bootstrap TOST ({} resamples)", n_bootstrap),
    })
}

/// Count proportion of bootstrap samples beyond a threshold.
fn count_proportion(sorted_diffs: &[f64], threshold: f64, less_than: bool) -> f64 {
    let n = sorted_diffs.len();
    if n == 0 {
        return 1.0;
    }

    let count = if less_than {
        // Count how many are <= threshold
        sorted_diffs.iter().filter(|&&d| d <= threshold).count()
    } else {
        // Count how many are >= threshold
        sorted_diffs.iter().filter(|&&d| d >= threshold).count()
    };

    count as f64 / n as f64
}

/// Validate inputs for bootstrap TOST.
fn validate_inputs(x: &[f64], y: &[f64], alpha: f64, n_bootstrap: usize) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }

    if y.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: y.len(),
        });
    }

    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    if n_bootstrap < 100 {
        return Err(StatError::InvalidParameter(format!(
            "n_bootstrap should be at least 100, got {}",
            n_bootstrap
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_tost_equivalent() {
        // Very similar groups
        let x = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0];
        let y = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_bootstrap(&x, &y, &bounds, 0.05, 1000, Some(42)).unwrap();

        // Small difference, likely equivalent
        assert!(result.estimate.abs() < 0.3);
    }

    #[test]
    fn test_bootstrap_tost_not_equivalent() {
        // Different groups
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let y = vec![15.0, 16.0, 17.0, 18.0, 19.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_bootstrap(&x, &y, &bounds, 0.05, 1000, Some(42)).unwrap();

        // Clear difference
        assert!(!result.equivalent);
    }

    #[test]
    fn test_bootstrap_reproducibility() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result1 = tost_bootstrap(&x, &y, &bounds, 0.05, 500, Some(12345)).unwrap();
        let result2 = tost_bootstrap(&x, &y, &bounds, 0.05, 500, Some(12345)).unwrap();

        assert!((result1.ci.0 - result2.ci.0).abs() < 1e-10);
        assert!((result1.ci.1 - result2.ci.1).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_insufficient_samples() {
        let x = vec![1.0];
        let y = vec![2.0, 3.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        assert!(tost_bootstrap(&x, &y, &bounds, 0.05, 1000, None).is_err());
    }

    #[test]
    fn test_bootstrap_too_few_resamples() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.1, 2.1, 3.1];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        assert!(tost_bootstrap(&x, &y, &bounds, 0.05, 50, None).is_err());
    }

    #[test]
    fn test_cohen_d_bounds() {
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1];
        let bounds = EquivalenceBounds::CohenD { d: 0.5 };

        let result = tost_bootstrap(&x, &y, &bounds, 0.05, 500, Some(42)).unwrap();

        // Bounds should be converted based on pooled SD
        assert!(result.bounds.0 < 0.0);
        assert!(result.bounds.1 > 0.0);
    }
}
