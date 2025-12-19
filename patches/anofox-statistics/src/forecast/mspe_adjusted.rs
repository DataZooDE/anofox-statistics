use crate::error::Result;
use crate::forecast::spa_common::{
    compute_means, compute_spa_pvalues, compute_standardized, compute_variances, find_best_model,
    validate_model_data,
};

/// Result of the MSPE-Adjusted SPA test
#[derive(Debug, Clone)]
pub struct MSPEAdjustedResult {
    /// The test statistic (maximum standardized Clark-West adjusted performance)
    pub statistic: f64,
    /// The consistent p-value (Hansen, 2005 style)
    pub p_value_consistent: f64,
    /// The upper p-value (more conservative)
    pub p_value_upper: f64,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Index of the best performing model (if any outperforms benchmark)
    pub best_model_idx: Option<usize>,
}

/// Perform the MSPE-Adjusted Superior Predictive Ability test.
///
/// This test combines the Clark-West adjustment for nested models with the
/// SPA bootstrap framework for multiple testing. It answers: "Is the best
/// of these K alternative models statistically significant, accounting for
/// both the nested model structure AND multiple testing?"
///
/// # Arguments
/// * `benchmark_errors` - Forecast errors from the benchmark (restricted) model (length T)
/// * `model_errors` - Forecast errors from K alternative (unrestricted) models (K x T)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_length` - Expected block length for stationary bootstrap
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `MSPEAdjustedResult` containing test statistics and p-values
///
/// # Algorithm
/// For each alternative model k, computes Clark-West adjusted differentials:
///   f_{t,k} = (e_benchmark,t² - e_k,t²) + (e_benchmark,t - e_k,t)²
///
/// Then uses stationary bootstrap to find the distribution of the maximum
/// t-statistic across all K models.
///
/// # References
/// * Clark, T.E. and McCracken, M.W. (2012) "Reality Checks and Comparisons
///   of Nested Predictive Models"
/// * Clark, T.E. and West, K.D. (2007) "Approximately Normal Tests for
///   Equal Predictive Accuracy in Nested Models"
/// * Hansen, P.R. (2005) "A Test for Superior Predictive Ability"
pub fn mspe_adjusted_spa(
    benchmark_errors: &[f64],
    model_errors: &[Vec<f64>],
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<MSPEAdjustedResult> {
    let t = benchmark_errors.len();

    // Validate inputs
    validate_model_data(t, model_errors, "alternative model")?;

    // Compute Clark-West adjusted differentials
    let f = compute_clark_west_differentials(benchmark_errors, model_errors);

    // Compute statistics
    let f_bar = compute_means(&f);
    let variances = compute_variances(&f, &f_bar);
    let standardized = compute_standardized(&f_bar, &variances, t);

    // Observed test statistic: max of standardized performance
    let observed = standardized
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Find best model
    let best_model_idx = find_best_model(&standardized);

    // Bootstrap p-values
    let (p_value_consistent, p_value_upper) =
        compute_spa_pvalues(&f, &f_bar, observed, n_bootstrap, block_length, seed);

    Ok(MSPEAdjustedResult {
        statistic: observed,
        p_value_consistent,
        p_value_upper,
        n_bootstrap,
        best_model_idx,
    })
}

/// Compute Clark-West adjusted differentials for nested model comparison.
///
/// For each model k: f_{t,k} = (e_b² - e_k²) + (e_b - e_k)²
fn compute_clark_west_differentials(benchmark: &[f64], models: &[Vec<f64>]) -> Vec<Vec<f64>> {
    models
        .iter()
        .map(|model_err| {
            benchmark
                .iter()
                .zip(model_err.iter())
                .map(|(e_b, e_k)| {
                    let squared_diff = e_b.powi(2) - e_k.powi(2);
                    let adjustment = (e_b - e_k).powi(2);
                    squared_diff + adjustment
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mspe_adjusted_clearly_better_model() {
        // Benchmark with high errors
        let benchmark: Vec<f64> = (0..100)
            .map(|i| 2.0 + (i as f64 * 0.1).sin() * 0.1)
            .collect();

        // Model with much lower errors
        let model: Vec<Vec<f64>> = vec![(0..100)
            .map(|i| 0.5 + (i as f64 * 0.1).cos() * 0.1)
            .collect()];

        let result = mspe_adjusted_spa(&benchmark, &model, 499, 5.0, Some(42)).unwrap();

        // Should reject null (model is better than benchmark)
        assert!(
            result.p_value_consistent < 0.05,
            "p_value {} should be < 0.05",
            result.p_value_consistent
        );
        assert!(result.best_model_idx == Some(0));
    }

    #[test]
    fn test_mspe_adjusted_no_better_model() {
        // Benchmark and model with similar performance
        let benchmark: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        // Models with similar errors (slight random variation)
        let model1: Vec<f64> = benchmark
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.01 * (i as f64).cos())
            .collect();
        let model2: Vec<f64> = benchmark
            .iter()
            .enumerate()
            .map(|(i, &x)| x - 0.01 * (i as f64).sin())
            .collect();

        let models = vec![model1, model2];

        let result = mspe_adjusted_spa(&benchmark, &models, 499, 5.0, Some(42)).unwrap();

        // Should not reject null (no model clearly better)
        assert!(
            result.p_value_upper > 0.01,
            "p_value_upper {} should be > 0.01",
            result.p_value_upper
        );
    }

    #[test]
    fn test_mspe_adjusted_multiple_models() {
        // Benchmark with moderate errors
        let benchmark: Vec<f64> = (0..100)
            .map(|i| 1.5 + (i as f64 * 0.1).sin() * 0.1)
            .collect();

        // Multiple models with varying performance
        let models = vec![
            // Slightly better
            (0..100)
                .map(|i| 1.2 + (i as f64 * 0.1).cos() * 0.1)
                .collect(),
            // Much better
            (0..100)
                .map(|i| 0.5 + (i as f64 * 0.1).sin() * 0.05)
                .collect(),
            // Worse
            (0..100)
                .map(|i| 2.0 + (i as f64 * 0.1).cos() * 0.1)
                .collect(),
        ];

        let result = mspe_adjusted_spa(&benchmark, &models, 499, 3.0, Some(42)).unwrap();

        // Model 1 (index 1) should be the best
        assert_eq!(result.best_model_idx, Some(1));
    }

    #[test]
    fn test_mspe_adjusted_empty_benchmark() {
        let benchmark: Vec<f64> = vec![];
        let models = vec![vec![1.0, 2.0, 3.0]];

        assert!(mspe_adjusted_spa(&benchmark, &models, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mspe_adjusted_no_models() {
        let benchmark: Vec<f64> = vec![1.0, 2.0, 3.0];
        let models: Vec<Vec<f64>> = vec![];

        assert!(mspe_adjusted_spa(&benchmark, &models, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mspe_adjusted_length_mismatch() {
        let benchmark: Vec<f64> = vec![1.0, 2.0, 3.0];
        let models = vec![vec![1.0, 2.0]]; // Different length

        assert!(mspe_adjusted_spa(&benchmark, &models, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mspe_adjusted_produces_different_results_than_spa() {
        // The Clark-West adjustment should produce different results
        // compared to a standard loss-based approach
        let benchmark: Vec<f64> = (0..100)
            .map(|i| 1.5 + (i as f64 * 0.1).sin() * 0.2)
            .collect();
        let model: Vec<Vec<f64>> = vec![(0..100)
            .map(|i| 1.0 + (i as f64 * 0.1).cos() * 0.2)
            .collect()];

        let result = mspe_adjusted_spa(&benchmark, &model, 499, 5.0, Some(42)).unwrap();

        // Just verify it runs and produces valid output
        assert!(result.statistic.is_finite());
        assert!(result.p_value_consistent >= 0.0 && result.p_value_consistent <= 1.0);
        assert!(result.p_value_upper >= 0.0 && result.p_value_upper <= 1.0);
    }
}
