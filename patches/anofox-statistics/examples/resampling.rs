//! Resampling methods example
//!
//! Run with: cargo run --example resampling

use anofox_statistics::{
    permutation_t_test, Alternative, CircularBlockBootstrap, PermutationEngine, StationaryBootstrap,
};

fn main() {
    println!("=== Resampling Methods ===\n");
    println!("Resampling methods provide distribution-free inference by repeatedly");
    println!("sampling from the observed data.\n");

    // ========== PERMUTATION T-TEST ==========
    println!("========== PERMUTATION T-TEST ==========\n");
    println!("Tests if two samples come from the same distribution by randomly");
    println!("shuffling group labels and computing the test statistic.\n");

    let x = vec![2.3, 2.5, 2.1, 2.8, 2.4, 2.6, 2.2, 2.7];
    let y = vec![1.8, 2.0, 1.7, 2.1, 1.9, 2.2, 1.6, 2.0];

    println!("Sample data:");
    println!("  Group X: {:?}", x);
    println!("  Group Y: {:?}", y);
    println!();

    let result = permutation_t_test(&x, &y, Alternative::TwoSided, 9999, Some(42)).unwrap();

    println!("Results (9999 permutations, seed=42 for reproducibility):");
    println!("  Observed t-statistic: {:.4}", result.statistic);
    println!("  p-value:              {:.4}", result.p_value);
    println!("  Permutations used:    {}", result.n_permutations);
    println!();
    println!(
        "  Interpretation: {}",
        if result.p_value < 0.05 {
            "Groups differ significantly (p < 0.05)"
        } else {
            "No significant difference (p >= 0.05)"
        }
    );
    println!();

    // One-sided test
    println!("One-sided test (H1: X > Y):");
    let result_greater = permutation_t_test(&x, &y, Alternative::Greater, 9999, Some(42)).unwrap();
    println!("  p-value: {:.4}", result_greater.p_value);
    println!(
        "  Decision: {}",
        if result_greater.p_value < 0.05 {
            "X is significantly greater than Y"
        } else {
            "No evidence X > Y"
        }
    );
    println!();

    // ========== PERMUTATION ENGINE (CUSTOM STATISTIC) ==========
    println!("========== PERMUTATION ENGINE (CUSTOM STATISTIC) ==========\n");
    println!("Use PermutationEngine for custom test statistics.\n");

    // Custom statistic: difference in medians
    let median_diff = |a: &[f64], b: &[f64]| -> f64 {
        let median = |v: &[f64]| -> f64 {
            let mut sorted: Vec<f64> = v.to_vec();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let n = sorted.len();
            if n % 2 == 1 {
                sorted[n / 2]
            } else {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            }
        };
        median(a) - median(b)
    };

    println!("Using difference in medians as test statistic:");

    let engine = PermutationEngine::new(9999).with_seed(42);
    let custom_result = engine
        .run(&x, &y, median_diff, Alternative::TwoSided)
        .unwrap();

    println!(
        "  Observed median difference: {:.4}",
        custom_result.statistic
    );
    println!("  p-value:                    {:.4}", custom_result.p_value);
    println!();

    // ========== STATIONARY BOOTSTRAP ==========
    println!("========== STATIONARY BOOTSTRAP ==========\n");
    println!("For time series data with unknown dependence structure.");
    println!("Uses random-length blocks (geometric distribution).");
    println!("Preserves temporal dependencies while resampling.\n");

    // Simulated time series (autocorrelated)
    let time_series: Vec<f64> = (0..50)
        .map(|i| {
            let t = i as f64 * 0.1;
            t.sin() + 0.5 * t.cos() + (i % 3) as f64 * 0.1
        })
        .collect();

    println!("Time series data ({} observations):", time_series.len());
    println!("  First 10 values: {:?}", &time_series[..10]);
    println!();

    let mut stationary = StationaryBootstrap::new(5.0, Some(42));
    let bootstrap_sample = stationary.sample(&time_series, time_series.len());

    println!("Stationary bootstrap (expected block length = 5.0):");
    println!("  Sample length: {}", bootstrap_sample.len());
    println!("  First 10 values: {:?}", &bootstrap_sample[..10]);
    println!();

    // Generate multiple samples for confidence intervals
    let n_bootstrap = 1000;
    let mut stationary_multi = StationaryBootstrap::new(5.0, Some(42));
    let samples = stationary_multi.samples(&time_series, time_series.len(), n_bootstrap);

    // Compute bootstrap mean of means
    let means: Vec<f64> = samples
        .iter()
        .map(|s| s.iter().sum::<f64>() / s.len() as f64)
        .collect();
    let mut sorted_means = means.clone();
    sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ci_lower = sorted_means[(0.025 * n_bootstrap as f64) as usize];
    let ci_upper = sorted_means[(0.975 * n_bootstrap as f64) as usize];
    let original_mean = time_series.iter().sum::<f64>() / time_series.len() as f64;

    println!(
        "Bootstrap confidence interval for the mean ({} samples):",
        n_bootstrap
    );
    println!("  Original mean: {:.4}", original_mean);
    println!("  95% CI:        [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!();

    // ========== CIRCULAR BLOCK BOOTSTRAP ==========
    println!("========== CIRCULAR BLOCK BOOTSTRAP ==========\n");
    println!("Uses fixed-length blocks with wrap-around at boundaries.");
    println!("Simpler than stationary bootstrap, good when block length is known.\n");

    let mut circular = CircularBlockBootstrap::new(5, Some(42));
    let circular_sample = circular.sample(&time_series, time_series.len());

    println!("Circular block bootstrap (block length = 5):");
    println!("  Sample length: {}", circular_sample.len());
    println!("  First 10 values: {:?}", &circular_sample[..10]);
    println!();

    // Show wrap-around behavior
    println!("Demonstrating wrap-around with small data:");
    let small_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut circular_small = CircularBlockBootstrap::new(3, Some(123));
    let small_sample = circular_small.sample(&small_data, 10);
    println!("  Original data: {:?}", small_data);
    println!("  Bootstrap sample (10 values): {:?}", small_sample);
    println!("  Note: Values wrap around from end to beginning.");
    println!();

    // ========== COMPARING BOOTSTRAP METHODS ==========
    println!("========== COMPARING BOOTSTRAP METHODS ==========\n");
    println!("When to use each method:\n");
    println!("  Stationary Bootstrap:");
    println!("    - Unknown dependence structure");
    println!("    - Adaptive block lengths");
    println!("    - More flexible for varying autocorrelation");
    println!();
    println!("  Circular Block Bootstrap:");
    println!("    - Known or estimated optimal block length");
    println!("    - Simpler implementation");
    println!("    - Fixed-length blocks");
    println!();
    println!("  Permutation Test:");
    println!("    - Testing group differences");
    println!("    - No temporal structure");
    println!("    - Distribution-free p-values");
}
