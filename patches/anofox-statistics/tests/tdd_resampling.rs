mod common;

use anofox_statistics::{
    permutation_t_test, Alternative, CircularBlockBootstrap, StationaryBootstrap,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-6;

// ============================================
// Permutation T-Test
// ============================================

#[test]
fn test_permutation_t_statistic() {
    let refs = common::load_reference_scalars("permutation_t.csv");
    let x = common::load_reference_vector("perm_x.csv");
    let y = common::load_reference_vector("perm_y.csv");

    let result = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(42))
        .expect("permutation_t_test should succeed");

    // Verify the t-statistic matches R
    assert_relative_eq!(result.statistic, refs["t_statistic"], epsilon = EPSILON);
}

#[test]
fn test_permutation_t_different_means_rejects_null() {
    // Clearly different means should have low p-value
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
    let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5];

    let result = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(42))
        .expect("permutation_t_test should succeed");

    assert!(
        result.p_value < 0.05,
        "p-value {} should be < 0.05 for clearly different means",
        result.p_value
    );
}

#[test]
fn test_permutation_t_similar_means_accepts_null() {
    // Similar means should have high p-value
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
    let y: Vec<f64> = vec![1.1, 2.1, 3.1, 4.1, 5.1, 1.6, 2.6, 3.6, 4.6, 5.6];

    let result = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(42))
        .expect("permutation_t_test should succeed");

    assert!(
        result.p_value > 0.1,
        "p-value {} should be > 0.1 for similar means",
        result.p_value
    );
}

#[test]
fn test_permutation_t_one_sided_less() {
    // x has lower mean than y
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = permutation_t_test(&x, &y, Alternative::Less, 999, Some(42))
        .expect("permutation_t_test should succeed");

    // H1: mean(x) < mean(y) should be rejected with low p-value
    assert!(
        result.p_value < 0.05,
        "p-value {} should be < 0.05 for Less alternative when x < y",
        result.p_value
    );
}

#[test]
fn test_permutation_t_empty_returns_error() {
    let x: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(permutation_t_test(&x, &y, Alternative::TwoSided, 100, None).is_err());
}

#[test]
fn test_permutation_t_insufficient_data_returns_error() {
    let x = vec![1.0];
    let y = vec![2.0, 3.0, 4.0];
    assert!(permutation_t_test(&x, &y, Alternative::TwoSided, 100, None).is_err());
}

// ============================================
// Stationary Bootstrap
// ============================================

#[test]
fn test_stationary_bootstrap_sample_length() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut bootstrap = StationaryBootstrap::new(3.0, Some(42));

    let sample = bootstrap.sample(&data, 10);

    assert_eq!(
        sample.len(),
        10,
        "Bootstrap sample should have requested length"
    );
}

#[test]
fn test_stationary_bootstrap_values_from_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut bootstrap = StationaryBootstrap::new(2.0, Some(42));

    let sample = bootstrap.sample(&data, 100);

    // All values should come from the original data
    for val in &sample {
        assert!(
            data.contains(val),
            "Bootstrap sample value {} should be from original data",
            val
        );
    }
}

#[test]
fn test_stationary_bootstrap_different_seeds_different_samples() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let mut bootstrap1 = StationaryBootstrap::new(3.0, Some(42));
    let mut bootstrap2 = StationaryBootstrap::new(3.0, Some(123));

    let sample1 = bootstrap1.sample(&data, 20);
    let sample2 = bootstrap2.sample(&data, 20);

    // Different seeds should produce different samples (with high probability)
    assert_ne!(
        sample1, sample2,
        "Different seeds should produce different samples"
    );
}

// ============================================
// Circular Block Bootstrap
// ============================================

#[test]
fn test_circular_block_bootstrap_sample_length() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut bootstrap = CircularBlockBootstrap::new(3, Some(42));

    let sample = bootstrap.sample(&data, 10);

    assert_eq!(
        sample.len(),
        10,
        "Bootstrap sample should have requested length"
    );
}

#[test]
fn test_circular_block_bootstrap_values_from_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut bootstrap = CircularBlockBootstrap::new(2, Some(42));

    let sample = bootstrap.sample(&data, 50);

    // All values should come from the original data
    for val in &sample {
        assert!(
            data.contains(val),
            "Bootstrap sample value {} should be from original data",
            val
        );
    }
}

#[test]
fn test_circular_block_bootstrap_reproducibility() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let mut bootstrap1 = CircularBlockBootstrap::new(3, Some(42));
    let mut bootstrap2 = CircularBlockBootstrap::new(3, Some(42));

    let sample1 = bootstrap1.sample(&data, 20);
    let sample2 = bootstrap2.sample(&data, 20);

    // Same seed should produce same samples
    assert_eq!(
        sample1, sample2,
        "Same seed should produce identical samples"
    );
}

#[test]
fn test_circular_block_bootstrap_wraps_around() {
    // With block size 5 and data length 5, blocks should wrap around
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut bootstrap = CircularBlockBootstrap::new(5, Some(42));

    // Generate many samples to ensure we can handle wrapping
    for _ in 0..100 {
        let sample = bootstrap.sample(&data, 10);
        assert_eq!(sample.len(), 10);
        for val in &sample {
            assert!(data.contains(val));
        }
    }
}
