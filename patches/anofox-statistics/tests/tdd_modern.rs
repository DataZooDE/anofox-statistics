mod common;

use anofox_statistics::{energy_distance_test_1d, mmd_test_1d};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-6;

// ============================================
// Energy Distance Test
// ============================================

#[test]
fn test_energy_distance_statistic() {
    let refs = common::load_reference_scalars("energy_distance.csv");
    let x = common::load_reference_vector("ed_x.csv");
    let y = common::load_reference_vector("ed_y.csv");

    let result = energy_distance_test_1d(&x, &y, 499, Some(42))
        .expect("energy_distance_test_1d should succeed");

    // Verify the energy distance statistic matches R
    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
}

#[test]
fn test_energy_distance_different_distributions_rejects_null() {
    // Clearly different distributions should have low p-value
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.1).collect();

    let result = energy_distance_test_1d(&x, &y, 499, Some(42))
        .expect("energy_distance_test_1d should succeed");

    assert!(
        result.p_value < 0.05,
        "p-value {} should be < 0.05 for clearly different distributions",
        result.p_value
    );
}

#[test]
fn test_energy_distance_similar_distributions_accepts_null() {
    // Similar distributions should have high p-value
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1 + 0.01).collect();
    let y: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();

    let result = energy_distance_test_1d(&x, &y, 499, Some(42))
        .expect("energy_distance_test_1d should succeed");

    assert!(
        result.p_value > 0.1,
        "p-value {} should be > 0.1 for similar distributions",
        result.p_value
    );
}

#[test]
fn test_energy_distance_empty_returns_error() {
    let x: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(energy_distance_test_1d(&x, &y, 100, None).is_err());
}

// ============================================
// MMD Test (Maximum Mean Discrepancy)
// ============================================

#[test]
fn test_mmd_different_distributions_rejects_null() {
    // Clearly different distributions should have low p-value
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.1).collect();

    let result = mmd_test_1d(&x, &y, 499, Some(42)).expect("mmd_test_1d should succeed");

    assert!(
        result.p_value < 0.05,
        "p-value {} should be < 0.05 for clearly different distributions",
        result.p_value
    );
}

#[test]
fn test_mmd_similar_distributions_accepts_null() {
    // Similar distributions should have high p-value
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1 + 0.01).collect();
    let y: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();

    let result = mmd_test_1d(&x, &y, 499, Some(42)).expect("mmd_test_1d should succeed");

    assert!(
        result.p_value > 0.1,
        "p-value {} should be > 0.1 for similar distributions",
        result.p_value
    );
}

#[test]
fn test_mmd_statistic_positive_for_different_distributions() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = mmd_test_1d(&x, &y, 99, Some(42)).expect("mmd_test_1d should succeed");

    assert!(
        result.statistic > 0.0,
        "MMD statistic should be positive for different distributions"
    );
}

#[test]
fn test_mmd_empty_returns_error() {
    let x: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(mmd_test_1d(&x, &y, 100, None).is_err());
}
