mod common;

use anofox_statistics::{
    tost_bootstrap, tost_correlation, tost_prop_one, tost_prop_two, tost_t_test_one_sample,
    tost_t_test_paired, tost_t_test_two_sample, tost_wilcoxon_paired, tost_wilcoxon_two_sample,
    tost_yuen, CorrelationTostMethod, EquivalenceBounds,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-6;

// ============================================
// One-Sample TOST t-test
// ============================================

#[test]
fn test_tost_one_sample_equivalent() {
    // Data with mean close to 0
    let x = vec![0.1, -0.1, 0.05, -0.05, 0.0, 0.08, -0.03, 0.02, 0.01, -0.02];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();

    // Should be equivalent - mean is ~0.0065, well within ±0.5
    assert!(result.equivalent);
    assert!(result.tost_p_value < 0.05);
    assert!(result.lower_test.rejected);
    assert!(result.upper_test.rejected);
}

#[test]
fn test_tost_one_sample_not_equivalent() {
    // Data with mean clearly outside bounds
    let x = vec![2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.05, 1.95, 2.1, 1.9];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).unwrap();

    // Should NOT be equivalent - mean ~2.0 is outside ±0.5
    assert!(!result.equivalent);
    assert!(result.tost_p_value > 0.05);
}

// ============================================
// Two-Sample TOST t-test
// ============================================

#[test]
fn test_tost_two_sample_equivalent() {
    // Two groups with very similar means
    let x = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0, 9.95, 10.05];
    let y = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1, 9.98, 10.02];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

    // Should be equivalent
    assert!(result.equivalent);
    assert!(result.estimate.abs() < 0.1);
}

#[test]
fn test_tost_two_sample_not_equivalent() {
    // Two groups with different means
    let x = vec![10.0, 10.1, 9.9, 10.2, 10.0, 10.05, 9.95, 10.1, 9.9, 10.0];
    let y = vec![12.0, 12.1, 11.9, 12.2, 12.0, 12.05, 11.95, 12.1, 11.9, 12.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

    // Should NOT be equivalent - difference ~2.0
    assert!(!result.equivalent);
}

#[test]
fn test_tost_two_sample_cohen_d_bounds() {
    // Test with Cohen's d bounds
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
    let bounds = EquivalenceBounds::CohenD { d: 0.5 };

    let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

    // Bounds should be computed using pooled SD
    assert!(result.bounds.0 < 0.0);
    assert!(result.bounds.1 > 0.0);
    // Small difference should be equivalent with d = 0.5 bounds
    assert!(result.estimate.abs() < 0.2);
}

// ============================================
// Paired TOST t-test
// ============================================

#[test]
fn test_tost_paired_equivalent() {
    // Paired data with small differences
    let before = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0, 14.0, 15.0];
    let after = vec![
        10.1, 11.9, 11.05, 13.1, 10.45, 11.55, 12.45, 10.05, 14.05, 14.95,
    ];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_t_test_paired(&before, &after, &bounds, 0.05).unwrap();

    // Should be equivalent - differences are tiny
    assert!(result.equivalent);
    assert!(result.estimate.abs() < 0.1);
}

// ============================================
// Correlation TOST
// ============================================

#[test]
fn test_tost_correlation_high_correlation_not_equivalent_to_zero() {
    // Data with strong positive correlation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![1.1, 2.2, 2.8, 4.1, 5.0, 6.2, 7.1, 7.9, 9.1, 10.0];

    let bounds = EquivalenceBounds::Symmetric { delta: 0.3 };
    let result =
        tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson).unwrap();

    // Strong correlation (~0.99) should NOT be equivalent to zero within ±0.3
    assert!(!result.equivalent);
    assert!(result.estimate > 0.9);
}

#[test]
fn test_tost_correlation_near_zero() {
    // Data with correlation close to zero (noise)
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0];

    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
    let result =
        tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson).unwrap();

    // Near-zero correlation should be close to equivalent
    assert!(result.estimate.abs() < 0.4);
}

// ============================================
// Proportion TOST
// ============================================

#[test]
fn test_tost_prop_one_equivalent() {
    // Proportion close to 0.5
    let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
    let result = tost_prop_one(52, 100, 0.5, &bounds, 0.05).unwrap();

    // 52% vs 50% with ±10% bounds
    assert!(result.estimate.abs() < 0.1);
    assert_relative_eq!(result.estimate, 0.02, epsilon = EPSILON);
}

#[test]
fn test_tost_prop_one_not_equivalent() {
    // Proportion far from null
    let bounds = EquivalenceBounds::Symmetric { delta: 0.05 };
    let result = tost_prop_one(70, 100, 0.5, &bounds, 0.05).unwrap();

    // 70% vs 50% with ±5% bounds should NOT be equivalent
    assert!(!result.equivalent);
    assert_relative_eq!(result.estimate, 0.20, epsilon = EPSILON);
}

#[test]
fn test_tost_prop_two_equivalent() {
    // Very similar proportions
    let bounds = EquivalenceBounds::Symmetric { delta: 0.1 };
    let result = tost_prop_two(51, 100, 49, 100, &bounds, 0.05).unwrap();

    // 51% vs 49% difference is 0.02
    assert_relative_eq!(result.estimate, 0.02, epsilon = EPSILON);
}

// ============================================
// Wilcoxon TOST
// ============================================

#[test]
fn test_tost_wilcoxon_paired_equivalent() {
    // Small differences
    let before = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0, 14.0, 15.0];
    let after = vec![
        10.1, 11.9, 11.05, 13.05, 10.45, 11.55, 12.45, 10.05, 14.05, 14.95,
    ];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_wilcoxon_paired(&before, &after, &bounds, 0.05).unwrap();

    // Small differences should produce small estimate
    assert!(result.estimate.abs() < 0.2);
}

#[test]
fn test_tost_wilcoxon_two_sample_not_equivalent() {
    // Clearly different groups
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
    let y = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_wilcoxon_two_sample(&x, &y, &bounds, 0.05).unwrap();

    // Clear difference
    assert!(!result.equivalent);
    assert!(result.estimate.abs() > 5.0);
}

// ============================================
// Bootstrap TOST
// ============================================

#[test]
fn test_tost_bootstrap_equivalent() {
    // Very similar groups
    let x = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0, 9.95, 10.05];
    let y = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1, 9.98, 10.02];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_bootstrap(&x, &y, &bounds, 0.05, 1000, Some(42)).unwrap();

    // Small difference
    assert!(result.estimate.abs() < 0.2);
}

#[test]
fn test_tost_bootstrap_not_equivalent() {
    // Different groups
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let y = vec![15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_bootstrap(&x, &y, &bounds, 0.05, 1000, Some(42)).unwrap();

    // Clear difference
    assert!(!result.equivalent);
}

#[test]
fn test_tost_bootstrap_reproducibility() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result1 = tost_bootstrap(&x, &y, &bounds, 0.05, 500, Some(12345)).unwrap();
    let result2 = tost_bootstrap(&x, &y, &bounds, 0.05, 500, Some(12345)).unwrap();

    // Same seed should give same results
    assert_relative_eq!(result1.ci.0, result2.ci.0, epsilon = EPSILON);
    assert_relative_eq!(result1.ci.1, result2.ci.1, epsilon = EPSILON);
    assert_relative_eq!(result1.estimate, result2.estimate, epsilon = EPSILON);
}

// ============================================
// Yuen TOST
// ============================================

#[test]
fn test_tost_yuen_equivalent() {
    // Similar groups
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

    // Small difference (~0.1)
    assert!(result.estimate.abs() < 0.3);
}

#[test]
fn test_tost_yuen_robust_to_outliers() {
    // Group x has an outlier
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 100.0];
    let y = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1];
    let bounds = EquivalenceBounds::Symmetric { delta: 2.0 };

    // With 20% trimming, the outlier should be removed
    let result = tost_yuen(&x, &y, &bounds, 0.05, 0.2).unwrap();

    // Trimmed means should be similar (outlier removed)
    // Without trimming, mean of x would be ~22.6, with trimming ~13.6
    assert!(result.estimate.abs() < 2.0);
}

// ============================================
// Edge Cases and Error Handling
// ============================================

#[test]
fn test_equivalence_bounds_validation() {
    // Test that invalid bounds are rejected
    assert!(EquivalenceBounds::symmetric(-0.5).is_err());
    assert!(EquivalenceBounds::raw(0.5, -0.5).is_err());
    assert!(EquivalenceBounds::cohen_d(0.0).is_err());
}

#[test]
fn test_tost_invalid_alpha() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 0.0).is_err());
    assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 1.0).is_err());
    assert!(tost_t_test_one_sample(&x, 0.0, &bounds, -0.1).is_err());
}

#[test]
fn test_tost_insufficient_data() {
    let x = vec![1.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

    assert!(tost_t_test_one_sample(&x, 0.0, &bounds, 0.05).is_err());
}

// ============================================
// CI Coverage Tests
// ============================================

#[test]
fn test_tost_ci_contains_estimate() {
    // The (1-2*alpha) CI should always contain the point estimate
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let y = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 2.0 };

    let result = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();

    assert!(result.ci.0 <= result.estimate);
    assert!(result.estimate <= result.ci.1);
}

#[test]
fn test_tost_symmetry() {
    // TOST should be symmetric: TOST(x, y) estimate = -TOST(y, x) estimate
    let x = vec![10.0, 11.0, 12.0, 13.0, 14.0];
    let y = vec![11.0, 12.0, 13.0, 14.0, 15.0];
    let bounds = EquivalenceBounds::Symmetric { delta: 2.0 };

    let result_xy = tost_t_test_two_sample(&x, &y, &bounds, 0.05, false).unwrap();
    let result_yx = tost_t_test_two_sample(&y, &x, &bounds, 0.05, false).unwrap();

    assert_relative_eq!(result_xy.estimate, -result_yx.estimate, epsilon = EPSILON);
}
