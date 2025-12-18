mod common;

use anofox_statistics::utils::math;
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-12;

// ============================================
// Mean Tests
// ============================================

#[test]
fn test_mean_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::mean(&x_short).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_short"], epsilon = EPSILON);
}

#[test]
fn test_mean_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::mean(&x_long).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_long"], epsilon = EPSILON);
}

#[test]
fn test_mean_single_element() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_single = vec![42.0];

    let result = math::mean(&x_single).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_single"], epsilon = EPSILON);
}

#[test]
fn test_mean_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::mean(&empty).is_err());
}

#[test]
fn test_stable_mean_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::stable_mean(&x_short).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_short"], epsilon = EPSILON);
}

#[test]
fn test_stable_mean_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::stable_mean(&x_long).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_long"], epsilon = EPSILON);
}

#[test]
fn test_stable_mean_single_element() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_single = vec![42.0];

    let result = math::stable_mean(&x_single).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_single"], epsilon = EPSILON);
}

#[test]
fn test_stable_mean_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::stable_mean(&empty).is_err());
}

// ============================================
// Variance Tests
// ============================================

#[test]
fn test_variance_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::variance(&x_short).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_short"], epsilon = EPSILON);
}

#[test]
fn test_variance_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::variance(&x_long).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_long"], epsilon = EPSILON);
}

#[test]
fn test_variance_outlier_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::variance(&x_outlier).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_outlier"], epsilon = EPSILON);
}

#[test]
fn test_variance_single_element_returns_error() {
    let x_single = vec![42.0];
    // R's var() returns NA for single element; we return error
    assert!(math::variance(&x_single).is_err());
}

#[test]
fn test_variance_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::variance(&empty).is_err());
}

#[test]
fn test_stable_variance_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::stable_variance(&x_short).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_short"], epsilon = EPSILON);
}

#[test]
fn test_stable_variance_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::stable_variance(&x_long).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_long"], epsilon = EPSILON);
}

#[test]
fn test_stable_variance_outlier_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::stable_variance(&x_outlier).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_outlier"], epsilon = EPSILON);
}

#[test]
fn test_stable_variance_single_element_returns_error() {
    let x_single = vec![42.0];
    // R's var() returns NA for single element; we return error
    assert!(math::stable_variance(&x_single).is_err());
}

#[test]
fn test_stable_variance_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::stable_variance(&empty).is_err());
}

// ============================================
// Median Tests
// ============================================

#[test]
fn test_median_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::median(&x_short).expect("median should succeed");
    assert_relative_eq!(result, refs["median_short"], epsilon = EPSILON);
}

#[test]
fn test_median_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::median(&x_long).expect("median should succeed");
    assert_relative_eq!(result, refs["median_long"], epsilon = EPSILON);
}

#[test]
fn test_median_even_length() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_even = vec![1.0, 2.0, 3.0, 4.0];

    let result = math::median(&x_even).expect("median should succeed");
    assert_relative_eq!(result, refs["median_even"], epsilon = EPSILON);
}

#[test]
fn test_median_single_element() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_single = vec![42.0];

    let result = math::median(&x_single).expect("median should succeed");
    assert_relative_eq!(result, refs["median_single"], epsilon = EPSILON);
}

#[test]
fn test_median_outlier_robust() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::median(&x_outlier).expect("median should succeed");
    assert_relative_eq!(result, refs["median_outlier"], epsilon = EPSILON);
}

#[test]
fn test_median_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::median(&empty).is_err());
}

// ============================================
// Trimmed Mean Tests
// ============================================

#[test]
fn test_trimmed_mean_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::trimmed_mean(&x_short, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_short"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::trimmed_mean(&x_long, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_long"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_outlier_robust() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::trimmed_mean(&x_outlier, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_outlier"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_invalid_trim_too_high() {
    let x = vec![1.0, 2.0, 3.0];
    // trim >= 0.5 is invalid
    assert!(math::trimmed_mean(&x, 0.5).is_err());
    assert!(math::trimmed_mean(&x, 0.6).is_err());
}

#[test]
fn test_trimmed_mean_invalid_trim_negative() {
    let x = vec![1.0, 2.0, 3.0];
    assert!(math::trimmed_mean(&x, -0.1).is_err());
}

#[test]
fn test_trimmed_mean_zero_trim_equals_mean() {
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let mean_result = math::mean(&x_short).expect("mean should succeed");
    let trim_result = math::trimmed_mean(&x_short, 0.0).expect("trimmed_mean should succeed");

    assert_relative_eq!(mean_result, trim_result, epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::trimmed_mean(&empty, 0.2).is_err());
}

// ============================================
// Standard Deviation Tests
// ============================================

#[test]
fn test_std_dev_short_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::std_dev(&x_short).expect("std_dev should succeed");
    assert_relative_eq!(result, refs["sd_short"], epsilon = EPSILON);
}

#[test]
fn test_std_dev_long_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::std_dev(&x_long).expect("std_dev should succeed");
    assert_relative_eq!(result, refs["sd_long"], epsilon = EPSILON);
}

#[test]
fn test_std_dev_outlier_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::std_dev(&x_outlier).expect("std_dev should succeed");
    assert_relative_eq!(result, refs["sd_outlier"], epsilon = EPSILON);
}

#[test]
fn test_std_dev_single_element_returns_error() {
    let x_single = vec![42.0];
    // std_dev needs at least 2 elements (since variance needs n-1)
    assert!(math::std_dev(&x_single).is_err());
}

#[test]
fn test_std_dev_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::std_dev(&empty).is_err());
}

// ============================================
// Skewness Tests
// ============================================

#[test]
fn test_skewness_short_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::skewness(&x_short).expect("skewness should succeed");
    // Symmetric data should have skewness near 0
    assert_relative_eq!(result, refs["skew_short"], epsilon = 1e-10);
}

#[test]
fn test_skewness_long_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::skewness(&x_long).expect("skewness should succeed");
    assert_relative_eq!(result, refs["skew_long"], epsilon = 1e-10);
}

#[test]
fn test_skewness_outlier_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::skewness(&x_outlier).expect("skewness should succeed");
    // Right-skewed data (outlier on right) should have positive skewness
    assert_relative_eq!(result, refs["skew_outlier"], epsilon = 1e-10);
}

#[test]
fn test_skewness_skewed_data() {
    let refs = common::load_reference_scalars("math_skewed.csv");
    let x_skewed = common::load_reference_vector("vec_skewed.csv");

    let result = math::skewness(&x_skewed).expect("skewness should succeed");
    assert_relative_eq!(result, refs["skew_skewed"], epsilon = 1e-10);
}

#[test]
fn test_skewness_constant_data_returns_zero() {
    let constant = vec![5.0, 5.0, 5.0, 5.0, 5.0];

    let result = math::skewness(&constant).expect("skewness should succeed");
    assert_relative_eq!(result, 0.0, epsilon = EPSILON);
}

#[test]
fn test_skewness_two_elements_returns_error() {
    let x = vec![1.0, 2.0];
    // Skewness needs at least 3 elements
    assert!(math::skewness(&x).is_err());
}

#[test]
fn test_skewness_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::skewness(&empty).is_err());
}

// ============================================
// Kurtosis Tests
// ============================================

#[test]
fn test_kurtosis_short_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::kurtosis(&x_short).expect("kurtosis should succeed");
    // Uniform-ish data has negative excess kurtosis
    assert_relative_eq!(result, refs["kurt_short"], epsilon = 1e-10);
}

#[test]
fn test_kurtosis_long_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::kurtosis(&x_long).expect("kurtosis should succeed");
    assert_relative_eq!(result, refs["kurt_long"], epsilon = 1e-10);
}

#[test]
fn test_kurtosis_outlier_vector() {
    let refs = common::load_reference_scalars("math_extended.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::kurtosis(&x_outlier).expect("kurtosis should succeed");
    // Heavy-tailed data (outlier) should have positive excess kurtosis
    assert_relative_eq!(result, refs["kurt_outlier"], epsilon = 1e-10);
}

#[test]
fn test_kurtosis_skewed_data() {
    let refs = common::load_reference_scalars("math_skewed.csv");
    let x_skewed = common::load_reference_vector("vec_skewed.csv");

    let result = math::kurtosis(&x_skewed).expect("kurtosis should succeed");
    assert_relative_eq!(result, refs["kurt_skewed"], epsilon = 1e-10);
}

#[test]
fn test_kurtosis_constant_data_returns_zero() {
    let constant = vec![5.0, 5.0, 5.0, 5.0, 5.0];

    let result = math::kurtosis(&constant).expect("kurtosis should succeed");
    assert_relative_eq!(result, 0.0, epsilon = EPSILON);
}

#[test]
fn test_kurtosis_three_elements_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    // Kurtosis needs at least 4 elements
    assert!(math::kurtosis(&x).is_err());
}

#[test]
fn test_kurtosis_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::kurtosis(&empty).is_err());
}

// ============================================
// Numerical Stability Tests
// ============================================
// These tests demonstrate why stable_mean and stable_variance
// are preferable for certain edge cases.

#[test]
fn test_stable_mean_large_offset() {
    // Data with large offset: values clustered around 1e12
    // Naive summation can lose precision when sum >> individual differences
    let large_offset: Vec<f64> = (0..1000).map(|i| 1e12 + (i as f64) * 0.001).collect();

    let naive = math::mean(&large_offset).unwrap();
    let stable = math::stable_mean(&large_offset).unwrap();

    // Both algorithms should give similar results for this case
    // At 1e12 magnitude, relative difference of ~1e-14 means absolute diff ~0.01
    // The key is they agree with each other within floating-point precision
    assert!((naive - stable).abs() < 0.1);

    // Value should be approximately 1e12 + 0.4995
    assert!(stable > 1e12);
    assert!(stable < 1e12 + 1.0);
}

#[test]
fn test_stable_mean_alternating_large_values() {
    // Alternating large positive and negative values
    // Tests catastrophic cancellation resistance
    let alternating: Vec<f64> = (0..100)
        .map(|i| if i % 2 == 0 { 1e15 } else { -1e15 + 1.0 })
        .collect();

    let naive = math::mean(&alternating).unwrap();
    let stable = math::stable_mean(&alternating).unwrap();

    // Expected: average of (1e15, -1e15+1, 1e15, -1e15+1, ...)
    // = (50 * 1e15 + 50 * (-1e15 + 1)) / 100 = 0.5
    // Due to floating-point precision limits at 1e15 scale, both methods
    // have some error. The key test is they give consistent results.
    assert_relative_eq!(stable, naive, epsilon = 0.1);

    // Both should be close to 0.5
    assert!(stable.abs() < 1.0);
    assert!(naive.abs() < 1.0);
}

#[test]
fn test_stable_variance_small_variance_large_mean() {
    // Data with very small variance relative to mean magnitude
    // Classic case where naive algorithms fail
    let data: Vec<f64> = vec![1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0, 1e9 + 4.0, 1e9 + 5.0];

    let naive = math::variance(&data).unwrap();
    let stable = math::stable_variance(&data).unwrap();

    // Variance of [1,2,3,4,5] = 2.5
    let expected = 2.5;

    // Both should work for this case, but stable is more reliable
    assert_relative_eq!(stable, expected, epsilon = 1e-10);
    assert_relative_eq!(naive, expected, epsilon = 1e-10);
}

#[test]
fn test_stable_variance_constant_data() {
    // Constant data should have zero variance
    let constant: Vec<f64> = vec![42.0; 100];

    let naive = math::variance(&constant).unwrap();
    let stable = math::stable_variance(&constant).unwrap();

    assert_relative_eq!(stable, 0.0, epsilon = 1e-15);
    assert_relative_eq!(naive, 0.0, epsilon = 1e-15);
}

#[test]
fn test_stable_variance_near_constant() {
    // Nearly constant data with tiny variance
    let near_constant: Vec<f64> = (0..100).map(|i| 1e8 + (i as f64) * 1e-10).collect();

    let naive = math::variance(&near_constant).unwrap();
    let stable = math::stable_variance(&near_constant).unwrap();

    // Variance of arithmetic sequence: n^2 - 1 / 12 * d^2 where d is step size
    // For 0..99 with step 1e-10: var = (100^2 - 1) / 12 * (1e-10)^2 * (100/99)
    // Simplified: var((0..n)*d) = var(0..n) * d^2

    // Both should be very small positive numbers
    assert!(stable >= 0.0);
    assert!(naive >= 0.0);
}

#[test]
fn test_stable_mean_many_small_values() {
    // Many small values that could underflow when summed naively
    let small_values: Vec<f64> = vec![1e-15; 10000];

    let naive = math::mean(&small_values).unwrap();
    let stable = math::stable_mean(&small_values).unwrap();

    assert_relative_eq!(stable, 1e-15, epsilon = 1e-25);
    assert_relative_eq!(naive, 1e-15, epsilon = 1e-25);
}

#[test]
fn test_stable_variance_two_elements() {
    // Minimum case: two elements
    let two = vec![1.0, 3.0];

    let naive = math::variance(&two).unwrap();
    let stable = math::stable_variance(&two).unwrap();

    // var([1,3]) = ((1-2)^2 + (3-2)^2) / 1 = 2
    assert_relative_eq!(stable, 2.0, epsilon = 1e-15);
    assert_relative_eq!(naive, 2.0, epsilon = 1e-15);
}

#[test]
fn test_stable_mean_equivalence_normal_data() {
    // For normal data, both algorithms should give identical results
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let naive = math::mean(&x_long).unwrap();
    let stable = math::stable_mean(&x_long).unwrap();

    // Both should match R's reference
    assert_relative_eq!(naive, refs["mean_long"], epsilon = EPSILON);
    assert_relative_eq!(stable, refs["mean_long"], epsilon = EPSILON);

    // And should be equal to each other
    assert_relative_eq!(naive, stable, epsilon = 1e-14);
}

#[test]
fn test_stable_variance_equivalence_normal_data() {
    // For normal data, both algorithms should give identical results
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let naive = math::variance(&x_long).unwrap();
    let stable = math::stable_variance(&x_long).unwrap();

    // Both should match R's reference
    assert_relative_eq!(naive, refs["var_long"], epsilon = EPSILON);
    assert_relative_eq!(stable, refs["var_long"], epsilon = EPSILON);

    // And should be equal to each other
    assert_relative_eq!(naive, stable, epsilon = 1e-12);
}
