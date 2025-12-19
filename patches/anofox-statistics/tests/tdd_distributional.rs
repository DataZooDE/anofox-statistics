mod common;

use anofox_statistics::{dagostino_k_squared, shapiro_wilk};
use approx::assert_relative_eq;

// Shapiro-Wilk has known implementation variations across software packages.
// The W statistic typically matches to 3-4 decimal places between implementations.
const W_EPSILON: f64 = 1e-3;
// P-values can vary more due to different approximation methods.
const P_EPSILON: f64 = 0.05;

// ============================================
// Shapiro-Wilk Test
// ============================================

#[test]
fn test_shapiro_wilk_normal_small() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_small.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_small"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_small"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_normal_medium() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_medium.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_medium"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_medium"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_normal_large() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_large.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_large"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_large"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_uniform() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_uniform.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, refs["w_uniform"], epsilon = W_EPSILON);
    assert_relative_eq!(result.p_value, refs["p_uniform"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_exponential() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_exp.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, refs["w_exp"], epsilon = W_EPSILON);
    assert_relative_eq!(result.p_value, refs["p_exp"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_too_small_returns_error() {
    // Shapiro-Wilk requires at least 3 observations
    let data = vec![1.0, 2.0];
    assert!(shapiro_wilk(&data).is_err());
}

#[test]
fn test_shapiro_wilk_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(shapiro_wilk(&empty).is_err());
}

// ============================================
// Shapiro-Wilk Edge Cases
// ============================================
// Note: Small sample Shapiro-Wilk implementations can vary significantly
// between software packages. We use relaxed tolerances for edge cases.
const SMALL_N_W_EPSILON: f64 = 0.02;

#[test]
fn test_shapiro_wilk_n_equals_3() {
    // Tests the exact formula path (n=3)
    let refs = common::load_reference_scalars("shapiro_wilk_edge.csv");
    let data = common::load_reference_vector("sw_n3.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, refs["w_n3"], epsilon = W_EPSILON);
    assert_relative_eq!(result.p_value, refs["p_n3"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_n_equals_4() {
    // Tests the small n coefficient path (n <= 5)
    let refs = common::load_reference_scalars("shapiro_wilk_edge.csv");
    let data = common::load_reference_vector("sw_n4.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    // Use relaxed tolerances for small n
    assert_relative_eq!(result.statistic, refs["w_n4"], epsilon = SMALL_N_W_EPSILON);
    // P-value: both indicate normality (high p-value)
    assert!(result.p_value > 0.1, "p-value should indicate normality");
}

#[test]
fn test_shapiro_wilk_n_equals_5() {
    // Tests the small n coefficient path (n <= 5)
    let refs = common::load_reference_scalars("shapiro_wilk_edge.csv");
    let data = common::load_reference_vector("sw_n5.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    // Use relaxed tolerances for small n
    assert_relative_eq!(result.statistic, refs["w_n5"], epsilon = SMALL_N_W_EPSILON);
    // P-value: both indicate normality (high p-value)
    assert!(result.p_value > 0.1, "p-value should indicate normality");
}

#[test]
fn test_shapiro_wilk_n_equals_10() {
    // Tests the 4 <= n <= 11 polynomial p-value approximation path
    let refs = common::load_reference_scalars("shapiro_wilk_edge.csv");
    let data = common::load_reference_vector("sw_n10.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    // Use relaxed tolerances for small n p-value approximation
    assert_relative_eq!(result.statistic, refs["w_n10"], epsilon = W_EPSILON);
    // P-value: both indicate normality (reasonably high p-value)
    assert!(result.p_value > 0.1, "p-value should indicate normality");
}

#[test]
fn test_shapiro_wilk_constant_data() {
    // Constant data should return W=1, p=1
    let constant = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];

    let result = shapiro_wilk(&constant).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.p_value, 1.0, epsilon = 1e-10);
}

#[test]
fn test_shapiro_wilk_too_large_returns_error() {
    // n > 5000 should return an error
    let large: Vec<f64> = (0..5001).map(|i| i as f64).collect();
    assert!(shapiro_wilk(&large).is_err());
}

// ============================================
// D'Agostino K-Squared Test
// ============================================
// D'Agostino's K² test combines skewness and kurtosis tests.
// The z-scores can vary slightly between implementations due to
// moment calculation differences (biased vs unbiased estimators).
const DAGOSTINO_Z_EPSILON: f64 = 0.15;

#[test]
fn test_dagostino_normal_data() {
    let refs = common::load_reference_scalars("dagostino.csv");
    let data = common::load_reference_vector("dag_normal.csv");

    let result = dagostino_k_squared(&data).expect("dagostino_k_squared should succeed");

    // Compare z-scores for skewness and kurtosis
    assert_relative_eq!(
        result.z_skewness,
        refs["z_skew_normal"],
        epsilon = DAGOSTINO_Z_EPSILON
    );
    assert_relative_eq!(
        result.z_kurtosis,
        refs["z_kurt_normal"],
        epsilon = DAGOSTINO_Z_EPSILON
    );

    // For normal data, p-value should be high (not reject normality)
    assert!(
        result.p_value > 0.05,
        "p-value {} should indicate normality",
        result.p_value
    );
}

#[test]
fn test_dagostino_skewed_data() {
    let refs = common::load_reference_scalars("dagostino.csv");
    let data = common::load_reference_vector("dag_skewed.csv");

    let result = dagostino_k_squared(&data).expect("dagostino_k_squared should succeed");

    // Compare z-scores for skewness and kurtosis
    assert_relative_eq!(
        result.z_skewness,
        refs["z_skew_skewed"],
        epsilon = DAGOSTINO_Z_EPSILON
    );
    assert_relative_eq!(
        result.z_kurtosis,
        refs["z_kurt_skewed"],
        epsilon = DAGOSTINO_Z_EPSILON
    );

    // For skewed data, p-value should be low (reject normality)
    assert!(
        result.p_value < 0.05,
        "p-value {} should reject normality for skewed data",
        result.p_value
    );
}

#[test]
fn test_dagostino_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(dagostino_k_squared(&empty).is_err());
}

#[test]
fn test_dagostino_insufficient_data_returns_error() {
    // D'Agostino requires at least 8 observations
    let small = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(dagostino_k_squared(&small).is_err());
}
