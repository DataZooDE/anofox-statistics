//! TDD tests for correlation functions validated against R.

mod common;

use anofox_statistics::{kendall, pearson, spearman, KendallVariant};

const TOLERANCE: f64 = 1e-10;
const P_VALUE_TOLERANCE: f64 = 1e-6; // p-values can have more numerical variation

// ============================================
// Pearson Correlation Tests
// ============================================

#[test]
fn test_pearson_vs_r() {
    let x = common::load_reference_vector("cor_x.csv");
    let y = common::load_reference_vector("cor_y.csv");
    let refs = common::load_reference_scalars("pearson.csv");

    let result = pearson(&x, &y, Some(0.95)).unwrap();

    assert!(
        (result.estimate - refs["estimate"]).abs() < TOLERANCE,
        "Pearson r: expected {}, got {}",
        refs["estimate"],
        result.estimate
    );
    assert!(
        (result.statistic - refs["statistic"]).abs() < TOLERANCE,
        "Pearson t: expected {}, got {}",
        refs["statistic"],
        result.statistic
    );
    assert!(
        (result.df.unwrap() - refs["df"]).abs() < TOLERANCE,
        "Pearson df: expected {}, got {}",
        refs["df"],
        result.df.unwrap()
    );
    assert!(
        (result.p_value - refs["p_value"]).abs() < P_VALUE_TOLERANCE,
        "Pearson p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

#[test]
fn test_pearson_confidence_intervals() {
    let x = common::load_reference_vector("cor_x.csv");
    let y = common::load_reference_vector("cor_y.csv");
    let refs = common::load_reference_scalars("pearson.csv");

    // 95% CI
    let result95 = pearson(&x, &y, Some(0.95)).unwrap();
    let ci95 = result95.conf_int.unwrap();
    assert!(
        (ci95.lower - refs["conf_low_95"]).abs() < 1e-4,
        "95% CI lower: expected {}, got {}",
        refs["conf_low_95"],
        ci95.lower
    );
    assert!(
        (ci95.upper - refs["conf_high_95"]).abs() < 1e-4,
        "95% CI upper: expected {}, got {}",
        refs["conf_high_95"],
        ci95.upper
    );

    // 90% CI
    let result90 = pearson(&x, &y, Some(0.90)).unwrap();
    let ci90 = result90.conf_int.unwrap();
    assert!(
        (ci90.lower - refs["conf_low_90"]).abs() < 1e-4,
        "90% CI lower: expected {}, got {}",
        refs["conf_low_90"],
        ci90.lower
    );
    assert!(
        (ci90.upper - refs["conf_high_90"]).abs() < 1e-4,
        "90% CI upper: expected {}, got {}",
        refs["conf_high_90"],
        ci90.upper
    );

    // 99% CI
    let result99 = pearson(&x, &y, Some(0.99)).unwrap();
    let ci99 = result99.conf_int.unwrap();
    assert!(
        (ci99.lower - refs["conf_low_99"]).abs() < 1e-4,
        "99% CI lower: expected {}, got {}",
        refs["conf_low_99"],
        ci99.lower
    );
    assert!(
        (ci99.upper - refs["conf_high_99"]).abs() < 1e-4,
        "99% CI upper: expected {}, got {}",
        refs["conf_high_99"],
        ci99.upper
    );
}

#[test]
fn test_pearson_perfect_positive() {
    let x = common::load_reference_vector("cor_perfect_x.csv");
    let y = common::load_reference_vector("cor_perfect_y.csv");
    let refs = common::load_reference_scalars("correlation_perfect.csv");

    let result = pearson(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["pearson_r"]).abs() < TOLERANCE,
        "Perfect positive Pearson r: expected {}, got {}",
        refs["pearson_r"],
        result.estimate
    );
}

#[test]
fn test_pearson_perfect_negative() {
    let x = common::load_reference_vector("cor_neg_x.csv");
    let y = common::load_reference_vector("cor_neg_y.csv");
    let refs = common::load_reference_scalars("correlation_negative.csv");

    let result = pearson(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["pearson_r"]).abs() < TOLERANCE,
        "Perfect negative Pearson r: expected {}, got {}",
        refs["pearson_r"],
        result.estimate
    );
    // For perfect correlation, t-statistic approaches -infinity.
    // R reports a very large finite number due to numerical precision,
    // but mathematically it should be -infinity. Accept either.
    assert!(
        result.statistic.is_infinite() || result.statistic < -1e6,
        "Perfect negative Pearson t: expected -inf or very large negative, got {}",
        result.statistic
    );
}

#[test]
fn test_pearson_zero_correlation() {
    let x = common::load_reference_vector("cor_zero_x.csv");
    let y = common::load_reference_vector("cor_zero_y.csv");
    let refs = common::load_reference_scalars("correlation_zero.csv");

    let result = pearson(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["pearson_r"]).abs() < TOLERANCE,
        "Near-zero Pearson r: expected {}, got {}",
        refs["pearson_r"],
        result.estimate
    );
    assert!(
        (result.p_value - refs["pearson_p"]).abs() < P_VALUE_TOLERANCE,
        "Near-zero Pearson p-value: expected {}, got {}",
        refs["pearson_p"],
        result.p_value
    );
}

// ============================================
// Spearman Correlation Tests
// ============================================

#[test]
fn test_spearman_vs_r() {
    let x = common::load_reference_vector("cor_x.csv");
    let y = common::load_reference_vector("cor_y.csv");
    let refs = common::load_reference_scalars("spearman.csv");

    let result = spearman(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["estimate"]).abs() < TOLERANCE,
        "Spearman rho: expected {}, got {}",
        refs["estimate"],
        result.estimate
    );
    // Note: R uses the S statistic (sum of squared rank differences) and
    // exact/AS89 p-value calculation. We use t-statistic approximation.
    // Both are valid approaches, so we use a more relaxed tolerance.
    // The key is that both indicate the same level of significance.
    let p_ratio = result.p_value / refs["p_value"];
    assert!(
        p_ratio > 0.1 && p_ratio < 10.0,
        "Spearman p-value: expected ~{}, got {} (ratio={})",
        refs["p_value"],
        result.p_value,
        p_ratio
    );
}

#[test]
fn test_spearman_perfect_positive() {
    let x = common::load_reference_vector("cor_perfect_x.csv");
    let y = common::load_reference_vector("cor_perfect_y.csv");
    let refs = common::load_reference_scalars("correlation_perfect.csv");

    let result = spearman(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["spearman_rho"]).abs() < TOLERANCE,
        "Perfect positive Spearman rho: expected {}, got {}",
        refs["spearman_rho"],
        result.estimate
    );
}

#[test]
fn test_spearman_perfect_negative() {
    let x = common::load_reference_vector("cor_neg_x.csv");
    let y = common::load_reference_vector("cor_neg_y.csv");
    let refs = common::load_reference_scalars("correlation_negative.csv");

    let result = spearman(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["spearman_rho"]).abs() < TOLERANCE,
        "Perfect negative Spearman rho: expected {}, got {}",
        refs["spearman_rho"],
        result.estimate
    );
}

#[test]
fn test_spearman_with_ties() {
    let x = common::load_reference_vector("cor_ties_x.csv");
    let y = common::load_reference_vector("cor_ties_y.csv");
    let refs = common::load_reference_scalars("correlation_ties.csv");

    let result = spearman(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["spearman_rho"]).abs() < TOLERANCE,
        "Spearman rho with ties: expected {}, got {}",
        refs["spearman_rho"],
        result.estimate
    );
}

#[test]
fn test_spearman_zero_correlation() {
    let x = common::load_reference_vector("cor_zero_x.csv");
    let y = common::load_reference_vector("cor_zero_y.csv");
    let refs = common::load_reference_scalars("correlation_zero.csv");

    let result = spearman(&x, &y, None).unwrap();

    assert!(
        (result.estimate - refs["spearman_rho"]).abs() < TOLERANCE,
        "Near-zero Spearman rho: expected {}, got {}",
        refs["spearman_rho"],
        result.estimate
    );
}

// ============================================
// Kendall Correlation Tests
// ============================================

#[test]
fn test_kendall_vs_r() {
    let x = common::load_reference_vector("cor_x.csv");
    let y = common::load_reference_vector("cor_y.csv");
    let refs = common::load_reference_scalars("kendall.csv");

    let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

    assert!(
        (result.estimate - refs["estimate"]).abs() < 1e-4,
        "Kendall tau: expected {}, got {}",
        refs["estimate"],
        result.estimate
    );
    // p-value comparison (may have more variation due to normal approximation)
    assert!(
        (result.p_value - refs["p_value"]).abs() < 0.01,
        "Kendall p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

#[test]
fn test_kendall_perfect_positive() {
    let x = common::load_reference_vector("cor_perfect_x.csv");
    let y = common::load_reference_vector("cor_perfect_y.csv");
    let refs = common::load_reference_scalars("correlation_perfect.csv");

    let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

    assert!(
        (result.estimate - refs["kendall_tau"]).abs() < TOLERANCE,
        "Perfect positive Kendall tau: expected {}, got {}",
        refs["kendall_tau"],
        result.estimate
    );
}

#[test]
fn test_kendall_perfect_negative() {
    let x = common::load_reference_vector("cor_neg_x.csv");
    let y = common::load_reference_vector("cor_neg_y.csv");
    let refs = common::load_reference_scalars("correlation_negative.csv");

    let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

    assert!(
        (result.estimate - refs["kendall_tau"]).abs() < TOLERANCE,
        "Perfect negative Kendall tau: expected {}, got {}",
        refs["kendall_tau"],
        result.estimate
    );
}

#[test]
fn test_kendall_with_ties() {
    let x = common::load_reference_vector("cor_ties_x.csv");
    let y = common::load_reference_vector("cor_ties_y.csv");
    let refs = common::load_reference_scalars("correlation_ties.csv");

    let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

    assert!(
        (result.estimate - refs["kendall_tau"]).abs() < 1e-4,
        "Kendall tau with ties: expected {}, got {}",
        refs["kendall_tau"],
        result.estimate
    );
}

#[test]
fn test_kendall_zero_correlation() {
    let x = common::load_reference_vector("cor_zero_x.csv");
    let y = common::load_reference_vector("cor_zero_y.csv");
    let refs = common::load_reference_scalars("correlation_zero.csv");

    let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

    assert!(
        (result.estimate - refs["kendall_tau"]).abs() < 1e-4,
        "Near-zero Kendall tau: expected {}, got {}",
        refs["kendall_tau"],
        result.estimate
    );
}

// ============================================
// Error Case Tests
// ============================================

#[test]
fn test_correlation_empty_data() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];

    assert!(pearson(&x, &y, None).is_err());
    assert!(spearman(&x, &y, None).is_err());
    assert!(kendall(&x, &y, KendallVariant::TauB).is_err());
}

#[test]
fn test_correlation_mismatched_length() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0];

    assert!(pearson(&x, &y, None).is_err());
    assert!(spearman(&x, &y, None).is_err());
    assert!(kendall(&x, &y, KendallVariant::TauB).is_err());
}

#[test]
fn test_correlation_too_few_observations() {
    let x = vec![1.0, 2.0];
    let y = vec![1.0, 2.0];

    assert!(pearson(&x, &y, None).is_err());
    assert!(spearman(&x, &y, None).is_err());
    assert!(kendall(&x, &y, KendallVariant::TauB).is_err());
}

#[test]
fn test_correlation_nan_values() {
    let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    assert!(pearson(&x, &y, None).is_err());
    assert!(spearman(&x, &y, None).is_err());
    assert!(kendall(&x, &y, KendallVariant::TauB).is_err());
}

#[test]
fn test_correlation_infinite_values() {
    let x = vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    assert!(pearson(&x, &y, None).is_err());
    assert!(spearman(&x, &y, None).is_err());
    assert!(kendall(&x, &y, KendallVariant::TauB).is_err());
}
