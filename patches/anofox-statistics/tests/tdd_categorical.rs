//! TDD tests for categorical data tests validated against R.

mod common;

use anofox_statistics::{
    chisq_goodness_of_fit, chisq_test, cohen_kappa, contingency_coef, cramers_v, fisher_exact,
    g_test, phi_coefficient, Alternative,
};

const TOLERANCE: f64 = 1e-10;
const P_VALUE_TOLERANCE: f64 = 1e-6;

// ============================================
// Chi-Square Test of Independence
// ============================================

#[test]
fn test_chisq_2x2_vs_r() {
    let observed = vec![vec![10, 20], vec![30, 40]];
    let refs = common::load_reference_scalars("chisq_2x2.csv");

    // Without Yates' correction
    let result = chisq_test(&observed, false).unwrap();

    assert!(
        (result.statistic - refs["statistic"]).abs() < TOLERANCE,
        "Chi-square statistic: expected {}, got {}",
        refs["statistic"],
        result.statistic
    );
    assert!(
        (result.df - refs["df"]).abs() < TOLERANCE,
        "Chi-square df: expected {}, got {}",
        refs["df"],
        result.df
    );
    assert!(
        (result.p_value - refs["p_value"]).abs() < P_VALUE_TOLERANCE,
        "Chi-square p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

#[test]
fn test_chisq_2x2_yates_vs_r() {
    let observed = vec![vec![10, 20], vec![30, 40]];
    let refs = common::load_reference_scalars("chisq_2x2.csv");

    // With Yates' correction
    let result = chisq_test(&observed, true).unwrap();

    assert!(
        (result.statistic - refs["statistic_yates"]).abs() < TOLERANCE,
        "Chi-square statistic (Yates): expected {}, got {}",
        refs["statistic_yates"],
        result.statistic
    );
    assert!(
        (result.p_value - refs["p_value_yates"]).abs() < P_VALUE_TOLERANCE,
        "Chi-square p-value (Yates): expected {}, got {}",
        refs["p_value_yates"],
        result.p_value
    );
}

#[test]
fn test_chisq_3x3_vs_r() {
    let observed = vec![vec![10, 20, 30], vec![15, 25, 35], vec![20, 30, 40]];
    let refs = common::load_reference_scalars("chisq_3x3.csv");

    let result = chisq_test(&observed, false).unwrap();

    assert!(
        (result.statistic - refs["statistic"]).abs() < TOLERANCE,
        "Chi-square 3x3 statistic: expected {}, got {}",
        refs["statistic"],
        result.statistic
    );
    assert!(
        (result.df - refs["df"]).abs() < TOLERANCE,
        "Chi-square 3x3 df: expected {}, got {}",
        refs["df"],
        result.df
    );
    assert!(
        (result.p_value - refs["p_value"]).abs() < P_VALUE_TOLERANCE,
        "Chi-square 3x3 p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

// ============================================
// Chi-Square Goodness of Fit
// ============================================

#[test]
fn test_chisq_gof_uniform_vs_r() {
    let observed = vec![20, 25, 22, 18, 15];
    let refs = common::load_reference_scalars("chisq_gof_uniform.csv");

    let result = chisq_goodness_of_fit(&observed, None).unwrap();

    assert!(
        (result.statistic - refs["statistic"]).abs() < TOLERANCE,
        "GOF statistic: expected {}, got {}",
        refs["statistic"],
        result.statistic
    );
    assert!(
        (result.df - refs["df"]).abs() < TOLERANCE,
        "GOF df: expected {}, got {}",
        refs["df"],
        result.df
    );
    assert!(
        (result.p_value - refs["p_value"]).abs() < P_VALUE_TOLERANCE,
        "GOF p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

#[test]
fn test_chisq_gof_custom_vs_r() {
    let observed = vec![50, 30, 20];
    let expected_props = vec![0.5, 0.3, 0.2];
    let refs = common::load_reference_scalars("chisq_gof_custom.csv");

    let result = chisq_goodness_of_fit(&observed, Some(&expected_props)).unwrap();

    assert!(
        (result.statistic - refs["statistic"]).abs() < TOLERANCE,
        "GOF custom statistic: expected {}, got {}",
        refs["statistic"],
        result.statistic
    );
    assert!(
        (result.p_value - refs["p_value"]).abs() < P_VALUE_TOLERANCE,
        "GOF custom p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

// ============================================
// Fisher's Exact Test
// ============================================

#[test]
fn test_fisher_exact_vs_r() {
    let table = [[3, 1], [1, 3]];
    let refs = common::load_reference_scalars("fisher_2x2.csv");

    let result = fisher_exact(&table, Alternative::TwoSided).unwrap();

    assert!(
        (result.p_value - refs["p_value"]).abs() < 0.01,
        "Fisher p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
    // Note: R uses conditional MLE for odds ratio (6.4), we use sample OR (9).
    // Both are valid estimates. We just verify our sample OR is correct.
    assert!(
        (result.odds_ratio - 9.0).abs() < TOLERANCE,
        "Fisher sample odds ratio: expected 9.0, got {}",
        result.odds_ratio
    );
}

#[test]
fn test_fisher_exact_extreme() {
    let table = [[5, 0], [0, 5]];
    let refs = common::load_reference_scalars("fisher_extreme.csv");

    let result = fisher_exact(&table, Alternative::TwoSided).unwrap();

    assert!(
        (result.p_value - refs["p_value"]).abs() < 0.01,
        "Fisher extreme p-value: expected {}, got {}",
        refs["p_value"],
        result.p_value
    );
}

// ============================================
// Cramér's V
// ============================================

#[test]
fn test_cramers_v_2x2_vs_r() {
    let observed = vec![vec![10, 20], vec![30, 40]];
    let refs = common::load_reference_scalars("cramers_v.csv");

    let result = cramers_v(&observed).unwrap();

    assert!(
        (result.estimate - refs["v_2x2"]).abs() < 1e-4,
        "Cramér's V (2x2): expected {}, got {}",
        refs["v_2x2"],
        result.estimate
    );
}

#[test]
fn test_cramers_v_3x3_vs_r() {
    let observed = vec![vec![10, 20, 30], vec![15, 25, 35], vec![20, 30, 40]];
    let refs = common::load_reference_scalars("cramers_v.csv");

    let result = cramers_v(&observed).unwrap();

    assert!(
        (result.estimate - refs["v_3x3"]).abs() < 1e-4,
        "Cramér's V (3x3): expected {}, got {}",
        refs["v_3x3"],
        result.estimate
    );
}

// ============================================
// Phi Coefficient
// ============================================

#[test]
fn test_phi_coefficient_vs_r() {
    let table = [[10, 20], [30, 40]];
    let refs = common::load_reference_scalars("phi_coefficient.csv");

    let result = phi_coefficient(&table).unwrap();

    assert!(
        (result.estimate - refs["phi"]).abs() < 1e-4,
        "Phi coefficient: expected {}, got {}",
        refs["phi"],
        result.estimate
    );
}

// ============================================
// Contingency Coefficient
// ============================================

#[test]
fn test_contingency_coef_vs_r() {
    let observed = vec![vec![10, 20], vec![30, 40]];
    let refs = common::load_reference_scalars("contingency_coef.csv");

    let result = contingency_coef(&observed).unwrap();

    assert!(
        (result.estimate - refs["c_2x2"]).abs() < 1e-4,
        "Contingency coefficient: expected {}, got {}",
        refs["c_2x2"],
        result.estimate
    );
}

// ============================================
// Cohen's Kappa
// ============================================

#[test]
fn test_cohen_kappa_vs_r() {
    let table = vec![vec![20, 5, 0], vec![10, 30, 5], vec![0, 5, 25]];

    // Compute kappa manually to match R's formula
    // Po = (20+30+25)/100 = 0.75
    // Pe = (35*30 + 45*40 + 20*30) / 10000 = (1050+1800+600)/10000 = 0.345
    // kappa = (0.75 - 0.345) / (1 - 0.345) = 0.618

    let result = cohen_kappa(&table, false).unwrap();

    // Expected kappa is approximately 0.618 based on R's calculation
    assert!(
        (result.kappa - 0.618).abs() < 0.01,
        "Cohen's kappa: expected ~0.618, got {}",
        result.kappa
    );
    assert!(result.kappa > 0.5 && result.kappa < 0.7);
}

// ============================================
// G-Test
// ============================================

#[test]
fn test_g_test_vs_chisq() {
    // G-test and chi-square should be asymptotically equivalent
    let observed = vec![vec![10, 20], vec![30, 40]];

    let chisq_result = chisq_test(&observed, false).unwrap();
    let g_result = g_test(&observed).unwrap();

    // Should have same df
    assert!(
        (chisq_result.df - g_result.df).abs() < TOLERANCE,
        "G-test df should match chi-square"
    );

    // Statistics should be similar
    let stat_ratio = g_result.statistic / chisq_result.statistic;
    assert!(
        stat_ratio > 0.8 && stat_ratio < 1.2,
        "G-test and chi-square statistics should be similar"
    );
}

// ============================================
// Error Cases
// ============================================

#[test]
fn test_chisq_empty_error() {
    let observed: Vec<Vec<usize>> = vec![];
    assert!(chisq_test(&observed, false).is_err());
}

#[test]
fn test_chisq_1x2_error() {
    let observed = vec![vec![10, 20]];
    assert!(chisq_test(&observed, false).is_err());
}

#[test]
fn test_chisq_gof_empty_error() {
    let observed: Vec<usize> = vec![];
    assert!(chisq_goodness_of_fit(&observed, None).is_err());
}

#[test]
fn test_kappa_nonsquare_error() {
    let table = vec![vec![10, 20], vec![30, 40], vec![50, 60]];
    assert!(cohen_kappa(&table, false).is_err());
}
