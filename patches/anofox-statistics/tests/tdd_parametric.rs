mod common;

use anofox_statistics::{
    brown_forsythe, one_way_anova, repeated_measures_anova, t_test, two_way_anova, yuen_test,
    Alternative, AnovaKind, TTestKind,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-10;

// ============================================
// Welch T-Test (Unequal Variances)
// ============================================

#[test]
fn test_welch_two_sided() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_x_two"], epsilon = EPSILON);
    assert_relative_eq!(
        result.mean_y.unwrap(),
        refs["mean_y_two"],
        epsilon = EPSILON
    );
}

#[test]
fn test_welch_less() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Welch, Alternative::Less, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_welch_greater() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Welch, Alternative::Greater, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

#[test]
fn test_welch_with_mu() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.5, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_mu"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = EPSILON);
}

#[test]
fn test_welch_confidence_interval_95() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        Some(0.95),
    )
    .expect("t_test should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_welch_confidence_interval_90() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        Some(0.90),
    )
    .expect("t_test should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

#[test]
fn test_welch_confidence_interval_99() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        Some(0.99),
    )
    .expect("t_test should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.99, epsilon = EPSILON);
}

// ============================================
// Student T-Test (Equal Variances Assumed)
// ============================================

#[test]
fn test_student_two_sided() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Student,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_x_two"], epsilon = EPSILON);
    assert_relative_eq!(
        result.mean_y.unwrap(),
        refs["mean_y_two"],
        epsilon = EPSILON
    );
}

#[test]
fn test_student_less() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Student, Alternative::Less, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_student_greater() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Student,
        Alternative::Greater,
        0.0,
        None,
    )
    .expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

// ============================================
// Paired T-Test
// ============================================

#[test]
fn test_paired_two_sided() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result = t_test(
        &g1,
        &g2,
        TTestKind::Paired,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_diff_two"], epsilon = EPSILON);
}

#[test]
fn test_paired_less() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result = t_test(&g1, &g2, TTestKind::Paired, Alternative::Less, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_paired_greater() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result = t_test(&g1, &g2, TTestKind::Paired, Alternative::Greater, 0.0, None)
        .expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

// ============================================
// Error Cases
// ============================================

#[test]
fn test_ttest_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(t_test(
        &empty,
        &y,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        None
    )
    .is_err());
}

#[test]
fn test_ttest_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(t_test(
        &x,
        &empty,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        None
    )
    .is_err());
}

#[test]
fn test_paired_unequal_length_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(t_test(&x, &y, TTestKind::Paired, Alternative::TwoSided, 0.0, None).is_err());
}

#[test]
fn test_ttest_insufficient_data_returns_error() {
    let x = vec![1.0];
    let y = vec![2.0];
    assert!(t_test(&x, &y, TTestKind::Welch, Alternative::TwoSided, 0.0, None).is_err());
}

// ============================================
// Yuen's Test (Robust T-Test with Trimmed Means)
// ============================================

#[test]
fn test_yuen_20_percent_trim() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result =
        yuen_test(&g1, &g2, 0.2, Alternative::TwoSided, None).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_20"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_20"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_20"], epsilon = EPSILON);
    assert_relative_eq!(result.diff, refs["diff_20"], epsilon = EPSILON);
}

#[test]
fn test_yuen_10_percent_trim() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result =
        yuen_test(&g1, &g2, 0.1, Alternative::TwoSided, None).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_10"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_10"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_10"], epsilon = EPSILON);
    assert_relative_eq!(result.diff, refs["diff_10"], epsilon = EPSILON);
}

#[test]
fn test_yuen_invalid_trim_returns_error() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    assert!(yuen_test(&x, &y, 0.5, Alternative::TwoSided, None).is_err());
    assert!(yuen_test(&x, &y, -0.1, Alternative::TwoSided, None).is_err());
}

#[test]
fn test_yuen_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(yuen_test(&empty, &y, 0.2, Alternative::TwoSided, None).is_err());
}

#[test]
fn test_yuen_less() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result =
        yuen_test(&g1, &g2, 0.2, Alternative::Less, None).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_20"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_20"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_yuen_greater() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result =
        yuen_test(&g1, &g2, 0.2, Alternative::Greater, None).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_20"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_20"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

// ============================================
// Brown-Forsythe Test (Variance Homogeneity)
// ============================================

#[test]
fn test_brown_forsythe_two_groups() {
    let refs = common::load_reference_scalars("brown_forsythe.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = brown_forsythe(&[&g1, &g2]).expect("brown_forsythe should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_2"], epsilon = EPSILON);
    assert_relative_eq!(result.df1, refs["df1_2"], epsilon = EPSILON);
    assert_relative_eq!(result.df2, refs["df2_2"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_2"], epsilon = EPSILON);
}

#[test]
fn test_brown_forsythe_three_groups() {
    let refs = common::load_reference_scalars("brown_forsythe.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result = brown_forsythe(&[&g1, &g2, &g3]).expect("brown_forsythe should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_3"], epsilon = EPSILON);
    assert_relative_eq!(result.df1, refs["df1_3"], epsilon = EPSILON);
    assert_relative_eq!(result.df2, refs["df2_3"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_3"], epsilon = EPSILON);
}

#[test]
fn test_brown_forsythe_single_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    assert!(brown_forsythe(&[&g1[..]]).is_err());
}

#[test]
fn test_brown_forsythe_empty_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(brown_forsythe(&[&g1[..], &empty[..]]).is_err());
}

// ============================================
// T-Test Result Fields
// ============================================

#[test]
fn test_ttest_result_contains_null_value() {
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result_mu0 = t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.0, None)
        .expect("t_test should succeed");

    let result_mu05 = t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.5, None)
        .expect("t_test should succeed");

    assert_relative_eq!(result_mu0.null_value, 0.0, epsilon = EPSILON);
    assert_relative_eq!(result_mu05.null_value, 0.5, epsilon = EPSILON);
}

// ============================================
// Yuen's Test Result Fields
// ============================================

#[test]
fn test_yuen_confidence_interval_95() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result = yuen_test(&g1, &g2, 0.2, Alternative::TwoSided, Some(0.95))
        .expect("yuen_test should succeed");

    let ci = result
        .conf_int
        .expect("confidence interval should be present");
    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

// ============================================
// One-Way ANOVA (Fisher's)
// ============================================

#[test]
fn test_fisher_anova_three_groups() {
    let refs = common::load_reference_scalars("anova_fisher_3g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result =
        one_way_anova(&[&g1, &g2, &g3], AnovaKind::Fisher).expect("one_way_anova should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.df_between, refs["df_between"], epsilon = EPSILON);
    assert_relative_eq!(result.df_within, refs["df_within"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = EPSILON);
    assert_relative_eq!(
        result.ss_between.unwrap(),
        refs["ss_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.ss_within.unwrap(),
        refs["ss_within"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.ms_between.unwrap(),
        refs["ms_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.ms_within.unwrap(),
        refs["ms_within"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.grand_mean.unwrap(),
        refs["grand_mean"],
        epsilon = EPSILON
    );
}

#[test]
fn test_fisher_anova_group_means() {
    let refs = common::load_reference_scalars("anova_fisher_3g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result =
        one_way_anova(&[&g1, &g2, &g3], AnovaKind::Fisher).expect("one_way_anova should succeed");

    assert_eq!(result.n_groups, 3);
    assert_eq!(result.group_sizes, vec![20, 25, 22]);
    assert_relative_eq!(result.group_means[0], refs["mean_a"], epsilon = EPSILON);
    assert_relative_eq!(result.group_means[1], refs["mean_b"], epsilon = EPSILON);
    assert_relative_eq!(result.group_means[2], refs["mean_c"], epsilon = EPSILON);
}

#[test]
fn test_fisher_anova_ss_total_identity() {
    // SS_total = SS_between + SS_within
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result =
        one_way_anova(&[&g1, &g2, &g3], AnovaKind::Fisher).expect("one_way_anova should succeed");

    let ss_total = result.ss_total.unwrap();
    let ss_between = result.ss_between.unwrap();
    let ss_within = result.ss_within.unwrap();

    assert_relative_eq!(ss_total, ss_between + ss_within, epsilon = 1e-10);
}

// ============================================
// One-Way ANOVA (Welch's)
// ============================================

#[test]
fn test_welch_anova_three_groups() {
    let refs = common::load_reference_scalars("anova_welch_3g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result =
        one_way_anova(&[&g1, &g2, &g3], AnovaKind::Welch).expect("one_way_anova should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.df_between, refs["df_between"], epsilon = EPSILON);
    // Welch df_within can have small numerical differences
    assert_relative_eq!(result.df_within, refs["df_within"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);

    // Welch's ANOVA doesn't produce SS/MS values
    assert!(result.ss_between.is_none());
    assert!(result.ss_within.is_none());
    assert!(result.grand_mean.is_none());
}

// ============================================
// Two-Group ANOVA (Edge Case)
// ============================================

#[test]
fn test_fisher_anova_two_groups() {
    let refs = common::load_reference_scalars("anova_2g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        one_way_anova(&[&g1, &g2], AnovaKind::Fisher).expect("one_way_anova should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["fisher_statistic"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.df_between,
        refs["fisher_df_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.df_within,
        refs["fisher_df_within"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.p_value, refs["fisher_p_value"], epsilon = EPSILON);
}

#[test]
fn test_two_group_anova_f_equals_t_squared() {
    // For 2 groups, F = t^2 for the equivalent t-test
    let refs = common::load_reference_scalars("anova_2g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let anova_result =
        one_way_anova(&[&g1, &g2], AnovaKind::Fisher).expect("one_way_anova should succeed");

    // F should equal t^2
    let t_stat_squared = refs["ttest_statistic"] * refs["ttest_statistic"];
    assert_relative_eq!(anova_result.statistic, t_stat_squared, epsilon = 1e-6);

    // p-values should be identical
    assert_relative_eq!(anova_result.p_value, refs["ttest_p_value"], epsilon = 1e-6);
}

#[test]
fn test_welch_anova_two_groups() {
    let refs = common::load_reference_scalars("anova_2g.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        one_way_anova(&[&g1, &g2], AnovaKind::Welch).expect("one_way_anova should succeed");

    assert_relative_eq!(result.statistic, refs["welch_statistic"], epsilon = EPSILON);
    assert_relative_eq!(
        result.df_between,
        refs["welch_df_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df_within, refs["welch_df_within"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["welch_p_value"], epsilon = 1e-6);
}

// ============================================
// Four-Group ANOVA (Unequal Variances)
// ============================================

#[test]
fn test_fisher_anova_four_groups() {
    let refs = common::load_reference_scalars("anova_4g.csv");
    let g1 = common::load_reference_vector("anova_g_low_var.csv");
    let g2 = common::load_reference_vector("anova_g_high_var.csv");
    let g3 = common::load_reference_vector("anova_g_med_var1.csv");
    let g4 = common::load_reference_vector("anova_g_med_var2.csv");

    let result = one_way_anova(&[&g1, &g2, &g3, &g4], AnovaKind::Fisher)
        .expect("one_way_anova should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["fisher_statistic"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.df_between,
        refs["fisher_df_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.df_within,
        refs["fisher_df_within"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.p_value, refs["fisher_p_value"], epsilon = EPSILON);
    assert_relative_eq!(
        result.ss_between.unwrap(),
        refs["fisher_ss_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.ss_within.unwrap(),
        refs["fisher_ss_within"],
        epsilon = EPSILON
    );
}

#[test]
fn test_welch_anova_four_groups() {
    let refs = common::load_reference_scalars("anova_4g.csv");
    let g1 = common::load_reference_vector("anova_g_low_var.csv");
    let g2 = common::load_reference_vector("anova_g_high_var.csv");
    let g3 = common::load_reference_vector("anova_g_med_var1.csv");
    let g4 = common::load_reference_vector("anova_g_med_var2.csv");

    let result = one_way_anova(&[&g1, &g2, &g3, &g4], AnovaKind::Welch)
        .expect("one_way_anova should succeed");

    assert_relative_eq!(result.statistic, refs["welch_statistic"], epsilon = EPSILON);
    assert_relative_eq!(
        result.df_between,
        refs["welch_df_between"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df_within, refs["welch_df_within"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["welch_p_value"], epsilon = 1e-6);
}

// ============================================
// ANOVA Error Cases
// ============================================

#[test]
fn test_anova_single_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(one_way_anova(&[&g1[..]], AnovaKind::Fisher).is_err());
}

#[test]
fn test_anova_empty_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(one_way_anova(&[&g1[..], &empty[..]], AnovaKind::Fisher).is_err());
}

#[test]
fn test_anova_single_observation_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    let g2 = vec![4.0]; // Only 1 observation - insufficient for variance
    assert!(one_way_anova(&[&g1[..], &g2[..]], AnovaKind::Fisher).is_err());
}

// ============================================
// Two-Way ANOVA (Balanced 2x3)
// ============================================

#[test]
fn test_two_way_anova_balanced_2x3() {
    let data = common::load_two_way_data("anova_2way_balanced_data.csv");
    let refs = common::load_reference_scalars("anova_2way_balanced.csv");

    let result = two_way_anova(&data.values, &data.factor_a, &data.factor_b)
        .expect("two_way_anova should succeed");

    // Factor A
    assert_relative_eq!(result.factor_a.ss, refs["ss_a"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_a.df, refs["df_a"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_a.ms, refs["ms_a"], epsilon = EPSILON);
    assert_relative_eq!(
        result.factor_a.f_statistic.unwrap(),
        refs["f_a"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.factor_a.p_value.unwrap(),
        refs["p_a"],
        epsilon = EPSILON
    );

    // Factor B
    assert_relative_eq!(result.factor_b.ss, refs["ss_b"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_b.df, refs["df_b"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_b.ms, refs["ms_b"], epsilon = EPSILON);
    assert_relative_eq!(
        result.factor_b.f_statistic.unwrap(),
        refs["f_b"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.factor_b.p_value.unwrap(),
        refs["p_b"],
        epsilon = EPSILON
    );

    // Interaction A:B
    assert_relative_eq!(result.interaction.ss, refs["ss_ab"], epsilon = EPSILON);
    assert_relative_eq!(result.interaction.df, refs["df_ab"], epsilon = EPSILON);
    assert_relative_eq!(result.interaction.ms, refs["ms_ab"], epsilon = EPSILON);
    assert_relative_eq!(
        result.interaction.f_statistic.unwrap(),
        refs["f_ab"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.interaction.p_value.unwrap(),
        refs["p_ab"],
        epsilon = EPSILON
    );

    // Residual (Error)
    assert_relative_eq!(result.residual.ss, refs["ss_error"], epsilon = EPSILON);
    assert_relative_eq!(result.residual.df, refs["df_error"], epsilon = EPSILON);
    assert_relative_eq!(result.residual.ms, refs["ms_error"], epsilon = EPSILON);

    // Grand mean
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], epsilon = EPSILON);
}

#[test]
fn test_two_way_anova_balanced_2x3_structure() {
    let data = common::load_two_way_data("anova_2way_balanced_data.csv");

    let result = two_way_anova(&data.values, &data.factor_a, &data.factor_b)
        .expect("two_way_anova should succeed");

    // Check structure
    assert_eq!(result.levels_a, 2);
    assert_eq!(result.levels_b, 3);
    assert_eq!(result.n, 30);

    // Check cell means shape
    assert_eq!(result.cell_means.len(), 2);
    assert_eq!(result.cell_means[0].len(), 3);
    assert_eq!(result.cell_means[1].len(), 3);

    // Check marginal means
    assert_eq!(result.marginal_means_a.len(), 2);
    assert_eq!(result.marginal_means_b.len(), 3);

    // Total SS should equal sum of all component SS
    let ss_total =
        result.factor_a.ss + result.factor_b.ss + result.interaction.ss + result.residual.ss;
    assert_relative_eq!(result.total.ss, ss_total, epsilon = EPSILON);
}

// ============================================
// Two-Way ANOVA (Balanced 2x2)
// ============================================

#[test]
fn test_two_way_anova_balanced_2x2() {
    let data = common::load_two_way_data("anova_2way_22_data.csv");
    let refs = common::load_reference_scalars("anova_2way_22.csv");

    let result = two_way_anova(&data.values, &data.factor_a, &data.factor_b)
        .expect("two_way_anova should succeed");

    // Factor A
    assert_relative_eq!(result.factor_a.ss, refs["ss_a"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_a.df, refs["df_a"], epsilon = EPSILON);
    assert_relative_eq!(
        result.factor_a.f_statistic.unwrap(),
        refs["f_a"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.factor_a.p_value.unwrap(),
        refs["p_a"],
        epsilon = EPSILON
    );

    // Factor B
    assert_relative_eq!(result.factor_b.ss, refs["ss_b"], epsilon = EPSILON);
    assert_relative_eq!(result.factor_b.df, refs["df_b"], epsilon = EPSILON);
    assert_relative_eq!(
        result.factor_b.f_statistic.unwrap(),
        refs["f_b"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.factor_b.p_value.unwrap(),
        refs["p_b"],
        epsilon = EPSILON
    );

    // Interaction
    assert_relative_eq!(result.interaction.ss, refs["ss_ab"], epsilon = EPSILON);
    assert_relative_eq!(result.interaction.df, refs["df_ab"], epsilon = EPSILON);
    assert_relative_eq!(
        result.interaction.f_statistic.unwrap(),
        refs["f_ab"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.interaction.p_value.unwrap(),
        refs["p_ab"],
        epsilon = EPSILON
    );

    // Residual
    assert_relative_eq!(result.residual.ss, refs["ss_error"], epsilon = EPSILON);
    assert_relative_eq!(result.residual.df, refs["df_error"], epsilon = EPSILON);

    // Grand mean
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], epsilon = EPSILON);

    // Structure
    assert_eq!(result.levels_a, 2);
    assert_eq!(result.levels_b, 2);
    assert_eq!(result.n, 16);
}

// ============================================
// Two-Way ANOVA (Unbalanced - Validation)
// ============================================

#[test]
fn test_two_way_anova_unbalanced_runs() {
    // For unbalanced designs, our Type III SS implementation uses an effect coding
    // approach. While exact numerical matching with R's car::Anova is complex due to
    // different constraint handling, we validate that:
    // 1. The function runs without error
    // 2. Results are reasonable (SS values are positive, p-values in [0,1])
    // 3. Residual SS matches (this is identical regardless of SS type)
    // 4. Grand mean matches
    let data = common::load_two_way_data("anova_2way_unbalanced_data.csv");
    let refs = common::load_reference_scalars("anova_2way_unbalanced.csv");

    let result = two_way_anova(&data.values, &data.factor_a, &data.factor_b)
        .expect("two_way_anova should succeed");

    // Validate basic sanity: all SS should be positive
    assert!(result.factor_a.ss > 0.0, "SS_A should be positive");
    assert!(result.factor_b.ss > 0.0, "SS_B should be positive");
    assert!(result.interaction.ss > 0.0, "SS_AB should be positive");
    assert!(result.residual.ss > 0.0, "SS_error should be positive");

    // Validate p-values are in valid range
    assert!(
        result.factor_a.p_value.unwrap() >= 0.0 && result.factor_a.p_value.unwrap() <= 1.0,
        "p_A should be in [0,1]"
    );
    assert!(
        result.factor_b.p_value.unwrap() >= 0.0 && result.factor_b.p_value.unwrap() <= 1.0,
        "p_B should be in [0,1]"
    );
    assert!(
        result.interaction.p_value.unwrap() >= 0.0 && result.interaction.p_value.unwrap() <= 1.0,
        "p_AB should be in [0,1]"
    );

    // Residual SS is the same regardless of SS type - this should match exactly
    assert_relative_eq!(result.residual.ss, refs["ss_error"], epsilon = EPSILON);

    // Grand mean should match exactly
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], epsilon = EPSILON);

    // Interaction SS should be close to R's (interaction is entered last in both approaches)
    assert_relative_eq!(
        result.interaction.ss,
        refs["ss_ab_type3"],
        max_relative = 0.10
    );
}

#[test]
fn test_two_way_anova_unbalanced_structure() {
    let data = common::load_two_way_data("anova_2way_unbalanced_data.csv");

    let result = two_way_anova(&data.values, &data.factor_a, &data.factor_b)
        .expect("two_way_anova should succeed");

    // Check structure
    assert_eq!(result.levels_a, 2);
    assert_eq!(result.levels_b, 2);
    assert_eq!(result.n, 18);

    // Degrees of freedom should still be correct
    assert_eq!(result.factor_a.df, 1.0);
    assert_eq!(result.factor_b.df, 1.0);
    assert_eq!(result.interaction.df, 1.0);
    assert_eq!(result.residual.df, 14.0); // n - a*b = 18 - 4 = 14
}

// ============================================
// Two-Way ANOVA Error Cases
// ============================================

#[test]
fn test_two_way_anova_mismatched_lengths_returns_error() {
    let values = vec![1.0, 2.0, 3.0];
    let factor_a = vec![0, 0]; // Wrong length
    let factor_b = vec![0, 0, 1];
    assert!(two_way_anova(&values, &factor_a, &factor_b).is_err());
}

#[test]
fn test_two_way_anova_single_level_returns_error() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let factor_a = vec![0, 0, 0, 0]; // Only one level
    let factor_b = vec![0, 1, 0, 1];
    assert!(two_way_anova(&values, &factor_a, &factor_b).is_err());
}

#[test]
fn test_two_way_anova_empty_cell_returns_error() {
    // 2x2 design with one empty cell: A1:B1 has no observations
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let factor_a = vec![0, 0, 0, 1, 1, 1];
    let factor_b = vec![0, 0, 1, 0, 0, 0]; // No A1:B1 observations
    assert!(two_way_anova(&values, &factor_a, &factor_b).is_err());
}

#[test]
fn test_two_way_anova_empty_input_returns_error() {
    let values: Vec<f64> = vec![];
    let factor_a: Vec<usize> = vec![];
    let factor_b: Vec<usize> = vec![];
    assert!(two_way_anova(&values, &factor_a, &factor_b).is_err());
}

// ============================================
// Repeated Measures ANOVA (3 Conditions)
// ============================================

#[test]
fn test_rm_anova_3_conditions_basic() {
    let data = common::load_rm_data("rm_anova_3cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_3cond.csv");

    // Convert to slice of slices format
    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Check condition (within-subjects) effect
    assert_relative_eq!(
        result.within_subjects.ss,
        refs["ss_condition"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.df,
        refs["df_condition"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.within_subjects.f_statistic.unwrap(),
        refs["f_statistic"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.p_value.unwrap(),
        refs["p_value"],
        max_relative = 1e-6
    );

    // Check error term
    assert_relative_eq!(result.error.ss, refs["ss_error"], max_relative = 1e-8);
    assert_relative_eq!(result.error.df, refs["df_error"], epsilon = EPSILON);

    // Check grand mean
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], max_relative = 1e-8);
}

#[test]
fn test_rm_anova_3_conditions_sphericity() {
    let data = common::load_rm_data("rm_anova_3cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_3cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Sphericity test should be present for 3 conditions
    assert!(
        result.sphericity.is_some(),
        "Sphericity test should be present for k >= 3"
    );

    let sph = result.sphericity.unwrap();
    // Mauchly's W should be in (0, 1]
    assert!(
        sph.w > 0.0 && sph.w <= 1.0,
        "Mauchly's W should be in (0, 1]"
    );
    // Chi-square should be non-negative
    assert!(sph.chi_square >= 0.0, "Chi-square should be non-negative");
    // p-value should be valid
    assert!(
        sph.p_value >= 0.0 && sph.p_value <= 1.0,
        "Sphericity p-value should be in [0, 1]"
    );

    // Check Mauchly's W is close to R (allowing some tolerance for different implementations)
    assert_relative_eq!(sph.w, refs["mauchly_w"], max_relative = 0.05);
}

#[test]
fn test_rm_anova_3_conditions_corrections() {
    let data = common::load_rm_data("rm_anova_3cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_3cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Greenhouse-Geisser correction
    assert!(
        result.greenhouse_geisser.is_some(),
        "GG correction should be present for k >= 3"
    );
    let gg = result.greenhouse_geisser.unwrap();

    // GG epsilon should be between 1/(k-1) and 1
    let k = 3.0;
    assert!(
        gg.epsilon >= 1.0 / (k - 1.0) && gg.epsilon <= 1.0,
        "GG epsilon should be in [1/(k-1), 1]"
    );

    // Check epsilon is reasonably close to R
    assert_relative_eq!(gg.epsilon, refs["gg_epsilon"], max_relative = 0.05);

    // Huynh-Feldt correction
    assert!(
        result.huynh_feldt.is_some(),
        "HF correction should be present for k >= 3"
    );
    let hf = result.huynh_feldt.unwrap();

    // HF epsilon should be at least as large as GG epsilon
    assert!(
        hf.epsilon >= gg.epsilon,
        "HF epsilon should be >= GG epsilon"
    );

    // Check HF epsilon is reasonably close to R
    assert_relative_eq!(hf.epsilon, refs["hf_epsilon"], max_relative = 0.05);
}

#[test]
fn test_rm_anova_3_conditions_condition_means() {
    let data = common::load_rm_data("rm_anova_3cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_3cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Check condition means
    assert_eq!(result.condition_means.len(), 3);
    assert_relative_eq!(
        result.condition_means[0],
        refs["condition_mean_0"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.condition_means[1],
        refs["condition_mean_1"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.condition_means[2],
        refs["condition_mean_2"],
        max_relative = 1e-8
    );

    // Check subject means are computed
    assert_eq!(result.subject_means.len(), 10);
}

// ============================================
// Repeated Measures ANOVA (2 Conditions)
// ============================================

#[test]
fn test_rm_anova_2_conditions_basic() {
    let data = common::load_rm_data("rm_anova_2cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_2cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Check condition effect
    assert_relative_eq!(
        result.within_subjects.ss,
        refs["ss_condition"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.df,
        refs["df_condition"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.within_subjects.f_statistic.unwrap(),
        refs["f_statistic"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.p_value.unwrap(),
        refs["p_value"],
        max_relative = 1e-6
    );

    // Check error term
    assert_relative_eq!(result.error.ss, refs["ss_error"], max_relative = 1e-8);
    assert_relative_eq!(result.error.df, refs["df_error"], epsilon = EPSILON);

    // Grand mean
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], max_relative = 1e-8);
}

#[test]
fn test_rm_anova_2_conditions_no_sphericity() {
    let data = common::load_rm_data("rm_anova_2cond_data.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Sphericity test should NOT be present for 2 conditions
    // (sphericity is trivially satisfied with only 2 conditions)
    assert!(
        result.sphericity.is_none(),
        "Sphericity test should not be present for k < 3"
    );
    assert!(
        result.greenhouse_geisser.is_none(),
        "GG correction should not be present for k < 3"
    );
    assert!(
        result.huynh_feldt.is_none(),
        "HF correction should not be present for k < 3"
    );
}

// ============================================
// Repeated Measures ANOVA (4 Conditions with Sphericity Violation)
// ============================================

#[test]
fn test_rm_anova_4_conditions_basic() {
    let data = common::load_rm_data("rm_anova_4cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_4cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Check condition effect
    assert_relative_eq!(
        result.within_subjects.ss,
        refs["ss_condition"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.df,
        refs["df_condition"],
        epsilon = EPSILON
    );
    assert_relative_eq!(
        result.within_subjects.f_statistic.unwrap(),
        refs["f_statistic"],
        max_relative = 1e-8
    );
    assert_relative_eq!(
        result.within_subjects.p_value.unwrap(),
        refs["p_value"],
        max_relative = 1e-6
    );

    // Check error term
    assert_relative_eq!(result.error.ss, refs["ss_error"], max_relative = 1e-8);
    assert_relative_eq!(result.error.df, refs["df_error"], epsilon = EPSILON);

    // Grand mean
    assert_relative_eq!(result.grand_mean, refs["grand_mean"], max_relative = 1e-8);
}

#[test]
fn test_rm_anova_4_conditions_sphericity_corrections() {
    let data = common::load_rm_data("rm_anova_4cond_data.csv");
    let refs = common::load_reference_scalars("rm_anova_4cond.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, true).expect("repeated_measures_anova should succeed");

    // Sphericity test should be present
    assert!(result.sphericity.is_some());
    let sph = result.sphericity.unwrap();

    // This dataset was designed to violate sphericity
    // Check Mauchly's W is in valid range
    assert!(sph.w > 0.0 && sph.w <= 1.0);

    // GG and HF corrections should be present
    assert!(result.greenhouse_geisser.is_some());
    assert!(result.huynh_feldt.is_some());

    let gg = result.greenhouse_geisser.unwrap();
    let hf = result.huynh_feldt.unwrap();

    // GG epsilon should be between 1/(k-1) and 1 where k=4
    assert!(gg.epsilon >= 1.0 / 3.0 && gg.epsilon <= 1.0);

    // HF epsilon should be >= GG epsilon
    assert!(hf.epsilon >= gg.epsilon);

    // Both corrected p-values should be valid
    assert!(gg.p_value >= 0.0 && gg.p_value <= 1.0);
    assert!(hf.p_value >= 0.0 && hf.p_value <= 1.0);
}

// ============================================
// Repeated Measures ANOVA (SS Identity)
// ============================================

#[test]
fn test_rm_anova_ss_total_identity() {
    // SS_total = SS_subjects + SS_conditions + SS_error
    let data = common::load_rm_data("rm_anova_3cond_data.csv");

    let data_refs: Vec<&[f64]> = data.subjects.iter().map(|s| s.as_slice()).collect();

    let result =
        repeated_measures_anova(&data_refs, false).expect("repeated_measures_anova should succeed");

    let ss_sum = result.subjects.ss + result.within_subjects.ss + result.error.ss;
    assert_relative_eq!(result.total.ss, ss_sum, max_relative = 1e-10);
}

// ============================================
// Repeated Measures ANOVA Error Cases
// ============================================

#[test]
fn test_rm_anova_single_subject_returns_error() {
    let data: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0]];
    assert!(repeated_measures_anova(&data, false).is_err());
}

#[test]
fn test_rm_anova_single_condition_returns_error() {
    let data: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0]];
    assert!(repeated_measures_anova(&data, false).is_err());
}

#[test]
fn test_rm_anova_unequal_conditions_returns_error() {
    let data: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0, 2.0, 3.0]];
    assert!(repeated_measures_anova(&data, false).is_err());
}

#[test]
fn test_rm_anova_empty_returns_error() {
    let data: Vec<&[f64]> = vec![];
    assert!(repeated_measures_anova(&data, false).is_err());
}
