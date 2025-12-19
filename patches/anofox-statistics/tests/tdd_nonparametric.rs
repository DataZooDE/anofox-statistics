mod common;

use anofox_statistics::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank, Alternative,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-10;

// ============================================
// Ranking Tests
// ============================================

#[test]
fn test_rank_simple() {
    let refs = common::load_reference_scalars("ranks.csv");
    let data = common::load_reference_vector("rank_simple.csv");

    let ranks = rank(&data).expect("rank should succeed");

    assert_eq!(ranks.len(), 8);
    assert_relative_eq!(ranks[0], refs["rank_simple_1"], epsilon = EPSILON);
    assert_relative_eq!(ranks[1], refs["rank_simple_2"], epsilon = EPSILON);
    assert_relative_eq!(ranks[2], refs["rank_simple_3"], epsilon = EPSILON);
    assert_relative_eq!(ranks[3], refs["rank_simple_4"], epsilon = EPSILON);
    assert_relative_eq!(ranks[4], refs["rank_simple_5"], epsilon = EPSILON);
    assert_relative_eq!(ranks[5], refs["rank_simple_6"], epsilon = EPSILON);
    assert_relative_eq!(ranks[6], refs["rank_simple_7"], epsilon = EPSILON);
    assert_relative_eq!(ranks[7], refs["rank_simple_8"], epsilon = EPSILON);
}

#[test]
fn test_rank_with_ties() {
    let refs = common::load_reference_scalars("ranks.csv");
    let data = common::load_reference_vector("rank_ties.csv");

    let ranks = rank(&data).expect("rank should succeed");

    assert_eq!(ranks.len(), 7);
    // Ties get average rank
    assert_relative_eq!(ranks[0], refs["rank_ties_1"], epsilon = EPSILON);
    assert_relative_eq!(ranks[1], refs["rank_ties_2"], epsilon = EPSILON);
    assert_relative_eq!(ranks[2], refs["rank_ties_3"], epsilon = EPSILON);
    assert_relative_eq!(ranks[3], refs["rank_ties_4"], epsilon = EPSILON);
    assert_relative_eq!(ranks[4], refs["rank_ties_5"], epsilon = EPSILON);
    assert_relative_eq!(ranks[5], refs["rank_ties_6"], epsilon = EPSILON);
    assert_relative_eq!(ranks[6], refs["rank_ties_7"], epsilon = EPSILON);
}

#[test]
fn test_rank_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(rank(&empty).is_err());
}

// ============================================
// Mann-Whitney U Test
// ============================================

#[test]
fn test_mann_whitney_u_two_sided() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_less() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_greater() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_corrected() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, true, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_corrected"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu_less() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu_greater() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_greater"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(mann_whitney_u(&empty, &y, Alternative::TwoSided, false, false, None, None).is_err());
}

#[test]
fn test_mann_whitney_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(mann_whitney_u(&x, &empty, Alternative::TwoSided, false, false, None, None).is_err());
}

// ============================================
// Wilcoxon Signed-Rank Test
// ============================================

#[test]
fn test_wilcoxon_signed_rank_two_sided() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_corrected() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, true, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_corrected"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_unequal_length_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None).is_err());
}

#[test]
fn test_wilcoxon_signed_rank_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(
        wilcoxon_signed_rank(&empty, &y, Alternative::TwoSided, false, false, None, None).is_err()
    );
}

// ============================================
// Kruskal-Wallis Test
// ============================================

#[test]
fn test_kruskal_wallis() {
    let refs = common::load_reference_scalars("kruskal_wallis.csv");
    let a = common::load_reference_vector("kw_a.csv");
    let b = common::load_reference_vector("kw_b.csv");
    let c = common::load_reference_vector("kw_c.csv");

    let result = kruskal_wallis(&[&a, &b, &c]).expect("kruskal_wallis should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_kruskal_wallis_single_group_returns_error() {
    let a = vec![1.0, 2.0, 3.0];
    assert!(kruskal_wallis(&[&a[..]]).is_err());
}

#[test]
fn test_kruskal_wallis_empty_group_returns_error() {
    let a = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(kruskal_wallis(&[&a[..], &empty[..]]).is_err());
}

// ============================================
// Brunner-Munzel Test
// ============================================

#[test]
fn test_brunner_munzel() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::TwoSided, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = 1e-6);
    assert_relative_eq!(result.df, refs["df"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
    assert_relative_eq!(result.estimate, refs["estimate"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_less() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::Less, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_greater() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::Greater, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_confidence_interval_95() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.05))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_confidence_interval_90() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.10))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_confidence_interval_99() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.01))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.99, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(brunner_munzel(&empty, &y, Alternative::TwoSided, None).is_err());
}

#[test]
fn test_brunner_munzel_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(brunner_munzel(&x, &empty, Alternative::TwoSided, None).is_err());
}

// ============================================
// Mann-Whitney U Exact Tests
// ============================================

#[test]
fn test_mann_whitney_u_exact_two_sided() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_exact_less() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_exact_greater() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_confidence_interval() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, Some(0.95), None)
        .expect("mann_whitney_u with CI should succeed");

    assert!(result.estimate.is_some());
    assert!(result.conf_int.is_some());

    let estimate = result.estimate.unwrap();
    let ci = result.conf_int.unwrap();

    assert_relative_eq!(estimate, refs["estimate"], epsilon = 1e-6);
    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_mann_whitney_u_confidence_interval_90() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, Some(0.90), None)
        .expect("mann_whitney_u with CI should succeed");

    let ci = result.conf_int.unwrap();

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

// ============================================
// Wilcoxon Signed-Rank Exact Tests
// ============================================

#[test]
fn test_wilcoxon_signed_rank_exact_two_sided() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_exact_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_exact_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_confidence_interval() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, Some(0.95), None)
        .expect("wilcoxon_signed_rank with CI should succeed");

    assert!(result.estimate.is_some());
    assert!(result.conf_int.is_some());

    let estimate = result.estimate.unwrap();
    let ci = result.conf_int.unwrap();

    assert_relative_eq!(estimate, refs["estimate"], epsilon = 1e-6);
    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_wilcoxon_signed_rank_confidence_interval_90() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, Some(0.90), None)
        .expect("wilcoxon_signed_rank with CI should succeed");

    let ci = result.conf_int.unwrap();

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

// ============================================
// Mann-Whitney U Result Fields
// ============================================

#[test]
fn test_mann_whitney_u_result_contains_null_value() {
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result_mu0 = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    let result_mu05 = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result_mu0.null_value, 0.0, epsilon = EPSILON);
    assert_relative_eq!(result_mu05.null_value, 0.5, epsilon = EPSILON);
}

// ============================================
// Wilcoxon Signed-Rank Result Fields
// ============================================

#[test]
fn test_wilcoxon_signed_rank_result_contains_null_value() {
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result_mu0 = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    let result_mu03 =
        wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, Some(0.3))
            .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result_mu0.null_value, 0.0, epsilon = EPSILON);
    assert_relative_eq!(result_mu03.null_value, 0.3, epsilon = EPSILON);
}
