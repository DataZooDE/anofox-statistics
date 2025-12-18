//! Parametric statistical tests example
//!
//! Run with: cargo run --example parametric

use anofox_statistics::{brown_forsythe, t_test, yuen_test, Alternative, TTestKind};

fn main() {
    println!("=== Parametric Statistical Tests ===\n");
    println!("Parametric tests assume data comes from a known distribution (usually normal).\n");

    // Sample data
    let before = vec![68.5, 72.3, 69.1, 71.8, 67.2, 73.5, 70.2, 68.9];
    let after = vec![70.2, 74.5, 71.3, 73.9, 69.4, 75.8, 72.1, 71.2];

    println!("Data: Weight measurements before and after a program");
    println!("  Before: {:?}", before);
    println!("  After:  {:?}", after);
    println!();

    // ========== T-TESTS ==========
    println!("========== T-TESTS ==========\n");

    // 1. Welch's t-test
    println!("--- 1. WELCH'S T-TEST ---");
    println!("For two independent samples with possibly unequal variances.");
    println!("This is the recommended default for comparing two groups.\n");

    let welch = t_test(
        &before,
        &after,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .unwrap();

    println!("  t-statistic: {:.4}", welch.statistic);
    println!("  df:          {:.2}", welch.df);
    println!("  p-value:     {:.4}", welch.p_value);
    println!("  Mean before: {:.4}", welch.mean_x);
    if let Some(mean_y) = welch.mean_y {
        println!("  Mean after:  {:.4}", mean_y);
    }
    println!();

    // 2. Student's t-test
    println!("--- 2. STUDENT'S T-TEST ---");
    println!("For two independent samples with equal variances (pooled variance).");
    println!("Use only when you're confident variances are equal.\n");

    let student = t_test(
        &before,
        &after,
        TTestKind::Student,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .unwrap();

    println!("  t-statistic: {:.4}", student.statistic);
    println!("  df:          {:.2}", student.df);
    println!("  p-value:     {:.4}", student.p_value);
    println!();

    // 3. Paired t-test
    println!("--- 3. PAIRED T-TEST ---");
    println!("For paired/matched samples (same subjects measured twice).");
    println!("Tests if the mean difference is zero.\n");

    let paired = t_test(
        &before,
        &after,
        TTestKind::Paired,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .unwrap();

    println!("  t-statistic:     {:.4}", paired.statistic);
    println!("  df:              {:.2}", paired.df);
    println!("  p-value:         {:.4}", paired.p_value);
    println!("  Mean difference: {:.4}", paired.mean_x);
    println!();

    // ========== ONE-SIDED ALTERNATIVES ==========
    println!("========== ONE-SIDED ALTERNATIVES ==========\n");

    println!("--- Testing if 'after' > 'before' ---\n");

    let greater = t_test(
        &after,
        &before,
        TTestKind::Welch,
        Alternative::Greater,
        0.0,
        None,
    )
    .unwrap();
    println!("  Alternative::Greater (H1: after > before)");
    println!("  t-statistic: {:.4}", greater.statistic);
    println!("  p-value:     {:.4}", greater.p_value);
    println!(
        "  Decision:    {}",
        if greater.p_value < 0.05 {
            "Reject H0 - 'after' is significantly greater"
        } else {
            "Fail to reject H0"
        }
    );
    println!();

    let less = t_test(
        &after,
        &before,
        TTestKind::Welch,
        Alternative::Less,
        0.0,
        None,
    )
    .unwrap();
    println!("  Alternative::Less (H1: after < before)");
    println!("  t-statistic: {:.4}", less.statistic);
    println!("  p-value:     {:.4}", less.p_value);
    println!(
        "  Decision:    {}",
        if less.p_value < 0.05 {
            "Reject H0 - 'after' is significantly less"
        } else {
            "Fail to reject H0"
        }
    );
    println!();

    // ========== YUEN'S ROBUST T-TEST ==========
    println!("========== YUEN'S ROBUST T-TEST ==========\n");
    println!("Uses trimmed means - robust against outliers.\n");

    // Data with an outlier
    let clean = vec![10.1, 10.3, 10.2, 10.5, 10.4, 10.1, 10.3, 10.2, 10.4, 10.3];
    let with_outlier = vec![10.2, 10.4, 10.3, 10.5, 50.0, 10.2, 10.4, 10.3, 10.5, 10.4];

    println!("Data with outlier:");
    println!("  Clean:        {:?}", clean);
    println!("  With outlier: {:?}", with_outlier);
    println!();

    // Standard t-test (affected by outlier)
    let ttest_outlier = t_test(
        &clean,
        &with_outlier,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .unwrap();

    println!("Standard Welch t-test (affected by outlier):");
    println!("  t-statistic: {:.4}", ttest_outlier.statistic);
    println!("  p-value:     {:.4}", ttest_outlier.p_value);
    println!("  Mean clean:       {:.4}", ttest_outlier.mean_x);
    if let Some(mean_y) = ttest_outlier.mean_y {
        println!("  Mean with outlier: {:.4}", mean_y);
    }
    println!();

    // Yuen's test (robust)
    let yuen = yuen_test(&clean, &with_outlier, 0.2, Alternative::TwoSided, None).unwrap();

    println!("Yuen's test (20% trimmed means - robust):");
    println!("  t-statistic: {:.4}", yuen.statistic);
    println!("  df:          {:.2}", yuen.df);
    println!("  p-value:     {:.4}", yuen.p_value);
    println!("  Difference:  {:.4}", yuen.diff);
    println!();
    println!("  Note: Yuen's test is less affected by the outlier.");
    println!();

    // ========== BROWN-FORSYTHE TEST ==========
    println!("========== BROWN-FORSYTHE TEST ==========\n");
    println!("Tests homogeneity of variances across groups.");
    println!("Use before ANOVA to check the equal variance assumption.\n");

    let group1: Vec<f64> = vec![4.2, 4.5, 4.1, 4.8, 4.3];
    let group2: Vec<f64> = vec![3.1, 3.4, 3.2, 3.6, 3.3, 3.5];
    let group3: Vec<f64> = vec![5.1, 5.4, 10.2, 5.8, 5.3]; // Higher variance

    println!("Three groups:");
    println!("  Group 1: {:?}", group1);
    println!("  Group 2: {:?}", group2);
    println!("  Group 3: {:?} (higher variance)", group3);
    println!();

    let groups: Vec<&[f64]> = vec![&group1, &group2, &group3];
    let bf = brown_forsythe(&groups).unwrap();

    println!("  F-statistic: {:.4}", bf.statistic);
    println!("  df1:         {:.0}", bf.df1);
    println!("  df2:         {:.0}", bf.df2);
    println!("  p-value:     {:.4}", bf.p_value);
    println!();
    println!(
        "  Interpretation: {}",
        if bf.p_value < 0.05 {
            "Variances are significantly different (p < 0.05)"
        } else {
            "No evidence of unequal variances (p >= 0.05)"
        }
    );
}
