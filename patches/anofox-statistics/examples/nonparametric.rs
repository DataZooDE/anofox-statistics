//! Nonparametric statistical tests example
//!
//! Run with: cargo run --example nonparametric

use anofox_statistics::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank, Alternative,
};

fn main() {
    println!("=== Nonparametric Statistical Tests ===\n");
    println!("Distribution-free tests that make no assumptions about the underlying distribution.");
    println!("Use when: data is ordinal, non-normal, or you want robust inference.\n");

    // ========== RANKING ==========
    println!("========== RANKING ==========\n");
    println!("Converts data to ranks. Ties receive average ranks.\n");

    let data_no_ties = vec![3.2, 1.5, 4.7, 2.1, 5.3, 2.8, 4.1, 3.9];
    let ranks_no_ties = rank(&data_no_ties).unwrap();

    println!("Data without ties:");
    println!("  Data:  {:?}", data_no_ties);
    println!("  Ranks: {:?}", ranks_no_ties);
    println!();

    let data_with_ties = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];
    let ranks_with_ties = rank(&data_with_ties).unwrap();

    println!("Data with ties (average rank method):");
    println!("  Data:  {:?}", data_with_ties);
    println!("  Ranks: {:?}", ranks_with_ties);
    println!("  Note: The three 3.0s share ranks 4, 5, 6 -> average = 5.0");
    println!();

    // ========== MANN-WHITNEY U TEST ==========
    println!("========== MANN-WHITNEY U TEST ==========\n");
    println!("Compares two independent groups (nonparametric alternative to t-test).");
    println!("Tests if one group tends to have larger values than the other.\n");

    let treatment = vec![5.0, 4.0, 5.0, 3.0, 4.0, 5.0, 4.0, 5.0];
    let control = vec![3.0, 2.0, 3.0, 4.0, 2.0, 3.0, 2.0, 3.0];

    println!("Satisfaction ratings (1-5 scale):");
    println!("  Treatment: {:?}", treatment);
    println!("  Control:   {:?}", control);
    println!();

    let mw = mann_whitney_u(
        &treatment,
        &control,
        Alternative::TwoSided,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    println!("  U statistic: {:.4}", mw.statistic);
    println!("  p-value:     {:.6}", mw.p_value);
    println!();
    println!(
        "  Interpretation: {}",
        if mw.p_value < 0.05 {
            "Groups differ significantly (p < 0.05)"
        } else {
            "No significant difference (p >= 0.05)"
        }
    );
    println!();

    // You can also use one-sided alternatives:
    // mann_whitney_u(&treatment, &control, Alternative::Greater, false) for H1: treatment > control
    println!();

    // ========== WILCOXON SIGNED-RANK TEST ==========
    println!("========== WILCOXON SIGNED-RANK TEST ==========\n");
    println!("For paired samples (nonparametric alternative to paired t-test).");
    println!("Tests if the median difference is zero.\n");

    let before = vec![8.0, 7.5, 9.0, 6.5, 8.5, 7.0, 9.5, 8.0];
    let after = vec![9.0, 8.5, 9.5, 7.5, 9.0, 8.0, 10.0, 9.0];

    println!("Pain scores before and after treatment:");
    println!("  Before: {:?}", before);
    println!("  After:  {:?}", after);
    println!();

    let wilcox = wilcoxon_signed_rank(
        &before,
        &after,
        Alternative::TwoSided,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    println!("  W statistic: {:.4}", wilcox.statistic);
    println!("  p-value:     {:.6}", wilcox.p_value);
    println!();
    println!(
        "  Interpretation: {}",
        if wilcox.p_value < 0.05 {
            "Significant change after treatment (p < 0.05)"
        } else {
            "No significant change (p >= 0.05)"
        }
    );
    println!();

    // ========== KRUSKAL-WALLIS TEST ==========
    println!("========== KRUSKAL-WALLIS TEST ==========\n");
    println!("Compares k independent groups (nonparametric alternative to one-way ANOVA).");
    println!("Tests if at least one group differs from the others.\n");

    let group_a = vec![5.2, 4.8, 5.1, 4.9, 5.0];
    let group_b = vec![3.1, 3.3, 3.0, 3.2, 3.4];
    let group_c = vec![4.1, 4.3, 4.0, 4.2, 4.4];

    println!("Three treatment groups:");
    println!("  Group A: {:?}", group_a);
    println!("  Group B: {:?}", group_b);
    println!("  Group C: {:?}", group_c);
    println!();

    let groups: Vec<&[f64]> = vec![&group_a, &group_b, &group_c];
    let kw = kruskal_wallis(&groups).unwrap();

    println!("  H statistic: {:.4}", kw.statistic);
    println!("  df:          {}", kw.df);
    println!("  p-value:     {:.6}", kw.p_value);
    println!();
    println!(
        "  Interpretation: {}",
        if kw.p_value < 0.05 {
            "At least one group differs significantly (p < 0.05)"
        } else {
            "No significant difference between groups (p >= 0.05)"
        }
    );
    println!();

    // ========== BRUNNER-MUNZEL TEST ==========
    println!("========== BRUNNER-MUNZEL TEST ==========\n");
    println!("Robust rank-based test for stochastic equality.");
    println!("Better than Mann-Whitney when variances are unequal.");
    println!("Estimates P(X < Y) + 0.5*P(X = Y) - the probability that a random");
    println!("observation from group Y exceeds one from group X.\n");

    let x = vec![1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0];
    let y = vec![3.0, 3.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 5.0, 4.0];

    println!("Two groups with different shapes:");
    println!("  X: {:?}", x);
    println!("  Y: {:?}", y);
    println!();

    let bm = brunner_munzel(&x, &y, Alternative::TwoSided, None).unwrap();

    println!("  t statistic: {:.4}", bm.statistic);
    println!("  df:          {:.2}", bm.df);
    println!("  p-value:     {:.6}", bm.p_value);
    println!("  estimate:    {:.4}", bm.estimate);
    println!();
    println!("  Interpretation:");
    println!(
        "    estimate = {:.4} means P(Y > X) = {:.1}%",
        bm.estimate,
        bm.estimate * 100.0
    );
    if bm.p_value < 0.05 {
        println!("    The groups differ significantly (p < 0.05)");
    } else {
        println!("    No significant difference (p >= 0.05)");
    }
}
