//! Quickstart example for anofox-statistics
//!
//! Run with: cargo run --example quickstart

use anofox_statistics::{
    mann_whitney_u, permutation_t_test, shapiro_wilk, t_test, Alternative, TTestKind,
};

fn main() {
    println!("=== anofox-statistics Quickstart ===\n");

    // Sample data: treatment vs control groups
    let treatment = vec![23.1, 25.4, 22.8, 24.5, 26.2, 23.9, 25.1, 24.7];
    let control = vec![19.2, 20.5, 18.8, 21.1, 19.7, 20.3, 18.9, 19.8];

    println!("Data:");
    println!("  Treatment: {:?}", treatment);
    println!("  Control:   {:?}", control);
    println!();

    // 1. Check normality first
    println!("--- 1. SHAPIRO-WILK NORMALITY TEST ---");
    println!("Tests whether data comes from a normal distribution.");
    println!();

    let sw_treatment = shapiro_wilk(&treatment).unwrap();
    let sw_control = shapiro_wilk(&control).unwrap();

    println!("  Treatment group:");
    println!("    W statistic: {:.4}", sw_treatment.statistic);
    println!("    p-value:     {:.4}", sw_treatment.p_value);
    println!(
        "    Decision:    {}",
        if sw_treatment.p_value > 0.05 {
            "Normal (p > 0.05)"
        } else {
            "Not normal (p <= 0.05)"
        }
    );
    println!();

    println!("  Control group:");
    println!("    W statistic: {:.4}", sw_control.statistic);
    println!("    p-value:     {:.4}", sw_control.p_value);
    println!(
        "    Decision:    {}",
        if sw_control.p_value > 0.05 {
            "Normal (p > 0.05)"
        } else {
            "Not normal (p <= 0.05)"
        }
    );
    println!();

    // 2. Welch's t-test (parametric)
    println!("--- 2. WELCH'S T-TEST ---");
    println!("Compares means of two independent groups (unequal variances assumed).");
    println!("Use when: Data is approximately normal.");
    println!();

    let ttest = t_test(
        &treatment,
        &control,
        TTestKind::Welch,
        Alternative::TwoSided,
        0.0,
        None,
    )
    .unwrap();

    println!("  t-statistic:       {:.4}", ttest.statistic);
    println!("  degrees of freedom: {:.2}", ttest.df);
    println!("  p-value:           {:.6}", ttest.p_value);
    println!("  Mean (treatment):  {:.4}", ttest.mean_x);
    if let Some(mean_y) = ttest.mean_y {
        println!("  Mean (control):    {:.4}", mean_y);
    }
    println!();
    println!(
        "  Interpretation: {}",
        if ttest.p_value < 0.05 {
            "Significant difference between groups (p < 0.05)"
        } else {
            "No significant difference (p >= 0.05)"
        }
    );
    println!();

    // 3. Mann-Whitney U test (nonparametric)
    println!("--- 3. MANN-WHITNEY U TEST ---");
    println!("Nonparametric alternative to the t-test.");
    println!("Use when: Data is ordinal or non-normal.");
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
            "Significant difference between groups (p < 0.05)"
        } else {
            "No significant difference (p >= 0.05)"
        }
    );
    println!();

    // 4. Permutation t-test (resampling)
    println!("--- 4. PERMUTATION T-TEST ---");
    println!("Distribution-free test using random permutations.");
    println!("Use when: You want exact inference without distributional assumptions.");
    println!();

    let perm =
        permutation_t_test(&treatment, &control, Alternative::TwoSided, 9999, Some(42)).unwrap();

    println!("  t-statistic:   {:.4}", perm.statistic);
    println!("  p-value:       {:.6}", perm.p_value);
    println!("  permutations:  9999");
    println!();
    println!(
        "  Interpretation: {}",
        if perm.p_value < 0.05 {
            "Significant difference between groups (p < 0.05)"
        } else {
            "No significant difference (p >= 0.05)"
        }
    );
    println!();

    println!("=== Summary ===");
    println!("All three tests (t-test, Mann-Whitney, permutation) agree:");
    let all_significant = ttest.p_value < 0.05 && mw.p_value < 0.05 && perm.p_value < 0.05;
    if all_significant {
        println!("  The treatment group has significantly higher values than the control group.");
    } else {
        println!("  Results are mixed or non-significant.");
    }
}
