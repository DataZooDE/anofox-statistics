//! Normality tests example
//!
//! Run with: cargo run --example normality

use anofox_statistics::{dagostino_k_squared, shapiro_wilk};

fn main() {
    println!("=== Normality Tests ===\n");
    println!("Testing whether data follows a normal distribution is crucial for");
    println!("choosing between parametric and nonparametric statistical methods.\n");

    // ========== SHAPIRO-WILK TEST ==========
    println!("========== SHAPIRO-WILK TEST ==========\n");
    println!("The most powerful test for normality, especially for small samples (n <= 5000).");
    println!("H0: Data comes from a normal distribution.");
    println!("H1: Data does not come from a normal distribution.\n");

    // Normal-looking data
    let normal_data = vec![
        -0.97, 0.26, -0.13, 0.05, -0.57, 0.53, 1.21, -1.15, 0.32, -0.45, 0.78, -0.89, 0.11, -0.23,
        0.67, -0.34, 0.45, -0.12, 0.98, -0.76,
    ];

    println!("1. Testing approximately normal data:");
    println!("   Data: {:?}", &normal_data[..10]);
    println!("   ... ({} observations total)\n", normal_data.len());

    let sw_normal = shapiro_wilk(&normal_data).unwrap();
    println!("   W statistic: {:.4}", sw_normal.statistic);
    println!("   p-value:     {:.4}", sw_normal.p_value);
    println!(
        "   Decision:    {}",
        if sw_normal.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal (p > 0.05)"
        } else {
            "REJECT H0 - Data is not normal (p <= 0.05)"
        }
    );
    println!();

    // Skewed data (exponential-like)
    let skewed_data = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0,
        100.0, 150.0, 200.0,
    ];

    println!("2. Testing right-skewed data (exponential-like):");
    println!("   Data: {:?}", &skewed_data[..10]);
    println!("   ... ({} observations total)\n", skewed_data.len());

    let sw_skewed = shapiro_wilk(&skewed_data).unwrap();
    println!("   W statistic: {:.4}", sw_skewed.statistic);
    println!("   p-value:     {:.6}", sw_skewed.p_value);
    println!(
        "   Decision:    {}",
        if sw_skewed.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal (p > 0.05)"
        } else {
            "REJECT H0 - Data is not normal (p <= 0.05)"
        }
    );
    println!("   Recommendation: Use nonparametric tests for this data.");
    println!();

    // Bimodal data
    let bimodal_data = vec![
        1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 0.8, 1.2, 1.0, 0.9, 5.0, 5.1, 5.2, 4.9, 5.0, 5.1, 4.8, 5.2,
        5.0, 4.9,
    ];

    println!("3. Testing bimodal data (two clusters):");
    println!("   Data: {:?}", &bimodal_data[..10]);
    println!("   ... ({} observations total)\n", bimodal_data.len());

    let sw_bimodal = shapiro_wilk(&bimodal_data).unwrap();
    println!("   W statistic: {:.4}", sw_bimodal.statistic);
    println!("   p-value:     {:.6}", sw_bimodal.p_value);
    println!(
        "   Decision:    {}",
        if sw_bimodal.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal (p > 0.05)"
        } else {
            "REJECT H0 - Data is not normal (p <= 0.05)"
        }
    );
    println!();

    // Uniform data
    let uniform_data = vec![
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 0.05, 0.99,
    ];

    println!("4. Testing uniform data:");
    println!("   Data: {:?}", &uniform_data[..10]);
    println!("   ... ({} observations total)\n", uniform_data.len());

    let sw_uniform = shapiro_wilk(&uniform_data).unwrap();
    println!("   W statistic: {:.4}", sw_uniform.statistic);
    println!("   p-value:     {:.4}", sw_uniform.p_value);
    println!(
        "   Decision:    {}",
        if sw_uniform.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal (p > 0.05)"
        } else {
            "REJECT H0 - Data is not normal (p <= 0.05)"
        }
    );
    println!();

    // ========== D'AGOSTINO'S K-SQUARED TEST ==========
    println!("========== D'AGOSTINO'S K-SQUARED TEST ==========\n");
    println!("Omnibus test based on skewness and kurtosis.");
    println!("Better for larger samples (n >= 20). Requires n >= 8.");
    println!("Provides separate z-scores for skewness and kurtosis.\n");

    println!("1. Testing approximately normal data:");
    let dag_normal = dagostino_k_squared(&normal_data).unwrap();
    println!("   z_skewness:  {:.4}", dag_normal.z_skewness);
    println!("   z_kurtosis:  {:.4}", dag_normal.z_kurtosis);
    println!("   p-value:     {:.4}", dag_normal.p_value);
    println!(
        "   Decision:    {}",
        if dag_normal.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal"
        } else {
            "REJECT H0 - Data is not normal"
        }
    );
    println!();

    println!("2. Testing right-skewed data:");
    let dag_skewed = dagostino_k_squared(&skewed_data).unwrap();
    println!(
        "   z_skewness:  {:.4} (high = right-skewed)",
        dag_skewed.z_skewness
    );
    println!("   z_kurtosis:  {:.4}", dag_skewed.z_kurtosis);
    println!("   p-value:     {:.6}", dag_skewed.p_value);
    println!(
        "   Decision:    {}",
        if dag_skewed.p_value > 0.05 {
            "FAIL TO REJECT H0 - Data appears normal"
        } else {
            "REJECT H0 - Data is not normal"
        }
    );
    println!();

    // ========== INTERPRETATION GUIDE ==========
    println!("========== INTERPRETATION GUIDE ==========\n");
    println!("Shapiro-Wilk W statistic:");
    println!("  - W close to 1.0 -> data is consistent with normality");
    println!("  - W much less than 1.0 -> data deviates from normality");
    println!();
    println!("D'Agostino z-scores:");
    println!("  - z_skewness > 2 or < -2 -> significant skewness");
    println!("  - z_kurtosis > 2 or < -2 -> significant kurtosis deviation");
    println!();
    println!("Decision rule (alpha = 0.05):");
    println!("  - p-value > 0.05: Cannot reject normality, use parametric tests");
    println!("  - p-value <= 0.05: Reject normality, consider nonparametric tests");
    println!();
    println!("Note: Failing to reject H0 doesn't prove normality - it means");
    println!("there's insufficient evidence against it. Visual inspection");
    println!("(Q-Q plots, histograms) should complement these tests.");
}
