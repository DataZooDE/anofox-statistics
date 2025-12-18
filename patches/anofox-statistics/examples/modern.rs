//! Modern distribution tests example
//!
//! Run with: cargo run --example modern

use anofox_statistics::{
    energy_distance_test, energy_distance_test_1d, mmd_test, mmd_test_1d, Kernel,
};

fn main() {
    println!("=== Modern Distribution Tests ===\n");
    println!("These tests can detect differences in distribution shape, not just location,");
    println!("and naturally extend to multivariate data.\n");

    // Sample data
    let normal = vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.15, -0.15];
    let shifted = vec![2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.1, 1.8, 2.15, 1.85];
    let similar = vec![
        0.15, -0.15, 0.25, -0.05, 0.25, -0.25, 0.05, -0.25, 0.2, -0.1,
    ];

    println!("Sample data (univariate):");
    println!("  Normal:  {:?}", normal);
    println!("  Shifted: {:?}", shifted);
    println!("  Similar: {:?}", similar);
    println!();

    // ========== ENERGY DISTANCE TEST ==========
    println!("========== ENERGY DISTANCE TEST ==========\n");
    println!("Based on Euclidean distances between observations.");
    println!("E(X,Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|");
    println!("Powerful against many types of distributional differences.\n");

    // Test different distributions
    println!("1. Comparing normal vs shifted (clearly different):");
    let energy_diff = energy_distance_test_1d(&normal, &shifted, 999, Some(42)).unwrap();
    println!("   Energy statistic: {:.4}", energy_diff.statistic);
    println!("   p-value:          {:.4}", energy_diff.p_value);
    println!("   Permutations:     {}", energy_diff.n_permutations);
    println!(
        "   Decision:         {}",
        if energy_diff.p_value < 0.05 {
            "REJECT H0 - Distributions differ (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    // Test similar distributions
    println!("2. Comparing normal vs similar:");
    let energy_sim = energy_distance_test_1d(&normal, &similar, 999, Some(42)).unwrap();
    println!("   Energy statistic: {:.4}", energy_sim.statistic);
    println!("   p-value:          {:.4}", energy_sim.p_value);
    println!(
        "   Decision:         {}",
        if energy_sim.p_value < 0.05 {
            "REJECT H0 - Distributions differ (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    // Multivariate example
    println!("3. Multivariate energy distance test (2D data):");
    let x_2d = vec![
        vec![1.0, 1.0],
        vec![1.2, 0.9],
        vec![0.9, 1.1],
        vec![1.1, 1.0],
        vec![1.0, 0.8],
    ];
    let y_2d = vec![
        vec![3.0, 3.0],
        vec![3.2, 2.9],
        vec![2.9, 3.1],
        vec![3.1, 3.0],
        vec![3.0, 2.8],
    ];

    println!("   Sample X: centered around (1, 1)");
    println!("   Sample Y: centered around (3, 3)");

    let energy_2d = energy_distance_test(&x_2d, &y_2d, 999, Some(42)).unwrap();
    println!("   Energy statistic: {:.4}", energy_2d.statistic);
    println!("   p-value:          {:.4}", energy_2d.p_value);
    println!(
        "   Decision:         {}",
        if energy_2d.p_value < 0.05 {
            "REJECT H0 - Distributions differ (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    // ========== MMD TEST ==========
    println!("========== MAXIMUM MEAN DISCREPANCY (MMD) TEST ==========\n");
    println!("Kernel-based test that embeds distributions into a reproducing");
    println!("kernel Hilbert space (RKHS). More sensitive to local differences.\n");

    // 1D test with automatic bandwidth
    println!("1. Univariate MMD test (automatic Gaussian bandwidth via median heuristic):");
    let mmd_diff = mmd_test_1d(&normal, &shifted, 999, Some(42)).unwrap();
    println!("   MMD^2 statistic: {:.6}", mmd_diff.statistic);
    println!("   p-value:         {:.4}", mmd_diff.p_value);
    println!(
        "   Decision:        {}",
        if mmd_diff.p_value < 0.05 {
            "REJECT H0 - Distributions differ (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    println!("2. Comparing similar distributions:");
    let mmd_sim = mmd_test_1d(&normal, &similar, 999, Some(42)).unwrap();
    println!("   MMD^2 statistic: {:.6}", mmd_sim.statistic);
    println!("   p-value:         {:.4}", mmd_sim.p_value);
    println!(
        "   Decision:        {}",
        if mmd_sim.p_value < 0.05 {
            "REJECT H0 - Distributions differ (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    // ========== DIFFERENT KERNELS ==========
    println!("========== COMPARING KERNEL TYPES ==========\n");
    println!("MMD can use different kernel functions:\n");

    // Convert to Vec<Vec<f64>> for mmd_test
    let normal_vec: Vec<Vec<f64>> = normal.iter().map(|&v| vec![v]).collect();
    let shifted_vec: Vec<Vec<f64>> = shifted.iter().map(|&v| vec![v]).collect();

    // Gaussian kernel
    let mmd_gaussian = mmd_test(
        &normal_vec,
        &shifted_vec,
        Kernel::Gaussian { bandwidth: 1.0 },
        999,
        Some(42),
    )
    .unwrap();
    println!("  Gaussian kernel (bandwidth=1.0):");
    println!(
        "    MMD^2: {:.6}, p-value: {:.4}",
        mmd_gaussian.statistic, mmd_gaussian.p_value
    );

    // Linear kernel
    let mmd_linear = mmd_test(&normal_vec, &shifted_vec, Kernel::Linear, 999, Some(42)).unwrap();
    println!("  Linear kernel:");
    println!(
        "    MMD^2: {:.6}, p-value: {:.4}",
        mmd_linear.statistic, mmd_linear.p_value
    );

    // Polynomial kernel
    let mmd_poly = mmd_test(
        &normal_vec,
        &shifted_vec,
        Kernel::Polynomial {
            degree: 2,
            scale: 1.0,
            offset: 1.0,
        },
        999,
        Some(42),
    )
    .unwrap();
    println!("  Polynomial kernel (degree=2, scale=1, offset=1):");
    println!(
        "    MMD^2: {:.6}, p-value: {:.4}",
        mmd_poly.statistic, mmd_poly.p_value
    );

    // Laplacian kernel
    let mmd_laplacian = mmd_test(
        &normal_vec,
        &shifted_vec,
        Kernel::Laplacian { bandwidth: 1.0 },
        999,
        Some(42),
    )
    .unwrap();
    println!("  Laplacian kernel (bandwidth=1.0):");
    println!(
        "    MMD^2: {:.6}, p-value: {:.4}",
        mmd_laplacian.statistic, mmd_laplacian.p_value
    );
    println!();

    // ========== WHEN TO USE EACH TEST ==========
    println!("========== WHEN TO USE EACH TEST ==========\n");
    println!("Energy Distance:");
    println!("  - Simpler, no kernel tuning required");
    println!("  - Good all-around test");
    println!("  - Based on Euclidean distances");
    println!();
    println!("MMD:");
    println!("  - More flexible with kernel choice");
    println!("  - Can be more powerful for specific alternatives");
    println!("  - Gaussian kernel: good for general use");
    println!("  - Linear kernel: detects mean differences");
    println!("  - Polynomial kernel: sensitive to higher moments");
    println!("  - Laplacian kernel: robust to outliers");
    println!();
    println!("Both tests:");
    println!("  - Work in arbitrary dimensions");
    println!("  - Use permutation p-values (distribution-free)");
    println!("  - Detect location and shape differences");
}
