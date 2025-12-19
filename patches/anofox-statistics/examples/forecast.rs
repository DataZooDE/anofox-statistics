//! Forecast evaluation tests example
//!
//! Run with: cargo run --example forecast

use anofox_statistics::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, spa_test, Alternative,
    LossFunction, MCSStatistic, VarEstimator,
};

fn main() {
    println!("=== Forecast Evaluation Tests ===\n");
    println!("These tests compare the predictive accuracy of forecasting models.");
    println!("Common in econometrics, finance, and time series analysis.\n");

    // Sample forecast errors
    let benchmark_errors = vec![
        0.5, -0.3, 0.4, -0.2, 0.6, -0.4, 0.3, -0.5, 0.4, -0.3, 0.5, -0.2, 0.4, -0.4, 0.3, -0.3,
    ];

    let model1_errors = vec![
        0.3, -0.2, 0.2, -0.1, 0.4, -0.2, 0.2, -0.3, 0.2, -0.2, 0.3, -0.1, 0.2, -0.2, 0.2, -0.2,
    ];

    let model2_errors = vec![
        0.4, -0.3, 0.3, -0.2, 0.5, -0.3, 0.3, -0.4, 0.3, -0.3, 0.4, -0.2, 0.3, -0.3, 0.2, -0.3,
    ];

    println!("Sample data: Forecast errors from competing models");
    println!("  Benchmark: {:?}", &benchmark_errors[..8]);
    println!("  Model 1:   {:?}", &model1_errors[..8]);
    println!("  Model 2:   {:?}", &model2_errors[..8]);
    println!("  ... ({} observations total)\n", benchmark_errors.len());

    // ========== DIEBOLD-MARIANO TEST ==========
    println!("========== DIEBOLD-MARIANO TEST ==========\n");
    println!("Tests if two forecasts have equal predictive accuracy.");
    println!("H0: E[L(e1)] = E[L(e2)] (equal predictive ability)\n");

    // Squared error loss
    println!("1. Benchmark vs Model 1 (Squared Error Loss, h=1):");
    let dm1 = diebold_mariano(
        &benchmark_errors,
        &model1_errors,
        LossFunction::SquaredError,
        1,
        Alternative::TwoSided,
        VarEstimator::Acf,
    )
    .unwrap();
    println!("   DM statistic: {:.4}", dm1.statistic);
    println!("   p-value:      {:.4}", dm1.p_value);
    println!(
        "   Decision:     {}",
        if dm1.p_value < 0.05 {
            "REJECT H0 - Models have different accuracy (p < 0.05)"
        } else {
            "FAIL TO REJECT H0 - No significant difference"
        }
    );
    println!();

    // Absolute error loss
    println!("2. Benchmark vs Model 1 (Absolute Error Loss, h=1):");
    let dm2 = diebold_mariano(
        &benchmark_errors,
        &model1_errors,
        LossFunction::AbsoluteError,
        1,
        Alternative::TwoSided,
        VarEstimator::Acf,
    )
    .unwrap();
    println!("   DM statistic: {:.4}", dm2.statistic);
    println!("   p-value:      {:.4}", dm2.p_value);
    println!();

    // With lag adjustment
    println!("3. Benchmark vs Model 1 (Squared Error, h=3 multi-step):");
    let dm3 = diebold_mariano(
        &benchmark_errors,
        &model1_errors,
        LossFunction::SquaredError,
        3,
        Alternative::TwoSided,
        VarEstimator::Acf,
    )
    .unwrap();
    println!("   DM statistic: {:.4}", dm3.statistic);
    println!("   p-value:      {:.4}", dm3.p_value);
    println!("   Note: h>1 adjusts for serial correlation in multi-step forecasts.");
    println!();

    // ========== CLARK-WEST TEST ==========
    println!("========== CLARK-WEST TEST ==========\n");
    println!("For comparing nested models (restricted vs unrestricted).");
    println!("Adjusts for bias that occurs when extra parameters are zero under H0.");
    println!("Example: AR(1) benchmark vs AR(2) alternative.\n");

    let cw = clark_west(&benchmark_errors, &model1_errors, 1).unwrap();
    println!("  CW statistic:        {:.4}", cw.statistic);
    println!("  p-value (one-sided): {:.4}", cw.p_value);
    println!("  p-value (two-sided): {:.4}", cw.p_value_two_sided);
    println!();
    println!(
        "  Interpretation: {}",
        if cw.p_value < 0.05 {
            "The alternative (unrestricted) model has superior predictive ability"
        } else {
            "No evidence that the alternative model outperforms the benchmark"
        }
    );
    println!();

    // ========== SUPERIOR PREDICTIVE ABILITY (SPA) TEST ==========
    println!("========== SUPERIOR PREDICTIVE ABILITY (SPA) TEST ==========\n");
    println!("Tests if ANY model beats the benchmark (multiple testing correction).");
    println!("H0: No model outperforms the benchmark.");
    println!("Addresses data snooping when comparing many models.\n");

    // Loss values (squared errors)
    let benchmark_losses: Vec<f64> = benchmark_errors.iter().map(|e| e * e).collect();
    let model_losses: Vec<Vec<f64>> = vec![
        model1_errors.iter().map(|e| e * e).collect(),
        model2_errors.iter().map(|e| e * e).collect(),
    ];

    let spa = spa_test(&benchmark_losses, &model_losses, 499, 3.0, Some(42)).unwrap();

    println!("  Comparing 2 models against benchmark:");
    println!("  SPA statistic:        {:.4}", spa.statistic);
    println!("  p-value (consistent): {:.4}", spa.p_value_consistent);
    println!("  p-value (upper):      {:.4}", spa.p_value_upper);
    println!("  Bootstrap samples:    {}", spa.n_bootstrap);
    if let Some(idx) = spa.best_model_idx {
        println!("  Best model index:     {} (Model {})", idx, idx + 1);
    }
    println!();
    println!(
        "  Interpretation: {}",
        if spa.p_value_consistent < 0.05 {
            "At least one model significantly outperforms the benchmark"
        } else {
            "No model significantly outperforms the benchmark"
        }
    );
    println!();

    // ========== MSPE-ADJUSTED SPA TEST ==========
    println!("========== MSPE-ADJUSTED SPA TEST ==========\n");
    println!("Combines Clark-West adjustment with SPA bootstrap.");
    println!("Use when comparing multiple NESTED models to a benchmark.\n");

    let mspe_models: Vec<Vec<f64>> = vec![model1_errors.clone(), model2_errors.clone()];

    let mspe = mspe_adjusted_spa(&benchmark_errors, &mspe_models, 499, 3.0, Some(42)).unwrap();

    println!("  MSPE statistic:       {:.4}", mspe.statistic);
    println!("  p-value (consistent): {:.4}", mspe.p_value_consistent);
    println!("  p-value (upper):      {:.4}", mspe.p_value_upper);
    if let Some(idx) = mspe.best_model_idx {
        println!("  Best model index:     {} (Model {})", idx, idx + 1);
    }
    println!();

    // ========== MODEL CONFIDENCE SET (MCS) ==========
    println!("========== MODEL CONFIDENCE SET (MCS) ==========\n");
    println!("Identifies the set of models containing the best with confidence.");
    println!("Sequential elimination of inferior models until p-value > alpha.\n");

    // All model losses for MCS
    let all_losses: Vec<Vec<f64>> = vec![
        benchmark_losses.clone(),
        model_losses[0].clone(),
        model_losses[1].clone(),
    ];

    println!("1. MCS with Range statistic (alpha = 0.10):");
    let mcs_range =
        model_confidence_set(&all_losses, 0.10, MCSStatistic::Range, 500, 3.0, Some(42)).unwrap();

    println!("   Models in MCS:      {:?}", mcs_range.included_models);
    println!("   Eliminated models:  {:?}", mcs_range.eliminated_models);
    println!("   MCS p-value:        {:.4}", mcs_range.mcs_p_value);
    println!();

    if !mcs_range.elimination_sequence.is_empty() {
        println!("   Elimination sequence:");
        for step in &mcs_range.elimination_sequence {
            println!(
                "     Model {}: p-value = {:.4} -> {}",
                step.model_idx,
                step.p_value,
                if step.eliminated {
                    "ELIMINATED"
                } else {
                    "kept"
                }
            );
        }
        println!();
    }

    println!("2. MCS with Max statistic (alpha = 0.10):");
    let mcs_max =
        model_confidence_set(&all_losses, 0.10, MCSStatistic::Max, 500, 3.0, Some(42)).unwrap();

    println!("   Models in MCS:      {:?}", mcs_max.included_models);
    println!("   Eliminated models:  {:?}", mcs_max.eliminated_models);
    println!("   MCS p-value:        {:.4}", mcs_max.mcs_p_value);
    println!();

    // ========== INTERPRETATION GUIDE ==========
    println!("========== WHEN TO USE EACH TEST ==========\n");
    println!("Diebold-Mariano:");
    println!("  - Comparing exactly two non-nested forecasts");
    println!("  - Standard test for forecast comparison");
    println!();
    println!("Clark-West:");
    println!("  - Comparing nested models (restricted vs unrestricted)");
    println!("  - e.g., AR(1) vs AR(2), or Random Walk vs more complex model");
    println!();
    println!("SPA Test:");
    println!("  - Comparing multiple non-nested models to a benchmark");
    println!("  - Controls for data snooping bias");
    println!();
    println!("MSPE-Adjusted SPA:");
    println!("  - Comparing multiple nested models to a benchmark");
    println!("  - Combines Clark-West adjustment with multiple testing");
    println!();
    println!("Model Confidence Set:");
    println!("  - Identifying all potentially best models");
    println!("  - No designated benchmark required");
    println!("  - Range statistic: single inferior model");
    println!("  - Max statistic: multiple inferior models");
}
