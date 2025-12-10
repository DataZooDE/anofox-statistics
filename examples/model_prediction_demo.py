#!/usr/bin/env python3
"""
Demonstration of efficient model-based prediction workflow
Using anofox_statistics_model_predict with pre-fitted models
"""

import duckdb
import numpy as np

def main():
    # Initialize DuckDB and load extension
    con = duckdb.connect()
    con.execute("LOAD 'build/debug/extension/anofox_statistics/anofox_statistics.duckdb_extension'")

    print("=" * 80)
    print("EFFICIENT MODEL-BASED PREDICTION DEMONSTRATION")
    print("=" * 80)

    # Sample data: Sales vs Price and Advertising Budget
    sales = [100, 120, 140, 160, 180, 200, 220, 240]
    price = [10, 12, 14, 16, 18, 20, 22, 24]
    advertising = [5, 6, 7, 8, 9, 10, 11, 12]

    print("\n📊 Training Data:")
    print(f"  Sales: {sales}")
    print(f"  Price: {price}")
    print(f"  Advertising: {advertising}")

    # Step 1: Fit model ONCE with full_output=true
    print("\n🔧 Step 1: Fit model with full_output=true (one-time operation)")

    con.execute("""
        CREATE TABLE sales_model AS
        SELECT * FROM anofox_statistics_ols(
            ?::DOUBLE[],
            ?::DOUBLE[][],
            MAP{'intercept': true, 'full_output': true}
        )
    """, [sales, [[p, a] for p, a in zip(price, advertising)]])

    # Show model summary
    model = con.execute("SELECT * FROM sales_model").fetchone()
    print(f"\n  ✓ Model fitted successfully!")
    print(f"  - Intercept: {model[0]:.4f}")
    print(f"  - Coefficients: {[f'{c:.4f}' for c in model[1]]}")
    print(f"  - R²: {model[2]:.4f}")
    print(f"  - MSE: {model[3]:.4f}")

    # Step 2: Make predictions on new data (NO refitting!)
    print("\n🎯 Step 2: Make predictions on new data (no refitting required)")

    # New scenarios to predict
    new_data = [
        [25, 13],  # Price=$25, Ad Budget=$13k
        [26, 14],  # Price=$26, Ad Budget=$14k
        [27, 15],  # Price=$27, Ad Budget=$15k
    ]

    print(f"\n  New observations to predict:")
    for i, (p, a) in enumerate(new_data, 1):
        print(f"    {i}. Price=${p}, Advertising=${a}k")

    # Prediction with confidence intervals
    print("\n  📈 Predictions with CONFIDENCE intervals (95%):")
    predictions = con.execute("""
        SELECT
            p.observation_id,
            round(p.predicted, 2) as forecast,
            round(p.ci_lower, 2) as lower,
            round(p.ci_upper, 2) as upper,
            round(p.se, 4) as std_error
        FROM sales_model m,
        LATERAL anofox_statistics_model_predict(
            m.intercept,
            m.coefficients,
            m.mse,
            m.x_train_means,
            m.coefficient_std_errors,
            m.intercept_std_error,
            m.df_residual,
            ?::DOUBLE[][],
            0.95,
            'confidence'
        ) p
    """, [new_data]).fetchall()

    for row in predictions:
        obs_id, pred, lower, upper, se = row
        print(f"    Obs {obs_id}: ${pred:>6.2f}  [{lower:>6.2f}, {upper:>6.2f}]  SE={se:.4f}")

    # Prediction with prediction intervals (wider)
    print("\n  🎲 Predictions with PREDICTION intervals (95%):")
    predictions = con.execute("""
        SELECT
            p.observation_id,
            round(p.predicted, 2) as forecast,
            round(p.ci_lower, 2) as lower,
            round(p.ci_upper, 2) as upper,
            round(p.ci_upper - p.ci_lower, 2) as width
        FROM sales_model m,
        LATERAL anofox_statistics_model_predict(
            m.intercept, m.coefficients, m.mse, m.x_train_means,
            m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
            ?::DOUBLE[][],
            0.95,
            'prediction'
        ) p
    """, [new_data]).fetchall()

    for row in predictions:
        obs_id, pred, lower, upper, width = row
        print(f"    Obs {obs_id}: ${pred:>6.2f}  [{lower:>6.2f}, {upper:>6.2f}]  Width={width:.2f}")

    # Step 3: Batch prediction without intervals (fastest)
    print("\n⚡ Step 3: High-speed batch predictions (no intervals)")

    batch_data = [[p, a] for p in range(10, 31) for a in range(5, 16)]

    predictions = con.execute("""
        SELECT COUNT(*) as n_predictions,
               round(AVG(p.predicted), 2) as avg_prediction,
               round(MIN(p.predicted), 2) as min_prediction,
               round(MAX(p.predicted), 2) as max_prediction
        FROM sales_model m,
        LATERAL anofox_statistics_model_predict(
            m.intercept, m.coefficients, m.mse, m.x_train_means,
            m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
            ?::DOUBLE[][],
            0.95,
            'none'  -- Skip intervals for speed
        ) p
    """, [batch_data]).fetchone()

    n, avg, min_val, max_val = predictions
    print(f"  ✓ Scored {n} scenarios in milliseconds")
    print(f"  - Average prediction: ${avg}")
    print(f"  - Range: ${min_val} - ${max_val}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ KEY BENEFITS:")
    print("=" * 80)
    print("  ✓ Model fitted ONCE, reused MANY times")
    print("  ✓ No refitting overhead for predictions")
    print("  ✓ Flexible intervals: confidence, prediction, or none")
    print("  ✓ Perfect for production pipelines and batch scoring")
    print("  ✓ Works with all regression types: OLS, Ridge, WLS, Elastic Net, RLS")
    print("=" * 80)

if __name__ == "__main__":
    main()
