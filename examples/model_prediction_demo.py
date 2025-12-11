#!/usr/bin/env python3
"""
Model Fit and Predict Demonstration
Using anofox_stats_ols_fit_agg and anofox_stats_predict

Usage: uv run examples/model_prediction_demo.py
"""

import duckdb


def main():
    # Initialize DuckDB (extension auto-loads when running from project with built extension)
    con = duckdb.connect()

    print("=" * 80)
    print("MODEL FIT AND PREDICT DEMONSTRATION")
    print("=" * 80)

    # Create training data
    print("\nStep 1: Create training data")
    print("-" * 60)

    con.execute("""
        CREATE TEMP TABLE training_data AS
        SELECT
            (100 + 10 * id + 5 * (id + random() * 2) + (random() * 10 - 5))::DOUBLE as sales,
            id::DOUBLE as price,
            (id + random() * 2)::DOUBLE as advertising
        FROM range(1, 21) t(id)
    """)

    print(con.execute("SELECT * FROM training_data LIMIT 5").fetchdf().to_string(index=False))

    # Fit OLS model
    print("\nStep 2: Fit OLS model using aggregate function")
    print("-" * 60)

    con.execute("""
        CREATE TEMP TABLE sales_model AS
        SELECT
            (fit).intercept as intercept,
            (fit).coefficients as coefficients,
            (fit).r_squared as r_squared,
            (fit).residual_std_error as std_error,
            (fit).n_observations as n_obs
        FROM (
            SELECT anofox_stats_ols_fit_agg(sales, [price, advertising]) as fit
            FROM training_data
        )
    """)

    model = con.execute("SELECT * FROM sales_model").fetchone()
    print(f"  Intercept: {model[0]:.4f}")
    print(f"  Coefficients: {[f'{c:.4f}' for c in model[1]]}")
    print(f"  R-squared: {model[2]:.4f}")
    print(f"  Std Error: {model[3]:.4f}")
    print(f"  N observations: {model[4]}")

    # Make predictions
    print("\nStep 3: Make predictions from stored model coefficients")
    print("-" * 60)

    new_scenarios = [
        (25.0, 13.0),
        (26.0, 14.0),
        (27.0, 15.0),
        (30.0, 18.0),
    ]

    print("  New scenarios:")
    for i, (price, ad) in enumerate(new_scenarios, 1):
        print(f"    {i}. Price=${price}, Advertising=${ad}k")

    # Batch predictions using linear formula: y = intercept + b1*x1 + b2*x2
    con.execute("""
        CREATE TEMP TABLE new_scenarios AS
        SELECT
            column0 as price,
            column1 as advertising,
            row_number() OVER () as scenario_id
        FROM (VALUES (25.0, 13.0), (26.0, 14.0), (27.0, 15.0), (30.0, 18.0))
    """)

    predictions = con.execute("""
        SELECT
            n.scenario_id,
            n.price,
            n.advertising,
            round(m.intercept + m.coefficients[1] * n.price + m.coefficients[2] * n.advertising, 2) as predicted
        FROM new_scenarios n
        CROSS JOIN sales_model m
        ORDER BY n.scenario_id
    """).fetchall()

    print("\n  Predictions (y = intercept + b1*price + b2*advertising):")
    for row in predictions:
        print(f"    Scenario {row[0]}: Price=${row[1]}, Ad=${row[2]}k -> Predicted Sales: ${row[3]}")

    # Window function comparison
    print("\nStep 4: Window function fit_predict (expanding window)")
    print("-" * 60)

    window_preds = con.execute("""
        SELECT
            id,
            round(price, 1) as price,
            round(advertising, 1) as ad,
            round(sales, 1) as actual,
            round((pred).yhat, 1) as predicted
        FROM (
            SELECT
                row_number() OVER () as id,
                price,
                advertising,
                sales,
                anofox_stats_ols_fit_predict(
                    sales,
                    [price, advertising],
                    {'fit_intercept': true}
                ) OVER (ORDER BY price ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as pred
            FROM training_data
        )
        WHERE pred IS NOT NULL
        ORDER BY id
        LIMIT 5
    """).fetchdf()

    print(window_preds.to_string(index=False))

    # Summary
    print("\n" + "=" * 80)
    print("FUNCTIONS DEMONSTRATED:")
    print("=" * 80)
    print("  - anofox_stats_ols_fit_agg: Fit OLS model via GROUP BY aggregation")
    print("  - anofox_stats_ols_fit_predict: Window function for expanding/rolling fit")
    print("")
    print("Prediction from stored coefficients: intercept + coef[1]*x1 + coef[2]*x2 + ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
