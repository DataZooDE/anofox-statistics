LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Export model coefficients for external scoring (using literal array sample)
COPY (
    WITH model AS (
        SELECT * FROM ols_inference(
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
            [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2],
             [1.3, 2.3, 3.3, 4.3], [1.4, 2.4, 3.4, 4.4], [1.5, 2.5, 3.5, 4.5],
             [1.6, 2.6, 3.6, 4.6], [1.7, 2.7, 3.7, 4.7], [1.8, 2.8, 3.8, 4.8],
             [1.9, 2.9, 3.9, 4.9]],
            0.95,
            true
        )
    )
    SELECT
        variable,
        estimate as coefficient,
        std_error,
        p_value,
        CURRENT_TIMESTAMP as model_trained_at,
        10 as training_observations
    FROM model
) TO 'model_coefficients.csv' (HEADER, DELIMITER ',');

-- Export predictions with confidence intervals
COPY (
    SELECT
        customer_id,
        predicted as predicted_ltv,
        ci_lower as ltv_conservative,
        ci_upper as ltv_optimistic
    FROM prediction_results
) TO 'customer_predictions.parquet' (FORMAT PARQUET);
