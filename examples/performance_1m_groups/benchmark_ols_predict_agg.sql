-- OLS Fit Predict Aggregate Benchmark: 1M Groups, 100M Rows, 3 Features
-- This benchmark tests the fit_predict aggregate function that:
-- - Fits model once per group on training rows (y IS NOT NULL)
-- - Returns predictions for ALL rows (including out-of-sample)
-- Usage: duckdb < benchmark_ols_predict_agg.sql

.timer on

-- Load the extension
LOAD 'anofox_statistics';

-- Generate test data with ~80% training rows (y not null) and 20% prediction rows (y null)
WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        i / 1000000 AS row_num,
        random() * 100 AS x1,
        random() * 50 AS x2,
        random() * 25 AS x3,
        CASE WHEN (i % 100) < 80 THEN random() * 100 ELSE NULL END AS y
    FROM generate_series(1, 100000000) t(i)
)
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT
        group_id,
        UNNEST(ols_fit_predict_agg(y, [x1, x2, x3])) AS pred
    FROM test_data
    GROUP BY group_id
) t;
