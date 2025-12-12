-- OLS Fit Predict Benchmark: 1M Groups, 100M Rows, 3 Features
-- Usage: duckdb < benchmark_ols.sql

.timer on

WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        i / 1000000 AS row_num,
        random() * 100 AS x1,
        random() * 50 AS x2,
        random() * 25 AS x3,
        random() * 100 AS y
    FROM generate_series(1, 100000000) t(i)
)
SELECT COUNT(*) AS total_predictions FROM (
    SELECT anofox_stats_ols_fit_predict(y, [x1, x2, x3], {'fit_intercept': true}) OVER (
        PARTITION BY group_id ORDER BY row_num
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pred
    FROM test_data
) t WHERE pred IS NOT NULL;
