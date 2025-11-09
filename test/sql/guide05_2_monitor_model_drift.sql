LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create model performance archive
CREATE TEMP TABLE model_results_archive AS
SELECT
    'product_sales_model' as model_name,
    0.85 as r_squared,
    (DATE '2024-01-01' + i * INTERVAL '1' DAY) as archived_at
FROM generate_series(1, 30) t(i);

-- Create current model results
CREATE TEMP TABLE current_model AS
SELECT
    0.82 as r_squared,
    0.81 as adj_r_squared;

-- Track model performance over time
CREATE TEMP TABLE model_performance_log AS
WITH previous_performance AS (
    SELECT r_squared as previous_r2
    FROM model_results_archive
    WHERE model_name = 'product_sales_model'
    ORDER BY archived_at DESC LIMIT 1 OFFSET 1
),
performance_with_previous AS (
    SELECT
        CURRENT_DATE as check_date,
        'product_sales_model' as model_name,
        r_squared as current_r2,
        (SELECT previous_r2 FROM previous_performance) as previous_r2
    FROM current_model
)
SELECT
    check_date,
    model_name,
    current_r2,
    previous_r2,
    CASE
        WHEN current_r2 < previous_r2 * 0.9 THEN 'DEGRADED'
        ELSE 'STABLE'
    END as status
FROM performance_with_previous;
