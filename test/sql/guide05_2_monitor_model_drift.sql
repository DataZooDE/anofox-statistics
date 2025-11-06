LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Track model performance over time
CREATE TABLE model_performance_log AS
SELECT
    CURRENT_DATE as check_date,
    'product_sales_model' as model_name,
    r_squared as current_r2,
    (SELECT r_squared FROM model_results_archive
     WHERE model_name = 'product_sales_model'
     ORDER BY archived_at DESC LIMIT 1 OFFSET 1) as previous_r2,
    CASE
        WHEN r_squared < previous_r2 * 0.9 THEN 'DEGRADED'
        ELSE 'STABLE'
    END as status
FROM current_model;
