LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Detect outliers and influential points using residual diagnostics
WITH data AS (
    SELECT
        [2.1::DOUBLE, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0] as y_actual,  -- Last point is outlier
        [2.0::DOUBLE, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0] as y_predicted  -- Fitted values
)
SELECT
    result.obs_id,
    ROUND(result.residual, 3) as residual,
    ROUND(result.std_residual, 3) as std_residual,
    result.is_outlier
FROM data,
LATERAL anofox_stats_residual_diagnostics(data.y_actual, data.y_predicted, 2.5) as result
ORDER BY ABS(result.std_residual) DESC
LIMIT 3;
