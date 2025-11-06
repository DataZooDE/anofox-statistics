LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Detect outliers and influential points using residual diagnostics
WITH data AS (
    SELECT
        [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0]::DOUBLE[] as y_actual,  -- Last point is outlier
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]::DOUBLE[] as y_predicted  -- Fitted values
)
SELECT
    obs_id,
    ROUND(residual, 3) as residual,
    ROUND(std_residual, 3) as std_residual,
    is_outlier
FROM data, anofox_statistics_residual_diagnostics(y_actual, y_predicted, 2.5)
ORDER BY ABS(std_residual) DESC
LIMIT 3;
