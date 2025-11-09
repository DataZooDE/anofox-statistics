LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Note: residual_diagnostics expects y_actual and y_predicted, not y and X
WITH predictions AS (
    SELECT
        [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[] as y_actual,
        [50.5, 54.8, 60.2, 64.9, 70.1, 74.7]::DOUBLE[] as y_predicted  -- Simulated predictions
)
SELECT
    obs_id,
    residual,
    std_residual,
    is_outlier  -- TRUE if |std_residual| > 2.5
FROM predictions, anofox_statistics_residual_diagnostics(
    y_actual,
    y_predicted,
    2.5  -- outlier_threshold
);
