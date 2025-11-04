-- Detect outliers and influential points (use positional parameters)
SELECT
    obs_id,
    ROUND(residual, 3) as residual,
    ROUND(leverage, 3) as leverage,
    ROUND(cooks_distance, 3) as cooks_d,
    is_outlier,
    is_influential
FROM residual_diagnostics(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0]::DOUBLE[], -- y (last point is outlier)
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][], -- x
    true,  -- add_intercept
    2.5,   -- outlier_threshold
    0.5    -- influence_threshold
)
ORDER BY cooks_distance DESC
LIMIT 3;
