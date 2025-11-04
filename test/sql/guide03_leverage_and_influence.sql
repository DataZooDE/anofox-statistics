-- Find most influential observations (using literal arrays)
SELECT
    obs_id,
    cooks_distance,
    leverage,
    dffits
FROM residual_diagnostics(
    [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],
    true,
    2.5,
    0.5
)
WHERE is_influential = TRUE
ORDER BY cooks_distance DESC;
