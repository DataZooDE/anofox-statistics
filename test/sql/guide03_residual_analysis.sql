SELECT
    obs_id,
    residual,
    std_residual,
    studentized_residual,
    is_outlier  -- TRUE if |studentized| > 2.5
FROM residual_diagnostics(
    [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[],      -- y: data
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],  -- x: features
    true,                                                  -- add_intercept
    2.5,                                                   -- outlier_threshold
    0.5                                                    -- influence_threshold
);
