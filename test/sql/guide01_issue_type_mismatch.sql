-- Ensure arrays are DOUBLE[] and use positional parameters
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0]::DOUBLE[],  -- y: Cast to DOUBLE[]
    [1.0, 2.0, 3.0]::DOUBLE[],  -- x1: Cast to DOUBLE[]
    true                         -- add_intercept
);
