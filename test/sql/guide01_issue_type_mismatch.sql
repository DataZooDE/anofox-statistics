LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Ensure arrays are DOUBLE[] and use positional parameters
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0]::DOUBLE[],  -- y: Cast to DOUBLE[]
    [1.0, 2.0, 3.0]::DOUBLE[],  -- x1: Cast to DOUBLE[]
    true                         -- add_intercept
);
