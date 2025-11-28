LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Ensure arrays are DOUBLE[] and use new API with 2D array + MAP
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0]::DOUBLE[],         -- y: Cast to DOUBLE[]
    [[1.0, 2.0, 3.0]]::DOUBLE[][],     -- X: 2D array (one feature)
    {'intercept': true}              -- options in MAP
);
