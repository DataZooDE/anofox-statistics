LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Variance proportional to x (new API with 2D array + MAP)
SELECT * FROM anofox_statistics_wls_fit(
    [50.0, 100.0, 150.0, 200.0, 250.0]::DOUBLE[],  -- y: sales
    [[10.0, 20.0, 30.0, 40.0, 50.0]]::DOUBLE[][],  -- X: 2D array (one feature)
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],      -- weights: proportional to size
    {'intercept': true}                          -- options in MAP
);
