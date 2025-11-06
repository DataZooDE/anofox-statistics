LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Simple linear regression using ridge with lambda=0 (equivalent to OLS)
SELECT * FROM anofox_statistics_ridge(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],      -- y
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],      -- x1
    0.0::DOUBLE,                               -- lambda=0 gives OLS
    true::BOOLEAN                              -- add_intercept
);
