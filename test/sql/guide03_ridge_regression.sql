LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Table function requires literal arrays (positional parameters)
SELECT * FROM anofox_statistics_ridge(
    [100.0, 98.0, 102.0, 97.0, 101.0]::DOUBLE[],  -- y: sales
    [10.0, 9.8, 10.2, 9.7, 10.1]::DOUBLE[],       -- x1: price
    [9.9, 9.7, 10.1, 9.8, 10.0]::DOUBLE[],        -- x2: competitors_price (correlated!)
    0.1::DOUBLE,                                    -- lambda (explicit cast required)
    true::BOOLEAN                                   -- add_intercept
);
