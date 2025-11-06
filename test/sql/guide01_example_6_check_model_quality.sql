LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Compare models (use positional parameters)
SELECT * FROM information_criteria(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 15.9]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][],
    true  -- add_intercept
);
