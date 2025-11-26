LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Table function requires literal arrays with 2D array + MAP
WITH data AS (
    SELECT
        [100.0::DOUBLE, 98.0, 102.0, 97.0, 101.0] as y,
        [
            [10.0::DOUBLE, 9.8, 10.2, 9.7, 10.1],
            [9.9::DOUBLE, 9.7, 10.1, 9.8, 10.0]
        ] as X
)
SELECT result.* FROM data,
LATERAL anofox_statistics_ridge_fit(
    data.y,
    data.X,
    MAP(['lambda', 'intercept'], [0.1::DOUBLE, 1.0::DOUBLE])
) as result;
