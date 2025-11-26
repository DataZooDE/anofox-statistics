LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Recursive Least Squares (new API with 2D array + MAP)
WITH data AS (
    SELECT
        [10.0::DOUBLE, 11.0, 12.0, 13.0, 14.0, 15.0] as y,
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0]] as X
)
SELECT result.* FROM data,
LATERAL anofox_statistics_rls_fit(
    data.y,
    data.X,
    MAP(['lambda', 'intercept'], [0.99::DOUBLE, 1.0::DOUBLE])
) as result;
