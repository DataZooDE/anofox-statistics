LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Simple linear regression using ridge with lambda=0 (equivalent to OLS)
-- Rewritten to avoid lateral join by using literal values in SELECT
WITH input AS (
    SELECT
        LIST_VALUE(1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0) as y,
        LIST_VALUE(LIST_VALUE(1.1::DOUBLE, 2.1, 2.9, 4.2, 4.8)) as X
)
SELECT result.* FROM input,
LATERAL anofox_statistics_ridge_fit(
    input.y,
    input.X,
    {'lambda': 0.0, 'intercept': true}
) as result;
