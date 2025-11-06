-- Test Technical Guide Examples

.bail on
.mode box

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT '========================================';
SELECT 'Testing Technical Guide Examples';
SELECT '========================================';

-- Unit Test Example
SELECT '--- Unit Test: Basic OLS ---';
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0]::DOUBLE[],
    [1.0, 2.0, 3.0]::DOUBLE[],
    true
);

-- Benchmark Test Example (using aggregate function)
SELECT '--- Benchmark Test: Large Dataset with Aggregate ---';
DROP TABLE IF EXISTS large_data;
CREATE TABLE large_data AS
SELECT
    i::DOUBLE as x,
    (i * 2.0 + RANDOM() * 0.1)::DOUBLE as y
FROM range(1, 1001) t(i);

SELECT
    (ols_fit_agg(y, x)).coefficient as coef,
    (ols_fit_agg(y, x)).r2 as r_squared
FROM large_data;

SELECT '========================================';
SELECT 'Technical Guide: ALL TESTS PASSED';
SELECT '========================================';
