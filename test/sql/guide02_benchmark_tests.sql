LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Generate large dataset
CREATE TABLE large_data AS
SELECT
    i::DOUBLE as x,
    (i * 2.0 + RANDOM() * 0.1)::DOUBLE as y
FROM range(1, 1000001) t(i);

-- Time execution with aggregate function (supports table inputs)
.timer on
SELECT
    (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as coef,
    (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).r2 as r2
FROM large_data;

-- Note: Table functions require literal array parameters, not subqueries.
-- For large datasets, use aggregate functions which can operate directly on tables.
