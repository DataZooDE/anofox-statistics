-- Generate large dataset
CREATE TABLE large_data AS
SELECT
    i::DOUBLE as x,
    (i * 2.0 + RANDOM() * 0.1)::DOUBLE as y
FROM range(1, 1000001) t(i);

-- Time execution with aggregate function (supports table inputs)
.timer on
SELECT
    (ols_fit_agg(y, x)).coefficient as coef,
    (ols_fit_agg(y, x)).r2 as r_squared
FROM large_data;

-- Note: Table functions require literal array parameters, not subqueries.
-- For large datasets, use aggregate functions which can operate directly on tables.
