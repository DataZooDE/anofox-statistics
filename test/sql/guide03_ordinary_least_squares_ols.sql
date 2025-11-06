LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Simple OLS with aggregate function (works directly with table data)
SELECT
    category,
    (ols_fit_agg(sales, price)).coefficient as price_effect,
    (ols_fit_agg(sales, price)).r2 as r_squared
FROM products
GROUP BY category;

-- Note: ols_fit_agg works directly with table columns.
-- For table functions with multiple predictors, use literal arrays (see Quick Start Guide).
