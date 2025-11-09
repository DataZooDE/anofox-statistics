LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample products data
CREATE TEMP TABLE products AS
SELECT
    CASE
        WHEN i <= 15 THEN 'electronics'
        WHEN i <= 30 THEN 'clothing'
        ELSE 'furniture'
    END as category,
    (100 + i * 5 + random() * 20)::DOUBLE as sales,
    (50 + i * 2 + random() * 10)::DOUBLE as price
FROM generate_series(1, 45) t(i);

-- Simple OLS with aggregate function (works directly with table data)
SELECT
    category,
    (ols_fit_agg(sales, price)).coefficient as price_effect,
    (ols_fit_agg(sales, price)).r2 as r_squared
FROM products
GROUP BY category;

-- Note: ols_fit_agg works directly with table columns.
-- For table functions with multiple predictors, use literal arrays (see Quick Start Guide).
