LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample data
CREATE TABLE sales AS
SELECT
    CASE WHEN i <= 10 THEN 'Product A' ELSE 'Product B' END as product,
    i::DOUBLE as price,
    (i * 2.0 + RANDOM() * 0.5)::DOUBLE as quantity
FROM range(1, 21) t(i);

-- Regression per product
SELECT
    product,
    (ols_fit_agg(quantity, price)).coefficient as price_elasticity,
    (ols_fit_agg(quantity, price)).r2 as fit_quality
FROM sales
GROUP BY product;
