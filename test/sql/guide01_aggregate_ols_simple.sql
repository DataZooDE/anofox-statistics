-- Quick Start Example: Simple OLS Aggregate with GROUP BY
-- Demonstrates basic per-group regression analysis

-- Sample data: sales by product category
CREATE TEMP TABLE product_sales AS
SELECT
    'electronics' as category, 100 as price, 250 as units_sold
UNION ALL SELECT 'electronics', 120, 230
UNION ALL SELECT 'electronics', 140, 210
UNION ALL SELECT 'electronics', 160, 190
UNION ALL SELECT 'electronics', 180, 170
UNION ALL SELECT 'furniture', 200, 180
UNION ALL SELECT 'furniture', 250, 165
UNION ALL SELECT 'furniture', 300, 150
UNION ALL SELECT 'furniture', 350, 135
UNION ALL SELECT 'furniture', 400, 120;

-- Run OLS regression for each category
SELECT
    category,
    result.coefficients[1] as price_elasticity,
    result.intercept,
    result.r2,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_ols_agg(
            units_sold::DOUBLE,
            [price::DOUBLE],
            {'intercept': true}
        ) as result
    FROM product_sales
    GROUP BY category
) sub;
