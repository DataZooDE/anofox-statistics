LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Advanced Use Case: Multi-Level Hierarchical Aggregation
-- Combine multiple GROUP BY levels with aggregates for complex analysis

-- Sample data: Sales across product hierarchy
CREATE TEMP TABLE sales_hierarchy AS
SELECT
    CASE i % 2 WHEN 0 THEN 'electronics' ELSE 'appliances' END as category,
    CASE
        WHEN i % 6 < 2 THEN 'smartphones'
        WHEN i % 6 < 4 THEN 'laptops'
        ELSE 'tablets'
    END as subcategory,
    CASE i % 3 WHEN 0 THEN 'north' WHEN 1 THEN 'south' ELSE 'west' END as region,
    DATE '2024-01-01' + INTERVAL (i) DAY as sale_date,
    (100 + i * 5 + random() * 50)::DOUBLE as price,
    (10 + i * 0.5 + random() * 5)::DOUBLE as marketing_cost,
    (50 + 0.8 * (100 + i * 5) - 2 * (10 + i * 0.5) + random() * 30)::DOUBLE as units
FROM generate_series(1, 90) as t(i);

-- Level 1: Product-level analysis
WITH product_models AS (
    SELECT
        category,
        subcategory,
        anofox_statistics_ols_agg(
            units,
            [price, marketing_cost],
            {'intercept': true}
        ) as model,
        COUNT(*) as n_sales
    FROM sales_hierarchy
    GROUP BY category, subcategory
),
-- Level 2: Category-level summary
category_summary AS (
    SELECT
        category,
        AVG(model.r2) as avg_r2,
        AVG(model.coefficients[1]) as avg_price_sensitivity,
        AVG(model.coefficients[2]) as avg_marketing_effectiveness,
        SUM(n_sales) as total_sales,
        COUNT(*) as n_subcategories
    FROM product_models
    GROUP BY category
),
-- Level 3: Regional product performance
regional_product AS (
    SELECT
        region,
        subcategory,
        anofox_statistics_ols_agg(
            units,
            [price],
            {'intercept': true}
        ) as regional_model
    FROM sales_hierarchy
    GROUP BY region, subcategory
)
-- Combine insights from multiple levels
SELECT
    pm.category,
    pm.subcategory,
    pm.model.r2 as product_fit,
    pm.model.coefficients[1] as price_effect,
    cs.avg_price_sensitivity as category_avg,
    cs.n_subcategories as competing_products,
    rp.region,
    rp.regional_model.r2 as regional_fit,
    -- Multi-level insights
    CASE
        WHEN ABS(pm.model.coefficients[1] - cs.avg_price_sensitivity) > 5
            THEN 'Outlier - different from category average'
        ELSE 'Typical for category'
    END as category_comparison,
    CASE
        WHEN pm.model.r2 > 0.7 AND rp.regional_model.r2 > 0.7
            THEN 'Strong predictability at all levels'
        WHEN pm.model.r2 < 0.5 OR rp.regional_model.r2 < 0.5
            THEN 'Weak model - investigate other factors'
        ELSE 'Moderate predictability'
    END as model_assessment
FROM product_models pm
JOIN category_summary cs ON pm.category = cs.category
JOIN regional_product rp ON pm.subcategory = rp.subcategory
ORDER BY pm.category, pm.subcategory, rp.region
LIMIT 20;
