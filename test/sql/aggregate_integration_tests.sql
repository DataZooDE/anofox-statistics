-- Integration tests for aggregate functions
-- Tests complex scenarios: window functions, CTEs, joins, subqueries

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create comprehensive test dataset
CREATE TABLE sales_data AS
SELECT
    DATE '2024-01-01' + INTERVAL (i) DAY as date,
    CASE
        WHEN i % 3 = 0 THEN 'product_a'
        WHEN i % 3 = 1 THEN 'product_b'
        ELSE 'product_c'
    END as product,
    CASE
        WHEN i % 4 = 0 THEN 'north'
        WHEN i % 4 = 1 THEN 'south'
        WHEN i % 4 = 2 THEN 'east'
        ELSE 'west'
    END as region,
    (100.0 + i * 2.0 + random() * 20.0)::DOUBLE as revenue,
    (50.0 + i * 0.5 + random() * 10.0)::DOUBLE as marketing_spend,
    (30.0 + i * 0.3 + random() * 5.0)::DOUBLE as product_cost,
    (1000.0 + i * 10.0)::DOUBLE as customer_count,
    (0.8 + random() * 0.4)::DOUBLE as weight
FROM generate_series(1, 90) as t(i);

SELECT '=== Test 1: Window functions with aggregates ===' as test_name;
-- Rolling regression using window function
SELECT
    date,
    product,
    revenue,
    model.coefficients[1] as marketing_roi,
    model.r_squared
FROM (
    SELECT
        date,
        product,
        revenue,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) OVER (
            PARTITION BY product
            ORDER BY date
            ROWS BETWEEN 15 PRECEDING AND CURRENT ROW
        ) as model
    FROM sales_data
) sub
WHERE date >= DATE '2024-01-20'
ORDER BY product, date
LIMIT 15;

SELECT '=== Test 2: Multiple aggregates in CTE pipeline ===' as test_name;
WITH base_models AS (
    SELECT
        product,
        region,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend, product_cost], {'intercept': true}) as ols_model,
        anofox_statistics_wls_fit_agg(revenue, [marketing_spend, product_cost], weight, {'intercept': true}) as wls_model,
        anofox_statistics_ridge_fit_agg(revenue, [marketing_spend, product_cost], {'lambda': 1.0, 'intercept': true}) as ridge_model
    FROM sales_data
    GROUP BY product, region
),
model_comparison AS (
    SELECT
        product,
        region,
        ols_model.r_squared as ols_r2,
        wls_model.r_squared as wls_r2,
        ridge_model.r_squared as ridge_r2,
        ols_model.coefficients[1] as ols_marketing_coef,
        wls_model.coefficients[1] as wls_marketing_coef,
        ridge_model.coefficients[1] as ridge_marketing_coef
    FROM base_models
)
SELECT
    product,
    region,
    ols_r2,
    wls_r2,
    ridge_r2,
    CASE
        WHEN wls_r2 > ols_r2 THEN 'WLS better (weights help)'
        ELSE 'OLS sufficient'
    END as model_recommendation
FROM model_comparison
ORDER BY product, region
LIMIT 10;

SELECT '=== Test 3: Subquery aggregation ===' as test_name;
SELECT
    main.product,
    main.avg_revenue,
    model.coefficients[1] as cost_sensitivity,
    model.r_squared,
    model.n_obs
FROM (
    SELECT
        product,
        AVG(revenue) as avg_revenue
    FROM sales_data
    GROUP BY product
) main
JOIN (
    SELECT
        product,
        anofox_statistics_ols_fit_agg(revenue, [product_cost], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) model ON main.product = model.product
ORDER BY main.product;

SELECT '=== Test 4: Combining GROUP BY and window functions ===' as test_name;
WITH daily_models AS (
    SELECT
        date,
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as daily_model
    FROM sales_data
    GROUP BY date, product
)
SELECT
    date,
    product,
    daily_model.r_squared as daily_r2,
    AVG(daily_model.r_squared) OVER (
        PARTITION BY product
        ORDER BY date
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) as rolling_avg_r2,
    daily_model.coefficients[1] as daily_coefficient
FROM daily_models
WHERE date >= DATE '2024-01-15'
ORDER BY product, date
LIMIT 20;

SELECT '=== Test 5: HAVING with aggregate conditions ===' as test_name;
SELECT
    product,
    region,
    model.r_squared,
    model.coefficients,
    model.n_obs
FROM (
    SELECT
        product,
        region,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend, product_cost], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product, region
    HAVING COUNT(*) >= 15
) sub
WHERE sub.model.r_squared > 0.5
ORDER BY sub.model.r_squared DESC;

SELECT '=== Test 6: Nested aggregation ===' as test_name;
WITH region_models AS (
    SELECT
        region,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model,
        COUNT(*) as n_observations
    FROM sales_data
    GROUP BY region
)
SELECT
    region,
    model.r_squared,
    model.coefficients[1] as marketing_effectiveness,
    n_observations,
    RANK() OVER (ORDER BY model.r_squared DESC) as performance_rank
FROM region_models
ORDER BY performance_rank;

SELECT '=== Test 7: UNION of aggregate results ===' as test_name;
SELECT
    'OLS' as method,
    product,
    model.r_squared,
    model.coefficients[1] as coef1
FROM (
    SELECT
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) sub
UNION ALL
SELECT
    'Ridge (lambda=1)' as method,
    product,
    model.r_squared,
    model.coefficients[1] as coef1
FROM (
    SELECT
        product,
        anofox_statistics_ridge_fit_agg(revenue, [marketing_spend], {'lambda': 1.0, 'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) sub
UNION ALL
SELECT
    'RLS (ff=0.95)' as method,
    product,
    model.r_squared,
    model.coefficients[1] as coef1
FROM (
    SELECT
        product,
        anofox_statistics_rls_fit_agg(revenue, [marketing_spend], {'forgetting_factor': 0.95, 'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) sub
ORDER BY product, method;

SELECT '=== Test 8: Self-join with aggregates ===' as test_name;
WITH product_models AS (
    SELECT
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product
)
SELECT
    a.product as product_a,
    b.product as product_b,
    a.model.r_squared as r2_a,
    b.model.r_squared as r2_b,
    ABS(a.model.coefficients[1] - b.model.coefficients[1]) as coef_difference
FROM product_models a
CROSS JOIN product_models b
WHERE a.product < b.product
ORDER BY coef_difference DESC;

SELECT '=== Test 9: Aggregate with CASE expressions ===' as test_name;
SELECT
    product,
    CASE
        WHEN model.r_squared > 0.8 THEN 'Excellent fit'
        WHEN model.r_squared > 0.6 THEN 'Good fit'
        WHEN model.r_squared > 0.4 THEN 'Moderate fit'
        ELSE 'Poor fit'
    END as model_quality,
    model.r_squared,
    model.coefficients,
    model.n_obs
FROM (
    SELECT
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend, product_cost], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) sub
ORDER BY model.r_squared DESC;

SELECT '=== Test 10: Complex filtering with aggregates ===' as test_name;
WITH initial_models AS (
    SELECT
        product,
        region,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model,
        AVG(revenue) as avg_revenue
    FROM sales_data
    WHERE revenue > 100
    GROUP BY product, region
    HAVING COUNT(*) >= 10
)
SELECT
    product,
    region,
    model.r_squared,
    model.coefficients[1] as marketing_roi,
    avg_revenue
FROM initial_models
WHERE model.r_squared > 0.3
ORDER BY model.r_squared DESC
LIMIT 15;

SELECT '=== Test 11: Aggregate results in calculated columns ===' as test_name;
SELECT
    product,
    model.r_squared,
    model.coefficients[1] * 100 as roi_percentage,
    model.intercept,
    model.r_squared * model.n_obs as weighted_quality_score,
    ROUND(model.r_squared, 3) as r2_rounded
FROM (
    SELECT
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product
) sub;

SELECT '=== Test 12: Time-series analysis with LAG ===' as test_name;
WITH daily_coefficients AS (
    SELECT
        date,
        product,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model
    FROM sales_data
    GROUP BY date, product
)
SELECT
    date,
    product,
    model.coefficients[1] as current_coef,
    LAG(model.coefficients[1]) OVER (PARTITION BY product ORDER BY date) as prev_coef,
    model.coefficients[1] - LAG(model.coefficients[1]) OVER (PARTITION BY product ORDER BY date) as coef_change
FROM daily_coefficients
WHERE date >= DATE '2024-01-10' AND date <= DATE '2024-01-20'
ORDER BY product, date;

SELECT '=== Test 13: Multi-level aggregation ===' as test_name;
WITH product_level AS (
    SELECT
        product,
        region,
        anofox_statistics_ols_fit_agg(revenue, [marketing_spend], {'intercept': true}) as model
    FROM sales_data
    GROUP BY product, region
),
summary AS (
    SELECT
        product,
        AVG(model.r_squared) as avg_r2_across_regions,
        MAX(model.r_squared) as max_r2,
        MIN(model.r_squared) as min_r2,
        COUNT(*) as num_regions
    FROM product_level
    GROUP BY product
)
SELECT * FROM summary
ORDER BY avg_r2_across_regions DESC;

-- Cleanup
DROP TABLE sales_data;

SELECT '=== All integration tests completed ===' as status;
