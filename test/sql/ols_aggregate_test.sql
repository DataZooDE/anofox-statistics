-- Test suite for anofox_statistics_ols_agg aggregate function

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create comprehensive test dataset
CREATE TABLE ols_agg_data AS
SELECT
    CASE
        WHEN i <= 15 THEN 'product_a'
        WHEN i <= 30 THEN 'product_b'
        ELSE 'product_c'
    END as product,
    CASE
        WHEN i % 3 = 0 THEN 'region_north'
        WHEN i % 3 = 1 THEN 'region_south'
        ELSE 'region_east'
    END as region,
    i::DOUBLE as time,
    (i * 1.5)::DOUBLE as price,
    (i * 0.8)::DOUBLE as marketing_spend,
    (100.0 + 2.5 * i - 0.5 * i * 1.5 + 1.2 * i * 0.8 + random() * 5.0)::DOUBLE as sales
FROM generate_series(1, 45) as t(i);

SELECT '=== Test 1: Basic OLS aggregate by product ===' as test_name;
SELECT
    product,
    result.coefficients[1] as price_effect,
    result.coefficients[2] as marketing_effect,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.n_obs
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

SELECT '=== Test 2: OLS aggregate by region ===' as test_name;
SELECT
    region,
    result.coefficients,
    result.intercept,
    result.r2
FROM (
    SELECT
        region,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY region
) sub
ORDER BY region;

SELECT '=== Test 3: OLS aggregate without intercept ===' as test_name;
SELECT
    product,
    result.intercept as should_be_zero,
    result.coefficients,
    result.r2
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': false}) as result
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

SELECT '=== Test 4: Multi-level grouping ===' as test_name;
SELECT
    product,
    region,
    result.n_obs,
    result.r2
FROM (
    SELECT
        product,
        region,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product, region
) sub
ORDER BY product, region;

SELECT '=== Test 5: Single predictor ===' as test_name;
SELECT
    product,
    result.coefficients[1] as price_coef,
    result.intercept,
    result.r2
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

SELECT '=== Test 6: Three predictors ===' as test_name;
SELECT
    product,
    result.coefficients,
    result.n_features,
    result.r2
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price, marketing_spend, time], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

SELECT '=== Test 7: Compare intercept vs no-intercept R² ===' as test_name;
SELECT
    product,
    with_intercept.r2 as r2_with_intercept,
    without_intercept.r2 as r2_without_intercept,
    (with_intercept.r2 - without_intercept.r2) as r2_difference
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price], {'intercept': true}) as with_intercept,
        anofox_statistics_ols_agg(sales, [price], {'intercept': false}) as without_intercept
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

SELECT '=== Test 8: Aggregate entire dataset (no grouping) ===' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.n_obs,
    result.n_features
FROM (
    SELECT
        anofox_statistics_ols_agg(sales, [price, marketing_spend, time], {'intercept': true}) as result
    FROM ols_agg_data
) sub;

SELECT '=== Test 9: HAVING clause with aggregate ===' as test_name;
SELECT
    product,
    result.r2,
    result.n_obs
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product
    HAVING COUNT(*) >= 10
) sub
ORDER BY result.r2 DESC;

SELECT '=== Test 10: Coefficient extraction ===' as test_name;
SELECT
    product,
    result.coefficients[1] as beta_1,
    result.coefficients[2] as beta_2,
    result.intercept as beta_0
FROM (
    SELECT
        product,
        anofox_statistics_ols_agg(sales, [price, marketing_spend], {'intercept': true}) as result
    FROM ols_agg_data
    GROUP BY product
) sub
ORDER BY product;

-- Cleanup
DROP TABLE ols_agg_data;

SELECT '=== OLS aggregate tests completed ===' as status;
