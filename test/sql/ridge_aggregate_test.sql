-- Test suite for anofox_stats_ridge_fit_agg aggregate function (Ridge Regression with L2 regularization)

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create test data with multicollinearity
CREATE TABLE ridge_agg_data AS
SELECT
    CASE
        WHEN i <= 15 THEN 'market_tech'
        WHEN i <= 30 THEN 'market_finance'
        ELSE 'market_healthcare'
    END as sector,
    i::DOUBLE as time_period,
    (100.0 + i * 2.0 + random() * 5.0)::DOUBLE as price,
    (100.0 + i * 2.05 + random() * 5.0)::DOUBLE as price_correlated,  -- Highly correlated with price
    (50.0 + i * 1.0 + random() * 3.0)::DOUBLE as volume,
    (200.0 + i * 3.0 + 0.5 * i * 2.0 - 0.3 * i * 1.0 + random() * 10.0)::DOUBLE as returns
FROM generate_series(1, 45) as t(i);

SELECT '=== Test 1: Ridge aggregate with lambda=0.1 ===' as test_name;
SELECT
    sector,
    result.coefficients,
    result.intercept,
    result.r2,
    result.lambda,
    result.n_obs
FROM (
    SELECT
        sector,
        anofox_stats_ridge_fit_agg(
            returns,
            [price, volume],
            {'lambda': 0.1, 'intercept': true}
        ) as result
    FROM ridge_agg_data
    GROUP BY sector
) sub
ORDER BY sector;

SELECT '=== Test 2: Ridge vs OLS comparison (multicollinearity) ===' as test_name;
SELECT
    sector,
    ridge.coefficients[1] as ridge_price_coef,
    ols.coefficients[1] as ols_price_coef,
    ridge.coefficients[2] as ridge_correlated_coef,
    ols.coefficients[2] as ols_correlated_coef,
    ridge.r2 as ridge_r2,
    ols.r2 as ols_r2
FROM (
    SELECT
        sector,
        anofox_stats_ridge_fit_agg(returns, [price, price_correlated], {'lambda': 1.0, 'intercept': true}) as ridge,
        anofox_stats_ols_fit_agg(returns, [price, price_correlated], {'intercept': true}) as ols
    FROM ridge_agg_data
    GROUP BY sector
) sub
ORDER BY sector;

SELECT '=== Test 3: Different lambda values ===' as test_name;
SELECT
    sector,
    lambda_0.lambda as lambda_value,
    lambda_0.r2 as r2_lambda_0,
    lambda_1.r2 as r2_lambda_1,
    lambda_10.r2 as r2_lambda_10,
    lambda_0.coefficients[1] as coef_lambda_0,
    lambda_1.coefficients[1] as coef_lambda_1,
    lambda_10.coefficients[1] as coef_lambda_10
FROM (
    SELECT
        sector,
        anofox_stats_ridge_fit_agg(returns, [price], {'lambda': 0.0, 'intercept': true}) as lambda_0,
        anofox_stats_ridge_fit_agg(returns, [price], {'lambda': 1.0, 'intercept': true}) as lambda_1,
        anofox_stats_ridge_fit_agg(returns, [price], {'lambda': 10.0, 'intercept': true}) as lambda_10
    FROM ridge_agg_data
    GROUP BY sector
) sub
ORDER BY sector;

SELECT '=== Test 4: Ridge without intercept ===' as test_name;
SELECT
    sector,
    result.intercept as should_be_zero,
    result.coefficients,
    result.r2,
    result.lambda
FROM (
    SELECT
        sector,
        anofox_stats_ridge_fit_agg(
            returns,
            [price, volume],
            {'lambda': 1.0, 'intercept': false}
        ) as result
    FROM ridge_agg_data
    GROUP BY sector
) sub
ORDER BY sector;

SELECT '=== Test 5: Multiple correlated predictors ===' as test_name;
CREATE TABLE multi_collinear_data AS
SELECT
    'group_a' as grp,
    i::DOUBLE as x1,
    (i + random() * 0.1)::DOUBLE as x2,  -- Nearly identical to x1
    (i + random() * 0.1)::DOUBLE as x3,  -- Nearly identical to x1
    (10.0 + 2.0 * i + random() * 2.0)::DOUBLE as y
FROM generate_series(1, 25) as t(i);

SELECT
    grp,
    result.coefficients,
    result.r2,
    result.adj_r2,
    result.lambda
FROM (
    SELECT
        grp,
        anofox_stats_ridge_fit_agg(y, [x1, x2, x3], {'lambda': 1.0, 'intercept': true}) as result
    FROM multi_collinear_data
    GROUP BY grp
) sub;

SELECT '=== Test 6: Regularization shrinkage effect ===' as test_name;
CREATE TABLE shrinkage_test AS
SELECT
    'test_group' as grp,
    i::DOUBLE as x,
    (100.0 + 5.0 * i)::DOUBLE as y
FROM generate_series(1, 20) as t(i);

SELECT
    'No regularization (lambda=0)' as scenario,
    result.coefficients[1] as coefficient
FROM (
    SELECT anofox_stats_ridge_fit_agg(y, [x], {'lambda': 0.0, 'intercept': true}) as result
    FROM shrinkage_test
) sub
UNION ALL
SELECT
    'Light regularization (lambda=1)' as scenario,
    result.coefficients[1] as coefficient
FROM (
    SELECT anofox_stats_ridge_fit_agg(y, [x], {'lambda': 1.0, 'intercept': true}) as result
    FROM shrinkage_test
) sub
UNION ALL
SELECT
    'Heavy regularization (lambda=100)' as scenario,
    result.coefficients[1] as coefficient
FROM (
    SELECT anofox_stats_ridge_fit_agg(y, [x], {'lambda': 100.0, 'intercept': true}) as result
    FROM shrinkage_test
) sub;

SELECT '=== Test 7: Entire dataset aggregation ===' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.lambda,
    result.n_obs
FROM (
    SELECT
        anofox_stats_ridge_fit_agg(returns, [price, volume], {'lambda': 0.5, 'intercept': true}) as result
    FROM ridge_agg_data
) sub;

SELECT '=== Test 8: Compare adjusted R² with regularization ===' as test_name;
SELECT
    sector,
    result.lambda,
    result.r2,
    result.adj_r2,
    (result.r2 - result.adj_r2) as r2_penalty
FROM (
    SELECT
        sector,
        anofox_stats_ridge_fit_agg(returns, [price, volume, price_correlated], {'lambda': 2.0, 'intercept': true}) as result
    FROM ridge_agg_data
    GROUP BY sector
) sub
ORDER BY sector;

-- Cleanup
DROP TABLE ridge_agg_data;
DROP TABLE multi_collinear_data;
DROP TABLE shrinkage_test;

SELECT '=== Ridge aggregate tests completed ===' as status;
