-- Test suite for anofox_statistics_wls_agg aggregate function (Weighted Least Squares)

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create test data with varying weights (heteroscedasticity scenario)
CREATE TABLE wls_agg_data AS
SELECT
    CASE
        WHEN i <= 12 THEN 'segment_premium'
        WHEN i <= 24 THEN 'segment_standard'
        ELSE 'segment_budget'
    END as customer_segment,
    i::DOUBLE as customer_tenure_months,
    (50.0 + i * 5.0 + random() * 10.0 * SQRT(i))::DOUBLE as monthly_revenue,
    -- Weight by inverse of variance (precision weights)
    (1.0 / (1.0 + i * 0.5))::DOUBLE as weight,
    i as customer_id
FROM generate_series(1, 36) as t(i);

SELECT '=== Test 1: WLS aggregate by segment with weights ===' as test_name;
SELECT
    customer_segment,
    result.coefficients[1] as tenure_effect,
    result.intercept,
    result.r2,
    result.weighted_mse,
    result.n_obs
FROM (
    SELECT
        customer_segment,
        anofox_statistics_wls_agg(
            monthly_revenue,
            [customer_tenure_months],
            weight,
            {'intercept': true}
        ) as result
    FROM wls_agg_data
    GROUP BY customer_segment
) sub
ORDER BY customer_segment;

SELECT '=== Test 2: WLS vs OLS comparison ===' as test_name;
SELECT
    customer_segment,
    wls.r2 as wls_r2,
    ols.r2 as ols_r2,
    wls.coefficients[1] as wls_coef,
    ols.coefficients[1] as ols_coef,
    (wls.r2 - ols.r2) as r2_improvement
FROM (
    SELECT
        customer_segment,
        anofox_statistics_wls_agg(monthly_revenue, [customer_tenure_months], weight, {'intercept': true}) as wls,
        anofox_statistics_ols_agg(monthly_revenue, [customer_tenure_months], {'intercept': true}) as ols
    FROM wls_agg_data
    GROUP BY customer_segment
) sub
ORDER BY customer_segment;

SELECT '=== Test 3: WLS without intercept ===' as test_name;
SELECT
    customer_segment,
    result.intercept as should_be_zero,
    result.coefficients,
    result.weighted_mse
FROM (
    SELECT
        customer_segment,
        anofox_statistics_wls_agg(
            monthly_revenue,
            [customer_tenure_months],
            weight,
            {'intercept': false}
        ) as result
    FROM wls_agg_data
    GROUP BY customer_segment
) sub
ORDER BY customer_segment;

SELECT '=== Test 4: Multiple predictors with weights ===' as test_name;
CREATE TABLE wls_multi_data AS
SELECT
    'test_group' as grp,
    i::DOUBLE as x1,
    (i * 2.0)::DOUBLE as x2,
    (i * 0.5)::DOUBLE as x3,
    (10.0 + 2.0 * i + 1.5 * i * 2.0 - 0.8 * i * 0.5 + random() * 3.0)::DOUBLE as y,
    (1.0 + 0.1 * i)::DOUBLE as w
FROM generate_series(1, 20) as t(i);

SELECT
    grp,
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2
FROM (
    SELECT
        grp,
        anofox_statistics_wls_agg(y, [x1, x2, x3], w, {'intercept': true}) as result
    FROM wls_multi_data
    GROUP BY grp
) sub;

SELECT '=== Test 5: Uniform weights (should match OLS) ===' as test_name;
CREATE TABLE uniform_weight_data AS
SELECT
    'group_a' as grp,
    i::DOUBLE as x,
    (5.0 + 2.0 * i)::DOUBLE as y,
    1.0 as weight  -- Uniform weights
FROM generate_series(1, 15) as t(i);

SELECT
    'WLS with uniform weights' as method,
    result.coefficients[1] as slope,
    result.intercept,
    result.r2
FROM (
    SELECT anofox_statistics_wls_agg(y, [x], weight, {'intercept': true}) as result
    FROM uniform_weight_data
) sub
UNION ALL
SELECT
    'OLS for comparison' as method,
    result.coefficients[1] as slope,
    result.intercept,
    result.r2
FROM (
    SELECT anofox_statistics_ols_agg(y, [x], {'intercept': true}) as result
    FROM uniform_weight_data
) sub;

SELECT '=== Test 6: High vs low weight observations ===' as test_name;
CREATE TABLE weight_impact_data AS
SELECT
    'scenario_1' as scenario,
    i::DOUBLE as x,
    (10.0 + 3.0 * i)::DOUBLE as y,
    CASE WHEN i <= 5 THEN 10.0 ELSE 1.0 END as weight  -- First 5 observations heavily weighted
FROM generate_series(1, 10) as t(i);

SELECT
    scenario,
    result.coefficients[1] as weighted_slope,
    result.intercept as weighted_intercept,
    result.weighted_mse
FROM (
    SELECT
        scenario,
        anofox_statistics_wls_agg(y, [x], weight, {'intercept': true}) as result
    FROM weight_impact_data
    GROUP BY scenario
) sub;

SELECT '=== Test 7: Zero and NULL weight handling ===' as test_name;
CREATE TABLE edge_weight_data AS
SELECT
    'test' as grp,
    i::DOUBLE as x,
    (i * 2.0)::DOUBLE as y,
    CASE
        WHEN i = 3 THEN 0.0  -- Zero weight
        WHEN i = 5 THEN NULL -- NULL weight
        ELSE 1.0
    END as weight
FROM generate_series(1, 8) as t(i);

SELECT
    grp,
    result.n_obs,
    result.coefficients
FROM (
    SELECT
        grp,
        anofox_statistics_wls_agg(y, [x], weight, {'intercept': true}) as result
    FROM edge_weight_data
    GROUP BY grp
) sub;

SELECT '=== Test 8: Entire dataset aggregation ===' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.r2,
    result.weighted_mse,
    result.n_obs
FROM (
    SELECT
        anofox_statistics_wls_agg(monthly_revenue, [customer_tenure_months], weight, {'intercept': true}) as result
    FROM wls_agg_data
) sub;

-- Cleanup
DROP TABLE wls_agg_data;
DROP TABLE wls_multi_data;
DROP TABLE uniform_weight_data;
DROP TABLE weight_impact_data;
DROP TABLE edge_weight_data;

SELECT '=== WLS aggregate tests completed ===' as status;
