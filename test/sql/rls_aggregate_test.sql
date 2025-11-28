-- Test suite for anofox_statistics_rls_fit_agg aggregate function (Recursive Least Squares)

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create test data simulating time-series with regime changes
CREATE TABLE rls_agg_data AS
SELECT
    CASE
        WHEN i <= 20 THEN 'regime_stable'
        WHEN i <= 40 THEN 'regime_volatile'
        ELSE 'regime_growth'
    END as market_regime,
    i::DOUBLE as time_index,
    -- Simulate regime-dependent relationships
    CASE
        WHEN i <= 20 THEN (50.0 + 2.0 * i + random() * 2.0)
        WHEN i <= 40 THEN (90.0 + 1.0 * i + random() * 8.0)
        ELSE (50.0 + 3.5 * i + random() * 3.0)
    END::DOUBLE as price,
    CASE
        WHEN i <= 20 THEN (30.0 + 1.5 * i + random() * 1.0)
        WHEN i <= 40 THEN (60.0 + 0.8 * i + random() * 5.0)
        ELSE (40.0 + 2.0 * i + random() * 2.0)
    END::DOUBLE as demand,
    -- Response variable
    CASE
        WHEN i <= 20 THEN (100.0 + 3.0 * i - 0.5 * i * 2.0 + random() * 5.0)
        WHEN i <= 40 THEN (120.0 + 2.0 * i - 0.3 * i * 1.0 + random() * 10.0)
        ELSE (80.0 + 4.0 * i - 0.7 * i * 3.5 + random() * 6.0)
    END::DOUBLE as sales
FROM generate_series(1, 60) as t(i);

SELECT '=== Test 1: RLS aggregate with forgetting_factor=1.0 (standard OLS) ===' as test_name;
SELECT
    market_regime,
    result.coefficients,
    result.intercept,
    result.r2,
    result.forgetting_factor,
    result.n_obs
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_fit_agg(
            sales,
            [price, demand],
            {'forgetting_factor': 1.0, 'intercept': true}
        ) as result
    FROM rls_agg_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

SELECT '=== Test 2: RLS vs OLS comparison ===' as test_name;
-- With forgetting_factor=1.0, RLS should be very similar to OLS
SELECT
    market_regime,
    rls.coefficients[1] as rls_price_coef,
    ols.coefficients[1] as ols_price_coef,
    rls.r2 as rls_r2,
    ols.r2 as ols_r2,
    ABS(rls.r2 - ols.r2) as r2_difference
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_fit_agg(sales, [price], {'forgetting_factor': 1.0, 'intercept': true}) as rls,
        anofox_statistics_ols_fit_agg(sales, [price], {'intercept': true}) as ols
    FROM rls_agg_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

SELECT '=== Test 3: Different forgetting factors ===' as test_name;
SELECT
    market_regime,
    ff_1_0.forgetting_factor as ff_1,
    ff_1_0.r2 as r2_ff_1_0,
    ff_0_95.r2 as r2_ff_0_95,
    ff_0_90.r2 as r2_ff_0_90,
    ff_1_0.coefficients[1] as coef_ff_1_0,
    ff_0_95.coefficients[1] as coef_ff_0_95,
    ff_0_90.coefficients[1] as coef_ff_0_90
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_fit_agg(sales, [price], {'forgetting_factor': 1.0, 'intercept': true}) as ff_1_0,
        anofox_statistics_rls_fit_agg(sales, [price], {'forgetting_factor': 0.95, 'intercept': true}) as ff_0_95,
        anofox_statistics_rls_fit_agg(sales, [price], {'forgetting_factor': 0.90, 'intercept': true}) as ff_0_90
    FROM rls_agg_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

SELECT '=== Test 4: RLS without intercept ===' as test_name;
SELECT
    market_regime,
    result.intercept as should_be_zero,
    result.coefficients,
    result.r2,
    result.forgetting_factor
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_fit_agg(
            sales,
            [price, demand],
            {'forgetting_factor': 0.98, 'intercept': false}
        ) as result
    FROM rls_agg_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

SELECT '=== Test 5: Adaptive learning (low forgetting factor) ===' as test_name;
-- Lower forgetting factor emphasizes recent observations
CREATE TABLE adaptive_data AS
SELECT
    'streaming_data' as source,
    i::DOUBLE as obs_index,
    CASE
        WHEN i <= 10 THEN (10.0 + 2.0 * i + random() * 1.0)  -- Initial relationship
        ELSE (50.0 + 5.0 * i + random() * 1.0)  -- Shifted relationship
    END::DOUBLE as x,
    CASE
        WHEN i <= 10 THEN (20.0 + 4.0 * i + random() * 2.0)
        ELSE (100.0 + 10.0 * i + random() * 2.0)
    END::DOUBLE as y
FROM generate_series(1, 20) as t(i);

SELECT
    source,
    result.coefficients[1] as adaptive_coef,
    result.intercept,
    result.r2,
    result.forgetting_factor
FROM (
    SELECT
        source,
        anofox_statistics_rls_fit_agg(y, [x], {'forgetting_factor': 0.85, 'intercept': true}) as result
    FROM adaptive_data
    GROUP BY source
) sub;

SELECT '=== Test 6: Multiple predictors with RLS ===' as test_name;
CREATE TABLE rls_multi_data AS
SELECT
    'multi_var_group' as grp,
    i::DOUBLE as x1,
    (i * 1.5)::DOUBLE as x2,
    (i * 0.8)::DOUBLE as x3,
    (15.0 + 2.0 * i + 1.5 * i * 1.5 - 0.5 * i * 0.8 + random() * 3.0)::DOUBLE as y
FROM generate_series(1, 30) as t(i);

SELECT
    grp,
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.n_obs
FROM (
    SELECT
        grp,
        anofox_statistics_rls_fit_agg(y, [x1, x2, x3], {'forgetting_factor': 0.97, 'intercept': true}) as result
    FROM rls_multi_data
    GROUP BY grp
) sub;

SELECT '=== Test 7: Entire dataset aggregation ===' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.r2,
    result.forgetting_factor,
    result.n_obs
FROM (
    SELECT
        anofox_statistics_rls_fit_agg(sales, [price, demand], {'forgetting_factor': 0.96, 'intercept': true}) as result
    FROM rls_agg_data
) sub;

SELECT '=== Test 8: Sensitivity to forgetting factor ===' as test_name;
CREATE TABLE sensitivity_data AS
SELECT
    'test_set' as grp,
    i::DOUBLE as x,
    (10.0 + 3.0 * i + random() * 2.0)::DOUBLE as y
FROM generate_series(1, 25) as t(i);

SELECT
    'Forgetting Factor: 1.00 (no decay)' as scenario,
    result.coefficients[1] as coefficient,
    result.r2
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x], {'forgetting_factor': 1.00, 'intercept': true}) as result
    FROM sensitivity_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.98 (slow decay)' as scenario,
    result.coefficients[1] as coefficient,
    result.r2
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x], {'forgetting_factor': 0.98, 'intercept': true}) as result
    FROM sensitivity_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.95 (moderate decay)' as scenario,
    result.coefficients[1] as coefficient,
    result.r2
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x], {'forgetting_factor': 0.95, 'intercept': true}) as result
    FROM sensitivity_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.90 (fast decay)' as scenario,
    result.coefficients[1] as coefficient,
    result.r2
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x], {'forgetting_factor': 0.90, 'intercept': true}) as result
    FROM sensitivity_data
) sub;

-- Cleanup
DROP TABLE rls_agg_data;
DROP TABLE adaptive_data;
DROP TABLE rls_multi_data;
DROP TABLE sensitivity_data;

SELECT '=== RLS aggregate tests completed ===' as status;
