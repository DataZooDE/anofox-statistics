# name: test/sql/window_functions_test.sql
# description: Test window functions (OVER clause) for all aggregate functions
# group: [anofox_statistics]

require anofox_statistics

# ==============================================================================
# Test Setup: Create sample time series data
# ==============================================================================

statement ok
CREATE TABLE ts_data AS
SELECT
    t::INTEGER as time,
    (10.0 + 2.0 * t + 0.5 * t * t + (random() - 0.5) * 2.0)::DOUBLE as y,
    t::DOUBLE as x1,
    (t * t)::DOUBLE as x2,
    (1.0 + 0.1 * t)::DOUBLE as weight
FROM generate_series(1, 20) s(t);

# ==============================================================================
# Test 1: OLS Window - Basic Rolling Window (5-row window)
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
    FROM ts_data
) WHERE ols_result IS NOT NULL;
----
16

# Verify that first 4 rows are NULL (insufficient data for window of 5)
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
    FROM ts_data
    WHERE time <= 4
) WHERE ols_result IS NULL;
----
4

# ==============================================================================
# Test 2: OLS Window - Expanding Window
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as ols_result
    FROM ts_data
) WHERE ols_result IS NOT NULL;
----
17

# First 3 rows should be NULL (need at least p+1=3 observations for 2 features)
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as ols_result
    FROM ts_data
    WHERE time <= 2
) WHERE ols_result IS NULL;
----
2

# ==============================================================================
# Test 3: OLS Window - Extract coefficients from window result
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ols_result.coefficients[1] as coef_x1,
        ols_result.coefficients[2] as coef_x2,
        ols_result.intercept as intercept,
        ols_result.r2 as r2
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    )
) WHERE coef_x1 IS NOT NULL AND r2 >= 0.0 AND r2 <= 1.0;
----
16

# ==============================================================================
# Test 4: WLS Window - Rolling window with weights
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_wls_agg(y, [x1, x2], weight, {'intercept': true})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as wls_result
    FROM ts_data
) WHERE wls_result IS NOT NULL;
----
16

# Verify WLS produces weighted_mse field
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        wls_result.weighted_mse as wmse
    FROM (
        SELECT
            time,
            anofox_statistics_wls_agg(y, [x1, x2], weight, {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as wls_result
        FROM ts_data
    )
) WHERE wmse IS NOT NULL AND wmse >= 0.0;
----
16

# ==============================================================================
# Test 5: Ridge Window - Rolling window with lambda
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ridge_agg(y, [x1, x2], {'intercept': true, 'lambda': 1.0})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ridge_result
    FROM ts_data
) WHERE ridge_result IS NOT NULL;
----
16

# Verify Ridge includes lambda in result
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ridge_result.lambda as lambda_val
    FROM (
        SELECT
            time,
            anofox_statistics_ridge_agg(y, [x1, x2], {'intercept': true, 'lambda': 1.0})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ridge_result
        FROM ts_data
    )
) WHERE lambda_val = 1.0;
----
16

# ==============================================================================
# Test 6: RLS Window - Rolling window with forgetting factor
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_rls_agg(y, [x1, x2], {'intercept': true, 'forgetting_factor': 0.99})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as rls_result
    FROM ts_data
) WHERE rls_result IS NOT NULL;
----
16

# Verify RLS includes forgetting_factor in result
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        rls_result.forgetting_factor as ff
    FROM (
        SELECT
            time,
            anofox_statistics_rls_agg(y, [x1, x2], {'intercept': true, 'forgetting_factor': 0.99})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as rls_result
        FROM ts_data
    )
) WHERE ABS(ff - 0.99) < 1e-10;
----
16

# ==============================================================================
# Test 7: Window with PARTITION BY
# ==============================================================================

statement ok
CREATE TABLE multi_category_data AS
SELECT
    category,
    t::INTEGER as time,
    (10.0 + 2.0 * t + (random() - 0.5) * 2.0)::DOUBLE as y,
    t::DOUBLE as x1,
    (t * 2.0)::DOUBLE as x2
FROM (
    SELECT 'A' as category, generate_series(1, 10) as t
    UNION ALL
    SELECT 'B' as category, generate_series(1, 10) as t
);

query I
SELECT COUNT(*) FROM (
    SELECT
        category,
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (PARTITION BY category ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as ols_result
    FROM multi_category_data
) WHERE ols_result IS NOT NULL;
----
16

# Verify both categories get results
query II
SELECT
    category,
    COUNT(*) as result_count
FROM (
    SELECT
        category,
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
            OVER (PARTITION BY category ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as ols_result
    FROM multi_category_data
)
WHERE ols_result IS NOT NULL
GROUP BY category
ORDER BY category;
----
A	8
B	8

# ==============================================================================
# Test 8: Intercept FALSE with window functions
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        anofox_statistics_ols_agg(y, [x1, x2], {'intercept': false})
            OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
    FROM ts_data
) WHERE ols_result IS NOT NULL;
----
16

# Verify intercept is exactly 0.0 when disabled
query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ols_result.intercept as intercept_val
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': false})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    )
) WHERE ABS(intercept_val) < 1e-10;
----
16

# ==============================================================================
# Test 9: Different window sizes produce different results
# ==============================================================================

# Window size 3 vs window size 5 should produce different coefficients
query I
SELECT COUNT(*) FROM (
    SELECT
        t1.time,
        t1.ols_result.coefficients[1] as coef_w3,
        t2.ols_result.coefficients[1] as coef_w5
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    ) t1
    JOIN (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    ) t2 ON t1.time = t2.time
    WHERE t1.ols_result IS NOT NULL AND t2.ols_result IS NOT NULL
) WHERE ABS(coef_w3 - coef_w5) > 1e-6;
----
13

# ==============================================================================
# Test 10: Window function n_obs field matches window size
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ols_result.n_obs as n_obs
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    )
) WHERE n_obs = 5;
----
16

# ==============================================================================
# Test 11: Expanding window n_obs grows monotonically
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ols_result.n_obs as n_obs,
        time as expected_n
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as ols_result
        FROM ts_data
    )
) WHERE n_obs = expected_n;
----
17

# ==============================================================================
# Test 12: All four methods work with same window specification
# ==============================================================================

query I
SELECT COUNT(*) FROM (
    SELECT
        time,
        ols.r2 as ols_r2,
        wls.r2 as wls_r2,
        ridge.r2 as ridge_r2,
        rls.r2 as rls_r2
    FROM (
        SELECT
            time,
            anofox_statistics_ols_agg(y, [x1, x2], {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ols
        FROM ts_data
    ) o
    JOIN (
        SELECT
            time,
            anofox_statistics_wls_agg(y, [x1, x2], weight, {'intercept': true})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as wls
        FROM ts_data
    ) w ON o.time = w.time
    JOIN (
        SELECT
            time,
            anofox_statistics_ridge_agg(y, [x1, x2], {'intercept': true, 'lambda': 0.1})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ridge
        FROM ts_data
    ) r ON o.time = r.time
    JOIN (
        SELECT
            time,
            anofox_statistics_rls_agg(y, [x1, x2], {'intercept': true, 'forgetting_factor': 1.0})
                OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as rls
        FROM ts_data
    ) rl ON o.time = rl.time
) WHERE ols_r2 IS NOT NULL AND wls_r2 IS NOT NULL AND ridge_r2 IS NOT NULL AND rls_r2 IS NOT NULL;
----
16

# ==============================================================================
# Cleanup
# ==============================================================================

statement ok
DROP TABLE ts_data;

statement ok
DROP TABLE multi_category_data;
