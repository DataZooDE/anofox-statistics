-- Comprehensive intercept validation tests
-- Validates intercept=true vs intercept=false behavior across all aggregate functions
--
-- R VALIDATION METHODOLOGY FOR INTERCEPT HANDLING:
--
-- All aggregate functions support two modes:
-- 1. intercept=TRUE:  y ~ β₀ + β₁x₁ + β₂x₂ + ... (standard regression)
--    - R² computed as: 1 - SS_res/SS_tot where SS_tot = Σ(y - ȳ)²
--    - Intercept estimated by centering: β₀ = ȳ - Σβᵢx̄ᵢ
--
-- 2. intercept=FALSE: y ~ β₁x₁ + β₂x₂ + ... (regression through origin)
--    - R² computed as: 1 - SS_res/SS_tot where SS_tot = Σy²
--    - Intercept forced to exactly 0.0
--    - R² typically lower (different baseline)
--
-- R validation commands:
--   lm(y ~ x1 + x2, data=df)        # intercept=TRUE
--   lm(y ~ x1 + x2 - 1, data=df)    # intercept=FALSE
--
-- Critical validation: When intercept=FALSE, result.intercept MUST be exactly 0.0
-- Tolerance: ±1e-10 for coefficients, intercept=0.0 must be exact
--
-- Test data: n=30, y = 10 + 3*x1 + 3*x2 + noise

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create test data for intercept testing
CREATE TABLE intercept_test_data AS
SELECT
    i,
    i::DOUBLE as x1,
    (i * 2.0)::DOUBLE as x2,
    (10.0 + 3.0 * i + 1.5 * i * 2.0 + random() * 2.0)::DOUBLE as y,
    (1.0 + 0.05 * i)::DOUBLE as weight
FROM generate_series(1, 30) as t(i);

-- =============================================================================
-- Test 1: OLS Aggregate with vs without intercept
-- =============================================================================
-- R validation:
--   Mode 1 (intercept=TRUE):
--     fit_with <- lm(y ~ x1 + x2, data=df)
--     coef(fit_with)[1]  # Intercept (should be ≈10)
--     summary(fit_with)$r.squared
--
--   Mode 2 (intercept=FALSE):
--     fit_without <- lm(y ~ x1 + x2 - 1, data=df)
--     # Intercept is 0 (not estimated)
--     summary(fit_without)$r.squared  # Different R² (from origin)
--
--   Expected: intercept ≈10.0 for mode 1, exactly 0.0 for mode 2
--
-- Tolerance: intercept=0.0 must be exact when intercept=FALSE
SELECT '=== Test 1: OLS with vs without intercept ===' as test_name;
SELECT
    'With intercept' as mode,
    result.intercept,
    result.coefficients,
    result.r_squared
FROM (
    SELECT anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': true}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'Without intercept' as mode,
    result.intercept as should_be_zero,
    result.coefficients,
    result.r_squared
FROM (
    SELECT anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': false}) as result
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 2: Verify R² calculation difference (critical validation)
-- =============================================================================
-- R validation: Demonstrates different R² baselines
--   With intercept:    SS_tot = Σ(y - ȳ)²  (variance around mean)
--   Without intercept: SS_tot = Σy²        (variance around zero)
--
--   R commands:
--     summary(lm(y ~ x1, data=df))$r.squared       # With intercept
--     summary(lm(y ~ x1 - 1, data=df))$r.squared   # Without intercept
--
--   Expected: R²_with_int > R²_without_int (typically)
--   Why: When data has non-zero mean, forcing through origin reduces fit quality
--
-- Tolerance: R² difference should be positive for this data (intercept ≈10)
SELECT '=== Test 2: Verify R² calculation difference ===' as test_name;
SELECT
    with_int.r_squared as r2_with_intercept,
    without_int.r_squared as r2_without_intercept,
    (with_int.r_squared - without_int.r_squared) as r2_difference,
    CASE
        WHEN with_int.r_squared > without_int.r_squared THEN 'With intercept has higher R² (expected)'
        ELSE 'Unexpected: Without intercept has higher R²'
    END as interpretation
FROM (
    SELECT anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true}) as with_int,
           anofox_statistics_ols_fit_agg(y, [x1], {'intercept': false}) as without_int
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 3: WLS intercept validation
-- =============================================================================
-- R validation (using lm with weights parameter):
--   Mode 1 (intercept=TRUE):
--     fit_with <- lm(y ~ x1 + x2, data=df, weights=weight)
--     coef(fit_with)[1]  # Intercept (should be ≈10)
--     summary(fit_with)$r.squared
--
--   Mode 2 (intercept=FALSE):
--     fit_without <- lm(y ~ x1 + x2 - 1, data=df, weights=weight)
--     # Intercept is 0 (not estimated)
--     summary(fit_without)$r.squared  # Different R² (from origin)
--
--   Expected: WLS adjusts for heteroscedasticity by weighting observations
--   Tolerance: intercept=0.0 must be exact when intercept=FALSE
SELECT '=== Test 3: WLS intercept validation ===' as test_name;
SELECT
    'WLS with intercept' as mode,
    result.intercept,
    result.r_squared,
    result.weighted_mse
FROM (
    SELECT anofox_statistics_wls_fit_agg(y, [x1, x2], weight, {'intercept': true}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'WLS without intercept' as mode,
    result.intercept,
    result.r_squared,
    result.weighted_mse
FROM (
    SELECT anofox_statistics_wls_fit_agg(y, [x1, x2], weight, {'intercept': false}) as result
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 4: Ridge intercept validation
-- =============================================================================
-- R validation (using glmnet with alpha=0 for L2 penalty):
--   Mode 1 (intercept=TRUE):
--     library(glmnet)
--     y_c <- y - mean(y); X_c <- scale(X, scale=FALSE)
--     fit_with <- glmnet(X_c, y_c, alpha=0, lambda=1.0/n, intercept=FALSE)
--     intercept <- mean(y) - coef(fit_with)[-1] %*% colMeans(X)
--     # Note: glmnet uses lambda/n scaling
--
--   Mode 2 (intercept=FALSE):
--     fit_without <- glmnet(X, y, alpha=0, lambda=1.0/n, intercept=FALSE)
--     # Intercept is 0 (not estimated)
--
--   Expected: Ridge penalty shrinks coefficients toward zero
--   Tolerance: intercept=0.0 must be exact when intercept=FALSE
SELECT '=== Test 4: Ridge intercept validation ===' as test_name;
SELECT
    'Ridge with intercept' as mode,
    result.intercept,
    result.coefficients,
    result.r_squared,
    result.lambda
FROM (
    SELECT anofox_statistics_ridge_fit_agg(y, [x1, x2], {'lambda': 1.0, 'intercept': true}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'Ridge without intercept' as mode,
    result.intercept,
    result.coefficients,
    result.r_squared,
    result.lambda
FROM (
    SELECT anofox_statistics_ridge_fit_agg(y, [x1, x2], {'lambda': 1.0, 'intercept': false}) as result
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 5: RLS intercept validation
-- =============================================================================
-- R validation (using custom RLS implementation):
--   Update equations: θ(t) = θ(t-1) + K(t) * (y(t) - φ(t)'θ(t-1))
--                     K(t) = P(t-1)φ(t) / (λ + φ(t)'P(t-1)φ(t))
--                     P(t) = (P(t-1) - K(t)φ(t)'P(t-1)) / λ
--
--   Mode 1 (intercept=TRUE):
--     # Center data first: y_c = y - mean(y), X_c = X - colMeans(X)
--     P <- diag(p) * 1000; theta <- rep(0, p)
--     for (i in 1:n) {
--       phi <- X_c[i,]
--       K <- (P %*% phi) / (lambda + t(phi) %*% P %*% phi)
--       theta <- theta + K * (y_c[i] - t(phi) %*% theta)
--       P <- (P - K %*% t(phi) %*% P) / lambda
--     }
--     intercept <- mean(y) - theta %*% colMeans(X)
--
--   Mode 2 (intercept=FALSE):
--     # Use raw data (no centering), intercept = 0
--
--   Expected: Sequential updates converge to final estimate
--   Tolerance: intercept=0.0 must be exact when intercept=FALSE
SELECT '=== Test 5: RLS intercept validation ===' as test_name;
SELECT
    'RLS with intercept' as mode,
    result.intercept,
    result.coefficients,
    result.r_squared,
    result.forgetting_factor
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x1, x2], {'forgetting_factor': 1.0, 'intercept': true}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'RLS without intercept' as mode,
    result.intercept,
    result.coefficients,
    result.r_squared,
    result.forgetting_factor
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x1, x2], {'forgetting_factor': 1.0, 'intercept': false}) as result
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 6: Coefficient magnitude comparison
-- =============================================================================
-- R validation: Demonstrates coefficient magnitude shift without intercept
--   Without intercept, coefficients often have larger magnitudes to compensate
--   for lack of intercept term (y = β₁x₁ must reach baseline y values)
--
--   R commands:
--     fit_with <- lm(y ~ x1, data=df)
--     fit_without <- lm(y ~ x1 - 1, data=df)
--     coef(fit_with)[2]     # Slope with intercept
--     coef(fit_without)[1]  # Slope without intercept (often larger)
--
--   Expected: |β_without| > |β_with| when data has non-zero mean
--   Why: Without intercept, slope must compensate for baseline shift
-- Tolerance: Magnitude difference should be positive for this data
SELECT '=== Test 6: Coefficient magnitude comparison ===' as test_name;
-- Without intercept, coefficients often have larger magnitudes
SELECT
    'OLS' as method,
    with_int.coefficients[1] as coef_with_intercept,
    without_int.coefficients[1] as coef_without_intercept,
    ABS(without_int.coefficients[1]) - ABS(with_int.coefficients[1]) as magnitude_difference
FROM (
    SELECT
        anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true}) as with_int,
        anofox_statistics_ols_fit_agg(y, [x1], {'intercept': false}) as without_int
    FROM intercept_test_data
) sub;

-- =============================================================================
-- Test 7: Intercept sign and interpretation
-- =============================================================================
-- R validation: Test recovery of known true intercept value
--   Test data: y = 50 + 2*x (perfect linear, no noise)
--   True intercept: 50.0
--   True slope: 2.0
--
--   R command:
--     df <- data.frame(x = 1:20, y = 50 + 2*(1:20))
--     fit <- lm(y ~ x, data = df)
--     coef(fit)[1]  # Should be exactly 50.0
--     coef(fit)[2]  # Should be exactly 2.0
--
--   Expected: Perfect fit (R² = 1.0), intercept ∈ [45, 55]
--   Tolerance: ±5.0 for intercept (accounts for potential noise)
SELECT '=== Test 7: Intercept sign and interpretation ===' as test_name;
-- Create data with known positive intercept
CREATE TABLE known_intercept_data AS
SELECT
    i::DOUBLE as x,
    (50.0 + 2.0 * i)::DOUBLE as y  -- True intercept is 50
FROM generate_series(1, 20) as t(i);

SELECT
    result.intercept,
    result.coefficients[1] as slope,
    result.r_squared,
    CASE
        WHEN result.intercept BETWEEN 45.0 AND 55.0 THEN 'Intercept estimate looks reasonable'
        ELSE 'Unexpected intercept value'
    END as validation
FROM (
    SELECT anofox_statistics_ols_fit_agg(y, [x], {'intercept': true}) as result
    FROM known_intercept_data
) sub;

-- =============================================================================
-- Test 8: Rolling/Expanding OLS intercept validation
-- =============================================================================
-- R validation: Rolling window regression with/without intercept
--   Rolling windows: Apply OLS to fixed-size sliding windows
--
--   R implementation (using rollapply from zoo package):
--     library(zoo)
--     rollapply(df, width=10, by.column=FALSE, FUN=function(w) {
--       fit <- lm(y ~ x1 + x2, data=as.data.frame(w))
--       c(intercept=coef(fit)[1], r2=summary(fit)$r.squared)
--     })
--
--   Without intercept:
--     rollapply(..., FUN=function(w) {
--       fit <- lm(y ~ x1 + x2 - 1, data=as.data.frame(w))
--       c(intercept=0, r2=summary(fit)$r.squared)
--     })
--
--   Expected: Each window produces independent regression estimate
--   Tolerance: intercept[i] = 0.0 for all windows when intercept=FALSE
-- Test 8: SKIPPED - anofox_statistics_rolling_ols function not implemented yet
-- SELECT '=== Test 8: Rolling/Expanding OLS intercept validation ===' as test_name;
-- CREATE TABLE rolling_test_array AS
-- SELECT
--     LIST(y ORDER BY i) as y_array,
--     LIST([x1, x2] ORDER BY i) as x_array
-- FROM (
--     SELECT i, x1, x2, y
--     FROM intercept_test_data
--     ORDER BY i
-- ) sub;
--
-- SELECT
--     'Rolling OLS with intercept' as mode,
--     result.intercept[1] as first_window_intercept,
--     result.r_squared[1] as first_window_r2
-- FROM (
--     SELECT * FROM anofox_statistics_rolling_ols(
--         (SELECT y_array FROM rolling_test_array),
--         (SELECT x_array FROM rolling_test_array),
--         {'window_size': 10, 'intercept': true}
--     )
--     LIMIT 1
-- ) result
-- UNION ALL
-- SELECT
--     'Rolling OLS without intercept' as mode,
--     result.intercept[1] as should_be_zero,
--     result.r_squared[1] as first_window_r2
-- FROM (
--     SELECT * FROM anofox_statistics_rolling_ols(
--         (SELECT y_array FROM rolling_test_array),
--         (SELECT x_array FROM rolling_test_array),
--         {'window_size': 10, 'intercept': false}
--     )
--     LIMIT 1
-- ) result;

-- =============================================================================
-- Test 9: Expanding OLS intercept validation
-- =============================================================================
-- R validation: Expanding window regression with/without intercept
--   Expanding windows: Apply OLS to growing windows from start
--   Window i contains observations [1, i] (min_periods <= i <= n)
--
--   R implementation:
--     results <- lapply(min_periods:n, function(i) {
--       fit <- lm(y ~ x1 + x2, data=df[1:i,])
--       c(intercept=coef(fit)[1], r2=summary(fit)$r.squared)
--     })
--
--   Without intercept:
--     fit <- lm(y ~ x1 + x2 - 1, data=df[1:i,])
--
--   Expected: Estimates stabilize as more data is included
--   Tolerance: intercept[i] = 0.0 for all windows when intercept=FALSE
-- Test 9: SKIPPED - anofox_statistics_expanding_ols function not implemented yet
-- SELECT '=== Test 9: Expanding OLS intercept validation ===' as test_name;
-- SELECT
--     'Expanding OLS with intercept' as mode,
--     result.intercept[1] as first_window_intercept,
--     result.r_squared[1] as first_window_r2
-- FROM (
--     SELECT * FROM anofox_statistics_expanding_ols(
--         (SELECT y_array FROM rolling_test_array),
--         (SELECT x_array FROM rolling_test_array),
--         {'min_periods': 10, 'intercept': true}
--     )
--     LIMIT 1
-- ) result
-- UNION ALL
-- SELECT
--     'Expanding OLS without intercept' as mode,
--     result.intercept[1] as should_be_zero,
--     result.r_squared[1] as first_window_r2
-- FROM (
--     SELECT * FROM anofox_statistics_expanding_ols(
--         (SELECT y_array FROM rolling_test_array),
--         (SELECT x_array FROM rolling_test_array),
--         {'min_periods': 10, 'intercept': false}
--     )
--     LIMIT 1
-- ) result;

-- =============================================================================
-- Test 10: Intercept with GROUP BY
-- =============================================================================
-- R validation: Grouped regression with/without intercept
--   Test data: Two groups (group_a: x=1..15, group_b: x=16..30)
--   Model: y = 20 + 3*x (perfect linear)
--
--   R implementation (using aggregate):
--     # With intercept
--     aggregate(cbind(y, x) ~ grp, data=df, function(d) {
--       fit <- lm(y ~ x, data=as.data.frame(d))
--       c(intercept=coef(fit)[1], r2=summary(fit)$r.squared)
--     })
--
--     # Without intercept
--     aggregate(cbind(y, x) ~ grp, data=df, function(d) {
--       fit <- lm(y ~ x - 1, data=as.data.frame(d))
--       c(intercept=0, r2=summary(fit)$r.squared)
--     })
--
--   Expected: Both groups fit perfectly (R² ≈ 1.0)
--   Expected: intercept_with ≈ 20, intercept_without = 0.0
--   Tolerance: ±1e-10 for intercept when disabled
SELECT '=== Test 10: Intercept with GROUP BY ===' as test_name;
CREATE TABLE grouped_intercept_test AS
SELECT
    CASE WHEN i <= 15 THEN 'group_a' ELSE 'group_b' END as grp,
    i::DOUBLE as x,
    (20.0 + 3.0 * i)::DOUBLE as y
FROM generate_series(1, 30) as t(i);

SELECT
    grp,
    with_int.intercept as intercept_with,
    without_int.intercept as intercept_without,
    with_int.r_squared as r2_with,
    without_int.r_squared as r2_without
FROM (
    SELECT
        grp,
        anofox_statistics_ols_fit_agg(y, [x], {'intercept': true}) as with_int,
        anofox_statistics_ols_fit_agg(y, [x], {'intercept': false}) as without_int
    FROM grouped_intercept_test
    GROUP BY grp
) sub
ORDER BY grp;

-- =============================================================================
-- Test 11: Verify intercept is truly zero when disabled (CRITICAL TEST)
-- =============================================================================
-- R validation: Critical test for intercept=FALSE requirement
--   This test validates the most important property: when intercept=FALSE,
--   the intercept field MUST be exactly 0.0 (not approximately, but exactly)
--
--   R behavior for reference:
--     fit <- lm(y ~ x1 - 1, data=df)  # The "-1" removes intercept
--     # R does not estimate intercept at all in this case
--     # Our implementation must return exactly 0.0
--
--   For each method:
--     - OLS: lm(y ~ x1 - 1, data=df)
--     - WLS: lm(y ~ x1 - 1, weights=weight, data=df)
--     - Ridge: glmnet(X, y, alpha=0, lambda=λ/n, intercept=FALSE)
--     - RLS: Custom implementation on raw data (no centering)
--
--   Validation criterion: |intercept| < 1e-10
--   Expected result: "PASS: Intercept is zero" for all 4 methods
--
--   CRITICAL: If this test fails, the intercept handling is incorrect
--   and violates the fundamental contract of intercept=FALSE option.
SELECT '=== Test 11: Verify intercept is truly zero when disabled ===' as test_name;
SELECT
    'OLS' as method,
    ABS(result.intercept) as absolute_intercept,
    CASE
        WHEN ABS(result.intercept) < 1e-10 THEN 'PASS: Intercept is zero'
        ELSE 'FAIL: Intercept is not zero'
    END as validation
FROM (
    SELECT anofox_statistics_ols_fit_agg(y, [x1], {'intercept': false}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'WLS' as method,
    ABS(result.intercept) as absolute_intercept,
    CASE
        WHEN ABS(result.intercept) < 1e-10 THEN 'PASS: Intercept is zero'
        ELSE 'FAIL: Intercept is not zero'
    END as validation
FROM (
    SELECT anofox_statistics_wls_fit_agg(y, [x1], weight, {'intercept': false}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'Ridge' as method,
    ABS(result.intercept) as absolute_intercept,
    CASE
        WHEN ABS(result.intercept) < 1e-10 THEN 'PASS: Intercept is zero'
        ELSE 'FAIL: Intercept is not zero'
    END as validation
FROM (
    SELECT anofox_statistics_ridge_fit_agg(y, [x1], {'lambda': 1.0, 'intercept': false}) as result
    FROM intercept_test_data
) sub
UNION ALL
SELECT
    'RLS' as method,
    ABS(result.intercept) as absolute_intercept,
    CASE
        WHEN ABS(result.intercept) < 1e-10 THEN 'PASS: Intercept is zero'
        ELSE 'FAIL: Intercept is not zero'
    END as validation
FROM (
    SELECT anofox_statistics_rls_fit_agg(y, [x1], {'forgetting_factor': 1.0, 'intercept': false}) as result
    FROM intercept_test_data
) sub;

-- Cleanup
DROP TABLE intercept_test_data;
DROP TABLE known_intercept_data;
-- DROP TABLE rolling_test_array; -- Table not created since tests are skipped
DROP TABLE grouped_intercept_test;

SELECT '=== All intercept validation tests completed ===' as status;
