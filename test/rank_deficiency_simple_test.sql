-- ======================================================================
-- Simple Rank-Deficiency Test Suite
-- Tests key functions with constant features
-- ======================================================================

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- ======================================================================
-- Test 1: OLS with Constant Feature (varargs API)
-- Expected: NULL for constant feature coefficient
-- ======================================================================
SELECT '=== Test 1: OLS with Constant Feature ===' as test_name;

SELECT * FROM anofox_statistics_ols(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    0.0,  -- lambda
    true  -- add_intercept
);

-- ======================================================================
-- Test 2: Ridge with Constant Feature (lambda=0.1)
-- Expected: NULL for constant feature coefficient
-- ======================================================================
SELECT '=== Test 2: Ridge with Constant Feature ===' as test_name;

SELECT * FROM anofox_statistics_ridge(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    0.1,  -- lambda
    true  -- add_intercept
);

-- ======================================================================
-- Test 3: WLS with Constant Feature
-- Expected: NULL for constant feature coefficient
-- ======================================================================
SELECT '=== Test 3: WLS with Constant Feature ===' as test_name;

SELECT * FROM anofox_statistics_wls(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    [1.0::DOUBLE, 1.0, 1.0, 1.0, 1.0],  -- weights
    true  -- add_intercept
);

-- ======================================================================
-- Test 4: RLS with Constant Feature
-- Expected: NULL for constant feature coefficient
-- ======================================================================
SELECT '=== Test 4: RLS with Constant Feature ===' as test_name;

SELECT * FROM anofox_statistics_rls(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    0.99::DOUBLE,  -- lambda (forgetting factor)
    true  -- add_intercept
);

-- ======================================================================
-- Test 5: Rolling OLS with Constant Feature
-- Expected: NULL for constant feature in coefficients array
-- ======================================================================
SELECT '=== Test 5: Rolling OLS with Constant ===' as test_name;

SELECT window_start, window_end, coefficients, intercept, r_squared
FROM anofox_statistics_rolling_ols(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    3::BIGINT,  -- window_size
    true  -- add_intercept
);

-- ======================================================================
-- Test 6: Expanding OLS with Constant Feature
-- Expected: NULL for constant feature in coefficients array
-- ======================================================================
SELECT '=== Test 6: Expanding OLS with Constant ===' as test_name;

SELECT window_start, window_end, coefficients, intercept, r_squared
FROM anofox_statistics_expanding_ols(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],  -- x1
    [5.0::DOUBLE, 5.0, 5.0, 5.0, 5.0, 5.0],  -- x2 (constant)
    3::BIGINT,  -- min_periods
    true  -- add_intercept
);

-- ======================================================================
-- Test 7: OLS with Perfect Multicollinearity
-- Expected: NULL for one of the collinear features
-- ======================================================================
SELECT '=== Test 7: OLS with Perfect Collinearity ===' as test_name;

SELECT * FROM anofox_statistics_ols(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- x1
    [2.0::DOUBLE, 4.0, 6.0, 8.0, 10.0],  -- x2 = 2*x1 (perfect collinearity)
    0.0,  -- lambda
    true  -- add_intercept
);

-- ======================================================================
-- Test 8: VIF with Constant Feature
-- Expected: NULL (nan) for constant feature
-- ======================================================================
SELECT '=== Test 8: VIF with Constant Feature ===' as test_name;

SELECT * FROM anofox_statistics_vif([
    [1.0::DOUBLE, 5.0, 1.1],
    [2.0, 5.0, 2.1],
    [3.0, 5.0, 2.9],
    [4.0, 5.0, 4.2],
    [5.0, 5.0, 4.8]
]);

-- ======================================================================
-- Test 9: VIF with Perfect Multicollinearity
-- Expected: Infinity for collinear features
-- ======================================================================
SELECT '=== Test 9: VIF with Perfect Collinearity ===' as test_name;

SELECT * FROM anofox_statistics_vif([
    [1.0::DOUBLE, 2.0, 1.1],
    [2.0, 4.0, 2.1],
    [3.0, 6.0, 2.9],
    [4.0, 8.0, 4.2],
    [5.0, 10.0, 4.8]
]);

-- ======================================================================
-- Test 10: Residual Diagnostics (Simplified API)
-- Expected: Computes residuals and detects outliers
-- Note: API changed to (y_actual, y_predicted, outlier_threshold)
-- ======================================================================
SELECT '=== Test 10: Residual Diagnostics Simplified API ===' as test_name;

SELECT obs_id,
       ROUND(residual, 6) as residual,
       ROUND(std_residual, 4) as std_residual,
       is_outlier
FROM anofox_statistics_residual_diagnostics(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],  -- y_actual
    [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.8],  -- y_predicted
    2.5   -- Outlier threshold
);

-- ======================================================================
-- Summary
-- ======================================================================
SELECT '=== Test Suite Complete ===' as test_name;
