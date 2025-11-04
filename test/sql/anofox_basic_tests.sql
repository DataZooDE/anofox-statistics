-- Comprehensive test suite for AnofoxStatistics DuckDB extension
-- Phase 1: Basic Functionality Tests

-- Test 1: OLS Fit - Basic functionality
SELECT 'Test 1: OLS Fit Basic' as test_name;
CREATE TABLE test_ols_data AS
    SELECT
        1 as x1, 2 as x2, 5 as y
    UNION ALL SELECT 2, 4, 10
    UNION ALL SELECT 3, 6, 15
    UNION ALL SELECT 4, 8, 20
    UNION ALL SELECT 5, 10, 25;

-- Basic OLS fit
SELECT
    coefficients,
    intercept,
    r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM anofox_ols_fit(test_ols_data, ['x1', 'x2'], 'y')
LIMIT 1;

-- Test 2: OLS with intercept disabled
SELECT 'Test 2: OLS Fit without intercept' as test_name;
SELECT
    coefficients,
    intercept,
    r_squared
FROM anofox_ols_fit(test_ols_data, ['x1', 'x2'], 'y', false)
LIMIT 1;

-- Test 3: Ridge Regression
SELECT 'Test 3: Ridge Regression' as test_name;
SELECT
    coefficients,
    intercept,
    r_squared,
    lambda
FROM anofox_ridge_fit(test_ols_data, ['x1', 'x2'], 'y', 1.0)
LIMIT 1;

-- Test 4: OLS Predict - Scalar function
SELECT 'Test 4: OLS Predict Scalar' as test_name;
CREATE TABLE test_ols_model AS
SELECT * FROM anofox_ols_fit(test_ols_data, ['x1', 'x2'], 'y') LIMIT 1;

SELECT
    x1, x2,
    anofox_ols_predict(
        ARRAY[0.0, 5.0],  -- coefficients from fitted model
        0.0,               -- intercept
        [x1, x2]           -- features
    ) as predicted
FROM test_ols_data;

-- Test 5: OLS Metrics
SELECT 'Test 5: OLS Metrics' as test_name;
CREATE TABLE predictions AS
SELECT
    x1, x2, y,
    anofox_ols_predict(ARRAY[5.0, 0.0], 0.0, [x1, x2]) as y_pred
FROM test_ols_data;

SELECT
    anofox_ols_r2(y, y_pred) as r2,
    anofox_ols_mse(y, y_pred) as mse,
    anofox_ols_rmse(y, y_pred) as rmse,
    anofox_ols_mae(y, y_pred) as mae
FROM predictions;

-- Test 6: Grouped OLS Fit
SELECT 'Test 6: Grouped OLS Fit' as test_name;
CREATE TABLE grouped_data AS
    SELECT 'A' as group_id, 1 as x1, 2 as x2, 5 as y
    UNION ALL SELECT 'A', 2, 4, 10
    UNION ALL SELECT 'A', 3, 6, 15
    UNION ALL SELECT 'B', 1, 2, 7
    UNION ALL SELECT 'B', 2, 4, 12
    UNION ALL SELECT 'B', 3, 6, 17;

SELECT
    group_id as group_key,
    coefficients,
    r_squared,
    n_obs
FROM anofox_grouped_ols_fit(
    grouped_data,
    ['group_id'],
    ['x1', 'x2'],
    'y'
)
ORDER BY group_id;

-- Test 7: Grouped Ridge
SELECT 'Test 7: Grouped Ridge Fit' as test_name;
SELECT
    group_id as group_key,
    coefficients,
    r_squared,
    lambda,
    n_obs
FROM anofox_grouped_ridge_fit(
    grouped_data,
    ['group_id'],
    ['x1', 'x2'],
    'y',
    0.5
)
ORDER BY group_id;

-- Test 8: Grouped Metrics
SELECT 'Test 8: Grouped Metrics' as test_name;
CREATE TABLE grouped_predictions AS
SELECT
    group_id, x1, x2, y,
    anofox_ols_predict(ARRAY[5.0, 0.0], 0.0, [x1, x2]) as y_pred
FROM grouped_data;

SELECT
    group_id as group_key,
    r_squared,
    mse,
    mae,
    n_obs
FROM anofox_grouped_metrics(
    grouped_predictions,
    ['group_id'],
    'y',
    'y_pred'
)
ORDER BY group_id;

-- Test 9: Rolling OLS
SELECT 'Test 9: Rolling OLS Fit' as test_name;
CREATE TABLE time_series_data AS
    SELECT 1 as t, 1 as x1, 2 as x2, 5 as y
    UNION ALL SELECT 2, 2, 4, 10
    UNION ALL SELECT 3, 3, 6, 15
    UNION ALL SELECT 4, 4, 8, 20
    UNION ALL SELECT 5, 5, 10, 25
    UNION ALL SELECT 6, 6, 12, 30;

SELECT
    window_id,
    coefficients,
    r_squared,
    window_start_idx,
    window_end_idx,
    n_obs
FROM anofox_rolling_ols_fit(
    time_series_data,
    ['x1', 'x2'],
    'y',
    3
)
ORDER BY window_id;

-- Test 10: Expanding OLS
SELECT 'Test 10: Expanding OLS Fit' as test_name;
SELECT
    window_id,
    coefficients,
    r_squared,
    window_start_idx,
    window_end_idx,
    n_obs
FROM anofox_expanding_ols_fit(
    time_series_data,
    ['x1', 'x2'],
    'y',
    2
)
ORDER BY window_id;

-- Test 11: RLS (Recursive Least Squares)
SELECT 'Test 11: RLS Fit' as test_name;
SELECT
    step,
    coefficients,
    prediction_error,
    mse_estimate,
    gain_norm,
    n_obs
FROM anofox_rls_fit(
    time_series_data,
    ['x1', 'x2'],
    'y',
    0.99,
    100.0
)
LIMIT 5;

-- Cleanup
DROP TABLE test_ols_data;
DROP TABLE test_ols_model;
DROP TABLE predictions;
DROP TABLE grouped_data;
DROP TABLE grouped_predictions;
DROP TABLE time_series_data;

SELECT 'All basic tests completed successfully!' as result;
