-- ============================================================================
-- OLS Single Series Examples
-- ============================================================================
-- Demonstrates basic OLS regression on a single dataset (no grouping).
-- Topics: Basic fit, inference, prediction, model diagnostics
--
-- IMPORTANT: The x parameter uses column-major format where each inner array
-- contains all values for ONE feature: [[feature1_vals], [feature2_vals], ...]
--
-- Run: ./build/release/duckdb < examples/ols_single_series.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Example 1: Basic OLS Fit with Array Inputs
-- ============================================================================
-- Simple regression: y = slope * x + intercept

SELECT '=== Example 1: Basic OLS Fit ===' AS section;

-- y = [10, 15, 20, 25, 30], x = [1, 2, 3, 4, 5] -> slope=5, intercept=5
SELECT
    result.coefficients,
    result.intercept,
    ROUND(result.r_squared, 4) AS r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0]::DOUBLE[],  -- y values
        [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][],   -- x values (one feature)
        {'intercept': true}
    ) AS result
);

-- ============================================================================
-- Example 2: Multiple Regression (3 predictors)
-- ============================================================================

SELECT '=== Example 2: Multiple Regression ===' AS section;

-- y = 5 + 2*x1 + 3*x2 - 1*x3 (approximately)
-- 8 observations, 3 features
SELECT
    result.coefficients,
    result.intercept,
    ROUND(result.r_squared, 4) AS r_squared,
    result.n_observations,
    result.n_features
FROM (
    SELECT anofox_stats_ols_fit(
        [15.0, 22.0, 31.0, 38.0, 45.0, 54.0, 61.0, 70.0]::DOUBLE[],  -- y
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],      -- x1 values
            [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0],    -- x2 values
            [3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0]       -- x3 values
        ]::DOUBLE[][],
        {'intercept': true}
    ) AS result
);

-- ============================================================================
-- Example 3: Full Inference Output (t-stats, p-values, CI)
-- ============================================================================

SELECT '=== Example 3: Full Inference Output ===' AS section;

SELECT
    result.coefficients,
    result.intercept,
    result.coefficient_std_errors,
    result.intercept_std_error,
    result.coefficient_t_values,
    result.intercept_t_value,
    result.coefficient_p_values,
    result.intercept_p_value,
    result.f_statistic,
    result.f_statistic_pvalue
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
);

-- ============================================================================
-- Example 4: Model Diagnostics (R², adj R², residual std error)
-- ============================================================================

SELECT '=== Example 4: Model Diagnostics ===' AS section;

SELECT
    ROUND(result.r_squared, 4) AS r_squared,
    ROUND(result.adj_r_squared, 4) AS adjusted_r_squared,
    ROUND(result.residual_std_error, 4) AS residual_std_error,
    result.n_observations AS observations,
    result.n_features AS features
FROM (
    SELECT anofox_stats_ols_fit(
        [12.5, 17.2, 21.8, 26.1, 31.5, 35.9, 41.2, 45.8]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]::DOUBLE[][],
        {'intercept': true}
    ) AS result
);

-- ============================================================================
-- Example 5: Prediction Using Fitted Coefficients
-- ============================================================================
-- After fitting, manually apply coefficients to predict new values

SELECT '=== Example 5: Prediction Using Fitted Coefficients ===' AS section;

WITH fitted AS (
    SELECT anofox_stats_ols_fit(
        [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][],
        {'intercept': true}
    ) AS result
),
new_data AS (
    SELECT x FROM (VALUES (6.0), (7.0), (8.0), (9.0), (10.0)) AS t(x)
)
SELECT
    nd.x,
    ROUND(f.result.intercept + f.result.coefficients[1] * nd.x, 2) AS predicted_y
FROM new_data nd, fitted f;

-- ============================================================================
-- Example 6: Regression Without Intercept
-- ============================================================================
-- Force line through origin (y = b*x, no intercept term)

SELECT '=== Example 6: Regression Without Intercept ===' AS section;

SELECT
    'With intercept' AS model,
    ROUND(result.intercept, 4) AS intercept,
    ROUND(result.coefficients[1], 4) AS slope,
    ROUND(result.r_squared, 4) AS r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        [5.0, 10.0, 15.0, 20.0, 25.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][],
        {'intercept': true}
    ) AS result
)
UNION ALL
SELECT
    'Without intercept' AS model,
    ROUND(result.intercept, 4) AS intercept,
    ROUND(result.coefficients[1], 4) AS slope,
    ROUND(result.r_squared, 4) AS r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        [5.0, 10.0, 15.0, 20.0, 25.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][],
        {'intercept': false}
    ) AS result
);

-- ============================================================================
-- Example 7: Fit from Table Data
-- ============================================================================
-- Aggregate data from a table into arrays, then fit

SELECT '=== Example 7: Fit from Table Data ===' AS section;

CREATE OR REPLACE TABLE sample_data AS
SELECT
    id,
    2.5 * id + 10.0 + (RANDOM() * 4.0 - 2.0) AS y,
    id::DOUBLE AS x
FROM range(1, 21) t(id);

-- Collect into column-major arrays then fit
WITH arrays AS (
    SELECT
        LIST(y ORDER BY id) AS y_arr,
        [LIST(x ORDER BY id)] AS x_arr  -- Note: wrap in outer array for column-major
    FROM sample_data
)
SELECT
    ROUND(result.intercept, 2) AS intercept,
    ROUND(result.coefficients[1], 2) AS slope,
    ROUND(result.r_squared, 4) AS r_squared,
    result.n_observations AS n_obs
FROM arrays, LATERAL (SELECT anofox_stats_ols_fit(y_arr, x_arr, {'intercept': true}) AS result);

-- ============================================================================
-- Example 8: Comparing Different Confidence Levels
-- ============================================================================

SELECT '=== Example 8: Different Confidence Levels ===' AS section;

SELECT
    '90%' AS confidence_level,
    ROUND(result.coefficient_ci_lower[1], 4) AS ci_lower,
    ROUND(result.coefficient_ci_upper[1], 4) AS ci_upper,
    ROUND(result.coefficient_ci_upper[1] - result.coefficient_ci_lower[1], 4) AS ci_width
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.90}
    ) AS result
)
UNION ALL
SELECT
    '95%' AS confidence_level,
    ROUND(result.coefficient_ci_lower[1], 4) AS ci_lower,
    ROUND(result.coefficient_ci_upper[1], 4) AS ci_upper,
    ROUND(result.coefficient_ci_upper[1] - result.coefficient_ci_lower[1], 4) AS ci_width
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
)
UNION ALL
SELECT
    '99%' AS confidence_level,
    ROUND(result.coefficient_ci_lower[1], 4) AS ci_lower,
    ROUND(result.coefficient_ci_upper[1], 4) AS ci_upper,
    ROUND(result.coefficient_ci_upper[1] - result.coefficient_ci_lower[1], 4) AS ci_width
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.99}
    ) AS result
);

-- Cleanup
DROP TABLE IF EXISTS sample_data;
