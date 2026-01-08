-- Extended Zero-Variance Bug Tests
-- Tests additional edge cases for constant feature handling across regression methods
-- Run with: duckdb -unsigned -c ".read test/sql/zero_variance_extended_tests.sql"

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- ============================================================================
-- TEST 1: All Features Constant (Intercept-Only Model)
-- When ALL features are constant, should return intercept = mean(y), coefficients = NaN
-- ============================================================================
SELECT '=== TEST 1: All Features Constant (Intercept-Only) ===' AS test_name;
CREATE OR REPLACE TABLE test1_all_constant AS
SELECT * FROM (VALUES
    (10.0, [5.0, 5.0, 5.0, 5.0]),
    (20.0, [5.0, 5.0, 5.0, 5.0]),
    (30.0, [5.0, 5.0, 5.0, 5.0]),
    (40.0, [5.0, 5.0, 5.0, 5.0]),
    (50.0, [5.0, 5.0, 5.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.intercept IS NOT NULL
            AND ABS(result.intercept - 30.0) < 0.01  -- mean(y) = (10+20+30+40+50)/5 = 30
            AND result.coefficients IS NOT NULL
            AND len(result.coefficients) = 4
            AND isnan(result.coefficients[1])
            AND isnan(result.coefficients[2])
            AND isnan(result.coefficients[3])
            AND isnan(result.coefficients[4])
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.intercept,
    result.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS result
    FROM test1_all_constant
);

-- ============================================================================
-- TEST 2: Inference Statistics NaN for Constant Columns
-- Verify std_errors, t_values, p_values are NaN where coefficients are NaN
-- ============================================================================
SELECT '=== TEST 2: Inference Statistics NaN for Constant Columns ===' AS test_name;
CREATE OR REPLACE TABLE test2_inference AS
SELECT * FROM (VALUES
    (10.0, [1.0, 5.0]),
    (20.0, [2.0, 5.0]),
    (30.0, [3.0, 5.0]),
    (40.0, [4.0, 5.0]),
    (50.0, [5.0, 5.0]),
    (60.0, [6.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.coefficients IS NOT NULL
            AND NOT isnan(result.coefficients[1])  -- First coefficient valid
            AND isnan(result.coefficients[2])       -- Second coefficient NaN (constant)
            AND result.std_errors IS NOT NULL
            AND NOT isnan(result.std_errors[1])     -- First std_error valid
            AND isnan(result.std_errors[2])         -- Second std_error NaN
            AND result.t_values IS NOT NULL
            AND NOT isnan(result.t_values[1])       -- First t_value valid
            AND isnan(result.t_values[2])           -- Second t_value NaN
            AND result.p_values IS NOT NULL
            AND NOT isnan(result.p_values[1])       -- First p_value valid
            AND isnan(result.p_values[2])           -- Second p_value NaN
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.coefficients,
    result.std_errors,
    result.t_values,
    result.p_values
FROM (
    SELECT ols_fit_agg(y, x, {'compute_inference': true}) AS result
    FROM test2_inference
);

-- ============================================================================
-- TEST 3: Boundary Condition (n_valid == min_obs)
-- 3 obs, 2 non-constant features, fit_intercept=true -> min_obs = 3 -> should PASS
-- ============================================================================
SELECT '=== TEST 3: Boundary Condition (n_valid == min_obs) ===' AS test_name;
CREATE OR REPLACE TABLE test3_boundary AS
SELECT * FROM (VALUES
    (10.0, [1.0, 2.0]),
    (20.0, [2.0, 4.0]),
    (30.0, [3.0, 6.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.coefficients IS NOT NULL
            AND result.n_observations = 3
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.n_observations,
    result.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS result
    FROM test3_boundary
);

-- ============================================================================
-- TEST 4: Mixed GROUP BY Patterns
-- Group A: all constant -> intercept-only
-- Group B: one constant -> one valid coefficient
-- Group C: none constant -> both valid coefficients
-- ============================================================================
SELECT '=== TEST 4: Mixed GROUP BY Patterns ===' AS test_name;
CREATE OR REPLACE TABLE test4_groups AS
SELECT * FROM (VALUES
    -- Group A: all features constant
    ('A', 10.0, [5.0, 5.0]),
    ('A', 20.0, [5.0, 5.0]),
    ('A', 30.0, [5.0, 5.0]),
    ('A', 40.0, [5.0, 5.0]),
    -- Group B: second feature constant
    ('B', 10.0, [1.0, 5.0]),
    ('B', 20.0, [2.0, 5.0]),
    ('B', 30.0, [3.0, 5.0]),
    ('B', 40.0, [4.0, 5.0]),
    -- Group C: no constant features (NOT collinear)
    ('C', 10.0, [1.0, 1.0]),
    ('C', 20.0, [2.0, 3.0]),
    ('C', 30.0, [3.0, 2.0]),
    ('C', 40.0, [4.0, 5.0])
) AS t(grp, y, x);

SELECT
    grp,
    CASE
        WHEN grp = 'A' AND result IS NOT NULL
            AND isnan(result.coefficients[1])
            AND isnan(result.coefficients[2])
            AND ABS(result.intercept - 25.0) < 0.01  -- mean(10,20,30,40) = 25
        THEN 'PASS'
        WHEN grp = 'B' AND result IS NOT NULL
            AND NOT isnan(result.coefficients[1])
            AND isnan(result.coefficients[2])
        THEN 'PASS'
        WHEN grp = 'C' AND result IS NOT NULL
            AND NOT isnan(result.coefficients[1])
            AND NOT isnan(result.coefficients[2])
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.intercept,
    result.coefficients
FROM (
    SELECT grp, ols_fit_agg(y, x) AS result
    FROM test4_groups
    GROUP BY grp
)
ORDER BY grp;

-- ============================================================================
-- TEST 5: Fit-Predict with All Constant Training Features
-- Training: all features constant -> intercept-only model
-- Test: all predictions should equal intercept (mean of training y)
-- NULL y values indicate test rows
-- ============================================================================
SELECT '=== TEST 5: Fit-Predict with All Constant Training ===' AS test_name;
CREATE OR REPLACE TABLE test5_fit_predict AS
SELECT * FROM (VALUES
    -- Training rows: all features constant
    (10.0, [5.0, 5.0]),
    (20.0, [5.0, 5.0]),
    (30.0, [5.0, 5.0]),
    (40.0, [5.0, 5.0]),
    -- Test rows (y = NULL): different feature values
    (NULL, [1.0, 2.0]),
    (NULL, [10.0, 20.0]),
    (NULL, [100.0, 200.0])
) AS t(y, x);

WITH predictions AS (
    SELECT ols_fit_predict_agg(y, x) OVER () AS pred_array
    FROM test5_fit_predict
    LIMIT 1
),
unnested AS (
    SELECT unnest(pred_array) AS pred FROM predictions
)
SELECT
    CASE
        WHEN COUNT(*) = 3
            AND MIN(pred.yhat) IS NOT NULL
            AND ABS(MIN(pred.yhat) - 25.0) < 0.01  -- All test predictions = intercept = 25
            AND ABS(MAX(pred.yhat) - 25.0) < 0.01
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    MIN(pred.yhat) AS min_pred,
    MAX(pred.yhat) AS max_pred
FROM unnested
WHERE NOT pred.is_training;

-- ============================================================================
-- TEST 6: Single Valid Feature Among Many Constants
-- 5 features: [constant, constant, VALID, constant, constant]
-- coefficients = [NaN, NaN, valid, NaN, NaN]
-- ============================================================================
SELECT '=== TEST 6: Single Valid Feature Among Many Constants ===' AS test_name;
CREATE OR REPLACE TABLE test6_single_valid AS
SELECT * FROM (VALUES
    (10.0, [5.0, 5.0, 1.0, 5.0, 5.0]),
    (20.0, [5.0, 5.0, 2.0, 5.0, 5.0]),
    (30.0, [5.0, 5.0, 3.0, 5.0, 5.0]),
    (40.0, [5.0, 5.0, 4.0, 5.0, 5.0]),
    (50.0, [5.0, 5.0, 5.0, 5.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.coefficients IS NOT NULL
            AND len(result.coefficients) = 5
            AND isnan(result.coefficients[1])      -- Constant -> NaN
            AND isnan(result.coefficients[2])      -- Constant -> NaN
            AND NOT isnan(result.coefficients[3])  -- Valid coefficient
            AND isnan(result.coefficients[4])      -- Constant -> NaN
            AND isnan(result.coefficients[5])      -- Constant -> NaN
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS result
    FROM test6_single_valid
);

-- ============================================================================
-- TEST 7: Prediction Value Accuracy
-- y = 10 * x1 + 0 * x2 (x2 constant)
-- Verify predictions are accurate
-- ============================================================================
SELECT '=== TEST 7: Prediction Value Accuracy ===' AS test_name;
CREATE OR REPLACE TABLE test7_accuracy AS
SELECT * FROM (VALUES
    (10.0, [1.0, 5.0]),
    (20.0, [2.0, 5.0]),
    (30.0, [3.0, 5.0]),
    (40.0, [4.0, 5.0]),
    (50.0, [5.0, 5.0]),
    -- Test rows
    (NULL, [6.0, 5.0]),
    (NULL, [7.0, 5.0]),
    (NULL, [10.0, 5.0])
) AS t(y, x);

WITH predictions AS (
    SELECT ols_fit_predict_agg(y, x) OVER () AS pred_array
    FROM test7_accuracy
    LIMIT 1
),
unnested AS (
    SELECT unnest(pred_array) AS pred FROM predictions
)
SELECT
    CASE
        WHEN COUNT(*) = 3
            -- Test predictions should be close to expected: 60, 70, 100
            AND ABS(MIN(pred.yhat) - 60.0) < 1.0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    MIN(pred.yhat) AS min_pred,
    MAX(pred.yhat) AS max_pred
FROM unnested
WHERE NOT pred.is_training;

-- ============================================================================
-- TEST 8: Near-Zero Variance Features
-- Feature with tiny variance below threshold (1e-10) -> treated as constant
-- ============================================================================
SELECT '=== TEST 8: Near-Zero Variance Features ===' AS test_name;
CREATE OR REPLACE TABLE test8_near_zero AS
SELECT * FROM (VALUES
    -- x1 varies normally, x2 has tiny variance (all same except tiny diff)
    (10.0, [1.0, 5.0]),
    (20.0, [2.0, 5.0 + 1e-12]),  -- Difference < 1e-10 threshold
    (30.0, [3.0, 5.0]),
    (40.0, [4.0, 5.0 + 1e-12]),
    (50.0, [5.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.coefficients IS NOT NULL
            AND NOT isnan(result.coefficients[1])  -- First has variance -> valid
            AND isnan(result.coefficients[2])      -- Second near-zero variance -> NaN
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS result
    FROM test8_near_zero
);

-- ============================================================================
-- TEST 9: WLS with Invalid Weights
-- Mix: positive (1.0), zero (0.0), negative (-1.0) weights
-- Only positive weights should be used
-- ============================================================================
SELECT '=== TEST 9: WLS with Invalid Weights ===' AS test_name;
CREATE OR REPLACE TABLE test9_wls_weights AS
SELECT * FROM (VALUES
    (10.0, [1.0], 1.0),   -- Valid weight
    (20.0, [2.0], 2.0),   -- Valid weight
    (30.0, [3.0], 0.0),   -- Zero weight - should be excluded
    (40.0, [4.0], -1.0),  -- Negative weight - should be excluded
    (50.0, [5.0], 1.5),   -- Valid weight
    (60.0, [6.0], 0.5)    -- Valid weight
) AS t(y, x, w);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.n_observations = 4  -- Only 4 valid (positive) weights
            AND result.coefficients IS NOT NULL
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.n_observations,
    result.coefficients
FROM (
    SELECT wls_fit_agg(y, x, w) AS result
    FROM test9_wls_weights
);

-- ============================================================================
-- TEST 10: Stress Test Many Features
-- 20 features: 3 non-constant (indices 0, 10, 19), 17 constant
-- ============================================================================
SELECT '=== TEST 10: Stress Test Many Features (20 features, 17 constant) ===' AS test_name;
-- 20 features: 3 non-constant (indices 0, 10, 19) with DIFFERENT values to avoid multicollinearity
CREATE OR REPLACE TABLE test10_stress AS
SELECT * FROM (VALUES
    (100.0, [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0]),
    (110.0, [2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 110.0]),
    (120.0, [3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 15.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 130.0]),
    (130.0, [4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 25.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 105.0]),
    (140.0, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 30.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 120.0]),
    (150.0, [6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 35.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 140.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.coefficients IS NOT NULL
            AND len(result.coefficients) = 20
            -- Feature 1 (index 0): non-constant -> valid
            AND NOT isnan(result.coefficients[1])
            -- Features 2-10 (indices 1-9): constant -> NaN
            AND isnan(result.coefficients[2])
            AND isnan(result.coefficients[3])
            AND isnan(result.coefficients[4])
            AND isnan(result.coefficients[5])
            AND isnan(result.coefficients[6])
            AND isnan(result.coefficients[7])
            AND isnan(result.coefficients[8])
            AND isnan(result.coefficients[9])
            AND isnan(result.coefficients[10])
            -- Feature 11 (index 10): non-constant -> valid
            AND NOT isnan(result.coefficients[11])
            -- Features 12-19 (indices 11-18): constant -> NaN
            AND isnan(result.coefficients[12])
            AND isnan(result.coefficients[13])
            AND isnan(result.coefficients[14])
            AND isnan(result.coefficients[15])
            AND isnan(result.coefficients[16])
            AND isnan(result.coefficients[17])
            AND isnan(result.coefficients[18])
            AND isnan(result.coefficients[19])
            -- Feature 20 (index 19): non-constant -> valid
            AND NOT isnan(result.coefficients[20])
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS result
    FROM test10_stress
);

-- ============================================================================
-- TEST 11: Ridge with All Constant Features
-- Verify Ridge also handles intercept-only model
-- ============================================================================
SELECT '=== TEST 11: Ridge with All Constant Features ===' AS test_name;
CREATE OR REPLACE TABLE test11_ridge_all_constant AS
SELECT * FROM (VALUES
    (10.0, [5.0, 5.0]),
    (20.0, [5.0, 5.0]),
    (30.0, [5.0, 5.0]),
    (40.0, [5.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.intercept IS NOT NULL
            AND ABS(result.intercept - 25.0) < 0.01  -- mean(10,20,30,40) = 25
            AND isnan(result.coefficients[1])
            AND isnan(result.coefficients[2])
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.intercept,
    result.coefficients
FROM (
    SELECT ridge_fit_agg(y, x, {'alpha': 0.1}) AS result
    FROM test11_ridge_all_constant
);

-- ============================================================================
-- TEST 12: Poisson with Constant Features
-- Poisson uses log link: intercept = log(mean(y))
-- ============================================================================
SELECT '=== TEST 12: Poisson with All Constant Features ===' AS test_name;
CREATE OR REPLACE TABLE test12_poisson_constant AS
SELECT * FROM (VALUES
    (10.0, [5.0, 5.0]),
    (20.0, [5.0, 5.0]),
    (30.0, [5.0, 5.0]),
    (40.0, [5.0, 5.0])
) AS t(y, x);

SELECT
    CASE
        WHEN result IS NOT NULL
            AND result.intercept IS NOT NULL
            -- For Poisson: intercept should be log(mean(y)) = log(25) ≈ 3.22
            AND ABS(result.intercept - ln(25.0)) < 0.01
            AND isnan(result.coefficients[1])
            AND isnan(result.coefficients[2])
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    result.intercept,
    ln(25.0) AS expected_intercept,
    result.coefficients
FROM (
    SELECT poisson_fit_agg(y, x) AS result
    FROM test12_poisson_constant
);

-- ============================================================================
-- CLEANUP
-- ============================================================================
DROP TABLE IF EXISTS test1_all_constant;
DROP TABLE IF EXISTS test2_inference;
DROP TABLE IF EXISTS test3_boundary;
DROP TABLE IF EXISTS test4_groups;
DROP TABLE IF EXISTS test5_fit_predict;
DROP TABLE IF EXISTS test6_single_valid;
DROP TABLE IF EXISTS test7_accuracy;
DROP TABLE IF EXISTS test8_near_zero;
DROP TABLE IF EXISTS test9_wls_weights;
DROP TABLE IF EXISTS test10_stress;
DROP TABLE IF EXISTS test11_ridge_all_constant;
DROP TABLE IF EXISTS test12_poisson_constant;

SELECT '============================================' AS separator;
SELECT '=== Extended Zero-Variance Tests Complete ===' AS summary;
SELECT '============================================' AS separator;
