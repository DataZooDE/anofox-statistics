-- =============================================================================
-- All Regression Methods: Zero-Variance Features & NaN Handling Tests
-- =============================================================================
-- Run with: duckdb -unsigned -c ".read test/sql/all_regression_zero_variance_bugs.sql"
--
-- Tests the same zero-variance scenarios across all regression methods:
-- - OLS, Ridge, ElasticNet, WLS, RLS, BLS, ALM, Poisson
-- =============================================================================

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- =============================================================================
-- TEST DATA: Multiple zero-variance features
-- =============================================================================
-- 5 observations, 4 features where 3 are constant (zero)
-- Expected: coefficients = [valid, NaN, NaN, NaN]

CREATE OR REPLACE TABLE test_data AS
SELECT * FROM (VALUES
    (10.0, [1.0, 0.0, 0.0, 0.0]),
    (20.0, [2.0, 0.0, 0.0, 0.0]),
    (30.0, [3.0, 0.0, 0.0, 0.0]),
    (40.0, [4.0, 0.0, 0.0, 0.0]),
    (50.0, [5.0, 0.0, 0.0, 0.0])
) AS t(y, x);

-- =============================================================================
-- TEST 1: OLS (already fixed - baseline)
-- =============================================================================
SELECT '=== TEST 1: OLS ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT ols_fit_agg(y, x) AS model FROM test_data);

-- =============================================================================
-- TEST 2: Ridge Regression
-- =============================================================================
SELECT '=== TEST 2: Ridge ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT ridge_fit_agg(y, x, {'alpha': 0.1}) AS model FROM test_data);

-- =============================================================================
-- TEST 3: ElasticNet Regression
-- =============================================================================
SELECT '=== TEST 3: ElasticNet ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT elasticnet_fit_agg(y, x, {'alpha': 0.1, 'l1_ratio': 0.5}) AS model FROM test_data);

-- =============================================================================
-- TEST 4: WLS (Weighted Least Squares)
-- =============================================================================
SELECT '=== TEST 4: WLS ===' AS test;
WITH wls_data AS (
    SELECT y, x, 1.0 AS weight FROM test_data
)
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT wls_fit_agg(y, x, weight) AS model FROM wls_data);

-- =============================================================================
-- TEST 5: RLS (Recursive Least Squares)
-- =============================================================================
SELECT '=== TEST 5: RLS ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT rls_fit_agg(y, x) AS model FROM test_data);

-- =============================================================================
-- TEST 6: BLS (Bounded Least Squares)
-- =============================================================================
SELECT '=== TEST 6: BLS ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT bls_fit_agg(y, x) AS model FROM test_data);

-- =============================================================================
-- TEST 7: ALM (Adaptive Linear Model)
-- =============================================================================
SELECT '=== TEST 7: ALM ===' AS test;
SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT alm_fit_agg(y, x) AS model FROM test_data);

-- =============================================================================
-- TEST 8: Poisson Regression (using count-like data)
-- =============================================================================
SELECT '=== TEST 8: Poisson ===' AS test;
CREATE OR REPLACE TABLE poisson_data AS
SELECT * FROM (VALUES
    (10, [1.0, 0.0, 0.0, 0.0]),
    (20, [2.0, 0.0, 0.0, 0.0]),
    (30, [3.0, 0.0, 0.0, 0.0]),
    (40, [4.0, 0.0, 0.0, 0.0]),
    (50, [5.0, 0.0, 0.0, 0.0])
) AS t(y, x);

SELECT
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (SELECT poisson_fit_agg(y::DOUBLE, x) AS model FROM poisson_data);

-- =============================================================================
-- TEST 9: GROUP BY with Ridge
-- =============================================================================
SELECT '=== TEST 9: Ridge with GROUP BY ===' AS test;
CREATE OR REPLACE TABLE group_data AS
SELECT * FROM (VALUES
    ('A', 1.0, [1.0, 0.0]),
    ('A', 2.0, [0.0, 1.0]),
    ('A', 3.0, [1.0, 0.0]),
    ('B', 10.0, [1.0, 0.0]),
    ('B', 20.0, [0.0, 1.0]),
    ('B', 30.0, [1.0, 0.0])
) AS t(group_id, y, x);

SELECT
    group_id,
    CASE WHEN model.coefficients IS NULL THEN 'FAIL' ELSE 'PASS' END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT group_id, ridge_fit_agg(y, x, {'alpha': 0.1}) AS model
    FROM group_data
    GROUP BY group_id
)
ORDER BY group_id;

-- =============================================================================
-- TEST 10: Fit-Predict with Ridge (NaN prediction handling)
-- =============================================================================
SELECT '=== TEST 10: Ridge Fit-Predict ===' AS test;
CREATE OR REPLACE TABLE predict_data AS
SELECT * FROM (VALUES
    (1.0,  [1.0, 0.0]),
    (2.0,  [2.0, 0.0]),
    (3.0,  [3.0, 0.0]),
    (4.0,  [4.0, 0.0]),
    (NULL, [5.0, 1.0]),  -- test row with non-zero second feature
    (NULL, [6.0, 1.0])
) AS t(y, x);

WITH predictions AS (
    SELECT ridge_fit_predict_agg(y, x, {'alpha': 0.1}) OVER () AS pred_array
    FROM predict_data
    LIMIT 1
),
unnested AS (
    SELECT unnest(pred_array) AS pred FROM predictions
)
SELECT
    CASE
        WHEN SUM(CASE WHEN pred.yhat IS NOT NULL AND NOT isnan(pred.yhat) THEN 1 ELSE 0 END) = 6
        THEN 'PASS - all predictions valid'
        ELSE 'FAIL - some predictions NaN'
    END AS status,
    SUM(CASE WHEN pred.is_training THEN 1 ELSE 0 END) AS train_count,
    SUM(CASE WHEN NOT pred.is_training THEN 1 ELSE 0 END) AS test_count
FROM unnested;

-- =============================================================================
-- CLEANUP
-- =============================================================================
DROP TABLE IF EXISTS test_data;
DROP TABLE IF EXISTS poisson_data;
DROP TABLE IF EXISTS group_data;
DROP TABLE IF EXISTS predict_data;

SELECT '=== ALL TESTS COMPLETE ===' AS summary;
