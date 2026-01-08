-- =============================================================================
-- DuckDB Regression Interface Tests: Zero-Variance Features & NaN Handling
-- =============================================================================
-- Run with: duckdb -unsigned -c ".read test/sql/ols_zero_variance_bugs.sql"
--
-- EXPECTED BEHAVIOR:
--   - Constant/zero-variance features -> coefficient = NaN (intentional)
--   - Multicollinear features -> coefficient = NaN (intentional)
--   - NaN coefficients contribute 0 to predictions (not propagate NaN)
--   - Model should NOT be NULL even with many zero-variance features
-- =============================================================================

-- Load extension from build directory (or use installed if running from elsewhere)
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- =============================================================================
-- TEST 1: Basic OLS - Should work
-- =============================================================================
SELECT '=== TEST 1: Basic OLS ===' AS test;

WITH data AS (
    SELECT * FROM (VALUES
        (1.0, [1.0, 0.0]),
        (2.0, [0.0, 1.0]),
        (3.0, [1.0, 0.0]),
        (4.0, [0.0, 1.0]),
        (5.0, [1.0, 0.0]),
        (6.0, [0.0, 1.0])
    ) AS t(y, x)
)
SELECT
    'PASS' AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

-- =============================================================================
-- TEST 2: Constant Feature (zero variance) - Should return NaN for that coef
-- =============================================================================
SELECT '=== TEST 2: Constant Feature ===' AS test;

WITH data AS (
    SELECT * FROM (VALUES
        (1.0, [1.0, 5.0]),  -- second feature is constant (5.0)
        (2.0, [2.0, 5.0]),
        (3.0, [3.0, 5.0]),
        (4.0, [4.0, 5.0]),
        (5.0, [5.0, 5.0])
    ) AS t(y, x)
)
SELECT
    CASE
        WHEN model.coefficients IS NULL THEN 'FAIL - NULL coefficients'
        WHEN isnan(model.coefficients[2]) AND NOT isnan(model.coefficients[1]) THEN 'PASS - NaN for constant, valid for other'
        ELSE 'CHECK - unexpected pattern'
    END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

-- =============================================================================
-- TEST 3: All-Zero Feature - Should handle gracefully
-- =============================================================================
SELECT '=== TEST 3: All-Zero Feature ===' AS test;

WITH data AS (
    SELECT * FROM (VALUES
        (1.0, [1.0, 0.0]),  -- second feature is all zeros
        (2.0, [2.0, 0.0]),
        (3.0, [3.0, 0.0]),
        (4.0, [4.0, 0.0]),
        (5.0, [5.0, 0.0])
    ) AS t(y, x)
)
SELECT
    CASE WHEN model.coefficients IS NOT NULL THEN 'PASS' ELSE 'FAIL - NULL coefficients' END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

-- =============================================================================
-- TEST 4: Multicollinear Features (x2 = 2*x1) - NaN expected for one
-- =============================================================================
SELECT '=== TEST 4: Multicollinear Features ===' AS test;

WITH data AS (
    SELECT * FROM (VALUES
        (1.0, [1.0, 2.0]),   -- x2 = 2 * x1
        (2.0, [2.0, 4.0]),
        (3.0, [3.0, 6.0]),
        (4.0, [4.0, 8.0]),
        (5.0, [5.0, 10.0])
    ) AS t(y, x)
)
SELECT
    CASE
        WHEN model.coefficients IS NULL THEN 'FAIL - NULL coefficients'
        WHEN list_contains(model.coefficients, 'NaN'::DOUBLE) THEN 'OK - NaN for collinear (expected)'
        ELSE 'PASS'
    END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

-- =============================================================================
-- TEST 5: Multiple All-Zero Features - BUG: Should NOT return NULL
-- =============================================================================
SELECT '=== TEST 5: Multiple All-Zero Features ===' AS test;

-- Expected: coefficients = [10.0, NaN, NaN, NaN] (first valid, rest NaN)
-- BUG: Currently returns NULL

WITH data AS (
    SELECT * FROM (VALUES
        (10.0, [1.0, 0.0, 0.0, 0.0]),
        (20.0, [2.0, 0.0, 0.0, 0.0]),
        (30.0, [3.0, 0.0, 0.0, 0.0]),
        (40.0, [4.0, 0.0, 0.0, 0.0]),
        (50.0, [5.0, 0.0, 0.0, 0.0])
    ) AS t(y, x)
)
SELECT
    CASE
        WHEN model.coefficients IS NULL THEN 'BUG - NULL coefficients (should be [valid, NaN, NaN, NaN])'
        WHEN NOT isnan(model.coefficients[1]) THEN 'PASS - first coef valid'
        ELSE 'CHECK'
    END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

-- =============================================================================
-- TEST 6: ols_predict_agg - BUG: NaN coefs should contribute 0, not propagate
-- =============================================================================
SELECT '=== TEST 6: ols_predict_agg with NULL y ===' AS test;

-- BUG: Predictions are NaN because NaN propagates instead of contributing 0

WITH data AS (
    SELECT * FROM (VALUES
        (1.0,  [1.0, 0.0]),
        (2.0,  [0.0, 1.0]),
        (3.0,  [1.0, 0.0]),
        (4.0,  [0.0, 1.0]),
        (NULL, [1.0, 0.0]),
        (NULL, [0.0, 1.0])
    ) AS t(y, x)
),
predictions AS (
    SELECT
        ols_fit_predict_agg(y, x) OVER () AS pred_array
    FROM data
    LIMIT 1
),
unnested AS (
    SELECT unnest(pred_array) AS pred FROM predictions
)
SELECT
    CASE
        WHEN SUM(CASE WHEN pred.yhat IS NOT NULL AND NOT isnan(pred.yhat) THEN 1 ELSE 0 END) = 6
        THEN 'PASS - all 6 predictions valid'
        WHEN SUM(CASE WHEN pred.yhat IS NOT NULL AND NOT isnan(pred.yhat) THEN 1 ELSE 0 END) = 0
        THEN 'BUG - all predictions NaN (NaN coefs should contribute 0)'
        ELSE 'PARTIAL'
    END AS status,
    SUM(CASE WHEN pred.is_training THEN 1 ELSE 0 END) AS train_count,
    SUM(CASE WHEN NOT pred.is_training THEN 1 ELSE 0 END) AS test_count,
    SUM(CASE WHEN pred.yhat IS NOT NULL AND NOT isnan(pred.yhat) THEN 1 ELSE 0 END) AS valid_predictions
FROM unnested;

-- =============================================================================
-- TEST 7: Constant feature appears only in test - NaN coef should contribute 0
-- =============================================================================
SELECT '=== TEST 7: Constant feature in training, non-zero in test ===' AS test;

-- Feature 2 is always 0 in train -> coefficient = NaN
-- Feature 2 is 1 in test -> NaN * 1 should contribute 0, not NaN

WITH data AS (
    SELECT * FROM (VALUES
        (1.0,  [1.0, 0.0]),
        (2.0,  [2.0, 0.0]),
        (3.0,  [3.0, 0.0]),
        (4.0,  [4.0, 0.0]),
        (NULL, [5.0, 1.0]),
        (NULL, [6.0, 1.0])
    ) AS t(y, x)
),
predictions AS (
    SELECT
        ols_fit_predict_agg(y, x) OVER () AS pred_array
    FROM data
    LIMIT 1
),
unnested AS (
    SELECT unnest(pred_array) AS pred FROM predictions
)
SELECT
    CASE
        WHEN SUM(CASE WHEN NOT pred.is_training AND pred.yhat IS NOT NULL AND NOT isnan(pred.yhat) THEN 1 ELSE 0 END) = 2
        THEN 'PASS - test predictions valid (NaN coef contributed 0)'
        ELSE 'BUG - test predictions NaN (NaN coef should contribute 0)'
    END AS status
FROM unnested;

-- =============================================================================
-- TEST 8: GROUP BY aggregation - BUG: Should not return NULL
-- =============================================================================
SELECT '=== TEST 8: GROUP BY aggregation ===' AS test;

-- BUG: GROUP BY returns NULL for all groups

WITH data AS (
    SELECT * FROM (VALUES
        ('A', 1.0, [1.0, 0.0]),
        ('A', 2.0, [0.0, 1.0]),
        ('A', 3.0, [1.0, 0.0]),
        ('B', 10.0, [1.0, 0.0]),
        ('B', 20.0, [0.0, 1.0]),
        ('B', 30.0, [1.0, 0.0])
    ) AS t(group_id, y, x)
)
SELECT
    group_id,
    CASE
        WHEN model.coefficients IS NULL THEN 'BUG - NULL coefficients'
        ELSE 'PASS'
    END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT group_id, ols_fit_agg(y, x) AS model
    FROM data
    GROUP BY group_id
)
ORDER BY group_id;

-- =============================================================================
-- TEST 9: Many zero-variance features - BUG: Should NOT return NULL
-- =============================================================================
SELECT '=== TEST 9: Many zero-variance features ===' AS test;

-- Simulates changepoint scenario: 11 features, 9 are all zeros
-- Expected: coefficients = [valid, valid, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
-- BUG: Currently returns NULL

WITH data AS (
    SELECT * FROM (VALUES
        (1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (2.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (3.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (4.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (5.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (6.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ) AS t(y, x)
)
SELECT
    CASE
        WHEN model.coefficients IS NULL THEN 'BUG - NULL coefficients (should have valid + NaN mix)'
        WHEN NOT isnan(model.coefficients[1])
        THEN 'PASS - first coef valid (x[0] and x[1] are multicollinear with intercept, so one is NaN)'
        ELSE 'CHECK'
    END AS status,
    model.intercept,
    model.coefficients
FROM (
    SELECT ols_fit_agg(y, x) AS model FROM data
);

SELECT '=== ALL TESTS COMPLETE ===' AS summary;
