-- ============================================================================
-- BLS Fit Predict Aggregate Examples
-- ============================================================================
-- Demonstrates bls_fit_predict_agg: fit Bounded Least Squares on training data,
-- predict all rows including future predictions (rows with NULL y).
--
-- Run: ./build/release/duckdb < examples/bls_fit_predict_agg.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset with Training and Prediction Rows
-- ============================================================================

CREATE OR REPLACE TABLE bounded_forecast AS
SELECT
    group_id,
    week,
    x,
    -- First 10 weeks have actual y (training), last 4 weeks are future (prediction)
    CASE WHEN week <= 10 THEN
        5.0 + 2.0 * x + (RANDOM() * 2 - 1)
    ELSE NULL END AS y
FROM (VALUES (1), (2)) AS g(group_id),
     generate_series(1, 14) AS w(week),
     LATERAL (SELECT week * 1.5 AS x);

-- ============================================================================
-- Example 1: Basic NNLS (Non-Negative Least Squares)
-- ============================================================================
-- Fit with coefficients >= 0, predict all rows

SELECT '=== Example 1: NNLS fit_predict_agg ===' AS section;

SELECT
    group_id,
    (pred).y AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training
FROM (
    SELECT
        group_id,
        UNNEST(bls_fit_predict_agg(
            y, [x],
            {'lower_bound': 0}
        )) AS pred
    FROM bounded_forecast
    GROUP BY group_id
) sub
WHERE group_id = 1
ORDER BY (pred).x[1];

-- ============================================================================
-- Example 2: Box-Constrained Regression
-- ============================================================================
-- Coefficients bounded between 0 and 5

SELECT '=== Example 2: Box-Constrained BLS ===' AS section;

SELECT
    group_id,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training
FROM (
    SELECT
        group_id,
        UNNEST(bls_fit_predict_agg(
            y, [x],
            {'lower_bound': 0, 'upper_bound': 5, 'intercept': true}
        )) AS pred
    FROM bounded_forecast
    GROUP BY group_id
) sub
WHERE NOT (pred).is_training
ORDER BY group_id;

-- ============================================================================
-- Example 3: With drop_y_zero_x Policy
-- ============================================================================

SELECT '=== Example 3: null_policy = drop_y_zero_x ===' AS section;

-- Create data with some zero x values
CREATE OR REPLACE TABLE zero_features AS
SELECT
    id,
    CASE WHEN id = 3 THEN 0.0 ELSE id * 1.0 END AS x,
    CASE WHEN id <= 10 THEN
        5.0 + 2.0 * id + (RANDOM() * 1)
    ELSE NULL END AS y
FROM generate_series(1, 14) AS t(id);

-- With drop_y_zero_x: excludes row 3 (x=0) from training
SELECT
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT UNNEST(bls_fit_predict_agg(y, [x], {'lower_bound': 0, 'null_policy': 'drop_y_zero_x'})) AS pred
    FROM zero_features
) sub;

-- ============================================================================
-- Example 4: Prediction Intervals
-- ============================================================================

SELECT '=== Example 4: Prediction Intervals ===' AS section;

SELECT
    group_id,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND((pred).yhat_lower, 2) AS lower_95,
    ROUND((pred).yhat_upper, 2) AS upper_95
FROM (
    SELECT
        group_id,
        UNNEST(bls_fit_predict_agg(
            y, [x],
            {'lower_bound': 0, 'confidence_level': 0.95}
        )) AS pred
    FROM bounded_forecast
    GROUP BY group_id
) sub
WHERE NOT (pred).is_training
ORDER BY group_id;

-- Cleanup
DROP TABLE IF EXISTS bounded_forecast;
DROP TABLE IF EXISTS zero_features;
