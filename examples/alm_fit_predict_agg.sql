-- ============================================================================
-- ALM Fit Predict Aggregate Examples
-- ============================================================================
-- Demonstrates alm_fit_predict_agg: fit Augmented Linear Model on training data,
-- predict all rows including future predictions (rows with NULL y).
-- ALM supports 24 error distributions for robust regression.
--
-- Run: ./build/release/duckdb < examples/alm_fit_predict_agg.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset with Training and Prediction Rows
-- ============================================================================

CREATE OR REPLACE TABLE robust_forecast AS
SELECT
    group_id,
    week,
    x,
    -- First 10 weeks have actual y (training), last 4 weeks are future (prediction)
    -- Add some outliers to demonstrate robust regression
    CASE WHEN week <= 10 THEN
        CASE WHEN week = 5 THEN 10.0 + 3.0 * x + 50.0  -- Outlier
             ELSE 10.0 + 3.0 * x + (RANDOM() * 4 - 2)
        END
    ELSE NULL END AS y
FROM (VALUES (1), (2)) AS g(group_id),
     generate_series(1, 14) AS w(week),
     LATERAL (SELECT week * 2.0 AS x);

-- ============================================================================
-- Example 1: Normal Distribution (Standard)
-- ============================================================================

SELECT '=== Example 1: Normal Distribution ===' AS section;

SELECT
    group_id,
    (pred).y AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training
FROM (
    SELECT
        group_id,
        UNNEST(alm_fit_predict_agg(
            y, [x],
            {'distribution': 'normal', 'intercept': true}
        )) AS pred
    FROM robust_forecast
    GROUP BY group_id
) sub
WHERE group_id = 1
ORDER BY (pred).x[1];

-- ============================================================================
-- Example 2: Laplace Distribution (Robust to Outliers)
-- ============================================================================

SELECT '=== Example 2: Laplace Distribution (Robust) ===' AS section;

SELECT
    group_id,
    (pred).y AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training
FROM (
    SELECT
        group_id,
        UNNEST(alm_fit_predict_agg(
            y, [x],
            {'distribution': 'laplace', 'intercept': true}
        )) AS pred
    FROM robust_forecast
    GROUP BY group_id
) sub
WHERE group_id = 1
ORDER BY (pred).x[1];

-- ============================================================================
-- Example 3: Student-t Distribution
-- ============================================================================

SELECT '=== Example 3: Student-t Distribution ===' AS section;

SELECT
    group_id,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training
FROM (
    SELECT
        group_id,
        UNNEST(alm_fit_predict_agg(
            y, [x],
            {'distribution': 'studentt', 'intercept': true}
        )) AS pred
    FROM robust_forecast
    GROUP BY group_id
) sub
WHERE NOT (pred).is_training
ORDER BY group_id;

-- ============================================================================
-- Example 4: With drop_y_zero_x Policy
-- ============================================================================

SELECT '=== Example 4: null_policy = drop_y_zero_x ===' AS section;

-- Create data with some zero x values
CREATE OR REPLACE TABLE zero_features AS
SELECT
    id,
    CASE WHEN id = 3 THEN 0.0 ELSE id * 1.0 END AS x,
    CASE WHEN id <= 10 THEN
        10.0 + 3.0 * id + (RANDOM() * 2)
    ELSE NULL END AS y
FROM generate_series(1, 14) AS t(id);

SELECT
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT UNNEST(alm_fit_predict_agg(y, [x], {'distribution': 'laplace', 'null_policy': 'drop_y_zero_x'})) AS pred
    FROM zero_features
) sub;

-- ============================================================================
-- Example 5: Prediction Intervals
-- ============================================================================

SELECT '=== Example 5: Prediction Intervals ===' AS section;

SELECT
    group_id,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND((pred).yhat_lower, 2) AS lower_95,
    ROUND((pred).yhat_upper, 2) AS upper_95
FROM (
    SELECT
        group_id,
        UNNEST(alm_fit_predict_agg(
            y, [x],
            {'distribution': 'laplace', 'confidence_level': 0.95}
        )) AS pred
    FROM robust_forecast
    GROUP BY group_id
) sub
WHERE NOT (pred).is_training
ORDER BY group_id;

-- ============================================================================
-- Example 6: Compare Normal vs Laplace on Outlier Data
-- ============================================================================

SELECT '=== Example 6: Normal vs Laplace Comparison ===' AS section;

WITH normal_pred AS (
    SELECT
        group_id,
        (pred).yhat AS yhat_normal,
        (pred).is_training
    FROM (
        SELECT group_id, UNNEST(alm_fit_predict_agg(y, [x], {'distribution': 'normal'})) AS pred
        FROM robust_forecast GROUP BY group_id
    ) sub
),
laplace_pred AS (
    SELECT
        group_id,
        (pred).yhat AS yhat_laplace,
        (pred).is_training
    FROM (
        SELECT group_id, UNNEST(alm_fit_predict_agg(y, [x], {'distribution': 'laplace'})) AS pred
        FROM robust_forecast GROUP BY group_id
    ) sub
)
SELECT
    'Normal' AS distribution,
    ROUND(AVG(yhat_normal), 2) AS avg_prediction
FROM normal_pred WHERE NOT is_training AND group_id = 1
UNION ALL
SELECT
    'Laplace' AS distribution,
    ROUND(AVG(yhat_laplace), 2) AS avg_prediction
FROM laplace_pred WHERE NOT is_training AND group_id = 1;

-- Cleanup
DROP TABLE IF EXISTS robust_forecast;
DROP TABLE IF EXISTS zero_features;
