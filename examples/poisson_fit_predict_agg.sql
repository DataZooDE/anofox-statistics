-- ============================================================================
-- Poisson Fit Predict Aggregate Examples
-- ============================================================================
-- Demonstrates poisson_fit_predict_agg: fit Poisson GLM on training data,
-- predict all rows including future predictions (rows with NULL y).
-- Poisson regression is used for count data (non-negative integers).
--
-- Run: ./build/release/duckdb < examples/poisson_fit_predict_agg.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset with Training and Prediction Rows
-- ============================================================================

CREATE OR REPLACE TABLE visitor_forecast AS
SELECT
    store_id,
    week,
    marketing_spend,
    -- First 10 weeks have actual visitor counts (training), last 4 weeks are future (prediction)
    -- True model: visitors ~ exp(2 + 0.05 * marketing_spend)
    CASE WHEN week <= 10 THEN
        ROUND(EXP(2.0 + 0.05 * marketing_spend) + (RANDOM() * 10 - 5))::INTEGER
    ELSE NULL END AS visitors
FROM (VALUES (1), (2), (3)) AS s(store_id),
     generate_series(1, 14) AS w(week),
     LATERAL (SELECT 20.0 + week * 5.0 + store_id * 10.0 AS marketing_spend);

-- ============================================================================
-- Example 1: Basic Poisson Regression with Log Link
-- ============================================================================

SELECT '=== Example 1: Basic Poisson fit_predict_agg ===' AS section;

SELECT
    store_id,
    (pred).y AS actual_visitors,
    ROUND((pred).yhat) AS predicted_visitors,
    (pred).is_training
FROM (
    SELECT
        store_id,
        UNNEST(poisson_fit_predict_agg(
            visitors, [marketing_spend],
            {'link': 'log', 'intercept': true}
        )) AS pred
    FROM visitor_forecast
    GROUP BY store_id
) sub
WHERE store_id = 1
ORDER BY (pred).x[1];

-- ============================================================================
-- Example 2: Separate Training vs Prediction Results
-- ============================================================================

SELECT '=== Example 2: Training vs Prediction Split ===' AS section;

WITH predictions AS (
    SELECT
        store_id,
        (pred).y AS actual,
        (pred).yhat AS predicted,
        (pred).is_training AS is_training
    FROM (
        SELECT
            store_id,
            UNNEST(poisson_fit_predict_agg(visitors, [marketing_spend])) AS pred
        FROM visitor_forecast
        GROUP BY store_id
    ) sub
)
SELECT
    store_id,
    CASE WHEN is_training THEN 'Training' ELSE 'Prediction' END AS dataset,
    COUNT(*) AS rows,
    ROUND(AVG(predicted)) AS avg_predicted
FROM predictions
GROUP BY store_id, is_training
ORDER BY store_id, is_training DESC;

-- ============================================================================
-- Example 3: Prediction Intervals for Future Values
-- ============================================================================

SELECT '=== Example 3: Prediction Intervals ===' AS section;

SELECT
    store_id,
    ROUND((pred).yhat) AS predicted,
    ROUND((pred).yhat_lower) AS lower_95,
    ROUND((pred).yhat_upper) AS upper_95
FROM (
    SELECT
        store_id,
        UNNEST(poisson_fit_predict_agg(
            visitors, [marketing_spend],
            {'link': 'log', 'confidence_level': 0.95}
        )) AS pred
    FROM visitor_forecast
    GROUP BY store_id
) sub
WHERE NOT (pred).is_training
ORDER BY store_id;

-- ============================================================================
-- Example 4: With drop_y_zero_x Policy
-- ============================================================================

SELECT '=== Example 4: null_policy = drop_y_zero_x ===' AS section;

-- Create data with some zero x values
CREATE OR REPLACE TABLE zero_marketing AS
SELECT
    id,
    CASE WHEN id = 3 THEN 0.0 ELSE id * 10.0 END AS marketing,
    CASE WHEN id <= 10 THEN
        ROUND(EXP(2.0 + 0.02 * id * 10.0) + RANDOM() * 5)::INTEGER
    ELSE NULL END AS visitors
FROM generate_series(1, 14) AS t(id);

-- With 'drop' policy: includes all training rows
SELECT
    'drop' AS policy,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows
FROM (
    SELECT UNNEST(poisson_fit_predict_agg(visitors, [marketing], {'null_policy': 'drop'})) AS pred
    FROM zero_marketing
) sub;

-- With 'drop_y_zero_x' policy: excludes row with marketing=0 from training
SELECT
    'drop_y_zero_x' AS policy,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows
FROM (
    SELECT UNNEST(poisson_fit_predict_agg(visitors, [marketing], {'null_policy': 'drop_y_zero_x'})) AS pred
    FROM zero_marketing
) sub;

-- ============================================================================
-- Example 5: Different Link Functions
-- ============================================================================

SELECT '=== Example 5: Link Function Comparison ===' AS section;

-- Log link (default) - canonical for Poisson
SELECT
    'log' AS link,
    ROUND(AVG((pred).yhat)) AS avg_predicted
FROM (
    SELECT UNNEST(poisson_fit_predict_agg(visitors, [marketing_spend], {'link': 'log'})) AS pred
    FROM visitor_forecast WHERE store_id = 1
) sub
WHERE NOT (pred).is_training;

-- Identity link
SELECT
    'identity' AS link,
    ROUND(AVG((pred).yhat)) AS avg_predicted
FROM (
    SELECT UNNEST(poisson_fit_predict_agg(visitors, [marketing_spend], {'link': 'identity'})) AS pred
    FROM visitor_forecast WHERE store_id = 1
) sub
WHERE NOT (pred).is_training;

-- Sqrt link
SELECT
    'sqrt' AS link,
    ROUND(AVG((pred).yhat)) AS avg_predicted
FROM (
    SELECT UNNEST(poisson_fit_predict_agg(visitors, [marketing_spend], {'link': 'sqrt'})) AS pred
    FROM visitor_forecast WHERE store_id = 1
) sub
WHERE NOT (pred).is_training;

-- ============================================================================
-- Example 6: Multiple Groups with Predictions
-- ============================================================================

SELECT '=== Example 6: Multiple Groups ===' AS section;

SELECT
    store_id,
    ROUND(AVG((pred).yhat)) AS avg_predicted,
    ROUND(MIN((pred).yhat)) AS min_predicted,
    ROUND(MAX((pred).yhat)) AS max_predicted
FROM (
    SELECT
        store_id,
        UNNEST(poisson_fit_predict_agg(
            visitors,
            [marketing_spend],
            {'link': 'log', 'intercept': true}
        )) AS pred
    FROM visitor_forecast
    GROUP BY store_id
) sub
WHERE NOT (pred).is_training
GROUP BY store_id
ORDER BY store_id;

-- Cleanup
DROP TABLE IF EXISTS visitor_forecast;
DROP TABLE IF EXISTS zero_marketing;
