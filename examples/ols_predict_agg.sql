-- ============================================================================
-- OLS Predict Aggregate Examples
-- ============================================================================
-- Demonstrates the predict_agg function: fit once per group, predict all rows.
-- Use case: Train on historical data (y not null), predict future (y null)
--
-- Run: ./build/release/duckdb < examples/ols_predict_agg.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset with Training and Prediction Rows
-- ============================================================================

CREATE OR REPLACE TABLE sales_forecast AS
SELECT
    store_id,
    week,
    marketing_spend,
    -- First 10 weeks have actual sales (training), last 4 weeks are future (prediction)
    CASE WHEN week <= 10 THEN
        100.0 + 2.5 * marketing_spend + store_id * 20.0 + (RANDOM() * 20 - 10)
    ELSE NULL END AS sales
FROM (VALUES (1), (2), (3)) AS s(store_id),
     generate_series(1, 14) AS w(week);

-- ============================================================================
-- Example 1: Basic predict_agg Usage
-- ============================================================================
-- Fit model on training rows (y not null), predict ALL rows including future

SELECT '=== Example 1: Basic predict_agg ===' AS section;

SELECT
    store_id,
    (pred).yhat AS predicted_sales,
    (pred).is_training AS is_training_row
FROM (
    SELECT
        store_id,
        UNNEST(ols_predict_agg(sales, [marketing_spend])) AS pred
    FROM sales_forecast
    GROUP BY store_id
) sub
WHERE store_id = 1
LIMIT 14;

-- ============================================================================
-- Example 2: Separate Training vs Prediction Results
-- ============================================================================

SELECT '=== Example 2: Training vs Prediction Split ===' AS section;

WITH predictions AS (
    SELECT
        store_id,
        (pred).y AS actual,
        (pred).x AS features,
        (pred).yhat AS predicted,
        (pred).yhat_lower AS ci_lower,
        (pred).yhat_upper AS ci_upper,
        (pred).is_training AS is_training
    FROM (
        SELECT
            store_id,
            UNNEST(ols_predict_agg(
                sales,
                [marketing_spend],
                {'intercept': true, 'confidence_level': 0.95}
            )) AS pred
        FROM sales_forecast
        GROUP BY store_id
    ) sub
)
SELECT
    store_id,
    CASE WHEN is_training THEN 'Training' ELSE 'Prediction' END AS dataset,
    COUNT(*) AS rows,
    ROUND(AVG(predicted), 2) AS avg_predicted
FROM predictions
GROUP BY store_id, is_training
ORDER BY store_id, is_training DESC;

-- ============================================================================
-- Example 3: Prediction Intervals for Future Values
-- ============================================================================

SELECT '=== Example 3: Prediction Intervals ===' AS section;

SELECT
    store_id,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND((pred).yhat_lower, 2) AS lower_95,
    ROUND((pred).yhat_upper, 2) AS upper_95,
    ROUND((pred).yhat_upper - (pred).yhat_lower, 2) AS interval_width
FROM (
    SELECT
        store_id,
        UNNEST(ols_predict_agg(
            sales,
            [marketing_spend],
            {'intercept': true, 'confidence_level': 0.95}
        )) AS pred
    FROM sales_forecast
    GROUP BY store_id
) sub
WHERE NOT (pred).is_training  -- Only future predictions
ORDER BY store_id;

-- ============================================================================
-- Example 4: Count Training vs Prediction Rows
-- ============================================================================

SELECT '=== Example 4: Row Counts ===' AS section;

SELECT
    store_id,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT
        store_id,
        UNNEST(ols_predict_agg(sales, [marketing_spend])) AS pred
    FROM sales_forecast
    GROUP BY store_id
) sub
GROUP BY store_id
ORDER BY store_id;

-- ============================================================================
-- Example 5: null_policy = 'drop' (Default)
-- ============================================================================
-- Rows with NULL y are excluded from training but included in predictions

SELECT '=== Example 5: null_policy = drop ===' AS section;

-- Create data with some missing training values
CREATE OR REPLACE TABLE sparse_training AS
SELECT
    store_id,
    week,
    marketing_spend,
    -- Make some training rows have NULL y
    CASE
        WHEN week <= 10 AND week % 3 != 0 THEN
            100.0 + 2.5 * marketing_spend + (RANDOM() * 10 - 5)
        ELSE NULL
    END AS sales
FROM (VALUES (1)) AS s(store_id),
     generate_series(1, 14) AS w(week);

SELECT
    (pred).y AS actual,
    (pred).x[1] AS marketing,
    ROUND((pred).yhat, 2) AS predicted,
    (pred).is_training AS is_training
FROM (
    SELECT
        UNNEST(ols_predict_agg(
            sales,
            [marketing_spend],
            {'intercept': true, 'null_policy': 'drop'}
        )) AS pred
    FROM sparse_training
) sub;

-- ============================================================================
-- Example 6: null_policy = 'drop_y_zero_x'
-- ============================================================================
-- Exclude rows where y is NULL OR any x feature is zero

SELECT '=== Example 6: null_policy = drop_y_zero_x ===' AS section;

-- Create data with some zero x values
CREATE OR REPLACE TABLE zero_features AS
SELECT
    id,
    x1,
    x2,
    -- y = 5 + 2*x1 + 3*x2
    CASE WHEN id <= 8 THEN
        5.0 + 2.0 * x1 + 3.0 * x2 + (RANDOM() * 2 - 1)
    ELSE NULL END AS y
FROM (
    SELECT
        id,
        CASE WHEN id = 3 THEN 0.0 ELSE id * 1.0 END AS x1,  -- Zero at id=3
        CASE WHEN id = 5 THEN 0.0 ELSE id * 0.5 END AS x2   -- Zero at id=5
    FROM generate_series(1, 12) AS t(id)
);

-- With 'drop' policy: includes rows with zero x values in training
SELECT
    'drop' AS policy,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows
FROM (
    SELECT UNNEST(ols_predict_agg(y, [x1, x2], {'null_policy': 'drop'})) AS pred
    FROM zero_features
) sub;

-- With 'drop_y_zero_x' policy: excludes rows with zero x values from training
SELECT
    'drop_y_zero_x' AS policy,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows
FROM (
    SELECT UNNEST(ols_predict_agg(y, [x1, x2], {'null_policy': 'drop_y_zero_x'})) AS pred
    FROM zero_features
) sub;

-- ============================================================================
-- Example 7: Multiple Groups with Predictions
-- ============================================================================

SELECT '=== Example 7: Multiple Groups ===' AS section;

SELECT
    store_id,
    ROUND(AVG((pred).yhat), 2) AS avg_predicted,
    ROUND(MIN((pred).yhat), 2) AS min_predicted,
    ROUND(MAX((pred).yhat), 2) AS max_predicted
FROM (
    SELECT
        store_id,
        UNNEST(ols_predict_agg(
            sales,
            [marketing_spend],
            {'intercept': true}
        )) AS pred
    FROM sales_forecast
    GROUP BY store_id
) sub
WHERE NOT (pred).is_training
GROUP BY store_id
ORDER BY store_id;

-- ============================================================================
-- Example 8: Combine with Original Data
-- ============================================================================

SELECT '=== Example 8: Join Predictions with Original Data ===' AS section;

WITH preds AS (
    SELECT
        store_id,
        ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY (pred).x[1]) AS rn,
        (pred).yhat AS predicted,
        (pred).is_training
    FROM (
        SELECT
            store_id,
            UNNEST(ols_predict_agg(sales, [marketing_spend])) AS pred
        FROM sales_forecast
        GROUP BY store_id
    ) sub
),
original AS (
    SELECT
        store_id,
        week,
        marketing_spend,
        sales,
        ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY marketing_spend) AS rn
    FROM sales_forecast
)
SELECT
    o.store_id,
    o.week,
    o.marketing_spend,
    o.sales AS actual,
    ROUND(p.predicted, 2) AS predicted,
    CASE WHEN p.is_training THEN 'Train' ELSE 'Predict' END AS type
FROM original o
JOIN preds p ON o.store_id = p.store_id AND o.rn = p.rn
WHERE o.store_id = 1
ORDER BY o.week;

-- ============================================================================
-- Example 9: Evaluate Model on Training Data
-- ============================================================================

SELECT '=== Example 9: Training Set Evaluation ===' AS section;

WITH results AS (
    SELECT
        store_id,
        (pred).y AS actual,
        (pred).yhat AS predicted,
        (pred).is_training
    FROM (
        SELECT
            store_id,
            UNNEST(ols_predict_agg(sales, [marketing_spend])) AS pred
        FROM sales_forecast
        GROUP BY store_id
    ) sub
    WHERE (pred).is_training
)
SELECT
    store_id,
    COUNT(*) AS n,
    ROUND(AVG(ABS(actual - predicted)), 4) AS mae,
    ROUND(SQRT(AVG(POW(actual - predicted, 2))), 4) AS rmse
FROM results
GROUP BY store_id
ORDER BY store_id;

-- ============================================================================
-- Example 10: Full Workflow - Train, Predict, Evaluate
-- ============================================================================

SELECT '=== Example 10: Full Workflow ===' AS section;

-- Create a more realistic dataset
CREATE OR REPLACE TABLE full_workflow AS
SELECT
    product_id,
    month,
    price,
    advertising,
    -- Sales depends on price (negative) and advertising (positive)
    -- First 8 months training, last 4 months prediction
    CASE WHEN month <= 8 THEN
        500.0 - 10.0 * price + 5.0 * advertising + product_id * 50.0 + (RANDOM() * 30 - 15)
    ELSE NULL END AS sales
FROM (VALUES (1), (2)) AS p(product_id),
     generate_series(1, 12) AS m(month),
     LATERAL (
         SELECT
             20.0 + month * 2.0 + (RANDOM() * 5) AS price,
             10.0 + month * 1.0 + (RANDOM() * 3) AS advertising
     ) features;

-- Fit and predict
SELECT
    product_id,
    'Overall' AS metric_type,
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT
        product_id,
        UNNEST(ols_predict_agg(
            sales,
            [price, advertising],
            {'intercept': true, 'confidence_level': 0.95}
        )) AS pred
    FROM full_workflow
    GROUP BY product_id
) sub
GROUP BY product_id
ORDER BY product_id;

-- Cleanup
DROP TABLE IF EXISTS sales_forecast;
DROP TABLE IF EXISTS sparse_training;
DROP TABLE IF EXISTS zero_features;
DROP TABLE IF EXISTS full_workflow;
