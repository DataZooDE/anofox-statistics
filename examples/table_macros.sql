-- ============================================================================
-- Table Macro Examples - The Preferred Way for Grouped Predictions
-- ============================================================================
-- Table macros provide user-friendly output for per-group model fitting and
-- prediction. They're simpler than the aggregate + UNNEST approach.
--
-- Available macros:
--   ols_fit_predict_by, ridge_fit_predict_by, wls_fit_predict_by,
--   elasticnet_fit_predict_by, lasso_fit_predict_by, bls_fit_predict_by,
--   pls_fit_predict_by, isotonic_fit_predict_by, quantile_fit_predict_by,
--   poisson_fit_predict_by, binomial_fit_predict_by
--
-- Run: ./build/release/duckdb < examples/table_macros.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Data
-- ============================================================================

CREATE OR REPLACE TABLE sales_data AS
SELECT
    region,
    month,
    100 + month * 5.0 +
    CASE region WHEN 'North' THEN 50 WHEN 'South' THEN 30 ELSE 10 END +
    (RANDOM() * 20 - 10) AS revenue,
    month * 10.0 + (RANDOM() * 5) AS marketing_spend,
    NULL AS is_training  -- Will be prediction rows
FROM (VALUES ('North'), ('South'), ('East')) AS r(region),
     generate_series(1, 12) AS t(month);

-- Mark some rows as training (NULL revenue = predict)
UPDATE sales_data SET revenue = NULL WHERE month > 10;

-- ============================================================================
-- Example 1: Basic OLS Grouped Prediction
-- ============================================================================
-- The simplest and most common use case

SELECT '=== Example 1: Basic OLS by Group ===' AS section;

SELECT * FROM ols_fit_predict_by(
    'sales_data',          -- source table
    'region',              -- group column
    'revenue',             -- y column
    ['marketing_spend']    -- x columns (list)
)
ORDER BY region, month;

-- ============================================================================
-- Example 2: With Model Options
-- ============================================================================

SELECT '=== Example 2: With Options ===' AS section;

SELECT * FROM ols_fit_predict_by(
    'sales_data',
    'region',
    'revenue',
    ['marketing_spend'],
    {'intercept': true, 'confidence_level': 0.95}  -- options map
)
ORDER BY region, month;

-- ============================================================================
-- Example 3: Filter Prediction Rows Only
-- ============================================================================
-- Useful when you only want to see predictions

SELECT '=== Example 3: Predictions Only ===' AS section;

SELECT *
FROM ols_fit_predict_by('sales_data', 'region', 'revenue', ['marketing_spend'])
WHERE NOT is_training
ORDER BY region, month;

-- ============================================================================
-- Example 4: Ridge Regression by Group
-- ============================================================================

SELECT '=== Example 4: Ridge by Group ===' AS section;

SELECT * FROM ridge_fit_predict_by(
    'sales_data',
    'region',
    'revenue',
    ['marketing_spend'],
    {'alpha': 0.1, 'intercept': true}
)
WHERE NOT is_training
ORDER BY region;

-- ============================================================================
-- Example 5: Multiple Features
-- ============================================================================

CREATE OR REPLACE TABLE multi_feature AS
SELECT
    product_id,
    month,
    100 + month * 5.0 + units * 2.0 + discount * -3.0 + (RANDOM() * 10) AS sales,
    month * 10 AS units,
    month * 0.5 AS discount
FROM (VALUES ('A'), ('B'), ('C')) AS p(product_id),
     generate_series(1, 15) AS t(month);

UPDATE multi_feature SET sales = NULL WHERE month > 12;

SELECT '=== Example 5: Multiple Features ===' AS section;

SELECT * FROM ols_fit_predict_by(
    'multi_feature',
    'product_id',
    'sales',
    ['units', 'discount']  -- multiple feature columns
)
ORDER BY product_id, month;

-- ============================================================================
-- Example 6: Comparison - Table Macro vs Aggregate Approach
-- ============================================================================
-- Table macro is simpler and produces better output

SELECT '=== Example 6: Comparison ===' AS section;

-- EASY WAY: Table macro (recommended)
SELECT '-- Table Macro (easy):' AS method;
SELECT region, month, yhat, is_training
FROM ols_fit_predict_by('sales_data', 'region', 'revenue', ['marketing_spend'])
WHERE NOT is_training
ORDER BY region, month;

-- VERBOSE WAY: Aggregate function + UNNEST
SELECT '-- Aggregate + UNNEST (verbose):' AS method;
WITH fit AS (
    SELECT
        region,
        UNNEST(ols_fit_predict_agg(revenue, [marketing_spend])) AS pred
    FROM sales_data
    GROUP BY region
)
SELECT
    region,
    (pred).yhat,
    (pred).is_training
FROM fit
WHERE NOT (pred).is_training
ORDER BY region;

-- ============================================================================
-- Example 7: Poisson Regression by Group (Count Data)
-- ============================================================================

CREATE OR REPLACE TABLE count_data AS
SELECT
    department,
    week,
    FLOOR(EXP(1.0 + 0.1 * week + RANDOM() * 0.5))::INTEGER AS incidents,
    week * 10 AS workload
FROM (VALUES ('IT'), ('HR'), ('Sales')) AS d(department),
     generate_series(1, 20) AS t(week);

UPDATE count_data SET incidents = NULL WHERE week > 15;

SELECT '=== Example 7: Poisson by Group ===' AS section;

SELECT * FROM poisson_fit_predict_by(
    'count_data',
    'department',
    'incidents',
    ['workload']
)
WHERE NOT is_training
ORDER BY department, week;

-- ============================================================================
-- Example 8: Quantile Regression by Group
-- ============================================================================

SELECT '=== Example 8: Quantile Regression ===' AS section;

SELECT * FROM quantile_fit_predict_by(
    'sales_data',
    'region',
    'revenue',
    ['marketing_spend'],
    {'quantile': 0.5}  -- Median regression
)
WHERE NOT is_training
ORDER BY region;

-- ============================================================================
-- Cleanup
-- ============================================================================

DROP TABLE IF EXISTS sales_data;
DROP TABLE IF EXISTS multi_feature;
DROP TABLE IF EXISTS count_data;
