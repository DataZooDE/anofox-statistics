-- ============================================================================
-- OLS Fit-Predict Examples with Artificial Datasets
-- ============================================================================
-- Shows the three ways to get per-row OLS predictions:
--   * ols_fit_predict_agg(y, x[, options])      -- fit once per group, predict all rows
--   * ols_fit_predict_by('table', g, y, [x...])   -- the same, as a table macro
--   * ols_fit_predict(y, x[, options]) OVER (...) -- expanding / rolling in-sample fits
-- Rows with NULL y are never used for fitting but always receive a prediction.
--
-- Run: ./build/release/duckdb < examples/example-fit-predict-ols.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Example 1: Train / Test Split via NULL y
-- ============================================================================
-- y = 2*x + 1 (+ small deterministic noise); rows 11-15 have y = NULL.

CREATE OR REPLACE TABLE simple_linear AS
SELECT
    id,
    id::DOUBLE AS x,
    CASE WHEN id <= 10 THEN 2.0 * id + 1.0 + (((id * 7) % 5) - 2.0) * 0.1 END AS y
FROM range(1, 16) t(id);

SELECT '=== Example 1: Fit on rows with y, predict all rows ===' AS section;

SELECT
    (p).y                     AS actual,
    ROUND((p).yhat, 3)        AS predicted,
    ROUND((p).yhat_lower, 3)  AS lower_95,
    ROUND((p).yhat_upper, 3)  AS upper_95,
    (p).is_training           AS is_training
FROM (SELECT unnest(ols_fit_predict_agg(y, [x])) AS p FROM simple_linear);

-- ============================================================================
-- Example 2: The Same with the Table Macro (keeps every source column)
-- ============================================================================

SELECT '=== Example 2: ols_fit_predict_by ===' AS section;

CREATE OR REPLACE TABLE simple_linear_grouped AS
SELECT 'all' AS grp, * FROM simple_linear;

SELECT id, x, y, ROUND(yhat, 3) AS yhat, is_training
FROM ols_fit_predict_by('simple_linear_grouped', grp, y, [x])
ORDER BY id;

-- ============================================================================
-- Example 3: Explicit Split Column
-- ============================================================================
-- The split variant trains only on rows whose split column is 'train'.

SELECT '=== Example 3: Split column ===' AS section;

CREATE OR REPLACE TABLE split_data AS
SELECT
    id,
    id::DOUBLE AS x,
    3.0 * id - 4.0 + (((id * 11) % 7) - 3.0) * 0.2 AS y,
    CASE WHEN id <= 12 THEN 'train' ELSE 'test' END AS split
FROM range(1, 21) t(id);

SELECT
    (p).is_training,
    COUNT(*)                                      AS n_rows,
    ROUND(AVG(ABS((p).y - (p).yhat)), 4)          AS mean_abs_error
FROM (SELECT unnest(ols_fit_predict_agg(y, [x], split)) AS p FROM split_data)
GROUP BY ALL
ORDER BY 1 DESC;

-- ============================================================================
-- Example 4: Multiple Linear Regression
-- ============================================================================
-- y = 2*x1 + 3*x2 + 10

SELECT '=== Example 4: Multiple regression ===' AS section;

CREATE OR REPLACE TABLE multi AS
SELECT
    id,
    id::DOUBLE              AS x1,
    ((id * 7) % 10)::DOUBLE AS x2,
    CASE WHEN id <= 15 THEN 2.0 * id + 3.0 * ((id * 7) % 10) + 10.0 END AS y
FROM range(1, 21) t(id);

SELECT
    ROUND(f.intercept, 3)   AS intercept,
    f.coefficients          AS coefficients,
    ROUND(f.r_squared, 4)   AS r_squared
FROM (SELECT ols_fit_agg(y, [x1, x2]) AS f FROM multi);

SELECT (p).y AS actual, ROUND((p).yhat, 2) AS predicted
FROM (SELECT unnest(ols_fit_predict_agg(y, [x1, x2])) AS p FROM multi)
WHERE NOT (p).is_training;

-- ============================================================================
-- Example 5: Expanding vs Rolling Windows
-- ============================================================================
-- The window function predicts the LAST ROW OF THE FRAME, so frames end at
-- CURRENT ROW. The slope changes at t = 20; the rolling fit adapts faster.

SELECT '=== Example 5: Expanding vs rolling ===' AS section;

CREATE OR REPLACE TABLE regime AS
SELECT
    t,
    t::DOUBLE AS x,
    CASE WHEN t <= 20 THEN 1.0 * t ELSE 20.0 + 3.0 * (t - 20) END AS y
FROM range(1, 31) r(t);

SELECT
    t,
    y,
    ROUND((ols_fit_predict(y, [x]) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 2) AS expanding,
    ROUND((ols_fit_predict(y, [x]) OVER (ORDER BY t ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)).yhat, 2)         AS rolling_5
FROM regime
ORDER BY t
OFFSET 17 LIMIT 8;

-- ============================================================================
-- Example 6: Per-Group Models
-- ============================================================================

SELECT '=== Example 6: Per-group models ===' AS section;

CREATE OR REPLACE TABLE grouped AS
SELECT
    g,
    id::DOUBLE AS x,
    CASE g WHEN 'A' THEN 2.0 * id + 1.0 WHEN 'B' THEN -1.0 * id + 50.0 END
        + (((id * 3) % 5) - 2.0) * 0.1 AS y
FROM (VALUES ('A'), ('B')) v(g), range(1, 11) t(id);

SELECT g, ROUND(f.coefficients[1], 3) AS slope, ROUND(f.intercept, 3) AS intercept
FROM (SELECT g, ols_fit_agg(y, [x]) AS f FROM grouped GROUP BY g)
ORDER BY g;

-- ============================================================================
-- Example 7: Without Intercept and Different Confidence Levels
-- ============================================================================

SELECT '=== Example 7: Options ===' AS section;

SELECT
    'with intercept'    AS model, ROUND((ols_fit_agg(y, [x])).coefficients[1], 4) AS slope
FROM simple_linear
UNION ALL
SELECT
    'without intercept', ROUND((ols_fit_agg(y, [x], {'fit_intercept': false})).coefficients[1], 4)
FROM simple_linear;

SELECT
    level,
    ROUND((p).yhat_upper - (p).yhat_lower, 3) AS interval_width_at_x15
FROM (
    SELECT '90%' AS level, unnest(ols_fit_predict_agg(y, [x], {'confidence_level': 0.90})) AS p FROM simple_linear
    UNION ALL
    SELECT '99%', unnest(ols_fit_predict_agg(y, [x], {'confidence_level': 0.99})) FROM simple_linear
)
WHERE (p).y IS NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY level ORDER BY (p).yhat DESC) = 1
ORDER BY level;

-- ============================================================================
-- Example 8: Coefficients with Inference
-- ============================================================================

SELECT '=== Example 8: Inference ===' AS section;

SELECT
    ROUND(f.coefficients[1], 4) AS slope,
    ROUND(f.std_errors[1], 4)   AS std_error,
    ROUND(f.t_values[1], 2)     AS t_value,
    f.p_values[1]               AS p_value
FROM (SELECT ols_fit_agg(y, [x], {'compute_inference': true}) AS f FROM simple_linear);

-- Cleanup
DROP TABLE IF EXISTS simple_linear;
DROP TABLE IF EXISTS simple_linear_grouped;
DROP TABLE IF EXISTS split_data;
DROP TABLE IF EXISTS multi;
DROP TABLE IF EXISTS regime;
DROP TABLE IF EXISTS grouped;
