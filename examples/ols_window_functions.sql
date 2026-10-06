-- ============================================================================
-- OLS Window Function Examples
-- ============================================================================
-- *_fit_predict(y, x[, options]) OVER (...) is a window aggregate: it fits on
-- the rows of the window frame (rows with NULL y are not trained on) and
-- returns STRUCT(yhat, yhat_lower, yhat_upper) for the LAST ROW OF THE FRAME.
--
-- Use frames that end at CURRENT ROW over a unique ORDER BY:
--   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW   (expanding)
--   ROWS BETWEEN k PRECEDING AND CURRENT ROW           (rolling)
-- Do NOT use:
--   ... AND 1 PRECEDING   -- predicts the previous row, not a one-step-ahead forecast
--   OVER (PARTITION BY g) without ORDER BY -- every row gets the same prediction
--   RANGE frames with ties
-- For one prediction per row of a whole group use ols_fit_predict_agg or
-- ols_fit_predict_by (see ols_predict_agg.sql).
--
-- Window functions exist for ols, ridge, elasticnet, wls (extra weight
-- argument), rls, huber, ransac and theil_sen.
--
-- Run: ./build/release/duckdb < examples/ols_window_functions.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Sample time series (deterministic pseudo-noise for reproducibility)
-- ============================================================================

CREATE OR REPLACE TABLE stock_prices AS
SELECT
    ticker,
    day,
    CASE ticker
        WHEN 'TECH'   THEN 100.0 + day * 1.5 + (((day * 37) % 11) - 5.0)
        WHEN 'BANK'   THEN 50.0 + day * 0.5 + (((day * 17) % 7) - 3.0) * 0.8
        WHEN 'RETAIL' THEN 30.0 + day * 0.3 + SIN(day * 0.2) * 5
    END AS price,
    (CASE ticker WHEN 'TECH' THEN 1000.0 WHEN 'BANK' THEN 500.0 ELSE 200.0 END
        + day * 10.0 + ((day * 13) % 9) * 5.0) AS volume_k
FROM (VALUES ('TECH'), ('BANK'), ('RETAIL')) AS t(ticker),
     generate_series(1, 30) AS d(day);

-- ============================================================================
-- Example 1: Expanding window (in-sample fit on all rows up to the current one)
-- ============================================================================

SELECT '=== Example 1: Expanding Window ===' AS section;

SELECT
    day,
    ROUND(price, 2)                AS price,
    ROUND((pred).yhat, 2)          AS fitted,
    ROUND(price - (pred).yhat, 2)  AS residual
FROM (
    SELECT
        day,
        price,
        ols_fit_predict(price, [day::DOUBLE]) OVER (
            ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS pred
    FROM stock_prices
    WHERE ticker = 'TECH'
)
ORDER BY day
LIMIT 10;

-- ============================================================================
-- Example 2: Rolling window (last 10 rows including the current one)
-- ============================================================================

SELECT '=== Example 2: Rolling Window ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS price,
    ROUND((ols_fit_predict(price, [day::DOUBLE]) OVER (
        ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 2) AS fitted_expanding,
    ROUND((ols_fit_predict(price, [day::DOUBLE]) OVER (
        ORDER BY day ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)).yhat, 2)         AS fitted_rolling_10
FROM stock_prices
WHERE ticker = 'RETAIL'
ORDER BY day
LIMIT 15;

-- ============================================================================
-- Example 3: Partitioned windows (one rolling model per ticker)
-- ============================================================================

SELECT '=== Example 3: Partitioned Windows ===' AS section;

-- Window functions are evaluated after WHERE, so filter in an outer query.
SELECT * FROM (
    SELECT
        ticker,
        day,
        ROUND(price, 2) AS price,
        ROUND((ols_fit_predict(price, [day::DOUBLE, volume_k]) OVER (
            PARTITION BY ticker ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 2) AS fitted
    FROM stock_prices
)
WHERE day >= 25
ORDER BY ticker, day;

-- ============================================================================
-- Example 4: Prediction intervals from the window function
-- ============================================================================

SELECT '=== Example 4: Intervals ===' AS section;

SELECT
    day,
    ROUND(price, 2)             AS price,
    ROUND((pred).yhat, 2)       AS fitted,
    ROUND((pred).yhat_lower, 2) AS lower_90,
    ROUND((pred).yhat_upper, 2) AS upper_90
FROM (
    SELECT
        day,
        price,
        ols_fit_predict(price, [day::DOUBLE], {'confidence_level': 0.90}) OVER (
            ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS pred
    FROM stock_prices
    WHERE ticker = 'BANK'
)
WHERE day > 25
ORDER BY day;

-- ============================================================================
-- Example 5: Forecasting appended rows (NULL y is predicted, not trained on)
-- ============================================================================
-- Rows whose y is NULL are excluded from fitting but still get a prediction
-- when they are the last row of the frame.

SELECT '=== Example 5: Forecast Rows with NULL y ===' AS section;

WITH extended AS (
    SELECT day, price FROM stock_prices WHERE ticker = 'TECH'
    UNION ALL
    SELECT day, NULL::DOUBLE FROM generate_series(31, 33) AS f(day)
)
SELECT * FROM (
    SELECT
        day,
        ROUND(price, 2) AS actual,
        ROUND((ols_fit_predict(price, [day::DOUBLE]) OVER (
            ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 2) AS prediction
    FROM extended
)
WHERE day >= 28
ORDER BY day;

-- ============================================================================
-- Example 6: Other models as window functions
-- ============================================================================

SELECT '=== Example 6: Ridge, Huber and WLS Windows ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS price,
    ROUND((ridge_fit_predict(price, [day::DOUBLE], {'alpha': 1.0}) OVER w).yhat, 2) AS ridge,
    ROUND((huber_fit_predict(price, [day::DOUBLE]) OVER w).yhat, 2)                 AS huber,
    ROUND((wls_fit_predict(price, [day::DOUBLE], volume_k) OVER w).yhat, 2)         AS wls
FROM stock_prices
WHERE ticker = 'TECH'
WINDOW w AS (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY day
LIMIT 8;

-- ============================================================================
-- Example 7: Coefficients over an expanding window with ols_fit_agg
-- ============================================================================
-- ols_fit_agg can also be used as a window aggregate to track how the
-- coefficients evolve as data accumulates.

SELECT '=== Example 7: Evolving Coefficients ===' AS section;

SELECT
    day,
    ROUND((ols_fit_agg(price, [day::DOUBLE]) OVER w).coefficients[1], 4) AS slope,
    ROUND((ols_fit_agg(price, [day::DOUBLE]) OVER w).r_squared, 4)       AS r_squared
FROM stock_prices
WHERE ticker = 'TECH'
WINDOW w AS (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY day
LIMIT 10;

-- ============================================================================
-- One-step-ahead forecasts
-- ============================================================================
-- To predict each row from a model fitted on EARLIER rows only, fit with
-- ols_fit_agg over a frame ending at 1 PRECEDING and score the current row
-- with predict(). predict() takes feature-major x and currently raises an
-- error when the coefficients are NULL (the first rows), so the query is
-- shown here commented out.
-- TODO(lead): model-aware predict
--
-- SELECT day, price,
--     predict([[day::DOUBLE]],
--             (ols_fit_agg(price, [day::DOUBLE]) OVER w).coefficients,
--             (ols_fit_agg(price, [day::DOUBLE]) OVER w).intercept)[1] AS forecast
-- FROM stock_prices
-- WHERE ticker = 'TECH'
-- WINDOW w AS (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- Cleanup
DROP TABLE IF EXISTS stock_prices;
