-- ============================================================================
-- OLS Window Functions Examples
-- ============================================================================
-- Demonstrates window-based regression using OVER clause.
-- Topics: Expanding windows, rolling windows, partitioned analysis, fit_predict
--
-- Run: ./build/release/duckdb < examples/ols_window_functions.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Time Series Dataset
-- ============================================================================

CREATE OR REPLACE TABLE stock_prices AS
SELECT
    ticker,
    day,
    -- Different stocks with different characteristics
    CASE ticker
        WHEN 'TECH' THEN 100.0 + day * 1.5 + (RANDOM() * 10 - 5)
        WHEN 'BANK' THEN 50.0 + day * 0.5 + (RANDOM() * 5 - 2.5)
        WHEN 'RETAIL' THEN 30.0 + day * 0.3 + SIN(day * 0.2) * 5 + (RANDOM() * 3 - 1.5)
    END AS price,
    CASE ticker
        WHEN 'TECH' THEN 1000000 + day * 10000 + (RANDOM() * 50000)::INTEGER
        WHEN 'BANK' THEN 500000 + day * 5000 + (RANDOM() * 25000)::INTEGER
        WHEN 'RETAIL' THEN 200000 + day * 2000 + (RANDOM() * 10000)::INTEGER
    END AS volume
FROM (VALUES ('TECH'), ('BANK'), ('RETAIL')) AS t(ticker),
     generate_series(1, 30) AS d(day);

-- ============================================================================
-- Example 1: Expanding Window (Train on All Preceding)
-- ============================================================================
-- As you move through time, the model is trained on all historical data

SELECT '=== Example 1: Expanding Window ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS price,
    ROUND(volume / 1000.0, 0) AS volume_k,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND(price - (pred).yhat, 2) AS error
FROM (
    SELECT
        day,
        price,
        volume,
        ols_fit_predict(
            price,
            [volume::DOUBLE],
            {'intercept': true}
        ) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS pred
    FROM stock_prices
    WHERE ticker = 'TECH'
)
WHERE day >= 5  -- Need some history first
ORDER BY day
LIMIT 15;

-- ============================================================================
-- Example 2: Fixed-Size Rolling Window
-- ============================================================================
-- Model trained on last N observations only (more responsive to recent trends)

SELECT '=== Example 2: Rolling Window (10 days) ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS actual,
    ROUND((pred_expand).yhat, 2) AS expanding_pred,
    ROUND((pred_roll).yhat, 2) AS rolling_pred,
    ROUND(ABS(price - (pred_expand).yhat), 2) AS expand_error,
    ROUND(ABS(price - (pred_roll).yhat), 2) AS roll_error
FROM (
    SELECT
        day,
        price,
        -- Expanding window
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pred_expand,
        -- Rolling window (last 10 days)
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN 9 PRECEDING AND 1 PRECEDING) AS pred_roll
    FROM stock_prices
    WHERE ticker = 'TECH'
)
WHERE day >= 12  -- Need full window
ORDER BY day;

-- ============================================================================
-- Example 3: Partitioned Analysis (Per-Group Time Series)
-- ============================================================================
-- Separate model for each ticker

SELECT '=== Example 3: Partitioned Analysis ===' AS section;

SELECT
    ticker,
    day,
    ROUND(price, 2) AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND((pred).yhat_lower, 2) AS lower_95,
    ROUND((pred).yhat_upper, 2) AS upper_95
FROM (
    SELECT
        ticker,
        day,
        price,
        ols_fit_predict(
            price,
            [day::DOUBLE],
            {'intercept': true, 'confidence_level': 0.95}
        ) OVER (
            PARTITION BY ticker
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS pred
    FROM stock_prices
)
WHERE day IN (15, 20, 25, 30)
ORDER BY ticker, day;

-- ============================================================================
-- Example 4: Prediction Intervals
-- ============================================================================
-- Get confidence/prediction intervals along with point estimates

SELECT '=== Example 4: Prediction Intervals ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND((pred).yhat_lower, 2) AS ci_lower,
    ROUND((pred).yhat_upper, 2) AS ci_upper,
    ROUND((pred).yhat_upper - (pred).yhat_lower, 2) AS interval_width,
    CASE WHEN price BETWEEN (pred).yhat_lower AND (pred).yhat_upper
         THEN 'Within CI' ELSE 'Outside CI' END AS in_interval
FROM (
    SELECT
        day,
        price,
        ols_fit_predict(
            price,
            [day::DOUBLE],
            {'intercept': true, 'confidence_level': 0.95}
        ) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS pred
    FROM stock_prices
    WHERE ticker = 'TECH'
)
WHERE day >= 5
ORDER BY day;

-- ============================================================================
-- Example 5: Multiple Features in Window Function
-- ============================================================================

SELECT '=== Example 5: Multiple Features ===' AS section;

SELECT
    day,
    ROUND(price, 2) AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    ROUND(price - (pred).yhat, 2) AS residual
FROM (
    SELECT
        day,
        price,
        ols_fit_predict(
            price,
            [day::DOUBLE, volume::DOUBLE],  -- Two features
            {'intercept': true}
        ) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS pred
    FROM stock_prices
    WHERE ticker = 'TECH'
)
WHERE day >= 5
ORDER BY day
LIMIT 15;

-- ============================================================================
-- Example 6: Fixed Mode - Train Once, Predict All
-- ============================================================================
-- Use fit_predict_mode='fixed' to fit on all training data, then predict all rows

SELECT '=== Example 6: Fixed Mode (Train Once) ===' AS section;

-- Create dataset with training (y not null) and test (y null) rows
CREATE OR REPLACE TABLE train_test AS
SELECT
    day,
    CASE WHEN day <= 20 THEN price ELSE NULL END AS y_train,
    price AS actual,
    volume
FROM stock_prices
WHERE ticker = 'TECH';

SELECT
    day,
    ROUND(actual, 2) AS actual,
    ROUND((pred).yhat, 2) AS predicted,
    CASE WHEN y_train IS NULL THEN 'Test' ELSE 'Train' END AS dataset
FROM (
    SELECT
        day,
        actual,
        y_train,
        ols_fit_predict(
            y_train,
            [day::DOUBLE],
            {'fit_predict_mode': 'fixed', 'intercept': true}
        ) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS pred
    FROM train_test
)
ORDER BY day;

-- ============================================================================
-- Example 7: Comparing Window Sizes
-- ============================================================================

SELECT '=== Example 7: Compare Window Sizes ===' AS section;

SELECT
    day,
    ROUND((pred_5).yhat, 2) AS window_5,
    ROUND((pred_10).yhat, 2) AS window_10,
    ROUND((pred_20).yhat, 2) AS window_20,
    ROUND((pred_all).yhat, 2) AS window_all
FROM (
    SELECT
        day,
        -- Window size 5
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS pred_5,
        -- Window size 10
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN 9 PRECEDING AND 1 PRECEDING) AS pred_10,
        -- Window size 20
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN 19 PRECEDING AND 1 PRECEDING) AS pred_20,
        -- All history (expanding)
        ols_fit_predict(price, [day::DOUBLE], {'intercept': true})
            OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pred_all
    FROM stock_prices
    WHERE ticker = 'TECH'
)
WHERE day >= 22
ORDER BY day;

-- ============================================================================
-- Example 8: Handling Missing Values (null_policy)
-- ============================================================================

SELECT '=== Example 8: Handling Missing Values ===' AS section;

-- Create data with some missing y values
CREATE OR REPLACE TABLE sparse_data AS
SELECT
    day,
    -- Make some y values NULL
    CASE WHEN day % 3 = 0 THEN NULL ELSE price END AS y,
    day::DOUBLE AS x,
    price AS actual
FROM stock_prices
WHERE ticker = 'TECH';

SELECT
    day,
    ROUND(actual, 2) AS actual,
    y IS NOT NULL AS y_present,
    ROUND((pred).yhat, 2) AS predicted
FROM (
    SELECT
        day,
        actual,
        y,
        ols_fit_predict(
            y,
            [x],
            {'intercept': true, 'null_policy': 'drop'}
        ) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS pred
    FROM sparse_data
)
WHERE day >= 5
ORDER BY day
LIMIT 15;

-- ============================================================================
-- Example 9: One-Step-Ahead Forecasting
-- ============================================================================
-- Train on rows up to t-1, predict for row t

SELECT '=== Example 9: One-Step-Ahead Forecast ===' AS section;

WITH forecasts AS (
    SELECT
        day,
        price AS actual,
        (pred).yhat AS forecast
    FROM (
        SELECT
            day,
            price,
            ols_fit_predict(
                price,
                [day::DOUBLE],
                {'intercept': true}
            ) OVER (
                ORDER BY day
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS pred
        FROM stock_prices
        WHERE ticker = 'TECH'
    )
    WHERE day >= 5
)
SELECT
    ROUND(AVG(ABS(actual - forecast)), 4) AS mae,
    ROUND(SQRT(AVG(POW(actual - forecast, 2))), 4) AS rmse,
    ROUND(AVG(ABS(actual - forecast) / actual) * 100, 2) AS mape_pct
FROM forecasts;

-- ============================================================================
-- Example 10: Model Performance Over Time
-- ============================================================================

SELECT '=== Example 10: Performance Over Time ===' AS section;

SELECT
    CASE
        WHEN day <= 10 THEN 'Days 1-10'
        WHEN day <= 20 THEN 'Days 11-20'
        ELSE 'Days 21-30'
    END AS period,
    COUNT(*) AS n,
    ROUND(AVG(ABS(actual - predicted)), 4) AS mae,
    ROUND(SQRT(AVG(POW(actual - predicted, 2))), 4) AS rmse
FROM (
    SELECT
        day,
        price AS actual,
        (pred).yhat AS predicted
    FROM (
        SELECT
            day,
            price,
            ols_fit_predict(
                price,
                [day::DOUBLE],
                {'intercept': true}
            ) OVER (
                ORDER BY day
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS pred
        FROM stock_prices
        WHERE ticker = 'TECH'
    )
    WHERE day >= 5
)
GROUP BY 1
ORDER BY 1;

-- Cleanup
DROP TABLE IF EXISTS stock_prices;
DROP TABLE IF EXISTS train_test;
DROP TABLE IF EXISTS sparse_data;
