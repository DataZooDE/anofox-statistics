-- ============================================================================
-- OLS Fit-Predict Examples with Artificial Dataset
-- ============================================================================
-- This file demonstrates all OLS (Ordinary Least Squares) fit-predict
-- capabilities using artificial datasets.

-- Load the extension
LOAD anofox_statistics;

-- ============================================================================
-- Example 1: Simple Linear Regression with Train/Test Split
-- ============================================================================
-- Create a dataset where y = 2*x + 1, with first 10 rows for training
-- and last 5 rows for prediction (y = NULL)

CREATE OR REPLACE TABLE simple_linear AS
SELECT
    CASE WHEN id <= 10 THEN (2.0 * id + 1.0 + (random() * 0.5 - 0.25))::DOUBLE ELSE NULL END as y,
    id::DOUBLE as x,
    id
FROM range(1, 16) t(id);

-- View the data
SELECT * FROM simple_linear;

-- Fit model on training data and predict on test data
-- The expanding window fits on all previous rows and predicts current row
SELECT
    id,
    x,
    y,
    ROUND((pred).yhat, 2) as yhat,           -- Point prediction
    ROUND((pred).yhat_lower, 2) as lower,    -- Lower confidence bound
    ROUND((pred).yhat_upper, 2) as upper     -- Upper confidence bound
FROM (
    SELECT
        id,
        x,
        y,
        anofox_statistics_fit_predict_ols(
            y,                              -- Dependent variable
            [x],                            -- Independent variables (array)
            MAP{'intercept': true}          -- Options: include intercept
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred
    FROM simple_linear
)
ORDER BY id;

-- ============================================================================
-- Example 2: Fixed Model - Train Once, Predict All
-- ============================================================================
-- Fit ONE model using all training data (y IS NOT NULL), then use that SAME
-- fixed model to predict on ALL rows (both training and test sets).
-- This is the classic train/test approach.
--
-- The 'fit_predict_mode': 'fixed' option tells the function to:
-- 1. Fit the model using ALL training data (where y IS NOT NULL)
-- 2. Use that same fixed model for ALL predictions (every row)
-- 3. Ignore the window frame - it uses all training data regardless of the frame
--
-- Note: You still need OVER (ORDER BY ... ROWS BETWEEN ...) for proper operation
-- (using just OVER () causes a DuckDB query profiler assertion error)
--
-- This is different from 'expanding' mode which fits a growing model as it
-- moves through the data.

CREATE OR REPLACE TABLE train_test_split AS
SELECT
    CASE WHEN id <= 12 THEN (2.5 * id + 3.0 + (random() * 2.0 - 1.0))::DOUBLE ELSE NULL END as y,
    id::DOUBLE as x,
    id
FROM range(1, 21) t(id);

-- View the data split
SELECT
    COUNT(*) as total_rows,
    SUM(CASE WHEN y IS NOT NULL THEN 1 ELSE 0 END) as training_rows,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as test_rows
FROM train_test_split;

-- Fit model on ALL training data (rows 1-12), then predict on ALL rows (1-20)
-- Using 'fit_predict_mode': 'fixed' ensures one model fit, not expanding window
SELECT
    id,
    x,
    y,
    ROUND((pred).yhat, 2) as yhat,
    ROUND((pred).yhat_lower, 2) as lower,
    ROUND((pred).yhat_upper, 2) as upper,
    CASE
        WHEN y IS NOT NULL THEN ROUND(ABS(y - (pred).yhat), 2)
        ELSE NULL
    END as abs_error,
    CASE WHEN y IS NULL THEN 'Test (Out-of-Sample)' ELSE 'Train (In-Sample)' END as dataset
FROM (
    SELECT
        id,
        x,
        y,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            {'fit_predict_mode': 'fixed', 'intercept': true}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred
    FROM train_test_split
)
ORDER BY id;

-- ============================================================================
-- Example 2b: Comparing Expanding vs Fixed Mode
-- ============================================================================
-- This example demonstrates the key difference between the two fit_predict_modes:
-- - 'expanding': Fits model on growing window (rows 1 to current-1), predicts current row
-- - 'fixed': Fits model ONCE on all training data, predicts all rows with same model

-- Create data where relationship changes over time to highlight the difference
CREATE OR REPLACE TABLE mode_comparison AS
SELECT
    id::DOUBLE as x,
    -- First 15 rows: y = 2*x + 5 (one regime)
    -- Last 5 rows: y = 4*x + 1 (different regime) - to show expanding adapts
    CASE
        WHEN id <= 15 THEN (2.0 * id + 5.0 + (random() * 1.0 - 0.5))::DOUBLE
        WHEN id <= 18 THEN (4.0 * id + 1.0 + (random() * 1.0 - 0.5))::DOUBLE
        ELSE NULL  -- Test data
    END as y,
    id
FROM range(1, 21) t(id);

-- Compare both modes side-by-side
SELECT
    id,
    x,
    y,
    ROUND((pred_expanding).yhat, 2) as expanding_pred,
    ROUND((pred_fixed).yhat, 2) as fixed_pred,
    ROUND((pred_expanding).yhat - (pred_fixed).yhat, 2) as difference,
    CASE
        WHEN id <= 15 THEN 'Regime 1 (slope=2)'
        WHEN id <= 18 THEN 'Regime 2 (slope=4)'
        ELSE 'Test Data'
    END as regime
FROM (
    SELECT
        id,
        x,
        y,
        -- Expanding mode: model grows/adapts as it moves through data
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            {'fit_predict_mode': 'expanding', 'intercept': true}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_expanding,
        -- Fixed mode: one model fit on ALL training data (rows 1-18)
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            {'fit_predict_mode': 'fixed', 'intercept': true}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_fixed
    FROM mode_comparison
)
ORDER BY id;

-- ============================================================================
-- Example 3: Multiple Linear Regression
-- ============================================================================
-- Dataset where y = 2*x1 + 3*x2 + 10, with independent features

CREATE OR REPLACE TABLE multiple_regression AS
SELECT
    id::DOUBLE as x1,
    (id + random() * 2.0)::DOUBLE as x2,  -- Independent feature
    CASE WHEN id <= 20 THEN
        (2.0 * id + 3.0 * (id + random() * 2.0) + 10.0 + (random() * 2.0 - 1.0))::DOUBLE
    ELSE NULL END as y,
    id
FROM range(1, 31) t(id);

-- Fit with multiple features
SELECT
    id,
    ROUND((pred).yhat, 2) as yhat,
    y as actual,
    CASE WHEN y IS NOT NULL
        THEN ROUND(ABS((pred).yhat - y), 2)
        ELSE NULL
    END as error
FROM (
    SELECT
        id,
        y,
        anofox_statistics_fit_predict_ols(
            y,
            [x1, x2],                       -- Multiple features
            MAP{'intercept': true}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred
    FROM multiple_regression
)
ORDER BY id;

-- ============================================================================
-- Example 4: Rolling Window Regression
-- ============================================================================
-- Use only the last N observations to fit the model
-- Useful for non-stationary data or detecting regime changes

CREATE OR REPLACE TABLE time_series AS
SELECT
    (2.0 * time_id + 5.0 + (random() * 3.0 - 1.5))::DOUBLE as y,
    time_id::DOUBLE as x,
    time_id
FROM range(1, 31) t(time_id);

-- Compare expanding window vs rolling window (size 10)
SELECT
    time_id,
    x,
    ROUND(y, 2) as actual,
    ROUND((pred_expanding).yhat, 2) as expanding_pred,
    ROUND((pred_rolling).yhat, 2) as rolling_pred
FROM (
    SELECT
        time_id,
        x,
        y,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': true}
        ) OVER (ORDER BY time_id) as pred_expanding,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': true}
        ) OVER (ORDER BY time_id ROWS BETWEEN 9 PRECEDING AND 1 PRECEDING) as pred_rolling
    FROM time_series
)
WHERE time_id >= 11  -- Only show after rolling window is full
ORDER BY time_id;

-- ============================================================================
-- Example 5: Grouped/Partitioned Regression
-- ============================================================================
-- Separate models for different groups

CREATE OR REPLACE TABLE grouped_data AS
SELECT
    (slope * id + intercept + (random() * 2.0 - 1.0))::DOUBLE as y,
    id::DOUBLE as x,
    group_name,
    slope,
    intercept,
    id
FROM range(1, 21) t(id),
(VALUES
    ('Group_A', 2.0, 5.0),
    ('Group_B', -1.5, 20.0),
    ('Group_C', 0.5, 10.0)
) g(group_name, slope, intercept);

-- Fit separate model for each group
SELECT
    group_name,
    id,
    ROUND(y, 2) as actual,
    ROUND((pred).yhat, 2) as predicted,
    ROUND((pred).yhat_lower, 2) as lower,
    ROUND((pred).yhat_upper, 2) as upper
FROM (
    SELECT
        group_name,
        id,
        y,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': true}
        ) OVER (PARTITION BY group_name ORDER BY id) as pred
    FROM grouped_data
)
ORDER BY group_name, id;

-- ============================================================================
-- Example 6: Regression Without Intercept
-- ============================================================================
-- Force the line through the origin (useful in some physical models)

CREATE OR REPLACE TABLE no_intercept_data AS
SELECT
    (3.5 * id + (random() * 1.0 - 0.5))::DOUBLE as y,
    id::DOUBLE as x,
    id
FROM range(1, 21) t(id);

-- Compare with and without intercept
SELECT
    id,
    ROUND(y, 2) as actual,
    ROUND((pred_with_intercept).yhat, 2) as with_intercept,
    ROUND((pred_no_intercept).yhat, 2) as no_intercept
FROM (
    SELECT
        id,
        y,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': true}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_with_intercept,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': false}
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_no_intercept
    FROM no_intercept_data
)
ORDER BY id;

-- ============================================================================
-- Example 7: Prediction Intervals with Different Confidence Levels
-- ============================================================================
-- Control the width of prediction intervals with alpha parameter

CREATE OR REPLACE TABLE confidence_demo AS
SELECT
    CASE WHEN id <= 15 THEN (2.0 * id + 10.0 + (random() * 4.0 - 2.0))::DOUBLE ELSE NULL END as y,
    id::DOUBLE as x,
    id
FROM range(1, 21) t(id);

-- Compare different confidence levels
SELECT
    id,
    ROUND((pred_95).yhat, 2) as prediction,
    ROUND((pred_95).yhat_lower, 2) as lower_95,
    ROUND((pred_95).yhat_upper, 2) as upper_95,
    ROUND((pred_99).yhat_lower, 2) as lower_99,
    ROUND((pred_99).yhat_upper, 2) as upper_99
FROM (
    SELECT
        id,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': 1, 'alpha': 0.05}  -- 95% confidence (default)
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_95,
        anofox_statistics_fit_predict_ols(
            y,
            [x],
            MAP{'intercept': 1, 'alpha': 0.01}  -- 99% confidence
        ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred_99
    FROM confidence_demo
)
WHERE id > 15  -- Show only predictions
ORDER BY id;

-- ============================================================================
-- Example 8: Forecasting Future Values
-- ============================================================================
-- Practical example: predict sales based on historical data

CREATE OR REPLACE TABLE sales_data AS
SELECT
    month,
    (100.0 + month * 10.0)::DOUBLE as marketing_spend,
    CASE
        WHEN month % 12 IN (11, 0, 1) THEN 'Winter'
        WHEN month % 12 IN (2, 3, 4) THEN 'Spring'
        WHEN month % 12 IN (5, 6, 7) THEN 'Summer'
        ELSE 'Fall'
    END as season,
    CASE
        WHEN month % 12 IN (11, 0, 1) THEN 100.0  -- Winter boost
        WHEN month % 12 IN (5, 6, 7) THEN 50.0     -- Summer boost
        ELSE 0.0
    END as season_effect,
    CASE WHEN month <= 12 THEN
        (500.0 + 2.5 * (100.0 + month * 10.0) +
         CASE WHEN month % 12 IN (11, 0, 1) THEN 100.0 WHEN month % 12 IN (5, 6, 7) THEN 50.0 ELSE 0.0 END +
         (random() * 50.0 - 25.0))::DOUBLE
    ELSE NULL END as sales
FROM range(1, 19) t(month);

-- Forecast next 6 months of sales
SELECT
    month,
    season,
    ROUND(marketing_spend, 0) as marketing_spend,
    ROUND(sales, 0) as actual_sales,
    ROUND((forecast).yhat, 0) as predicted_sales,
    ROUND((forecast).yhat_lower, 0) as lower_bound,
    ROUND((forecast).yhat_upper, 0) as upper_bound,
    CASE
        WHEN sales IS NULL THEN 'Forecast'
        ELSE 'Historical'
    END as type
FROM (
    SELECT
        month,
        season,
        marketing_spend,
        sales,
        anofox_statistics_fit_predict_ols(
            sales,
            [marketing_spend],
            MAP{'intercept': 1, 'alpha': 0.05}
        ) OVER (ORDER BY month) as forecast
    FROM sales_data
)
ORDER BY month;

-- ============================================================================
-- Example 9: Model Quality Assessment
-- ============================================================================
-- Extract predictions and calculate R-squared on training set

CREATE OR REPLACE TABLE model_quality AS
SELECT
    id::DOUBLE as x1,
    (id + random() * 5.0)::DOUBLE as x2,  -- Independent feature
    (2.5 * id + 1.5 * (id + random() * 5.0) + 5.0 + (random() * 3.0 - 1.5))::DOUBLE as y,
    id
FROM range(1, 31) t(id);

-- Calculate predictions and residuals
WITH predictions AS (
    SELECT
        y,
        (pred).yhat as y_pred
    FROM (
        SELECT
            y,
            anofox_statistics_fit_predict_ols(
                y,
                [x1, x2],
                MAP{'intercept': true}
            ) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pred
        FROM model_quality
    )
    WHERE y IS NOT NULL
),
y_mean AS (
    SELECT AVG(y) as mean_y FROM predictions
),
stats AS (
    SELECT
        SUM(POW(y - y_pred, 2)) as ss_res,              -- Residual sum of squares
        SUM(POW(y - (SELECT mean_y FROM y_mean), 2)) as ss_tot,  -- Total sum of squares
        COUNT(*) as n
    FROM predictions
)
SELECT
    ROUND(1.0 - ss_res / ss_tot, 4) as r_squared,
    ROUND(SQRT(ss_res / n), 4) as rmse
FROM stats;

-- ============================================================================
-- Example 10: Extracting Model Coefficients
-- ============================================================================
-- Use anofox_statistics_ols_agg to fit a model and extract coefficients,
-- intercept, R-squared, and other statistics

CREATE OR REPLACE TABLE coef_demo AS
SELECT
    id::DOUBLE as x1,
    (id + random() * 3.0)::DOUBLE as x2,
    (2.0 * id + 1.5 * (id + random() * 3.0) + 10.0 + (random() * 5.0 - 2.5))::DOUBLE as y,
    id
FROM range(1, 51) t(id);

-- Fit model and extract all statistics
SELECT
    'Model Statistics' as description,
    (fit).intercept as intercept,
    (fit).coefficients as coefficients,
    (fit).r2 as r_squared,
    (fit).adj_r2 as adj_r_squared,
    (fit).mse as mean_squared_error,
    (fit).n_obs as n_observations,
    (fit).df_residual as degrees_of_freedom
FROM (
    SELECT anofox_statistics_ols_agg(y, [x1, x2], MAP{'intercept': 1}) as fit
    FROM coef_demo
);

-- Extract just the coefficients with interpretation
SELECT
    'Coefficient Interpretation' as description,
    ROUND((fit).intercept, 4) as intercept,
    ROUND((fit).coefficients[1], 4) as beta_x1,
    ROUND((fit).coefficients[2], 4) as beta_x2,
    'y = ' || ROUND((fit).intercept, 2)::VARCHAR ||
    ' + ' || ROUND((fit).coefficients[1], 2)::VARCHAR || '*x1' ||
    ' + ' || ROUND((fit).coefficients[2], 2)::VARCHAR || '*x2' as equation
FROM (
    SELECT anofox_statistics_ols_agg(y, [x1, x2], MAP{'intercept': 1}) as fit
    FROM coef_demo
);

-- Coefficients with standard errors and t-statistics
SELECT
    'Statistical Significance' as description,
    ROUND((fit).coefficients[1], 4) as beta_x1,
    ROUND((fit).coefficient_std_errors[1], 4) as se_x1,
    ROUND((fit).coefficients[1] / (fit).coefficient_std_errors[1], 4) as t_stat_x1,
    ROUND((fit).coefficients[2], 4) as beta_x2,
    ROUND((fit).coefficient_std_errors[2], 4) as se_x2,
    ROUND((fit).coefficients[2] / (fit).coefficient_std_errors[2], 4) as t_stat_x2
FROM (
    SELECT anofox_statistics_ols_agg(y, [x1, x2], MAP{'intercept': 1}) as fit
    FROM coef_demo
);

-- Fit separate models for different groups and compare coefficients
CREATE OR REPLACE TABLE grouped_coef AS
SELECT
    id::DOUBLE as x,
    (slope * id + intercept + (random() * 3.0 - 1.5))::DOUBLE as y,
    group_name
FROM range(1, 31) t(id),
(VALUES ('Group_A', 2.5, 10.0), ('Group_B', -1.2, 25.0)) g(group_name, slope, intercept);

-- Compare coefficients across groups
SELECT
    group_name,
    ROUND((fit).intercept, 2) as intercept,
    ROUND((fit).coefficients[1], 2) as slope,
    ROUND((fit).r2, 4) as r_squared,
    (fit).n_obs as n
FROM (
    SELECT
        group_name,
        anofox_statistics_ols_agg(y, [x], MAP{'intercept': 1}) as fit
    FROM grouped_coef
    GROUP BY group_name
)
ORDER BY group_name;

-- ============================================================================
-- Cleanup
-- ============================================================================

DROP TABLE IF EXISTS simple_linear;
DROP TABLE IF EXISTS train_test_split;
DROP TABLE IF EXISTS mode_comparison;
DROP TABLE IF EXISTS multiple_regression;
DROP TABLE IF EXISTS time_series;
DROP TABLE IF EXISTS grouped_data;
DROP TABLE IF EXISTS no_intercept_data;
DROP TABLE IF EXISTS confidence_demo;
DROP TABLE IF EXISTS sales_data;
DROP TABLE IF EXISTS model_quality;
DROP TABLE IF EXISTS coef_demo;
DROP TABLE IF EXISTS grouped_coef;
