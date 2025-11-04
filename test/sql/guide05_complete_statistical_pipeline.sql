-- Create sample training data
CREATE OR REPLACE TABLE retail_stores AS
SELECT
    i as store_id,
    2024 as year,
    (50000 + i * 1000 +
     i * 200 * RANDOM() +           -- advertising effect
     i * 50 * RANDOM() +            -- store size effect
     -1000 * (i % 10) +             -- competitor distance effect
     i * 30 * RANDOM() +            -- income effect
     RANDOM() * 5000)::DOUBLE as sale_amount,
    (i * 200 + RANDOM() * 1000)::DOUBLE as advertising_spend,
    (1000 + i * 100 + RANDOM() * 500)::DOUBLE as store_size_sqft,
    (i % 10 + RANDOM() * 5)::DOUBLE as competitor_distance_miles,
    (40000 + i * 1000 + RANDOM() * 10000)::DOUBLE as local_income_median
FROM range(1, 51) t(i);

-- Create new stores table for predictions
CREATE OR REPLACE TABLE new_stores AS
SELECT
    100 + i as store_id,
    (i * 250 + 500)::DOUBLE as advertising_spend,
    (1200 + i * 150)::DOUBLE as store_size_sqft,
    (3 + i * 0.5)::DOUBLE as competitor_distance_miles,
    (45000 + i * 2000)::DOUBLE as local_income_median
FROM range(1, 6) t(i);

-- Stage 1: Data Preparation & Quality Checks
WITH training_data AS (
    SELECT
        store_id,
        sale_amount::DOUBLE as y,
        advertising_spend::DOUBLE as x1,
        store_size_sqft::DOUBLE as x2,
        competitor_distance_miles::DOUBLE as x3,
        local_income_median::DOUBLE as x4
    FROM retail_stores
    WHERE year = 2024 AND sale_amount > 0
),

-- Stage 2: Fit Models on Training Data
-- Using aggregate functions which work directly with table data
full_model AS (
    SELECT
        COUNT(*) as n_train,
        AVG(y) as mean_sales,
        AVG(x1) as mean_advertising,
        -- Primary model: advertising spend predicts sales
        (ols_fit_agg(y, x1)).coefficient as beta_advertising,
        (ols_fit_agg(y, x1)).r2 as model_r2,
        (ols_fit_agg(y, x1)).std_error as model_std_error,
        -- Additional univariate models for comparison
        (ols_fit_agg(y, x2)).coefficient as beta_store_size,
        (ols_fit_agg(y, x2)).r2 as r2_store_size,
        (ols_fit_agg(y, x3)).coefficient as beta_competitor,
        (ols_fit_agg(y, x3)).r2 as r2_competitor,
        (ols_fit_agg(y, x4)).coefficient as beta_income,
        (ols_fit_agg(y, x4)).r2 as r2_income
    FROM training_data
),

-- Calculate intercept from the model (intercept = mean_y - slope * mean_x)
model_params AS (
    SELECT
        *,
        mean_sales - beta_advertising * mean_advertising as intercept
    FROM full_model
),

-- Stage 3: Compute Fitted Values and Residuals from the Model
-- Using the actual model results to calculate predictions
fitted_residuals AS (
    SELECT
        t.store_id,
        t.y as actual_sales,
        -- Predict using the fitted model parameters
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * t.x1 as predicted_sales,
        -- Calculate residual
        t.y - ((SELECT intercept FROM model_params) +
               (SELECT beta_advertising FROM model_params) * t.x1) as residual,
        t.x1,
        t.x2,
        t.x3,
        t.x4
    FROM training_data t
),

-- Stage 4: Identify Outliers Based on Actual Model Residuals
-- Using residuals calculated from our fitted model
outlier_detection AS (
    SELECT
        store_id,
        actual_sales,
        predicted_sales,
        residual,
        -- Standardized residual (using actual model std_error)
        residual / (SELECT model_std_error FROM model_params) as std_residual,
        -- Flag outliers (|std_residual| > 2.5)
        ABS(residual / (SELECT model_std_error FROM model_params)) > 2.5 as is_outlier
    FROM fitted_residuals
),

-- Stage 5: Make Predictions for New Stores
-- Using the fitted model coefficients to predict sales for new stores
predictions AS (
    SELECT
        n.store_id,
        n.advertising_spend,
        n.store_size_sqft,
        n.competitor_distance_miles,
        n.local_income_median,
        -- Predict using the fitted model
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend as predicted_sales,
        -- Confidence interval (approximate: ±1.96 * std_error)
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend -
        1.96 * (SELECT model_std_error FROM model_params) as ci_lower,
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend +
        1.96 * (SELECT model_std_error FROM model_params) as ci_upper
    FROM new_stores n
),

-- Stage 6: Model Performance Summary
-- Using results from previous stages
performance_summary AS (
    SELECT
        'Training' as dataset,
        n_train as n_obs,
        ROUND(model_r2, 4) as r_squared,
        ROUND(model_std_error, 4) as std_error,
        (SELECT COUNT(*) FROM outlier_detection WHERE is_outlier) as n_outliers
    FROM model_params
    UNION ALL
    SELECT
        'Prediction' as dataset,
        COUNT(*) as n_obs,
        NULL as r_squared,
        NULL as std_error,
        NULL as n_outliers
    FROM predictions
),

-- Stage 7: Model Coefficients Report
-- Using actual fitted coefficients
coefficients_report AS (
    SELECT
        'Intercept' as predictor,
        ROUND(intercept, 2) as coefficient,
        NULL as univariate_r2
    FROM model_params
    UNION ALL
    SELECT
        'Advertising Spend',
        ROUND(beta_advertising, 2),
        ROUND(model_r2, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Store Size',
        ROUND(beta_store_size, 2),
        ROUND(r2_store_size, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Competitor Distance',
        ROUND(beta_competitor, 2),
        ROUND(r2_competitor, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Local Income',
        ROUND(beta_income, 2),
        ROUND(r2_income, 4)
    FROM model_params
)

-- Final Integrated Report
SELECT '=== MODEL PERFORMANCE ===' as report_section, NULL as detail_1, NULL as detail_2
UNION ALL
SELECT 'Dataset', CAST(dataset AS VARCHAR), CAST(n_obs AS VARCHAR)
FROM performance_summary
UNION ALL
SELECT 'R-Squared', NULL, CAST(r_squared AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT 'Std Error', NULL, CAST(std_error AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT 'Outliers Detected', NULL, CAST(n_outliers AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT '', NULL, NULL
UNION ALL
SELECT '=== MODEL COEFFICIENTS ===' as report_section, NULL, NULL
UNION ALL
SELECT predictor, CAST(coefficient AS VARCHAR), CAST(univariate_r2 AS VARCHAR)
FROM coefficients_report
UNION ALL
SELECT '', NULL, NULL
UNION ALL
SELECT '=== PREDICTIONS FOR NEW STORES ===' as report_section, NULL, NULL
UNION ALL
SELECT
    'Store ' || CAST(store_id AS VARCHAR),
    'Predicted: $' || CAST(ROUND(predicted_sales, 0) AS VARCHAR),
    'CI: $' || CAST(ROUND(ci_lower, 0) AS VARCHAR) || ' - $' || CAST(ROUND(ci_upper, 0) AS VARCHAR)
FROM predictions
ORDER BY report_section, detail_1;
