-- ============================================================================
-- MODEL FIT AND PREDICT DEMONSTRATION
-- Using anofox_stats_ols_fit_agg and anofox_stats_predict
--
-- Run with: ./build/release/duckdb < examples/model_prediction_demo.sql
-- ============================================================================

.timer on

.print ''
.print '================================================================================'
.print 'MODEL FIT AND PREDICT DEMONSTRATION'
.print '================================================================================'

.print ''
.print 'STEP 1: Create training data'
.print '--------------------------------------------------------------------------------'

CREATE TEMP TABLE training_data AS
SELECT
    (100 + 10 * id + 5 * (id + random() * 2) + (random() * 10 - 5))::DOUBLE as sales,
    id::DOUBLE as price,
    (id + random() * 2)::DOUBLE as advertising
FROM range(1, 21) t(id);

SELECT * FROM training_data LIMIT 10;

.print ''
.print 'STEP 2: Fit OLS model using aggregate function'
.print '--------------------------------------------------------------------------------'

CREATE TEMP TABLE sales_model AS
SELECT
    (fit).intercept as intercept,
    (fit).coefficients as coefficients,
    (fit).r_squared as r_squared,
    (fit).residual_std_error as residual_std_error,
    (fit).n_observations as n_obs
FROM (
    SELECT anofox_stats_ols_fit_agg(sales, [price, advertising]) as fit
    FROM training_data
);

SELECT
    round(intercept, 4) as intercept,
    list_transform(coefficients, x -> round(x, 4)) as coefficients,
    round(r_squared, 4) as r_squared,
    round(residual_std_error, 4) as std_error,
    n_obs
FROM sales_model;

.print ''
.print 'STEP 3: Make predictions from stored model coefficients'
.print '--------------------------------------------------------------------------------'

-- New scenarios to predict
CREATE TEMP TABLE new_scenarios AS
SELECT
    25.0::DOUBLE as price,
    13.0::DOUBLE as advertising,
    1 as scenario_id
UNION ALL SELECT 26.0, 14.0, 2
UNION ALL SELECT 27.0, 15.0, 3
UNION ALL SELECT 30.0, 18.0, 4;

-- Predict using: intercept + coef[1]*x1 + coef[2]*x2
SELECT
    n.scenario_id,
    n.price,
    n.advertising,
    round(m.intercept + m.coefficients[1] * n.price + m.coefficients[2] * n.advertising, 2) as predicted_sales
FROM new_scenarios n
CROSS JOIN sales_model m
ORDER BY n.scenario_id;

.print ''
.print 'STEP 4: Verify predictions match the data pattern'
.print '--------------------------------------------------------------------------------'

-- Show that predictions follow the linear relationship: y = intercept + b1*price + b2*advertising
SELECT
    n.scenario_id,
    n.price as price_x1,
    n.advertising as ad_x2,
    round(m.intercept, 2) as intercept,
    round(m.coefficients[1], 2) as b1_price,
    round(m.coefficients[2], 2) as b2_ad,
    round(m.intercept + m.coefficients[1] * n.price + m.coefficients[2] * n.advertising, 2) as predicted
FROM new_scenarios n
CROSS JOIN sales_model m
ORDER BY n.scenario_id;

.print ''
.print 'STEP 5: Compare with window function fit_predict'
.print '--------------------------------------------------------------------------------'

-- Use fit_predict for expanding window predictions
SELECT
    id,
    round(price, 1) as price,
    round(advertising, 1) as ad_spend,
    round(sales, 1) as actual,
    round((pred).yhat, 1) as predicted,
    round((pred).std_error, 3) as std_error
FROM (
    SELECT
        row_number() OVER () as id,
        price,
        advertising,
        sales,
        anofox_stats_ols_fit_predict(
            sales,
            [price, advertising],
            {'fit_intercept': true}
        ) OVER (ORDER BY price ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as pred
    FROM training_data
)
WHERE pred IS NOT NULL
ORDER BY id
LIMIT 10;

.print ''
.print '================================================================================'
.print 'SUMMARY'
.print '================================================================================'
.print 'Functions demonstrated:'
.print '  - anofox_stats_ols_fit_agg: Fit OLS model via GROUP BY aggregation'
.print '  - anofox_stats_ols_fit_predict: Window function for expanding/rolling fit'
.print ''
.print 'Prediction from stored coefficients: intercept + coef[1]*x1 + coef[2]*x2 + ...'
.print '================================================================================'
.print ''

-- Cleanup
DROP TABLE training_data;
DROP TABLE sales_model;
DROP TABLE new_scenarios;
