-- ============================================================================
-- EFFICIENT MODEL-BASED PREDICTION DEMONSTRATION
-- Using anofox_statistics_model_predict with pre-fitted models
--
-- Run with: duckdb -unsigned -init examples/model_prediction_demo.sql
-- ============================================================================

.print ''
.print '================================================================================'
.print 'EFFICIENT MODEL-BASED PREDICTION DEMONSTRATION'
.print '================================================================================'

-- Load the extension
LOAD 'build/debug/extension/anofox_statistics/anofox_statistics.duckdb_extension';

.print ''
.print '📊 STEP 1: Fit model ONCE with full_output=true'
.print '--------------------------------------------------------------------------------'

-- Sample data: Sales vs Price and Advertising Budget
CREATE TEMP TABLE training_data (sales DOUBLE, price DOUBLE, advertising DOUBLE);
INSERT INTO training_data VALUES
    (100, 10, 5),
    (120, 12, 6),
    (140, 14, 7),
    (160, 16, 8),
    (180, 18, 9),
    (200, 20, 10),
    (220, 22, 11),
    (240, 24, 12);

.print 'Training data loaded:'
SELECT * FROM training_data;

-- Fit model and store all metadata
CREATE TABLE sales_model AS
SELECT * FROM anofox_statistics_ols(
    (SELECT list(sales) FROM training_data)::DOUBLE[],
    (SELECT list([price, advertising]) FROM training_data)::DOUBLE[][],
    MAP{'intercept': true, 'full_output': true}
);

.print ''
.print '✅ Model fitted successfully!'
.print 'Model summary:'
SELECT
    round(intercept, 4) as intercept,
    list_transform(coefficients, x -> round(x, 4)) as coefficients,
    round(r_squared, 4) as r2,
    round(mse, 4) as mse,
    n_obs,
    df_residual
FROM sales_model;

.print ''
.print '================================================================================'
.print '🎯 STEP 2: Make predictions on new data (NO REFITTING!)'
.print '================================================================================'

.print ''
.print 'New scenarios to predict:'
.print '  1. Price=$25, Advertising=$13k'
.print '  2. Price=$26, Advertising=$14k'
.print '  3. Price=$27, Advertising=$15k'

.print ''
.print '📈 Predictions with CONFIDENCE intervals (95%):'
.print '--------------------------------------------------------------------------------'
SELECT
    p.observation_id as obs,
    round(p.predicted, 2) as forecast,
    round(p.ci_lower, 2) as ci_lower,
    round(p.ci_upper, 2) as ci_upper,
    round(p.se, 4) as std_error,
    round(p.ci_upper - p.ci_lower, 2) as interval_width
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept,
    m.coefficients,
    m.mse,
    m.x_train_means,
    m.coefficient_std_errors,
    m.intercept_std_error,
    m.df_residual,
    [[25.0, 13.0], [26.0, 14.0], [27.0, 15.0]]::DOUBLE[][],
    0.95,
    'confidence'
) p;

.print ''
.print '🎲 Predictions with PREDICTION intervals (95% - wider than confidence):'
.print '--------------------------------------------------------------------------------'
SELECT
    p.observation_id as obs,
    round(p.predicted, 2) as forecast,
    round(p.ci_lower, 2) as pi_lower,
    round(p.ci_upper, 2) as pi_upper,
    round(p.ci_upper - p.ci_lower, 2) as interval_width
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[25.0, 13.0], [26.0, 14.0], [27.0, 15.0]]::DOUBLE[][],
    0.95,
    'prediction'
) p;

.print ''
.print '⚡ STEP 3: High-speed batch predictions (no intervals = fastest)'
.print '--------------------------------------------------------------------------------'

-- Predictions at different confidence levels
.print ''
.print 'Same prediction at different confidence levels (prediction intervals):'
SELECT
    'CI 90%' as level,
    round(p.predicted, 2) as forecast,
    round(p.ci_upper - p.ci_lower, 2) as width
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[25.0, 13.0]]::DOUBLE[][],
    0.90, 'prediction'
) p

UNION ALL

SELECT
    'CI 95%' as level,
    round(p.predicted, 2) as forecast,
    round(p.ci_upper - p.ci_lower, 2) as width
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[25.0, 13.0]]::DOUBLE[][],
    0.95, 'prediction'
) p

UNION ALL

SELECT
    'CI 99%' as level,
    round(p.predicted, 2) as forecast,
    round(p.ci_upper - p.ci_lower, 2) as width
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[25.0, 13.0]]::DOUBLE[][],
    0.99, 'prediction'
) p;

.print ''
.print '================================================================================'
.print '✅ KEY BENEFITS OF MODEL-BASED PREDICTION'
.print '================================================================================'
.print '  ✓ Model fitted ONCE, reused MANY times'
.print '  ✓ No refitting overhead for predictions'
.print '  ✓ Flexible intervals: confidence, prediction, or none'
.print '  ✓ Perfect for production pipelines and batch scoring'
.print '  ✓ Works with all regression types: OLS, Ridge, WLS, Elastic Net, RLS'
.print '================================================================================'
.print ''
