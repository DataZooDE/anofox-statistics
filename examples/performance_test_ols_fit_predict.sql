-- ============================================================================
-- Performance Test: OLS Fit-Predict Window Functions
-- ============================================================================
-- This script generates a large dataset for performance testing of
-- anofox_statistics_ols_fit_predict window functions.
--
-- Dataset characteristics:
-- - Multiple groups with time-series style sequential data
-- - Each group has its own true coefficient vector
-- - Includes NULL values in y for prediction demonstration
-- - Sequential obs_id for proper window function ordering
--
-- Usage:
-- 1. Adjust configuration parameters below
-- 2. Run the entire script in DuckDB
-- 3. Observe timing output for performance analysis
-- ============================================================================

-- ============================================================================
-- CONFIGURATION PARAMETERS (easily changeable)
-- ============================================================================
SET VARIABLE n_groups = 10000;          -- Number of groups
SET VARIABLE n_obs_per_group = 100;     -- Observations per group
SET VARIABLE n_features = 8;            -- Number of features (x1, x2, ..., x8)
SET VARIABLE noise_std = 2.0;           -- Standard deviation of Gaussian noise
SET VARIABLE null_fraction = 0.1;       -- Fraction of y values to set as NULL (for prediction)

-- ============================================================================
-- STEP 1: Generate Random Coefficients for Each Group
-- ============================================================================
-- Each group gets its own "true" linear relationship with randomly sampled
-- coefficients. This creates heterogeneous data across groups.

.print '============================================================================'
.print 'Generating group-specific coefficients...'
.print '============================================================================'

CREATE OR REPLACE TABLE group_coefficients AS
SELECT
    group_id,
    -- Intercept: random value between -10 and 10
    (random() * 20.0 - 10.0) as beta_0,
    -- Coefficients for x1-x8: random values between -5 and 5
    (random() * 10.0 - 5.0) as beta_1,
    (random() * 10.0 - 5.0) as beta_2,
    (random() * 10.0 - 5.0) as beta_3,
    (random() * 10.0 - 5.0) as beta_4,
    (random() * 10.0 - 5.0) as beta_5,
    (random() * 10.0 - 5.0) as beta_6,
    (random() * 10.0 - 5.0) as beta_7,
    (random() * 10.0 - 5.0) as beta_8
FROM range(1, getvariable('n_groups') + 1) t(group_id);

.print 'Generated coefficients for ' || getvariable('n_groups') || ' groups'
.print ''

-- ============================================================================
-- STEP 2: Generate Sequential Observations for Each Group
-- ============================================================================
-- For each group, generate n_obs_per_group observations with:
-- - Sequential obs_id (important for window functions)
-- - Features x1-x8: random uniform values
-- - Response y: calculated from true linear relationship + noise
-- - Some y values set to NULL for prediction testing

.print '============================================================================'
.print 'Generating sequential observations...'
.print '============================================================================'

CREATE OR REPLACE TABLE performance_data AS
SELECT
    gc.group_id,
    obs.obs_id,
    -- Generate features x1 through x8 (uniform random between -10 and 10)
    (random() * 20.0 - 10.0)::DOUBLE as x1,
    (random() * 20.0 - 10.0)::DOUBLE as x2,
    (random() * 20.0 - 10.0)::DOUBLE as x3,
    (random() * 20.0 - 10.0)::DOUBLE as x4,
    (random() * 20.0 - 10.0)::DOUBLE as x5,
    (random() * 20.0 - 10.0)::DOUBLE as x6,
    (random() * 20.0 - 10.0)::DOUBLE as x7,
    (random() * 20.0 - 10.0)::DOUBLE as x8,
    -- Store coefficients for y calculation
    gc.beta_0, gc.beta_1, gc.beta_2, gc.beta_3, gc.beta_4,
    gc.beta_5, gc.beta_6, gc.beta_7, gc.beta_8
FROM
    group_coefficients gc
    CROSS JOIN range(1, getvariable('n_obs_per_group') + 1) obs(obs_id);

-- Add response variable y with true linear relationship + noise
-- Set some y values to NULL for prediction demonstration
CREATE OR REPLACE TABLE performance_data AS
SELECT
    group_id,
    obs_id,
    x1, x2, x3, x4, x5, x6, x7, x8,
    -- y = beta_0 + sum(beta_i * x_i) + noise, or NULL for prediction
    CASE
        WHEN random() < getvariable('null_fraction') THEN NULL
        ELSE (
            beta_0 +
            beta_1 * x1 +
            beta_2 * x2 +
            beta_3 * x3 +
            beta_4 * x4 +
            beta_5 * x5 +
            beta_6 * x6 +
            beta_7 * x7 +
            beta_8 * x8 +
            -- Gaussian noise using Box-Muller transform
            sqrt(-2.0 * ln(random())) * cos(2.0 * pi() * random()) * getvariable('noise_std')
        )::DOUBLE
    END as y,
    -- Store true y (without NULL) for validation
    (beta_0 +
     beta_1 * x1 +
     beta_2 * x2 +
     beta_3 * x3 +
     beta_4 * x4 +
     beta_5 * x5 +
     beta_6 * x6 +
     beta_7 * x7 +
     beta_8 * x8 +
     sqrt(-2.0 * ln(random())) * cos(2.0 * pi() * random()) * getvariable('noise_std')
    )::DOUBLE as y_true
FROM performance_data;

-- Report dataset size and NULL statistics
.print ''
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT group_id) as n_groups,
    COUNT(*) / COUNT(DISTINCT group_id) as obs_per_group,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_count,
    ROUND(100.0 * SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_percent
FROM performance_data;

.print ''
.print 'Dataset generated successfully!'
.print ''

-- ============================================================================
-- STEP 3: Performance Test - Expanding Window (Single Group)
-- ============================================================================
-- Fit-predict with expanding window on one group to test basic functionality

.print '============================================================================'
.print 'PERFORMANCE TEST 1: Expanding Window (Single Group)'
.print '============================================================================'
.print 'Mode: expanding - model refits on each new observation'
.timer on

CREATE OR REPLACE TABLE predictions_expanding_single AS
SELECT
    group_id,
    obs_id,
    y,
    x1, x2, x3, x4, x5, x6, x7, x8,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'expanding'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as pred
FROM performance_data
WHERE group_id = 1;

.timer off

-- Show sample predictions
.print ''
.print 'Sample predictions (first 10 rows):'
SELECT
    obs_id,
    y,
    pred.yhat as predicted_y,
    pred.yhat_lower as lower_bound,
    pred.yhat_upper as upper_bound,
    CASE WHEN y IS NULL THEN 'PREDICTED' ELSE 'FITTED' END as type
FROM predictions_expanding_single
WHERE obs_id <= 10
ORDER BY obs_id;

.print ''

-- ============================================================================
-- STEP 4: Performance Test - Fixed Window (Single Group)
-- ============================================================================
-- Fit-predict with fixed window on one group

.print '============================================================================'
.print 'PERFORMANCE TEST 2: Fixed Window (Single Group)'
.print '============================================================================'
.print 'Mode: fixed - fits once on training data, predicts all'
.timer on

CREATE OR REPLACE TABLE predictions_fixed_single AS
SELECT
    group_id,
    obs_id,
    y,
    x1, x2, x3, x4, x5, x6, x7, x8,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'fixed'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
    ) as pred
FROM performance_data
WHERE group_id = 1;

.timer off

.print ''
.print 'Sample predictions (rows with NULL y):'
SELECT
    obs_id,
    y,
    pred.yhat as predicted_y,
    pred.yhat_lower as lower_bound,
    pred.yhat_upper as upper_bound
FROM predictions_fixed_single
WHERE y IS NULL
ORDER BY obs_id
LIMIT 10;

.print ''

-- ============================================================================
-- STEP 5: Performance Test - Expanding Window (Multiple Groups)
-- ============================================================================
-- Fit-predict with expanding window on 100 groups

.print '============================================================================'
.print 'PERFORMANCE TEST 3: Expanding Window (100 Groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE predictions_expanding_multi AS
SELECT
    group_id,
    obs_id,
    y,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'expanding'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as pred
FROM performance_data
WHERE group_id <= 100;

.timer off

-- Report results
SELECT
    COUNT(*) as total_predictions,
    COUNT(DISTINCT group_id) as n_groups,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_predictions
FROM predictions_expanding_multi;

.print ''

-- ============================================================================
-- STEP 6: Performance Test - Fixed Window (Multiple Groups)
-- ============================================================================
-- Fit-predict with fixed window on 100 groups

.print '============================================================================'
.print 'PERFORMANCE TEST 4: Fixed Window (100 Groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE predictions_fixed_multi AS
SELECT
    group_id,
    obs_id,
    y,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'fixed'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
    ) as pred
FROM performance_data
WHERE group_id <= 100;

.timer off

-- Report results
SELECT
    COUNT(*) as total_predictions,
    COUNT(DISTINCT group_id) as n_groups,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_predictions
FROM predictions_fixed_multi;

.print ''

-- ============================================================================
-- STEP 7: Performance Test - Expanding Window (ALL Groups)
-- ============================================================================
-- Fit-predict with expanding window on ALL groups (most demanding test)

.print '============================================================================'
.print 'PERFORMANCE TEST 5: Expanding Window (ALL ' || getvariable('n_groups') || ' Groups)'
.print '============================================================================'
.print 'WARNING: This may take several minutes for 10,000 groups!'
.timer on

CREATE OR REPLACE TABLE predictions_expanding_all AS
SELECT
    group_id,
    obs_id,
    y,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'expanding'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as pred
FROM performance_data;

.timer off

-- Report results
SELECT
    COUNT(*) as total_predictions,
    COUNT(DISTINCT group_id) as n_groups,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_predictions,
    AVG(pred.yhat) as avg_prediction,
    STDDEV(pred.yhat) as stddev_prediction
FROM predictions_expanding_all;

.print ''

-- ============================================================================
-- STEP 8: Performance Test - Fixed Window (ALL Groups)
-- ============================================================================
-- Fit-predict with fixed window on ALL groups

.print '============================================================================'
.print 'PERFORMANCE TEST 6: Fixed Window (ALL ' || getvariable('n_groups') || ' Groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE predictions_fixed_all AS
SELECT
    group_id,
    obs_id,
    y,
    anofox_statistics_ols_fit_predict(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'fit_predict_mode': 'fixed'}
    ) OVER (
        PARTITION BY group_id
        ORDER BY obs_id
    ) as pred
FROM performance_data;

.timer off

-- Report results
SELECT
    COUNT(*) as total_predictions,
    COUNT(DISTINCT group_id) as n_groups,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_predictions,
    AVG(pred.yhat) as avg_prediction,
    STDDEV(pred.yhat) as stddev_prediction
FROM predictions_fixed_all;

.print ''

-- ============================================================================
-- STEP 9: Validation - Prediction Accuracy for NULL Values
-- ============================================================================
-- Compare predictions with true values (y_true) for NULL y observations

.print '============================================================================'
.print 'VALIDATION: Prediction Accuracy (Group 1, NULL values only)'
.print '============================================================================'

WITH validation AS (
    SELECT
        pd.obs_id,
        pd.y_true,
        pred.pred.yhat as predicted_y,
        pred.pred.yhat_lower,
        pred.pred.yhat_upper,
        ABS(pd.y_true - pred.pred.yhat) as abs_error,
        CASE
            WHEN pd.y_true BETWEEN pred.pred.yhat_lower AND pred.pred.yhat_upper
            THEN 1 ELSE 0
        END as in_interval
    FROM performance_data pd
    JOIN predictions_expanding_single pred
        ON pd.group_id = pred.group_id AND pd.obs_id = pred.obs_id
    WHERE pd.y IS NULL AND pd.group_id = 1
)
SELECT
    COUNT(*) as n_predictions,
    ROUND(AVG(abs_error), 4) as mean_abs_error,
    ROUND(STDDEV(abs_error), 4) as std_abs_error,
    ROUND(MIN(abs_error), 4) as min_abs_error,
    ROUND(MAX(abs_error), 4) as max_abs_error,
    ROUND(100.0 * AVG(in_interval), 2) as coverage_percent
FROM validation;

.print ''
.print 'Note: coverage_percent should be close to 95% for 95% prediction intervals'
.print ''

-- ============================================================================
-- STEP 10: Validation - Compare Fixed vs Expanding Window
-- ============================================================================
-- Compare predictions between fixed and expanding modes

.print '============================================================================'
.print 'VALIDATION: Fixed vs Expanding Window Predictions (Group 1)'
.print '============================================================================'

SELECT
    pf.obs_id,
    pf.y,
    pf.pred.yhat as fixed_prediction,
    pe.pred.yhat as expanding_prediction,
    ABS(pf.pred.yhat - pe.pred.yhat) as prediction_diff
FROM predictions_fixed_single pf
JOIN predictions_expanding_single pe
    ON pf.group_id = pe.group_id AND pf.obs_id = pe.obs_id
WHERE pf.obs_id <= 20
ORDER BY pf.obs_id;

.print ''
.print 'Note: Fixed mode predictions are often more stable (uses all training data)'
.print '      Expanding mode adapts as new data arrives (updates coefficients)'
.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
.print '============================================================================'
.print 'PERFORMANCE TEST SUMMARY'
.print '============================================================================'
.print 'Configuration:'
.print '  - Groups: ' || getvariable('n_groups')
.print '  - Observations per group: ' || getvariable('n_obs_per_group')
.print '  - Total rows: ' || (getvariable('n_groups') * getvariable('n_obs_per_group'))
.print '  - Features: ' || getvariable('n_features')
.print '  - Noise std dev: ' || getvariable('noise_std')
.print '  - NULL fraction: ' || getvariable('null_fraction')
.print ''
.print 'Tests completed:'
.print '  1. Expanding window - single group'
.print '  2. Fixed window - single group'
.print '  3. Expanding window - 100 groups'
.print '  4. Fixed window - 100 groups'
.print '  5. Expanding window - all groups'
.print '  6. Fixed window - all groups'
.print ''
.print 'Tables created:'
.print '  - group_coefficients: True coefficients for each group'
.print '  - performance_data: Sequential observations with NULL y values'
.print '  - predictions_expanding_single: Expanding window predictions (1 group)'
.print '  - predictions_fixed_single: Fixed window predictions (1 group)'
.print '  - predictions_expanding_multi: Expanding window predictions (100 groups)'
.print '  - predictions_fixed_multi: Fixed window predictions (100 groups)'
.print '  - predictions_expanding_all: Expanding window predictions (all groups)'
.print '  - predictions_fixed_all: Fixed window predictions (all groups)'
.print '============================================================================'
