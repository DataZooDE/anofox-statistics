-- ============================================================================
-- Performance Test: OLS Fit-Predict Window Functions
-- ============================================================================
-- This script loads a pre-generated dataset and tests the performance of
-- anofox_statistics_ols_fit_predict window functions.
--
-- Dataset characteristics:
-- - Multiple groups with time-series style sequential data
-- - Each group has its own true coefficient vector
-- - Includes NULL values in y for prediction demonstration
-- - Sequential obs_id for proper window function ordering
--
-- Prerequisites:
-- 1. Run generate_test_data.sql first to create the parquet file
--
-- Usage:
-- 1. Run the entire script in DuckDB
-- 2. Observe timing output for performance analysis
-- 3. Results are saved to examples/performance_test/results/
-- ============================================================================

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- ============================================================================
-- STEP 1: Load Performance Data from Parquet File
-- ============================================================================

.print '============================================================================'
.print 'Loading performance data from parquet file...'
.print '============================================================================'

CREATE OR REPLACE TABLE performance_data AS
SELECT * FROM 'examples/performance_test/data/performance_data_fit_predict.parquet';

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
.print 'Dataset loaded successfully!'
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
-- STEP 11: Save Results to Parquet Files
-- ============================================================================

.print '============================================================================'
.print 'Saving results to parquet files...'
.print '============================================================================'

-- Save expanding window predictions (group 1)
COPY (
    SELECT
        group_id,
        obs_id,
        y,
        pred.yhat as yhat,
        pred.yhat_lower,
        pred.yhat_upper
    FROM predictions_expanding_single
) TO 'examples/performance_test/results/sql_predictions_expanding_single.parquet' (FORMAT PARQUET);

-- Save fixed window predictions (group 1)
COPY (
    SELECT
        group_id,
        obs_id,
        y,
        pred.yhat as yhat,
        pred.yhat_lower,
        pred.yhat_upper
    FROM predictions_fixed_single
) TO 'examples/performance_test/results/sql_predictions_fixed_single.parquet' (FORMAT PARQUET);

-- Save expanding window predictions (100 groups)
COPY (
    SELECT
        group_id,
        obs_id,
        y,
        pred.yhat as yhat,
        pred.yhat_lower,
        pred.yhat_upper
    FROM predictions_expanding_multi
) TO 'examples/performance_test/results/sql_predictions_expanding_multi.parquet' (FORMAT PARQUET);

-- Save fixed window predictions (100 groups)
COPY (
    SELECT
        group_id,
        obs_id,
        y,
        pred.yhat as yhat,
        pred.yhat_lower,
        pred.yhat_upper
    FROM predictions_fixed_multi
) TO 'examples/performance_test/results/sql_predictions_fixed_multi.parquet' (FORMAT PARQUET);

.print 'Results saved to examples/performance_test/results/'
.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
.print '============================================================================'
.print 'PERFORMANCE TEST SUMMARY'
.print '============================================================================'
.print 'Tests completed:'
.print '  1. Expanding window - single group'
.print '  2. Fixed window - single group'
.print '  3. Expanding window - 100 groups'
.print '  4. Fixed window - 100 groups'
.print '  5. Expanding window - all groups'
.print '  6. Fixed window - all groups'
.print ''
.print 'Tables created:'
.print '  - performance_data: Sequential observations with NULL y values'
.print '  - predictions_expanding_single: Expanding window predictions (1 group)'
.print '  - predictions_fixed_single: Fixed window predictions (1 group)'
.print '  - predictions_expanding_multi: Expanding window predictions (100 groups)'
.print '  - predictions_fixed_multi: Fixed window predictions (100 groups)'
.print '  - predictions_expanding_all: Expanding window predictions (all groups)'
.print '  - predictions_fixed_all: Fixed window predictions (all groups)'
.print ''
.print 'Results saved to:'
.print '  - examples/performance_test/results/sql_predictions_expanding_single.parquet'
.print '  - examples/performance_test/results/sql_predictions_fixed_single.parquet'
.print '  - examples/performance_test/results/sql_predictions_expanding_multi.parquet'
.print '  - examples/performance_test/results/sql_predictions_fixed_multi.parquet'
.print '============================================================================'
