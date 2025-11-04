-- Phase 2 Validation: Dataset 1 (Simple Synthetic y = 2x + 1)
-- Expected: intercept=1.0, slope=2.0, R²=1.0

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

.mode box
.width 12 12 12 14 12 12

SELECT '========================================';
SELECT 'DATASET 1: Simple Synthetic (y = 2x + 1)';
SELECT '========================================';
SELECT '';

-- Load data
CREATE TABLE simple_data AS SELECT * FROM read_csv_auto('validation/data/simple_test.csv');
CREATE TABLE simple_new AS SELECT * FROM read_csv_auto('validation/data/simple_test_new.csv');

SELECT 'Training Data:';
SELECT * FROM simple_data ORDER BY x;
SELECT '';

-- Test 1: OLS Inference
SELECT '========== TEST 1: OLS Inference ==========';
SELECT 'Expected: intercept=1.0, slope=2.0, both significant';
SELECT '';

SELECT * FROM ols_inference(
    [3.0, 5.0, 7.0, 9.0, 11.0]::DOUBLE[],  -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],  -- X
    0.95,   -- confidence_level
    true    -- add_intercept
) ORDER BY variable;

SELECT '';

-- Test 2: Prediction Intervals
SELECT '========== TEST 2: Prediction Intervals ==========';
SELECT 'Expected predictions: x=6 -> y=13, x=7 -> y=15';
SELECT '';

SELECT * FROM ols_predict_interval(
    [3.0, 5.0, 7.0, 9.0, 11.0]::DOUBLE[],  -- y_train
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],  -- X_train
    [[6.0], [7.0]]::DOUBLE[][],  -- X_new
    0.95,           -- confidence_level
    'prediction',   -- interval_type
    true            -- add_intercept
);

SELECT '';

-- Test 3: Residual Diagnostics
SELECT '========== TEST 3: Residual Diagnostics ==========';
SELECT 'Expected: All residuals ~0 (perfect fit), no outliers/influential points';
SELECT '';

SELECT * FROM residual_diagnostics(
    [3.0, 5.0, 7.0, 9.0, 11.0]::DOUBLE[],  -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],  -- X
    true,   -- add_intercept
    2.5,    -- outlier_threshold
    0.5     -- influence_threshold
);
