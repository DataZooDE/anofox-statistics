-- Test all SQL examples from guides
-- This script tests each example systematically

.bail on
.mode box

-- Load extension
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT '========================================';
SELECT 'Testing Quick Start Guide Examples';
SELECT '========================================';

-- Example 1: Simple Linear Regression
SELECT '--- Example 1: Simple Linear Regression ---';
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],
    true
);

-- Example 2: Get p-values and Significance
SELECT '--- Example 2: Get p-values and Significance ---';
SELECT
    variable,
    ROUND(estimate, 4) as coefficient,
    ROUND(p_value, 4) as p_value,
    significant
FROM ols_inference(
    [2.1, 4.0, 6.1, 7.9, 10.2]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    0.95,
    true
);

-- Example 3: Regression Per Group
SELECT '--- Example 3: Regression Per Group ---';
DROP TABLE IF EXISTS sales;
CREATE TABLE sales AS
SELECT
    CASE WHEN i <= 10 THEN 'Product A' ELSE 'Product B' END as product,
    i::DOUBLE as price,
    (i * 2.0 + RANDOM() * 0.5)::DOUBLE as quantity
FROM range(1, 21) t(i);

SELECT
    product,
    (ols_fit_agg(quantity, price)).coefficient as price_elasticity,
    (ols_fit_agg(quantity, price)).r2 as fit_quality
FROM sales
GROUP BY product;

-- Example 4: Time-Series Rolling Regression
SELECT '--- Example 4: Time-Series Rolling Regression ---';
DROP TABLE IF EXISTS time_series;
CREATE TABLE time_series AS
SELECT
    i as time_idx,
    (i * 1.5 + RANDOM() * 0.3)::DOUBLE as value
FROM range(1, 51) t(i);

SELECT
    time_idx,
    value,
    ols_coeff_agg(value, time_idx) OVER (
        ORDER BY time_idx
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as rolling_trend
FROM time_series
WHERE time_idx >= 10
LIMIT 5;

-- Example 5: Make Predictions
SELECT '--- Example 5: Make Predictions ---';
SELECT * FROM ols_predict_interval(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    [[6.0], [7.0], [8.0]]::DOUBLE[][],
    0.95,
    'prediction',
    true
);

-- Example 6: Check Model Quality
SELECT '--- Example 6: Check Model Quality ---';
SELECT * FROM information_criteria(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 15.9]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][],
    true
);

-- Example 7: Detect Outliers
SELECT '--- Example 7: Detect Outliers ---';
SELECT
    obs_id,
    ROUND(residual, 3) as residual,
    ROUND(leverage, 3) as leverage,
    ROUND(cooks_distance, 3) as cooks_d,
    is_outlier,
    is_influential
FROM residual_diagnostics(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][],
    true,
    2.5,
    0.5
)
ORDER BY cooks_distance DESC
LIMIT 3;

-- Pattern 4: Full statistical workflow
SELECT '--- Pattern 4: Full statistical workflow ---';
WITH inference AS (
    SELECT * FROM ols_inference(
        [2.1, 4.0, 6.1, 7.9, 10.2, 11.8]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],
        0.95,
        true
    )
),
quality AS (
    SELECT * FROM information_criteria(
        [2.1, 4.0, 6.1, 7.9, 10.2, 11.8]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],
        true
    )
),
diagnostics AS (
    SELECT * FROM residual_diagnostics(
        [2.1, 4.0, 6.1, 7.9, 10.2, 11.8]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],
        true,
        2.5,
        0.5
    )
)
SELECT * FROM inference, quality, diagnostics;

SELECT '========================================';
SELECT 'Quick Start Guide: ALL TESTS PASSED';
SELECT '========================================';
