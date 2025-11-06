-- Test Statistics Guide Examples

.bail on
.mode box

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT '========================================';
SELECT 'Testing Statistics Guide Examples';
SELECT '========================================';

-- Example: OLS Regression with aggregate function
SELECT '--- OLS Regression Example (aggregate function) ---';
DROP TABLE IF EXISTS sales_data;
CREATE TABLE sales_data AS
SELECT
    i as id,
    (100 - i * 2.0 + i * 0.5 + RANDOM() * 5)::DOUBLE as sales,
    i::DOUBLE as price,
    (i * 0.5)::DOUBLE as advertising
FROM range(1, 51) t(i);

SELECT
    (ols_fit_agg_array(sales, [price, advertising])).r2 as r_squared
FROM sales_data
LIMIT 1;

-- Example: Ridge Regression
SELECT '--- Ridge Regression Example ---';
SELECT * FROM anofox_statistics_ridge(
    [100.0, 98.0, 102.0, 97.0, 101.0]::DOUBLE[],
    [10.0, 9.8, 10.2, 9.7, 10.1]::DOUBLE[],
    [9.9, 9.7, 10.1, 9.8, 10.0]::DOUBLE[],
    0.1::DOUBLE,
    true::BOOLEAN
);

-- Example: WLS
SELECT '--- Weighted Least Squares Example ---';
SELECT * FROM anofox_statistics_wls(
    [50.0, 100.0, 150.0, 200.0, 250.0]::DOUBLE[],
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],
    true
);

-- Example: OLS Inference
SELECT '--- OLS Inference Example ---';
SELECT
    variable,
    estimate,
    std_error,
    t_statistic,
    p_value,
    significant
FROM ols_inference(
    [65.0, 72.0, 78.0, 85.0, 92.0, 88.0]::DOUBLE[],
    [[3.0, 7.0], [4.0, 8.0], [5.0, 7.0], [6.0, 8.0], [7.0, 9.0], [6.5, 7.5]]::DOUBLE[][],
    0.95,
    true
);

-- Example: Prediction Intervals
SELECT '--- Prediction Intervals Example ---';
SELECT
    predicted,
    ci_lower,
    ci_upper
FROM ols_predict_interval(
    [50.0, 55.0, 60.0, 65.0, 70.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    [[6.0], [7.0], [8.0]]::DOUBLE[][],
    0.95,
    'prediction',
    true
);

-- Example: Residual Diagnostics
SELECT '--- Residual Diagnostics Example ---';
SELECT
    obs_id,
    residual,
    std_residual,
    is_outlier
FROM anofox_statistics_residual_diagnostics(
    [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]::DOUBLE[][],
    true,
    2.5,
    0.5
)
LIMIT 3;

-- Example: VIF
SELECT '--- VIF Example ---';
SELECT
    variable_name,
    vif
FROM anofox_statistics_vif(
    [[10.0, 9.9, 10.1], [20.0, 19.8, 20.2], [30.0, 29.9, 30.1], [40.0, 39.7, 40.3]]::DOUBLE[][]
);

-- Example: Normality Test
SELECT '--- Normality Test Example ---';
SELECT * FROM anofox_statistics_normality_test(
    [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.0, 0.1, -0.1, 0.2]::DOUBLE[],
    0.05
);

-- Example: Information Criteria Model Comparison
SELECT '--- Information Criteria Example ---';
WITH model1 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],
        [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]::DOUBLE[][],
        true
    )
),
model2 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],
        [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0], [15.0, 10.0]]::DOUBLE[][],
        true
    )
)
SELECT
    'Model 1 (price only)' as model,
    aic, bic, r_squared FROM model1
UNION ALL
SELECT
    'Model 2 (price + ads)',
    aic, bic, r_squared FROM model2;

-- Example: RLS
SELECT '--- Recursive Least Squares Example ---';
SELECT * FROM anofox_statistics_rls(
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]::DOUBLE[],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],
    0.99::DOUBLE,
    true::BOOLEAN
);

-- Example: Rolling Window
SELECT '--- Rolling Window Example ---';
DROP TABLE IF EXISTS time_series;
CREATE TABLE time_series AS
SELECT
    i as date,
    (i * 1.5 + RANDOM() * 0.3)::DOUBLE as sales,
    i as price
FROM range(1, 51) t(i);

SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as rolling_elasticity
FROM time_series
WHERE date >= 31
LIMIT 5;

-- Example: Hypothesis Testing Workflow
SELECT '--- Hypothesis Testing Workflow ---';
SELECT
    variable,
    estimate as effect,
    p_value,
    CASE
        WHEN p_value < 0.001 THEN 'Highly significant ***'
        WHEN p_value < 0.01 THEN 'Very significant **'
        WHEN p_value < 0.05 THEN 'Significant *'
        ELSE 'Not significant'
    END as significance_level
FROM ols_inference(
    [100.0, 95.0, 92.0, 88.0, 85.0]::DOUBLE[],
    [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0]]::DOUBLE[][],
    0.95,
    true
)
WHERE variable = 'x1';

SELECT '========================================';
SELECT 'Statistics Guide: ALL TESTS PASSED';
SELECT '========================================';
