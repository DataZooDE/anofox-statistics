LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample data table
CREATE OR REPLACE TABLE workflow_sample AS
SELECT * FROM (VALUES
    (2.1, 1.0),
    (4.0, 2.0),
    (6.1, 3.0),
    (7.9, 4.0),
    (10.2, 5.0),
    (11.8, 6.0)
) AS t(y, x);

-- Complete statistical workflow
WITH
-- Step 1: Fit model and compute statistics
model_fit AS (
    SELECT
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as slope,
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).r2 as r2,
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).residual_standard_error as std_error,
        COUNT(*) as n_obs
    FROM workflow_sample
),

-- Step 2: Compute fitted values and residuals using the model
fitted_values AS (
    SELECT
        y,
        x,
        (SELECT slope FROM model_fit) * x as fitted,
        y - (SELECT slope FROM model_fit) * x as residual
    FROM workflow_sample
),

-- Step 3: Identify outliers based on residuals
outlier_check AS (
    SELECT
        y,
        x,
        residual,
        residual / (SELECT std_error FROM model_fit) as standardized_residual,
        ABS(residual / (SELECT std_error FROM model_fit)) > 2.5 as is_outlier
    FROM fitted_values
)

-- Display comprehensive results
SELECT
    'Model Summary' as section,
    'R²' as metric,
    ROUND(r2, 4)::VARCHAR as value
FROM model_fit
UNION ALL
SELECT 'Model Summary', 'Slope', ROUND(slope, 4)::VARCHAR FROM model_fit
UNION ALL
SELECT 'Model Summary', 'Std Error', ROUND(std_error, 4)::VARCHAR FROM model_fit
UNION ALL
SELECT 'Model Summary', 'N Observations', n_obs::VARCHAR FROM model_fit
UNION ALL
SELECT
    'Diagnostics',
    'Outliers Detected',
    COUNT(*)::VARCHAR
FROM outlier_check
WHERE is_outlier
UNION ALL
SELECT
    'Diagnostics',
    'Max |Standardized Residual|',
    ROUND(MAX(ABS(standardized_residual)), 2)::VARCHAR
FROM outlier_check
ORDER BY section, metric;
