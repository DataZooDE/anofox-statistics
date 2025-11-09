LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample data with multiple predictors
CREATE OR REPLACE TABLE model_comparison_data AS
WITH raw_data AS (
    SELECT
        i,
        (RANDOM() * 10)::DOUBLE as x1,
        (RANDOM() * 10)::DOUBLE as x2,
        (RANDOM() * 10)::DOUBLE as x3,  -- Noise
        (RANDOM() * 10)::DOUBLE as x4  -- Noise
    FROM range(1, 101) t(i)
)
SELECT
    i as obs_id,
    (10 + x1 * 2.5 + x2 * 0.5 + RANDOM() * 5)::DOUBLE as y,  -- x1 is strong, x2 is weak
    x1,
    x2,
    x3,
    x4
FROM raw_data;

-- Compare simple vs complex models using aggregate functions
-- Simple model: just x1
WITH simple_model AS (
    SELECT
        'Simple Model (x1 only)' as model_type,
        ROUND((ols_fit_agg(y, x1)).r2, 4) as r_squared,
        COUNT(*) as n_obs,
        1 as n_predictors
    FROM model_comparison_data
),
-- Complex model: analyze multiple predictors individually
complex_predictors AS (
    SELECT 'x1' as var, (ols_fit_agg(y, x1)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x2' as var, (ols_fit_agg(y, x2)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x3' as var, (ols_fit_agg(y, x3)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x4' as var, (ols_fit_agg(y, x4)).r2 as r2 FROM model_comparison_data
),
complex_summary AS (
    SELECT
        'Complex Model (all vars)' as model_type,
        ROUND(MAX(r2), 4) as r_squared,  -- Best predictor's R²
        (SELECT COUNT(*) FROM model_comparison_data) as n_obs,
        4 as n_predictors
    FROM complex_predictors
)
SELECT
    model_type,
    r_squared,
    n_predictors,
    CASE
        WHEN r_squared > 0.8 THEN 'Excellent'
        WHEN r_squared > 0.6 THEN 'Good'
        WHEN r_squared > 0.4 THEN 'Fair'
        ELSE 'Poor'
    END as model_quality
FROM simple_model
UNION ALL
SELECT model_type, r_squared, n_predictors,
    CASE
        WHEN r_squared > 0.8 THEN 'Excellent'
        WHEN r_squared > 0.6 THEN 'Good'
        WHEN r_squared > 0.4 THEN 'Fair'
        ELSE 'Poor'
    END as model_quality
FROM complex_summary
ORDER BY r_squared DESC;  -- Higher R² is better for comparison
