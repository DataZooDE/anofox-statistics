-- ============================================================================
-- OLS Diagnostics Examples
-- ============================================================================
-- Demonstrates model diagnostic capabilities.
-- Topics: VIF (multicollinearity), Jarque-Bera (normality), Residuals, AIC/BIC
--
-- Run: ./build/release/duckdb < examples/ols_diagnostics.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Datasets
-- ============================================================================

-- Dataset with multicollinearity (correlated features)
CREATE OR REPLACE TABLE collinear_data AS
SELECT
    id,
    x1,
    -- x2 is highly correlated with x1
    x1 * 2.0 + (RANDOM() * 2 - 1) AS x2,
    -- x3 is independent
    RANDOM() * 100 AS x3,
    -- y depends on all three
    5.0 + 2.0 * x1 + 1.0 * (x1 * 2.0) + 0.5 * (RANDOM() * 100) + (RANDOM() * 10 - 5) AS y
FROM (
    SELECT id, id * 1.0 AS x1
    FROM generate_series(1, 50) AS t(id)
);

-- Dataset for normality testing (normal residuals)
CREATE OR REPLACE TABLE normal_data AS
SELECT
    id,
    x,
    -- y = 10 + 3*x + normal noise
    10.0 + 3.0 * x + (RANDOM() + RANDOM() + RANDOM() + RANDOM() - 2.0) * 5 AS y
FROM (
    SELECT id, id * 1.0 AS x
    FROM generate_series(1, 100) AS t(id)
);

-- Dataset for normality testing (skewed residuals)
CREATE OR REPLACE TABLE skewed_data AS
SELECT
    id,
    x,
    -- y = 10 + 3*x + skewed noise (always positive)
    10.0 + 3.0 * x + ABS(RANDOM() * 20) AS y
FROM (
    SELECT id, id * 1.0 AS x
    FROM generate_series(1, 100) AS t(id)
);

-- ============================================================================
-- Example 1: VIF for Multicollinearity Detection
-- ============================================================================

SELECT '=== Example 1: VIF Multicollinearity Detection ===' AS section;

-- VIF values for collinear features
-- VIF > 5 indicates moderate correlation
-- VIF > 10 indicates high correlation
SELECT
    vif([
        (SELECT LIST(x1) FROM collinear_data),
        (SELECT LIST(x2) FROM collinear_data),
        (SELECT LIST(x3) FROM collinear_data)
    ]) AS vif_values;

-- Interpret VIF values
WITH vif_results AS (
    SELECT UNNEST(vif([
        (SELECT LIST(x1) FROM collinear_data),
        (SELECT LIST(x2) FROM collinear_data),
        (SELECT LIST(x3) FROM collinear_data)
    ])) AS vif_value,
    UNNEST(['x1', 'x2', 'x3']) AS variable
)
SELECT
    variable,
    ROUND(vif_value, 2) AS vif,
    CASE
        WHEN vif_value < 5 THEN 'Low (OK)'
        WHEN vif_value < 10 THEN 'Moderate (Warning)'
        ELSE 'High (Problematic)'
    END AS multicollinearity
FROM vif_results;

-- ============================================================================
-- Example 2: VIF Aggregate Function
-- ============================================================================

SELECT '=== Example 2: VIF Aggregate ===' AS section;

SELECT
    vif_agg([x1, x2, x3]) AS vif_values
FROM collinear_data;

-- ============================================================================
-- Example 3: Jarque-Bera Normality Test
-- ============================================================================

SELECT '=== Example 3: Jarque-Bera Normality Test ===' AS section;

-- First, calculate residuals from OLS fit
WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true}
    )
),
residuals AS (
    SELECT
        y - (f.intercept + f.coefficients[1] * x) AS residual
    FROM normal_data, fitted f
)
SELECT
    'Normal data' AS dataset,
    ROUND((jb).statistic, 4) AS jb_statistic,
    ROUND((jb).p_value, 4) AS p_value,
    CASE
        WHEN (jb).p_value > 0.05 THEN 'Normal (fail to reject H0)'
        ELSE 'Non-normal (reject H0)'
    END AS interpretation
FROM (
    SELECT jarque_bera((SELECT LIST(residual) FROM residuals)) AS jb
);

-- Test with skewed data
WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM skewed_data),
        (SELECT LIST([x] ORDER BY id) FROM skewed_data),
        {'intercept': true}
    )
),
residuals AS (
    SELECT
        y - (f.intercept + f.coefficients[1] * x) AS residual
    FROM skewed_data, fitted f
)
SELECT
    'Skewed data' AS dataset,
    ROUND((jb).statistic, 4) AS jb_statistic,
    ROUND((jb).p_value, 4) AS p_value,
    CASE
        WHEN (jb).p_value > 0.05 THEN 'Normal (fail to reject H0)'
        ELSE 'Non-normal (reject H0)'
    END AS interpretation
FROM (
    SELECT jarque_bera((SELECT LIST(residual) FROM residuals)) AS jb
);

-- ============================================================================
-- Example 4: Jarque-Bera with Skewness and Kurtosis
-- ============================================================================

SELECT '=== Example 4: JB with Skewness/Kurtosis ===' AS section;

WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true}
    )
),
residuals AS (
    SELECT
        y - (f.intercept + f.coefficients[1] * x) AS residual
    FROM normal_data, fitted f
)
SELECT
    ROUND((jb).statistic, 4) AS jb_statistic,
    ROUND((jb).p_value, 4) AS p_value,
    ROUND((jb).skewness, 4) AS skewness,
    ROUND((jb).kurtosis, 4) AS kurtosis,
    ROUND((jb).n, 0) AS n_obs
FROM (
    SELECT jarque_bera((SELECT LIST(residual) FROM residuals)) AS jb
);

-- ============================================================================
-- Example 5: Jarque-Bera Aggregate Function
-- ============================================================================

SELECT '=== Example 5: JB Aggregate ===' AS section;

WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true}
    )
)
SELECT
    jarque_bera_agg(y - (f.intercept + f.coefficients[1] * x)) AS jb_result
FROM normal_data, fitted f;

-- ============================================================================
-- Example 6: AIC for Model Comparison
-- ============================================================================

SELECT '=== Example 6: AIC Model Comparison ===' AS section;

-- Create data for model comparison
CREATE OR REPLACE TABLE model_compare AS
SELECT
    id,
    x1,
    x1 * x1 AS x2,  -- quadratic term
    RANDOM() * 50 AS x3,  -- noise variable
    -- True model: y = 5 + 2*x1 + 0.1*x1^2 + noise
    5.0 + 2.0 * x1 + 0.1 * x1 * x1 + (RANDOM() * 10 - 5) AS y
FROM (
    SELECT id, id * 0.5 AS x1
    FROM generate_series(1, 50) AS t(id)
);

-- Model 1: Just x1 (underfitting)
WITH fit1 AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM model_compare),
        (SELECT LIST([x1] ORDER BY id) FROM model_compare),
        {'intercept': true}
    )
)
SELECT
    'Model 1: y ~ x1' AS model,
    ROUND(r2, 4) AS r_squared,
    ROUND(aic(mse * n_obs, n_obs, 2), 2) AS aic_value,
    ROUND(bic(mse * n_obs, n_obs, 2), 2) AS bic_value
FROM fit1
UNION ALL
-- Model 2: x1 + x2 (correct model)
SELECT
    'Model 2: y ~ x1 + x2' AS model,
    ROUND(r2, 4) AS r_squared,
    ROUND(aic(mse * n_obs, n_obs, 3), 2) AS aic_value,
    ROUND(bic(mse * n_obs, n_obs, 3), 2) AS bic_value
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM model_compare),
    (SELECT LIST([x1, x2] ORDER BY id) FROM model_compare),
    {'intercept': true}
)
UNION ALL
-- Model 3: x1 + x2 + x3 (overfitting)
SELECT
    'Model 3: y ~ x1 + x2 + x3' AS model,
    ROUND(r2, 4) AS r_squared,
    ROUND(aic(mse * n_obs, n_obs, 4), 2) AS aic_value,
    ROUND(bic(mse * n_obs, n_obs, 4), 2) AS bic_value
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM model_compare),
    (SELECT LIST([x1, x2, x3] ORDER BY id) FROM model_compare),
    {'intercept': true}
);

-- ============================================================================
-- Example 7: Residual Analysis
-- ============================================================================

SELECT '=== Example 7: Residual Analysis ===' AS section;

WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true}
    )
),
residuals AS (
    SELECT
        id,
        x,
        y,
        f.intercept + f.coefficients[1] * x AS y_hat,
        y - (f.intercept + f.coefficients[1] * x) AS residual
    FROM normal_data, fitted f
)
SELECT
    'Residual Statistics' AS metric,
    ROUND(AVG(residual), 4) AS mean,
    ROUND(STDDEV(residual), 4) AS std_dev,
    ROUND(MIN(residual), 4) AS min,
    ROUND(MAX(residual), 4) AS max
FROM residuals;

-- ============================================================================
-- Example 8: Standardized Residuals for Outlier Detection
-- ============================================================================

SELECT '=== Example 8: Outlier Detection ===' AS section;

WITH fitted AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true}
    )
),
residuals AS (
    SELECT
        id,
        x,
        y,
        y - (f.intercept + f.coefficients[1] * x) AS residual,
        f.residual_std_error AS se
    FROM normal_data, fitted f
)
SELECT
    id,
    ROUND(x, 2) AS x,
    ROUND(y, 2) AS y,
    ROUND(residual, 2) AS residual,
    ROUND(residual / se, 2) AS std_residual,
    CASE
        WHEN ABS(residual / se) > 3 THEN 'Outlier (|z| > 3)'
        WHEN ABS(residual / se) > 2 THEN 'Potential outlier (|z| > 2)'
        ELSE 'Normal'
    END AS status
FROM residuals
WHERE ABS(residual / se) > 2
ORDER BY ABS(residual / se) DESC;

-- ============================================================================
-- Example 9: Per-Group Diagnostics
-- ============================================================================

SELECT '=== Example 9: Per-Group Diagnostics ===' AS section;

CREATE OR REPLACE TABLE grouped_diag AS
SELECT
    category,
    id,
    x1,
    -- Different noise levels per category
    CASE category
        WHEN 'Low noise' THEN 10.0 + 2.0 * x1 + (RANDOM() * 2 - 1)
        WHEN 'High noise' THEN 10.0 + 2.0 * x1 + (RANDOM() * 20 - 10)
    END AS y,
    x1 * 2.0 + (RANDOM() * 2 - 1) AS x2  -- correlated feature
FROM (VALUES ('Low noise'), ('High noise')) AS c(category),
     (SELECT id, id * 1.0 AS x1 FROM generate_series(1, 30) AS t(id)) AS data;

-- Per-group VIF
SELECT
    category,
    vif_agg([x1, x2]) AS vif_values
FROM grouped_diag
GROUP BY category;

-- ============================================================================
-- Example 10: Complete Diagnostic Report
-- ============================================================================

SELECT '=== Example 10: Complete Diagnostic Report ===' AS section;

WITH fit AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM normal_data),
        (SELECT LIST([x] ORDER BY id) FROM normal_data),
        {'intercept': true, 'full_output': true}
    )
),
residuals AS (
    SELECT
        y - (f.intercept + f.coefficients[1] * x) AS residual
    FROM normal_data, fitted f
    LATERAL (SELECT * FROM fit) f
),
jb AS (
    SELECT jarque_bera((SELECT LIST(residual) FROM residuals)) AS result
)
SELECT '=== MODEL FIT ===' AS report
UNION ALL
SELECT 'R-squared: ' || ROUND((SELECT r2 FROM fit), 4)::VARCHAR
UNION ALL
SELECT 'Adj R-squared: ' || ROUND((SELECT adj_r2 FROM fit), 4)::VARCHAR
UNION ALL
SELECT 'Residual Std Error: ' || ROUND((SELECT residual_std_error FROM fit), 4)::VARCHAR
UNION ALL
SELECT 'F-statistic: ' || ROUND((SELECT f_statistic FROM fit), 4)::VARCHAR || ' (p=' || ROUND((SELECT f_statistic_pvalue FROM fit), 6)::VARCHAR || ')'
UNION ALL
SELECT ''
UNION ALL
SELECT '=== NORMALITY TEST ===' AS report
UNION ALL
SELECT 'Jarque-Bera: ' || ROUND((SELECT (result).statistic FROM jb), 4)::VARCHAR
UNION ALL
SELECT 'p-value: ' || ROUND((SELECT (result).p_value FROM jb), 4)::VARCHAR
UNION ALL
SELECT 'Skewness: ' || ROUND((SELECT (result).skewness FROM jb), 4)::VARCHAR
UNION ALL
SELECT 'Kurtosis: ' || ROUND((SELECT (result).kurtosis FROM jb), 4)::VARCHAR
UNION ALL
SELECT ''
UNION ALL
SELECT '=== MODEL SELECTION ===' AS report
UNION ALL
SELECT 'AIC: ' || ROUND(aic((SELECT mse * n_obs FROM fit), (SELECT n_obs FROM fit), 2), 2)::VARCHAR
UNION ALL
SELECT 'BIC: ' || ROUND(bic((SELECT mse * n_obs FROM fit), (SELECT n_obs FROM fit), 2), 2)::VARCHAR;

-- Cleanup
DROP TABLE IF EXISTS collinear_data;
DROP TABLE IF EXISTS normal_data;
DROP TABLE IF EXISTS skewed_data;
DROP TABLE IF EXISTS model_compare;
DROP TABLE IF EXISTS grouped_diag;
