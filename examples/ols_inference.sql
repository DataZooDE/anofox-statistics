-- ============================================================================
-- OLS Statistical Inference Examples
-- ============================================================================
-- Demonstrates statistical inference capabilities: t-tests, p-values, CIs, F-test
-- Topics: Coefficient significance, confidence intervals, model significance
--
-- Run: ./build/release/duckdb < examples/ols_inference.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset
-- ============================================================================

CREATE OR REPLACE TABLE inference_data AS
SELECT
    id,
    -- y = 10 + 2.5*x1 + 0.1*x2 + noise
    -- x1 has strong effect, x2 has weak effect
    10.0 + 2.5 * x1 + 0.1 * x2 + (RANDOM() * 10 - 5) AS y,
    x1,
    x2
FROM (
    SELECT
        id,
        id * 1.0 AS x1,
        id * 10.0 + (RANDOM() * 50) AS x2
    FROM generate_series(1, 50) AS t(id)
);

-- ============================================================================
-- Example 1: Full Inference Output
-- ============================================================================

SELECT '=== Example 1: Full Inference Output ===' AS section;

SELECT
    -- Coefficients
    ROUND(intercept, 4) AS intercept,
    coefficients AS coefs,

    -- Standard errors
    ROUND(intercept_std_error, 4) AS intercept_se,
    coefficient_std_errors AS coef_se,

    -- t-statistics
    ROUND(intercept_t_value, 4) AS intercept_t,
    coefficient_t_values AS coef_t,

    -- p-values
    ROUND(intercept_p_value, 6) AS intercept_p,
    coefficient_p_values AS coef_p

FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true}
);

-- ============================================================================
-- Example 2: Confidence Intervals for Coefficients
-- ============================================================================

SELECT '=== Example 2: Confidence Intervals ===' AS section;

SELECT
    'Intercept' AS parameter,
    ROUND(intercept, 4) AS estimate,
    ROUND(intercept_ci_lower, 4) AS ci_lower,
    ROUND(intercept_ci_upper, 4) AS ci_upper,
    ROUND(intercept_ci_upper - intercept_ci_lower, 4) AS ci_width
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
)
UNION ALL
SELECT
    'x1' AS parameter,
    ROUND(coefficients[1], 4) AS estimate,
    ROUND(coefficient_ci_lower[1], 4) AS ci_lower,
    ROUND(coefficient_ci_upper[1], 4) AS ci_upper,
    ROUND(coefficient_ci_upper[1] - coefficient_ci_lower[1], 4) AS ci_width
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
)
UNION ALL
SELECT
    'x2' AS parameter,
    ROUND(coefficients[2], 4) AS estimate,
    ROUND(coefficient_ci_lower[2], 4) AS ci_lower,
    ROUND(coefficient_ci_upper[2], 4) AS ci_upper,
    ROUND(coefficient_ci_upper[2] - coefficient_ci_lower[2], 4) AS ci_width
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
);

-- ============================================================================
-- Example 3: Significance Stars (p-value interpretation)
-- ============================================================================

SELECT '=== Example 3: Significance Stars ===' AS section;

WITH fit AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM inference_data),
        (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
        {'intercept': true, 'full_output': true}
    )
)
SELECT
    'Intercept' AS variable,
    ROUND(intercept, 4) AS estimate,
    ROUND(intercept_p_value, 6) AS p_value,
    CASE
        WHEN intercept_p_value < 0.001 THEN '***'
        WHEN intercept_p_value < 0.01 THEN '**'
        WHEN intercept_p_value < 0.05 THEN '*'
        WHEN intercept_p_value < 0.1 THEN '.'
        ELSE ''
    END AS significance
FROM fit
UNION ALL
SELECT
    'x1' AS variable,
    ROUND(coefficients[1], 4) AS estimate,
    ROUND(coefficient_p_values[1], 6) AS p_value,
    CASE
        WHEN coefficient_p_values[1] < 0.001 THEN '***'
        WHEN coefficient_p_values[1] < 0.01 THEN '**'
        WHEN coefficient_p_values[1] < 0.05 THEN '*'
        WHEN coefficient_p_values[1] < 0.1 THEN '.'
        ELSE ''
    END AS significance
FROM fit
UNION ALL
SELECT
    'x2' AS variable,
    ROUND(coefficients[2], 4) AS estimate,
    ROUND(coefficient_p_values[2], 6) AS p_value,
    CASE
        WHEN coefficient_p_values[2] < 0.001 THEN '***'
        WHEN coefficient_p_values[2] < 0.01 THEN '**'
        WHEN coefficient_p_values[2] < 0.05 THEN '*'
        WHEN coefficient_p_values[2] < 0.1 THEN '.'
        ELSE ''
    END AS significance
FROM fit;

-- ============================================================================
-- Example 4: F-Statistic (Overall Model Significance)
-- ============================================================================

SELECT '=== Example 4: F-Statistic ===' AS section;

SELECT
    ROUND(f_statistic, 4) AS f_statistic,
    ROUND(f_statistic_pvalue, 8) AS f_pvalue,
    CASE
        WHEN f_statistic_pvalue < 0.001 THEN 'Highly significant (p < 0.001)'
        WHEN f_statistic_pvalue < 0.01 THEN 'Very significant (p < 0.01)'
        WHEN f_statistic_pvalue < 0.05 THEN 'Significant (p < 0.05)'
        ELSE 'Not significant (p >= 0.05)'
    END AS model_significance,
    df_residual AS df_residual,
    n_obs - df_residual - 1 AS df_model
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true}
);

-- ============================================================================
-- Example 5: Comparing Different Confidence Levels
-- ============================================================================

SELECT '=== Example 5: Confidence Level Comparison ===' AS section;

-- 90% CI
SELECT
    '90%' AS conf_level,
    ROUND(coefficients[1], 4) AS x1_estimate,
    ROUND(coefficient_ci_lower[1], 4) AS x1_ci_lower,
    ROUND(coefficient_ci_upper[1], 4) AS x1_ci_upper
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.90}
)
UNION ALL
-- 95% CI
SELECT
    '95%' AS conf_level,
    ROUND(coefficients[1], 4) AS x1_estimate,
    ROUND(coefficient_ci_lower[1], 4) AS x1_ci_lower,
    ROUND(coefficient_ci_upper[1], 4) AS x1_ci_upper
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
)
UNION ALL
-- 99% CI
SELECT
    '99%' AS conf_level,
    ROUND(coefficients[1], 4) AS x1_estimate,
    ROUND(coefficient_ci_lower[1], 4) AS x1_ci_lower,
    ROUND(coefficient_ci_upper[1], 4) AS x1_ci_upper
FROM ols_fit(
    (SELECT LIST(y ORDER BY id) FROM inference_data),
    (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
    {'intercept': true, 'full_output': true, 'confidence_level': 0.99}
);

-- ============================================================================
-- Example 6: Per-Group Inference
-- ============================================================================

SELECT '=== Example 6: Per-Group Inference ===' AS section;

CREATE OR REPLACE TABLE grouped_inference AS
SELECT
    category,
    id,
    -- Different true relationships per group
    CASE category
        WHEN 'A' THEN 5.0 + 3.0 * x + (RANDOM() * 4 - 2)
        WHEN 'B' THEN 20.0 + 0.5 * x + (RANDOM() * 8 - 4)
        WHEN 'C' THEN 10.0 + 1.5 * x + (RANDOM() * 2 - 1)
    END AS y,
    x
FROM (VALUES ('A'), ('B'), ('C')) AS c(category),
     (SELECT id, id * 1.0 AS x FROM generate_series(1, 20) AS t(id)) AS data;

SELECT
    category,
    ROUND(result.coefficients[1], 4) AS slope,
    ROUND(result.coefficient_std_errors[1], 4) AS slope_se,
    ROUND(result.coefficient_t_values[1], 4) AS slope_t,
    ROUND(result.coefficient_p_values[1], 6) AS slope_p,
    CASE
        WHEN result.coefficient_p_values[1] < 0.001 THEN '***'
        WHEN result.coefficient_p_values[1] < 0.01 THEN '**'
        WHEN result.coefficient_p_values[1] < 0.05 THEN '*'
        ELSE 'ns'
    END AS sig
FROM (
    SELECT
        category,
        ols_fit_agg(y, [x], {'intercept': true, 'full_output': true}) AS result
    FROM grouped_inference
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 7: Testing Hypothesis H0: beta = 0
-- ============================================================================

SELECT '=== Example 7: Hypothesis Testing ===' AS section;

WITH fit AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM inference_data),
        (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
        {'intercept': true, 'full_output': true}
    )
)
SELECT
    variable,
    estimate,
    std_error,
    t_value,
    p_value,
    CASE
        WHEN p_value < 0.05 THEN 'Reject H0: coefficient IS significantly different from 0'
        ELSE 'Fail to reject H0: coefficient is NOT significantly different from 0'
    END AS conclusion
FROM (
    SELECT 'x1' AS variable,
           ROUND(coefficients[1], 4) AS estimate,
           ROUND(coefficient_std_errors[1], 4) AS std_error,
           ROUND(coefficient_t_values[1], 4) AS t_value,
           ROUND(coefficient_p_values[1], 6) AS p_value
    FROM fit
    UNION ALL
    SELECT 'x2' AS variable,
           ROUND(coefficients[2], 4) AS estimate,
           ROUND(coefficient_std_errors[2], 4) AS std_error,
           ROUND(coefficient_t_values[2], 4) AS t_value,
           ROUND(coefficient_p_values[2], 6) AS p_value
    FROM fit
);

-- ============================================================================
-- Example 8: Standard Error and Sample Size Relationship
-- ============================================================================

SELECT '=== Example 8: Sample Size Effect on SE ===' AS section;

-- Fit with different sample sizes
WITH samples AS (
    SELECT
        n,
        ols_fit(
            (SELECT LIST(y ORDER BY id) FROM inference_data WHERE id <= n),
            (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data WHERE id <= n),
            {'intercept': true, 'full_output': true}
        ) AS result
    FROM (VALUES (10), (20), (30), (40), (50)) AS t(n)
)
SELECT
    n AS sample_size,
    ROUND((result).coefficients[1], 4) AS x1_estimate,
    ROUND((result).coefficient_std_errors[1], 4) AS x1_std_error,
    ROUND((result).coefficient_ci_upper[1] - (result).coefficient_ci_lower[1], 4) AS x1_ci_width
FROM samples
ORDER BY n;

-- ============================================================================
-- Example 9: Model Summary Table
-- ============================================================================

SELECT '=== Example 9: Model Summary Table ===' AS section;

WITH fit AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM inference_data),
        (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
    )
)
SELECT '--- Coefficients ---' AS section
UNION ALL
SELECT
    LPAD(variable, 12) || ' | ' ||
    LPAD(estimate::VARCHAR, 10) || ' | ' ||
    LPAD(std_err::VARCHAR, 10) || ' | ' ||
    LPAD(t_val::VARCHAR, 10) || ' | ' ||
    LPAD(p_val::VARCHAR, 12) || ' | ' ||
    sig
FROM (
    SELECT
        'Intercept' AS variable,
        ROUND(intercept, 4)::VARCHAR AS estimate,
        ROUND(intercept_std_error, 4)::VARCHAR AS std_err,
        ROUND(intercept_t_value, 4)::VARCHAR AS t_val,
        ROUND(intercept_p_value, 6)::VARCHAR AS p_val,
        CASE WHEN intercept_p_value < 0.001 THEN '***'
             WHEN intercept_p_value < 0.01 THEN '**'
             WHEN intercept_p_value < 0.05 THEN '*' ELSE '' END AS sig
    FROM fit
    UNION ALL
    SELECT
        'x1' AS variable,
        ROUND(coefficients[1], 4)::VARCHAR,
        ROUND(coefficient_std_errors[1], 4)::VARCHAR,
        ROUND(coefficient_t_values[1], 4)::VARCHAR,
        ROUND(coefficient_p_values[1], 6)::VARCHAR,
        CASE WHEN coefficient_p_values[1] < 0.001 THEN '***'
             WHEN coefficient_p_values[1] < 0.01 THEN '**'
             WHEN coefficient_p_values[1] < 0.05 THEN '*' ELSE '' END
    FROM fit
    UNION ALL
    SELECT
        'x2' AS variable,
        ROUND(coefficients[2], 4)::VARCHAR,
        ROUND(coefficient_std_errors[2], 4)::VARCHAR,
        ROUND(coefficient_t_values[2], 4)::VARCHAR,
        ROUND(coefficient_p_values[2], 6)::VARCHAR,
        CASE WHEN coefficient_p_values[2] < 0.001 THEN '***'
             WHEN coefficient_p_values[2] < 0.01 THEN '**'
             WHEN coefficient_p_values[2] < 0.05 THEN '*' ELSE '' END
    FROM fit
) summary
UNION ALL
SELECT '--- Model Statistics ---' AS section
UNION ALL
SELECT 'R-squared: ' || ROUND(r2, 4)::VARCHAR FROM fit
UNION ALL
SELECT 'Adj. R-squared: ' || ROUND(adj_r2, 4)::VARCHAR FROM fit
UNION ALL
SELECT 'F-statistic: ' || ROUND(f_statistic, 4)::VARCHAR || ' (p = ' || ROUND(f_statistic_pvalue, 6)::VARCHAR || ')' FROM fit
UNION ALL
SELECT 'Observations: ' || n_obs::VARCHAR FROM fit;

-- ============================================================================
-- Example 10: Checking if Zero is in Confidence Interval
-- ============================================================================

SELECT '=== Example 10: Zero in CI Check ===' AS section;

WITH fit AS (
    SELECT * FROM ols_fit(
        (SELECT LIST(y ORDER BY id) FROM inference_data),
        (SELECT LIST([x1, x2] ORDER BY id) FROM inference_data),
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
    )
)
SELECT
    variable,
    ROUND(estimate, 4) AS estimate,
    ROUND(ci_lower, 4) AS ci_lower,
    ROUND(ci_upper, 4) AS ci_upper,
    CASE
        WHEN ci_lower > 0 THEN 'Significant positive (0 not in CI)'
        WHEN ci_upper < 0 THEN 'Significant negative (0 not in CI)'
        ELSE 'Not significant (0 in CI)'
    END AS interpretation
FROM (
    SELECT 'x1' AS variable,
           coefficients[1] AS estimate,
           coefficient_ci_lower[1] AS ci_lower,
           coefficient_ci_upper[1] AS ci_upper
    FROM fit
    UNION ALL
    SELECT 'x2' AS variable,
           coefficients[2] AS estimate,
           coefficient_ci_lower[2] AS ci_lower,
           coefficient_ci_upper[2] AS ci_upper
    FROM fit
);

-- Cleanup
DROP TABLE IF EXISTS inference_data;
DROP TABLE IF EXISTS grouped_inference;
