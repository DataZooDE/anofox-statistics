-- ============================================================================
-- OLS Statistical Inference Examples
-- ============================================================================
-- Demonstrates statistical inference capabilities: t-tests, p-values, CIs, F-test
-- Topics: Coefficient significance, confidence intervals, model significance
--
-- NOTE: Inference arrays (std_errors, t_values, p_values, ci_lower, ci_upper)
-- contain values for coefficients only, NOT for intercept.
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
    ROUND(result.intercept, 4) AS intercept,
    result.coefficients AS coefs,

    -- Standard errors (for coefficients only)
    result.std_errors AS coef_se,

    -- t-statistics (for coefficients only)
    result.t_values AS coef_t,

    -- p-values (for coefficients only)
    result.p_values AS coef_p
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true}
    ) AS result
);

-- ============================================================================
-- Example 2: Confidence Intervals for Coefficients
-- ============================================================================

SELECT '=== Example 2: Confidence Intervals ===' AS section;

-- ci_lower/ci_upper arrays contain CIs for coefficients only
SELECT
    'x1' AS parameter,
    ROUND(result.coefficients[1], 4) AS estimate,
    ROUND(result.ci_lower[1], 4) AS ci_lower,
    ROUND(result.ci_upper[1], 4) AS ci_upper,
    ROUND(result.ci_upper[1] - result.ci_lower[1], 4) AS ci_width
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
);

-- ============================================================================
-- Example 3: Significance Stars (p-value interpretation)
-- ============================================================================

SELECT '=== Example 3: Significance Stars ===' AS section;

WITH fit AS (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true}
    ) AS result
)
SELECT
    'x1' AS variable,
    ROUND(result.coefficients[1], 4) AS estimate,
    ROUND(result.p_values[1], 6) AS p_value,
    CASE
        WHEN result.p_values[1] < 0.001 THEN '***'
        WHEN result.p_values[1] < 0.01 THEN '**'
        WHEN result.p_values[1] < 0.05 THEN '*'
        WHEN result.p_values[1] < 0.1 THEN '.'
        ELSE ''
    END AS significance
FROM fit;

-- ============================================================================
-- Example 4: F-Statistic (Overall Model Significance)
-- ============================================================================

SELECT '=== Example 4: F-Statistic ===' AS section;

SELECT
    ROUND(result.f_statistic, 4) AS f_statistic,
    ROUND(result.f_pvalue, 8) AS f_pvalue,
    CASE
        WHEN result.f_pvalue < 0.001 THEN 'Highly significant (p < 0.001)'
        WHEN result.f_pvalue < 0.01 THEN 'Very significant (p < 0.01)'
        WHEN result.f_pvalue < 0.05 THEN 'Significant (p < 0.05)'
        ELSE 'Not significant (p >= 0.05)'
    END AS model_significance
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true}
    ) AS result
);

-- ============================================================================
-- Example 5: Comparing Different Confidence Levels
-- ============================================================================

SELECT '=== Example 5: Confidence Level Comparison ===' AS section;

-- 90% CI
SELECT
    '90%' AS conf_level,
    ROUND(result.coefficients[1], 4) AS x1_estimate,
    ROUND(result.ci_lower[1], 4) AS x1_ci_lower,
    ROUND(result.ci_upper[1], 4) AS x1_ci_upper
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.90}
    ) AS result
)
UNION ALL
-- 95% CI
SELECT
    '95%' AS conf_level,
    ROUND(result.coefficients[1], 4) AS x1_estimate,
    ROUND(result.ci_lower[1], 4) AS x1_ci_lower,
    ROUND(result.ci_upper[1], 4) AS x1_ci_upper
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
)
UNION ALL
-- 99% CI
SELECT
    '99%' AS conf_level,
    ROUND(result.coefficients[1], 4) AS x1_estimate,
    ROUND(result.ci_lower[1], 4) AS x1_ci_lower,
    ROUND(result.ci_upper[1], 4) AS x1_ci_upper
FROM (
    SELECT anofox_stats_ols_fit(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]::DOUBLE[],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.99}
    ) AS result
);

-- ============================================================================
-- Example 6: Per-Group Inference with Aggregate Function
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
    ROUND(result.std_errors[1], 4) AS slope_se,
    ROUND(result.t_values[1], 4) AS slope_t,
    ROUND(result.p_values[1], 6) AS slope_p,
    CASE
        WHEN result.p_values[1] < 0.001 THEN '***'
        WHEN result.p_values[1] < 0.01 THEN '**'
        WHEN result.p_values[1] < 0.05 THEN '*'
        ELSE 'ns'
    END AS sig
FROM (
    SELECT
        category,
        ols_fit_agg(y, [x], {'intercept': true, 'compute_inference': true}) AS result
    FROM grouped_inference
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 7: Testing Hypothesis H0: beta = 0
-- ============================================================================

SELECT '=== Example 7: Hypothesis Testing ===' AS section;

WITH fit AS (
    SELECT anofox_stats_ols_fit(
        [15.0, 22.0, 31.0, 38.0, 45.0, 54.0, 61.0, 70.0]::DOUBLE[],
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0]
        ]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true}
    ) AS result
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
           ROUND(result.coefficients[1], 4) AS estimate,
           ROUND(result.std_errors[1], 4) AS std_error,
           ROUND(result.t_values[1], 4) AS t_value,
           ROUND(result.p_values[1], 6) AS p_value
    FROM fit
    UNION ALL
    SELECT 'x2' AS variable,
           ROUND(result.coefficients[2], 4) AS estimate,
           ROUND(result.std_errors[2], 4) AS std_error,
           ROUND(result.t_values[2], 4) AS t_value,
           ROUND(result.p_values[2], 6) AS p_value
    FROM fit
);

-- ============================================================================
-- Example 8: Model Summary Statistics
-- ============================================================================

SELECT '=== Example 8: Model Summary Statistics ===' AS section;

SELECT
    ROUND(result.r_squared, 4) AS r_squared,
    ROUND(result.adj_r_squared, 4) AS adj_r_squared,
    ROUND(result.residual_std_error, 4) AS residual_std_error,
    result.n_observations,
    result.n_features,
    ROUND(result.f_statistic, 4) AS f_statistic,
    ROUND(result.f_pvalue, 6) AS f_pvalue
FROM (
    SELECT anofox_stats_ols_fit(
        [15.0, 22.0, 31.0, 38.0, 45.0, 54.0, 61.0, 70.0]::DOUBLE[],
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0]
        ]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true}
    ) AS result
);

-- ============================================================================
-- Example 9: Coefficient Summary Table
-- ============================================================================

SELECT '=== Example 9: Coefficient Summary Table ===' AS section;

WITH fit AS (
    SELECT anofox_stats_ols_fit(
        [15.0, 22.0, 31.0, 38.0, 45.0, 54.0, 61.0, 70.0]::DOUBLE[],
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0]
        ]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
)
SELECT
    variable,
    estimate,
    std_error,
    t_value,
    p_value,
    sig
FROM (
    SELECT
        'x1' AS variable,
        ROUND(result.coefficients[1], 4) AS estimate,
        ROUND(result.std_errors[1], 4) AS std_error,
        ROUND(result.t_values[1], 4) AS t_value,
        ROUND(result.p_values[1], 6) AS p_value,
        CASE WHEN result.p_values[1] < 0.001 THEN '***'
             WHEN result.p_values[1] < 0.01 THEN '**'
             WHEN result.p_values[1] < 0.05 THEN '*' ELSE '' END AS sig
    FROM fit
    UNION ALL
    SELECT
        'x2' AS variable,
        ROUND(result.coefficients[2], 4),
        ROUND(result.std_errors[2], 4),
        ROUND(result.t_values[2], 4),
        ROUND(result.p_values[2], 6),
        CASE WHEN result.p_values[2] < 0.001 THEN '***'
             WHEN result.p_values[2] < 0.01 THEN '**'
             WHEN result.p_values[2] < 0.05 THEN '*' ELSE '' END
    FROM fit
);

-- ============================================================================
-- Example 10: Checking if Zero is in Confidence Interval
-- ============================================================================

SELECT '=== Example 10: Zero in CI Check ===' AS section;

WITH fit AS (
    SELECT anofox_stats_ols_fit(
        [15.0, 22.0, 31.0, 38.0, 45.0, 54.0, 61.0, 70.0]::DOUBLE[],
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0]
        ]::DOUBLE[][],
        {'intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) AS result
)
SELECT
    variable,
    ROUND(estimate, 4) AS estimate,
    ROUND(ci_low, 4) AS ci_lower,
    ROUND(ci_high, 4) AS ci_upper,
    CASE
        WHEN ci_low > 0 THEN 'Significant positive (0 not in CI)'
        WHEN ci_high < 0 THEN 'Significant negative (0 not in CI)'
        ELSE 'Not significant (0 in CI)'
    END AS interpretation
FROM (
    SELECT 'x1' AS variable,
           result.coefficients[1] AS estimate,
           result.ci_lower[1] AS ci_low,
           result.ci_upper[1] AS ci_high
    FROM fit
    UNION ALL
    SELECT 'x2' AS variable,
           result.coefficients[2] AS estimate,
           result.ci_lower[2] AS ci_low,
           result.ci_upper[2] AS ci_high
    FROM fit
);

-- Cleanup
DROP TABLE IF EXISTS inference_data;
DROP TABLE IF EXISTS grouped_inference;
