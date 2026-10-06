-- ============================================================================
-- OLS Statistical Inference Examples
-- ============================================================================
-- Demonstrates statistical inference: standard errors, t-tests, p-values,
-- confidence intervals, the overall F-test and heteroscedasticity-robust SEs.
--
-- Inference fields (std_errors, t_values, p_values, ci_lower, ci_upper,
-- f_statistic, f_pvalue) are only present with {'compute_inference': true}.
-- They refer to the slope coefficients; the intercept has no inference fields.
--
-- Run: ./build/release/duckdb < examples/ols_inference.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset
-- ============================================================================
-- y = 10 + 2.5*x1 + 0.1*x2 + noise  (x1 has a strong effect, x2 a weak one)
-- A deterministic pseudo-noise term keeps the output reproducible.

CREATE OR REPLACE TABLE inference_data AS
SELECT
    id,
    10.0 + 2.5 * x1 + 0.1 * x2 + (((id * 37) % 11) - 5.0) AS y,
    x1,
    x2,
    CASE WHEN id % 2 = 0 THEN 'even' ELSE 'odd' END AS segment
FROM (
    SELECT id, id * 1.0 AS x1, id * 10.0 + ((id * 53) % 50) AS x2
    FROM generate_series(1, 50) AS t(id)
);

-- ============================================================================
-- Example 1: Full Inference Output
-- ============================================================================

SELECT '=== Example 1: Full Inference Output ===' AS section;

SELECT
    ROUND(fit.intercept, 4)  AS intercept,
    fit.coefficients         AS coefs,
    fit.std_errors           AS coef_se,
    fit.t_values             AS coef_t,
    fit.p_values             AS coef_p
FROM (
    SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS fit
    FROM inference_data
);

-- ============================================================================
-- Example 2: Confidence Intervals for Coefficients
-- ============================================================================

SELECT '=== Example 2: Confidence Intervals ===' AS section;

WITH fit AS (
    SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'confidence_level': 0.95}) AS f
    FROM inference_data
)
SELECT
    name                                  AS parameter,
    ROUND(f.coefficients[i], 4)           AS estimate,
    ROUND(f.ci_lower[i], 4)               AS ci_lower,
    ROUND(f.ci_upper[i], 4)               AS ci_upper,
    ROUND(f.ci_upper[i] - f.ci_lower[i], 4) AS ci_width
FROM fit, (VALUES (1, 'x1'), (2, 'x2')) AS p(i, name)
ORDER BY i;

-- ============================================================================
-- Example 3: Significance Stars
-- ============================================================================

SELECT '=== Example 3: Significance Stars ===' AS section;

WITH fit AS (
    SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS f
    FROM inference_data
)
SELECT
    name                         AS parameter,
    ROUND(f.coefficients[i], 4)  AS estimate,
    ROUND(f.p_values[i], 6)      AS p_value,
    CASE WHEN f.p_values[i] < 0.001 THEN '***'
         WHEN f.p_values[i] < 0.01  THEN '**'
         WHEN f.p_values[i] < 0.05  THEN '*'
         WHEN f.p_values[i] < 0.1   THEN '.'
         ELSE '' END             AS significance
FROM fit, (VALUES (1, 'x1'), (2, 'x2')) AS p(i, name)
ORDER BY i;

-- ============================================================================
-- Example 4: Overall Model Significance (F-test)
-- ============================================================================

SELECT '=== Example 4: F-test ===' AS section;

SELECT
    ROUND(f.r_squared, 4)     AS r_squared,
    ROUND(f.adj_r_squared, 4) AS adj_r_squared,
    ROUND(f.f_statistic, 2)   AS f_statistic,
    f.f_pvalue                AS f_pvalue,
    f.n_observations          AS n
FROM (
    SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS f
    FROM inference_data
);

-- ============================================================================
-- Example 5: Different Confidence Levels
-- ============================================================================

SELECT '=== Example 5: Confidence Levels ===' AS section;

SELECT '90%' AS level,
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.90})).ci_lower[1], 4) AS lower,
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.90})).ci_upper[1], 4) AS upper
FROM inference_data
UNION ALL
SELECT '95%',
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.95})).ci_lower[1], 4),
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.95})).ci_upper[1], 4)
FROM inference_data
UNION ALL
SELECT '99%',
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.99})).ci_lower[1], 4),
       ROUND((ols_fit_agg(y, [x1], {'compute_inference': true, 'confidence_level': 0.99})).ci_upper[1], 4)
FROM inference_data;

-- ============================================================================
-- Example 6: Heteroscedasticity-Robust Standard Errors (HC0-HC3)
-- ============================================================================

SELECT '=== Example 6: Robust Standard Errors ===' AS section;

SELECT hc AS hc_type, se
FROM (
    SELECT 'none' AS hc, (ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': 'none'})).std_errors AS se FROM inference_data
    UNION ALL
    SELECT 'hc0', (ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': 'hc0'})).std_errors FROM inference_data
    UNION ALL
    SELECT 'hc1', (ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': 'hc1'})).std_errors FROM inference_data
    UNION ALL
    SELECT 'hc3', (ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': 'hc3'})).std_errors FROM inference_data
);

-- ============================================================================
-- Example 7: Per-Group Inference with GROUP BY
-- ============================================================================

SELECT '=== Example 7: Per-Group Inference ===' AS section;

SELECT
    segment,
    ROUND(f.coefficients[1], 4) AS x1_effect,
    ROUND(f.p_values[1], 6)     AS x1_p_value,
    f.p_values[1] < 0.05        AS x1_significant
FROM (
    SELECT segment, ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS f
    FROM inference_data
    GROUP BY segment
)
ORDER BY segment;

-- ============================================================================
-- Example 8: Scalar Form on Arrays (column-major x)
-- ============================================================================
-- ols_fit takes y DOUBLE[] and x DOUBLE[][] where each inner list is ONE
-- feature column across all observations.

SELECT '=== Example 8: Scalar ols_fit ===' AS section;

SELECT
    ROUND(f.coefficients[1], 4) AS slope,
    ROUND(f.p_values[1], 6)     AS p_value
FROM (
    SELECT ols_fit(
        LIST(y ORDER BY id),
        [LIST(x1 ORDER BY id)],
        {'compute_inference': true}
    ) AS f
    FROM inference_data
);

-- Cleanup
DROP TABLE IF EXISTS inference_data;
