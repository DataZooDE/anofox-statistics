-- ============================================================================
-- OLS Diagnostics Examples
-- ============================================================================
-- Demonstrates model diagnostics.
-- Topics: VIF (multicollinearity), Jarque-Bera (normality), residual analysis,
--         AIC/BIC model comparison
--
-- Run: ./build/release/duckdb < examples/ols_diagnostics.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Datasets (deterministic pseudo-noise for reproducibility)
-- ============================================================================

-- Dataset with multicollinearity: x2 is almost 2 * x1, x3 is unrelated
CREATE OR REPLACE TABLE collinear_data AS
SELECT
    id,
    x1,
    x1 * 2.0 + (((id * 7) % 5) - 2.0) * 0.3 AS x2,
    ((id * 31) % 97)::DOUBLE                AS x3,
    5.0 + 2.0 * x1 + 0.5 * ((id * 31) % 97) + (((id * 13) % 9) - 4.0) AS y
FROM (SELECT id, id * 1.0 AS x1 FROM generate_series(1, 50) AS t(id));

-- Dataset with symmetric noise and one with skewed (always positive) noise
CREATE OR REPLACE TABLE noise_data AS
SELECT
    id,
    id * 1.0 AS x,
    10.0 + 3.0 * id + (((id * 37) % 11) - 5.0)           AS y_normal,
    10.0 + 3.0 * id + POW(((id * 37) % 11) / 2.0, 3)     AS y_skewed
FROM generate_series(1, 100) AS t(id);

-- ============================================================================
-- Example 1: VIF for Multicollinearity Detection (scalar, column-major)
-- ============================================================================

SELECT '=== Example 1: VIF (scalar) ===' AS section;

-- VIF > 5 indicates moderate, VIF > 10 high multicollinearity
WITH v AS (
    SELECT vif([LIST(x1 ORDER BY id), LIST(x2 ORDER BY id), LIST(x3 ORDER BY id)]) AS vifs
    FROM collinear_data
)
SELECT
    name                      AS variable,
    ROUND(vifs[i], 2)         AS vif,
    CASE WHEN vifs[i] < 5  THEN 'Low (OK)'
         WHEN vifs[i] < 10 THEN 'Moderate (Warning)'
         ELSE 'High (Problematic)' END AS multicollinearity
FROM v, (VALUES (1, 'x1'), (2, 'x2'), (3, 'x3')) AS p(i, name)
ORDER BY i;

-- ============================================================================
-- Example 2: VIF Aggregate Function
-- ============================================================================

SELECT '=== Example 2: VIF Aggregate ===' AS section;

SELECT vif_agg([x1, x2, x3]) AS vif_values FROM collinear_data;

-- ============================================================================
-- Example 3: Jarque-Bera Normality Test on Residuals
-- ============================================================================
-- Residuals come from ols_fit_predict_by, which returns every row with yhat.

SELECT '=== Example 3: Jarque-Bera Normality Test ===' AS section;

CREATE OR REPLACE TABLE noise_long AS
SELECT 'normal' AS kind, id, x, y_normal AS y FROM noise_data
UNION ALL
SELECT 'skewed', id, x, y_skewed FROM noise_data;

SELECT
    kind,
    ROUND((jarque_bera_agg(y - yhat)).statistic, 3) AS jb_statistic,
    ROUND((jarque_bera_agg(y - yhat)).p_value, 6)   AS p_value,
    ROUND((jarque_bera_agg(y - yhat)).skewness, 3)  AS skewness,
    CASE WHEN (jarque_bera_agg(y - yhat)).p_value > 0.05
         THEN 'Consistent with normality' ELSE 'Non-normal residuals' END AS conclusion
FROM ols_fit_predict_by('noise_long', kind, y, [x])
GROUP BY kind
ORDER BY kind;

-- Scalar form on an array
SELECT (jarque_bera([1.2, 0.8, -0.3, 0.1, -1.1, 0.4, -0.2, 0.9, -0.7, 0.0])).p_value AS jb_p_value;

-- ============================================================================
-- Example 4: Residual Diagnostics (standardized, studentized, leverage)
-- ============================================================================
-- The scalar form residuals_diagnostics(y, y_hat, x, residual_std_error,
-- include_studentized) returns standardized and studentized residuals and
-- leverage. x is column-major (one inner list per feature).

SELECT '=== Example 4: Residual Diagnostics ===' AS section;

WITH p AS (
    SELECT id, x, y, yhat
    FROM ols_fit_predict_by('noise_long', kind, y, [x])
    WHERE kind = 'normal'
),
s AS (
    SELECT (ols_fit_agg(y, [x])).residual_std_error AS rse FROM p
),
d AS (
    SELECT residuals_diagnostics(
        LIST(y ORDER BY id), LIST(yhat ORDER BY id), [LIST(x ORDER BY id)],
        (SELECT rse FROM s), true
    ) AS r
    FROM p
)
SELECT
    ROUND(list_max(list_transform(r.standardized, lambda v: abs(v))), 3) AS max_abs_standardized,
    ROUND(list_max(r.leverage), 4)                                       AS max_leverage,
    len(list_filter(r.studentized, lambda v: abs(v) > 2))                AS n_flagged_abs_gt_2
FROM d;

-- ============================================================================
-- Example 5: AIC / BIC Model Comparison
-- ============================================================================
-- aic(rss, n, k) and bic(rss, n, k); RSS = residual_std_error^2 * (n - k).

SELECT '=== Example 5: AIC/BIC Model Comparison ===' AS section;

WITH fits AS (
    SELECT 'y ~ x1'           AS model, 2 AS k, ols_fit_agg(y, [x1])         AS f FROM collinear_data
    UNION ALL
    SELECT 'y ~ x1 + x3',     3,        ols_fit_agg(y, [x1, x3])                FROM collinear_data
    UNION ALL
    SELECT 'y ~ x1 + x2 + x3', 4,       ols_fit_agg(y, [x1, x2, x3])            FROM collinear_data
)
SELECT
    model,
    ROUND(f.r_squared, 4)     AS r_squared,
    ROUND(f.adj_r_squared, 4) AS adj_r_squared,
    ROUND(aic(POW(f.residual_std_error, 2) * (f.n_observations - k), f.n_observations, k), 2) AS aic,
    ROUND(bic(POW(f.residual_std_error, 2) * (f.n_observations - k), f.n_observations, k), 2) AS bic
FROM fits
ORDER BY aic;

-- Cleanup
DROP TABLE IF EXISTS collinear_data;
DROP TABLE IF EXISTS noise_data;
DROP TABLE IF EXISTS noise_long;
