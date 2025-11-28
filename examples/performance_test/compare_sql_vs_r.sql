-- ============================================================================
-- Compare SQL vs R Results
-- ============================================================================
-- This script compares the results from SQL (DuckDB) and R (lm) implementations
-- to validate that both produce equivalent results.
--
-- Prerequisites:
-- 1. Run generate_test_data.sql
-- 2. Run performance_test_ols_fit_predict.sql
-- 3. Run performance_test_ols_fit_predict.R
-- 4. Run performance_test_ols_aggregate.sql
-- 5. Run performance_test_ols_aggregate.R
-- ============================================================================

.print '============================================================================'
.print 'COMPARING SQL (DuckDB) vs R (lm) RESULTS'
.print '============================================================================'
.print ''

-- ============================================================================
-- PART 1: Compare Fit-Predict Results (Expanding Window, Single Group)
-- ============================================================================

.print '============================================================================'
.print 'PART 1: Fit-Predict Expanding Window (Group 1)'
.print '============================================================================'

-- Load both results
CREATE OR REPLACE TABLE sql_pred_exp AS
  SELECT * FROM 'examples/performance_test/results/sql_predictions_expanding_single.parquet';

CREATE OR REPLACE TABLE r_pred_exp AS
  SELECT * FROM 'examples/performance_test/results/r_predictions_expanding_single.parquet';

-- Compare predictions
.print ''
.print 'Statistical comparison of predictions:'

SELECT
    COUNT(*) as n_predictions,
    AVG(ABS(s.yhat - r.yhat)) as mean_abs_diff,
    MAX(ABS(s.yhat - r.yhat)) as max_abs_diff,
    STDDEV(s.yhat - r.yhat) as stddev_diff,
    SUM(CASE WHEN ABS(s.yhat - r.yhat) < 0.01 THEN 1 ELSE 0 END) as n_close_match,
    ROUND(100.0 * SUM(CASE WHEN ABS(s.yhat - r.yhat) < 0.01 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_close_match
FROM sql_pred_exp s
JOIN r_pred_exp r ON s.obs_id = r.obs_id
WHERE s.yhat IS NOT NULL AND r.yhat IS NOT NULL;

.print ''
.print 'Sample comparison (first 10 predictions):'

SELECT
    s.obs_id,
    ROUND(s.yhat, 4) as sql_yhat,
    ROUND(r.yhat, 4) as r_yhat,
    ROUND(ABS(s.yhat - r.yhat), 6) as abs_diff,
    CASE
        WHEN ABS(s.yhat - r.yhat) < 0.01 THEN 'MATCH'
        WHEN ABS(s.yhat - r.yhat) < 0.1 THEN 'CLOSE'
        ELSE 'DIFFER'
    END as status
FROM sql_pred_exp s
JOIN r_pred_exp r ON s.obs_id = r.obs_id
WHERE s.yhat IS NOT NULL AND r.yhat IS NOT NULL
ORDER BY s.obs_id
LIMIT 10;

.print ''

-- ============================================================================
-- PART 2: Compare Fit-Predict Results (Fixed Window, Single Group)
-- ============================================================================

.print '============================================================================'
.print 'PART 2: Fit-Predict Fixed Window (Group 1)'
.print '============================================================================'

-- Load both results
CREATE OR REPLACE TABLE sql_pred_fix AS
  SELECT * FROM 'examples/performance_test/results/sql_predictions_fixed_single.parquet';

CREATE OR REPLACE TABLE r_pred_fix AS
  SELECT * FROM 'examples/performance_test/results/r_predictions_fixed_single.parquet';

-- Compare predictions
.print ''
.print 'Statistical comparison of predictions:'

SELECT
    COUNT(*) as n_predictions,
    AVG(ABS(s.yhat - r.yhat)) as mean_abs_diff,
    MAX(ABS(s.yhat - r.yhat)) as max_abs_diff,
    STDDEV(s.yhat - r.yhat) as stddev_diff,
    SUM(CASE WHEN ABS(s.yhat - r.yhat) < 0.01 THEN 1 ELSE 0 END) as n_close_match,
    ROUND(100.0 * SUM(CASE WHEN ABS(s.yhat - r.yhat) < 0.01 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_close_match
FROM sql_pred_fix s
JOIN r_pred_fix r ON s.obs_id = r.obs_id
WHERE s.yhat IS NOT NULL AND r.yhat IS NOT NULL;

.print ''

-- ============================================================================
-- PART 3: Compare Aggregate Results (All Groups)
-- ============================================================================

.print '============================================================================'
.print 'PART 3: Aggregate Models (All Groups)'
.print '============================================================================'

-- Load both results
CREATE OR REPLACE TABLE sql_models AS
  SELECT * FROM 'examples/performance_test/results/sql_group_models.parquet';

CREATE OR REPLACE TABLE r_models AS
  SELECT * FROM 'examples/performance_test/results/r_group_models.parquet';

-- Compare model statistics
.print ''
.print 'Statistical comparison of model coefficients (intercept):'

SELECT
    COUNT(*) as n_groups,
    AVG(ABS(s.intercept - r.intercept)) as mean_abs_diff_intercept,
    MAX(ABS(s.intercept - r.intercept)) as max_abs_diff_intercept,
    STDDEV(s.intercept - r.intercept) as stddev_diff_intercept,
    SUM(CASE WHEN ABS(s.intercept - r.intercept) < 0.001 THEN 1 ELSE 0 END) as n_close_match,
    ROUND(100.0 * SUM(CASE WHEN ABS(s.intercept - r.intercept) < 0.001 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_close_match
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id;

.print ''
.print 'Statistical comparison of R-squared:'

SELECT
    COUNT(*) as n_groups,
    AVG(ABS(s.r2 - r.r2)) as mean_abs_diff_r2,
    MAX(ABS(s.r2 - r.r2)) as max_abs_diff_r2,
    STDDEV(s.r2 - r.r2) as stddev_diff_r2,
    SUM(CASE WHEN ABS(s.r2 - r.r2) < 0.0001 THEN 1 ELSE 0 END) as n_close_match,
    ROUND(100.0 * SUM(CASE WHEN ABS(s.r2 - r.r2) < 0.0001 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_close_match
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id;

.print ''
.print 'Sample comparison (group 1):'

SELECT
    'Intercept' as parameter,
    ROUND(s.intercept, 6) as sql_value,
    ROUND(r.intercept, 6) as r_value,
    ROUND(ABS(s.intercept - r.intercept), 9) as abs_diff
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'x1',
    ROUND(s.coefficients[1], 6),
    ROUND(r.coef_x1, 6),
    ROUND(ABS(s.coefficients[1] - r.coef_x1), 9)
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'x2',
    ROUND(s.coefficients[2], 6),
    ROUND(r.coef_x2, 6),
    ROUND(ABS(s.coefficients[2] - r.coef_x2), 9)
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'R-squared',
    ROUND(s.r2, 6),
    ROUND(r.r2, 6),
    ROUND(ABS(s.r2 - r.r2), 9)
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'Adj R-squared',
    ROUND(s.adj_r2, 6),
    ROUND(r.adj_r2, 6),
    ROUND(ABS(s.adj_r2 - r.adj_r2), 9)
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1;

.print ''

-- ============================================================================
-- PART 4: Compare Full Statistical Output
-- ============================================================================

.print '============================================================================'
.print 'PART 4: Full Statistical Output Comparison'
.print '============================================================================'

-- Load full output results
CREATE OR REPLACE TABLE sql_models_full AS
  SELECT * FROM 'examples/performance_test/results/sql_group_models_full.parquet';

CREATE OR REPLACE TABLE r_models_full AS
  SELECT * FROM 'examples/performance_test/results/r_group_models_full.parquet';

.print ''
.print 'Comparison of advanced statistics (group 1):'

SELECT
    'F-statistic' as statistic,
    ROUND(s.f_statistic, 6) as sql_value,
    ROUND(r.f_statistic, 6) as r_value,
    ROUND(ABS(s.f_statistic - r.f_statistic), 9) as abs_diff
FROM sql_models_full s
JOIN r_models_full r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'AIC',
    ROUND(s.aic, 6),
    ROUND(r.aic, 6),
    ROUND(ABS(s.aic - r.aic), 9)
FROM sql_models_full s
JOIN r_models_full r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'BIC',
    ROUND(s.bic, 6),
    ROUND(r.bic, 6),
    ROUND(ABS(s.bic - r.bic), 9)
FROM sql_models_full s
JOIN r_models_full r ON s.group_id = r.group_id
WHERE s.group_id = 1;

.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================

.print '============================================================================'
.print 'COMPARISON SUMMARY'
.print '============================================================================'
.print ''
.print 'Datasets compared:'
.print '  1. Fit-Predict Expanding Window (Single Group)'
.print '  2. Fit-Predict Fixed Window (Single Group)'
.print '  3. Aggregate Models (All Groups)'
.print '  4. Full Statistical Output'
.print ''
.print 'Interpretation:'
.print '  - abs_diff < 0.001: Excellent match (likely due to floating-point precision)'
.print '  - abs_diff < 0.01: Good match (acceptable numerical differences)'
.print '  - abs_diff >= 0.01: Potential discrepancy (investigate further)'
.print ''
.print 'Notes:'
.print '  - Small differences expected due to:'
.print '    * Different numerical libraries (DuckDB vs R)'
.print '    * Floating-point arithmetic precision'
.print '    * Different QR decomposition implementations'
.print '  - Large differences may indicate implementation bugs'
.print '============================================================================'
