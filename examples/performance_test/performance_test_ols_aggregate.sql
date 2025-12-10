-- ============================================================================
-- Performance Test: OLS Aggregate Functions with GROUP BY
-- ============================================================================
-- This script loads a pre-generated dataset and tests the performance of
-- anofox_stats_ols_fit_agg with GROUP BY functionality.
--
-- Dataset characteristics:
-- - Multiple groups with heterogeneous linear relationships
-- - Each group has its own true coefficient vector
-- - Gaussian noise added to simulate realistic data
-- - No NULL values (required for aggregate functions)
--
-- Prerequisites:
-- 1. Run generate_test_data.sql first to create the parquet file
--
-- Usage:
-- 1. Run the entire script in DuckDB
-- 2. Observe timing output for performance analysis
-- 3. Results are saved to examples/performance_test/results/
-- ============================================================================

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- ============================================================================
-- STEP 1: Load Performance Data from Parquet File
-- ============================================================================

.print '============================================================================'
.print 'Loading performance data from parquet file...'
.print '============================================================================'

CREATE OR REPLACE TABLE performance_data AS
SELECT * FROM 'examples/performance_test/data/performance_data_aggregate.parquet';

-- Report dataset size
.print ''
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT group_id) as n_groups,
    COUNT(*) / COUNT(DISTINCT group_id) as obs_per_group
FROM performance_data;

.print ''
.print 'Dataset loaded successfully!'
.print ''

-- ============================================================================
-- STEP 3: Performance Test - GROUP BY with All Groups
-- ============================================================================
-- Fit OLS model for each group using anofox_stats_ols_fit_agg

.print '============================================================================'
.print 'PERFORMANCE TEST 1: GROUP BY Aggregation (All Groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE group_models AS
SELECT
    group_id,
    anofox_stats_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        {'intercept': true, 'confidence_level': 0.95}
    ) as model
FROM performance_data
GROUP BY group_id;

.timer off

-- Display sample results
.print ''
.print 'Sample results (first 5 groups):'
SELECT
    group_id,
    model.coefficients as coefficients,
    model.r2 as r_squared,
    model.adj_r2 as adj_r_squared,
    model.n_obs as n_obs
FROM group_models
WHERE group_id <= 5
ORDER BY group_id;

.print ''

-- ============================================================================
-- STEP 4: Performance Test - GROUP BY with Full Statistical Output
-- ============================================================================
-- Same as above but with full_output=true for comprehensive statistics

.print '============================================================================'
.print 'PERFORMANCE TEST 2: GROUP BY with Full Statistical Output'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE group_models_full AS
SELECT
    group_id,
    anofox_stats_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        {'intercept': true, 'confidence_level': 0.95, 'full_output': true}
    ) as model
FROM performance_data
GROUP BY group_id;

.timer off

-- Display comprehensive statistics for first group
.print ''
.print 'Sample comprehensive output (group 1):'
SELECT
    model.coefficients as coefficients,
    model.coefficient_std_errors as std_errors,
    model.coefficient_t_statistics as t_statistics,
    model.coefficient_p_values as p_values,
    model.r2 as r_squared,
    model.adj_r2 as adj_r_squared,
    model.f_statistic as f_statistic,
    model.aic as aic,
    model.bic as bic
FROM group_models_full
WHERE group_id = 1;

.print ''

-- ============================================================================
-- STEP 5: Performance Test - Subset of Groups
-- ============================================================================
-- Test performance on smaller subset (100 groups)

.print '============================================================================'
.print 'PERFORMANCE TEST 3: GROUP BY on Subset (100 groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE subset_models AS
SELECT
    group_id,
    anofox_stats_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        {'intercept': true}
    ) as model
FROM performance_data
WHERE group_id <= 100
GROUP BY group_id;

.timer off

.print 'Fitted models for 100 groups'
.print ''

-- ============================================================================
-- STEP 6: Validation - Compare with True Coefficients
-- ============================================================================
-- Check if estimated coefficients are close to true coefficients for first group

.print '============================================================================'
.print 'VALIDATION: Estimated vs True Coefficients (Group 1)'
.print '============================================================================'

SELECT
    'Intercept' as parameter,
    gc.beta_0 as true_value,
    gm.model.intercept as estimated_value,
    gm.model.intercept - gc.beta_0 as error
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x1',
    gc.beta_1,
    gm.model.coefficients[1],
    gm.model.coefficients[1] - gc.beta_1
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x2',
    gc.beta_2,
    gm.model.coefficients[2],
    gm.model.coefficients[2] - gc.beta_2
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x3',
    gc.beta_3,
    gm.model.coefficients[3],
    gm.model.coefficients[3] - gc.beta_3
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x4',
    gc.beta_4,
    gm.model.coefficients[4],
    gm.model.coefficients[4] - gc.beta_4
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x5',
    gc.beta_5,
    gm.model.coefficients[5],
    gm.model.coefficients[5] - gc.beta_5
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x6',
    gc.beta_6,
    gm.model.coefficients[6],
    gm.model.coefficients[6] - gc.beta_6
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x7',
    gc.beta_7,
    gm.model.coefficients[7],
    gm.model.coefficients[7] - gc.beta_7
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x8',
    gc.beta_8,
    gm.model.coefficients[8],
    gm.model.coefficients[8] - gc.beta_8
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1;

.print ''
.print 'Note: Small errors are expected due to random noise in the data'
.print ''

-- ============================================================================
-- STEP 7: Save Results to Parquet Files
-- ============================================================================

.print '============================================================================'
.print 'Saving results to parquet files...'
.print '============================================================================'

-- Save basic model results (all groups)
COPY (
    SELECT
        group_id,
        model.intercept,
        model.coefficients,
        model.r2,
        model.adj_r2,
        model.n_obs,
        model.df_model,
        model.df_residual
    FROM group_models
) TO 'examples/performance_test/results/sql_group_models.parquet' (FORMAT PARQUET);

-- Save full model results (all groups)
COPY (
    SELECT
        group_id,
        model.intercept,
        model.coefficients,
        model.coefficient_std_errors,
        model.coefficient_t_statistics,
        model.coefficient_p_values,
        model.r2,
        model.adj_r2,
        model.f_statistic,
        model.f_statistic_p_value,
        model.aic,
        model.bic,
        model.n_obs,
        model.df_model,
        model.df_residual
    FROM group_models_full
) TO 'examples/performance_test/results/sql_group_models_full.parquet' (FORMAT PARQUET);

.print 'Results saved to examples/performance_test/results/'
.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
.print '============================================================================'
.print 'PERFORMANCE TEST SUMMARY'
.print '============================================================================'
.print 'Tests completed:'
.print '  1. GROUP BY aggregation on all groups (basic output)'
.print '  2. GROUP BY aggregation on all groups (full output)'
.print '  3. GROUP BY aggregation on subset (100 groups)'
.print ''
.print 'Tables created:'
.print '  - performance_data: Raw observations with features and response'
.print '  - group_models: Fitted models (basic output)'
.print '  - group_models_full: Fitted models (full statistical output)'
.print '  - subset_models: Fitted models for subset'
.print ''
.print 'Results saved to:'
.print '  - examples/performance_test/results/sql_group_models.parquet'
.print '  - examples/performance_test/results/sql_group_models_full.parquet'
.print '============================================================================'
