-- ============================================================================
-- Performance Test: OLS Aggregate Functions with GROUP BY
-- ============================================================================
-- This script generates a large dataset for performance testing of
-- anofox_statistics_ols_fit_agg with GROUP BY functionality.
--
-- Dataset characteristics:
-- - Multiple groups with heterogeneous linear relationships
-- - Each group has its own true coefficient vector
-- - Gaussian noise added to simulate realistic data
-- - No NULL values (required for aggregate functions)
--
-- Usage:
-- 1. Adjust configuration parameters below
-- 2. Run the entire script in DuckDB
-- 3. Observe timing output for performance analysis
-- ============================================================================

-- ============================================================================
-- CONFIGURATION PARAMETERS (easily changeable)
-- ============================================================================
SET VARIABLE n_groups = 10000;          -- Number of groups
SET VARIABLE n_obs_per_group = 100;     -- Observations per group
SET VARIABLE n_features = 8;            -- Number of features (x1, x2, ..., x8)
SET VARIABLE noise_std = 2.0;           -- Standard deviation of Gaussian noise

-- ============================================================================
-- STEP 1: Generate Random Coefficients for Each Group
-- ============================================================================
-- Each group gets its own "true" linear relationship with randomly sampled
-- coefficients. This creates heterogeneous data across groups.

.print '============================================================================'
.print 'Generating group-specific coefficients...'
.print '============================================================================'

CREATE OR REPLACE TABLE group_coefficients AS
SELECT
    group_id,
    -- Intercept: random value between -10 and 10
    (random() * 20.0 - 10.0) as beta_0,
    -- Coefficient for x1: random value between -5 and 5
    (random() * 10.0 - 5.0) as beta_1,
    -- Coefficient for x2
    (random() * 10.0 - 5.0) as beta_2,
    -- Coefficient for x3
    (random() * 10.0 - 5.0) as beta_3,
    -- Coefficient for x4
    (random() * 10.0 - 5.0) as beta_4,
    -- Coefficient for x5
    (random() * 10.0 - 5.0) as beta_5,
    -- Coefficient for x6
    (random() * 10.0 - 5.0) as beta_6,
    -- Coefficient for x7
    (random() * 10.0 - 5.0) as beta_7,
    -- Coefficient for x8
    (random() * 10.0 - 5.0) as beta_8
FROM range(1, getvariable('n_groups') + 1) t(group_id);

.print 'Generated coefficients for ' || getvariable('n_groups') || ' groups'
.print ''

-- ============================================================================
-- STEP 2: Generate Observations for Each Group
-- ============================================================================
-- For each group, generate n_obs_per_group observations with:
-- - Features x1-x8: random uniform values between -10 and 10
-- - Response y: calculated from true linear relationship + Gaussian noise

.print '============================================================================'
.print 'Generating observations...'
.print '============================================================================'

CREATE OR REPLACE TABLE performance_data AS
SELECT
    gc.group_id,
    obs.obs_id,
    -- Generate features x1 through x8 (uniform random between -10 and 10)
    (random() * 20.0 - 10.0)::DOUBLE as x1,
    (random() * 20.0 - 10.0)::DOUBLE as x2,
    (random() * 20.0 - 10.0)::DOUBLE as x3,
    (random() * 20.0 - 10.0)::DOUBLE as x4,
    (random() * 20.0 - 10.0)::DOUBLE as x5,
    (random() * 20.0 - 10.0)::DOUBLE as x6,
    (random() * 20.0 - 10.0)::DOUBLE as x7,
    (random() * 20.0 - 10.0)::DOUBLE as x8,
    -- Store coefficients for y calculation
    gc.beta_0, gc.beta_1, gc.beta_2, gc.beta_3, gc.beta_4,
    gc.beta_5, gc.beta_6, gc.beta_7, gc.beta_8
FROM
    group_coefficients gc
    CROSS JOIN range(1, getvariable('n_obs_per_group') + 1) obs(obs_id);

-- Add response variable y with true linear relationship + noise
CREATE OR REPLACE TABLE performance_data AS
SELECT
    group_id,
    obs_id,
    x1, x2, x3, x4, x5, x6, x7, x8,
    -- y = beta_0 + beta_1*x1 + beta_2*x2 + ... + beta_8*x8 + noise
    (beta_0 +
     beta_1 * x1 +
     beta_2 * x2 +
     beta_3 * x3 +
     beta_4 * x4 +
     beta_5 * x5 +
     beta_6 * x6 +
     beta_7 * x7 +
     beta_8 * x8 +
     -- Gaussian noise using Box-Muller transform
     sqrt(-2.0 * ln(random())) * cos(2.0 * pi() * random()) * getvariable('noise_std')
    )::DOUBLE as y
FROM performance_data;

-- Report dataset size
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT group_id) as n_groups,
    COUNT(*) / COUNT(DISTINCT group_id) as obs_per_group
FROM performance_data;

.print ''
.print 'Dataset generated successfully!'
.print ''

-- ============================================================================
-- STEP 3: Performance Test - GROUP BY with All Groups
-- ============================================================================
-- Fit OLS model for each group using anofox_statistics_ols_fit_agg

.print '============================================================================'
.print 'PERFORMANCE TEST 1: GROUP BY Aggregation (All Groups)'
.print '============================================================================'
.timer on

CREATE OR REPLACE TABLE group_models AS
SELECT
    group_id,
    anofox_statistics_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'confidence_level': 0.95}
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
    model.r_squared as r_squared,
    model.adj_r_squared as adj_r_squared,
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
    anofox_statistics_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true, 'confidence_level': 0.95, 'full_output': true}
    ) as model
FROM performance_data
GROUP BY group_id;

.timer off

-- Display comprehensive statistics for first group
.print ''
.print 'Sample comprehensive output (group 1):'
SELECT
    model.coefficients as coefficients,
    model.std_errors as std_errors,
    model.t_statistics as t_statistics,
    model.p_values as p_values,
    model.r_squared as r_squared,
    model.adj_r_squared as adj_r_squared,
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
    anofox_statistics_ols_fit_agg(
        y,
        [x1, x2, x3, x4, x5, x6, x7, x8],
        MAP{'intercept': true}
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
    gm.model.coefficients[1] as estimated_value,
    gm.model.coefficients[1] - gc.beta_0 as error
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x1',
    gc.beta_1,
    gm.model.coefficients[2],
    gm.model.coefficients[2] - gc.beta_1
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x2',
    gc.beta_2,
    gm.model.coefficients[3],
    gm.model.coefficients[3] - gc.beta_2
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x3',
    gc.beta_3,
    gm.model.coefficients[4],
    gm.model.coefficients[4] - gc.beta_3
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x4',
    gc.beta_4,
    gm.model.coefficients[5],
    gm.model.coefficients[5] - gc.beta_4
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x5',
    gc.beta_5,
    gm.model.coefficients[6],
    gm.model.coefficients[6] - gc.beta_5
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x6',
    gc.beta_6,
    gm.model.coefficients[7],
    gm.model.coefficients[7] - gc.beta_6
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x7',
    gc.beta_7,
    gm.model.coefficients[8],
    gm.model.coefficients[8] - gc.beta_7
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1

UNION ALL

SELECT
    'x8',
    gc.beta_8,
    gm.model.coefficients[9],
    gm.model.coefficients[9] - gc.beta_8
FROM group_coefficients gc
JOIN group_models gm ON gc.group_id = gm.group_id
WHERE gc.group_id = 1;

.print ''
.print 'Note: Small errors are expected due to random noise in the data'
.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
.print '============================================================================'
.print 'PERFORMANCE TEST SUMMARY'
.print '============================================================================'
.print 'Configuration:'
.print '  - Groups: ' || getvariable('n_groups')
.print '  - Observations per group: ' || getvariable('n_obs_per_group')
.print '  - Total rows: ' || (getvariable('n_groups') * getvariable('n_obs_per_group'))
.print '  - Features: ' || getvariable('n_features')
.print '  - Noise std dev: ' || getvariable('noise_std')
.print ''
.print 'Tests completed:'
.print '  1. GROUP BY aggregation on all groups (basic output)'
.print '  2. GROUP BY aggregation on all groups (full output)'
.print '  3. GROUP BY aggregation on subset (100 groups)'
.print ''
.print 'Tables created:'
.print '  - group_coefficients: True coefficients for each group'
.print '  - performance_data: Raw observations with features and response'
.print '  - group_models: Fitted models (basic output)'
.print '  - group_models_full: Fitted models (full statistical output)'
.print '  - subset_models: Fitted models for subset'
.print '============================================================================'
