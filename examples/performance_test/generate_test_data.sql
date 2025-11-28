-- ============================================================================
-- Generate Test Data for Performance Testing
-- ============================================================================
-- This script generates two test datasets and saves them as parquet files:
-- 1. performance_data_fit_predict.parquet - for window function tests
-- 2. performance_data_aggregate.parquet - for GROUP BY aggregate tests
--
-- Both datasets can be loaded by SQL and R scripts for consistent testing.
-- ============================================================================

-- ============================================================================
-- CONFIGURATION PARAMETERS
-- ============================================================================
SET VARIABLE n_groups = 10000;          -- Number of groups
SET VARIABLE n_obs_per_group = 100;     -- Observations per group
SET VARIABLE n_features = 8;            -- Number of features (x1, x2, ..., x8)
SET VARIABLE noise_std = 2.0;           -- Standard deviation of Gaussian noise
SET VARIABLE null_fraction = 0.1;       -- Fraction of y values to set as NULL (fit-predict only)

.print '============================================================================'
.print 'GENERATING TEST DATASETS'
.print '============================================================================'
.print 'Configuration:'
.print '  - Groups: ' || getvariable('n_groups')
.print '  - Observations per group: ' || getvariable('n_obs_per_group')
.print '  - Total rows: ' || (getvariable('n_groups') * getvariable('n_obs_per_group'))
.print '  - Features: ' || getvariable('n_features')
.print '  - Noise std dev: ' || getvariable('noise_std')
.print '  - NULL fraction (fit-predict): ' || getvariable('null_fraction')
.print ''

-- ============================================================================
-- STEP 1: Generate Random Coefficients for Each Group
-- ============================================================================
.print 'Generating group-specific coefficients...'

CREATE OR REPLACE TABLE group_coefficients AS
SELECT
    group_id,
    -- Intercept: random value between -10 and 10
    (random() * 20.0 - 10.0) as beta_0,
    -- Coefficients for x1-x8: random values between -5 and 5
    (random() * 10.0 - 5.0) as beta_1,
    (random() * 10.0 - 5.0) as beta_2,
    (random() * 10.0 - 5.0) as beta_3,
    (random() * 10.0 - 5.0) as beta_4,
    (random() * 10.0 - 5.0) as beta_5,
    (random() * 10.0 - 5.0) as beta_6,
    (random() * 10.0 - 5.0) as beta_7,
    (random() * 10.0 - 5.0) as beta_8
FROM range(1, getvariable('n_groups') + 1) t(group_id);

.print 'Generated coefficients for ' || getvariable('n_groups') || ' groups'
.print ''

-- ============================================================================
-- STEP 2: Generate Base Dataset (used for both test cases)
-- ============================================================================
.print 'Generating base observations...'

CREATE OR REPLACE TABLE base_data AS
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

.print 'Generated ' || (SELECT COUNT(*) FROM base_data) || ' base observations'
.print ''

-- ============================================================================
-- STEP 3: Create Dataset for Fit-Predict Tests (with NULL values)
-- ============================================================================
.print 'Creating fit-predict dataset with NULL values...'

CREATE OR REPLACE TABLE performance_data_fit_predict AS
SELECT
    group_id,
    obs_id,
    x1, x2, x3, x4, x5, x6, x7, x8,
    -- y = beta_0 + sum(beta_i * x_i) + noise, or NULL for prediction
    CASE
        WHEN random() < getvariable('null_fraction') THEN NULL
        ELSE (
            beta_0 +
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
        )::DOUBLE
    END as y,
    -- Store true y (without NULL) for validation
    (beta_0 +
     beta_1 * x1 +
     beta_2 * x2 +
     beta_3 * x3 +
     beta_4 * x4 +
     beta_5 * x5 +
     beta_6 * x6 +
     beta_7 * x7 +
     beta_8 * x8 +
     sqrt(-2.0 * ln(random())) * cos(2.0 * pi() * random()) * getvariable('noise_std')
    )::DOUBLE as y_true,
    -- Store true coefficients for validation
    beta_0, beta_1, beta_2, beta_3, beta_4,
    beta_5, beta_6, beta_7, beta_8
FROM base_data;

-- Report statistics
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT group_id) as n_groups,
    COUNT(*) / COUNT(DISTINCT group_id) as obs_per_group,
    SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_count,
    ROUND(100.0 * SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_percent
FROM performance_data_fit_predict;

.print ''

-- ============================================================================
-- STEP 4: Create Dataset for Aggregate Tests (no NULL values)
-- ============================================================================
.print 'Creating aggregate dataset (no NULL values)...'

CREATE OR REPLACE TABLE performance_data_aggregate AS
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
    )::DOUBLE as y,
    -- Store true coefficients for validation
    beta_0, beta_1, beta_2, beta_3, beta_4,
    beta_5, beta_6, beta_7, beta_8
FROM base_data;

-- Report statistics
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT group_id) as n_groups,
    COUNT(*) / COUNT(DISTINCT group_id) as obs_per_group
FROM performance_data_aggregate;

.print ''

-- ============================================================================
-- STEP 5: Save Datasets as Parquet Files
-- ============================================================================
.print 'Saving datasets as parquet files...'

-- Save fit-predict dataset
COPY performance_data_fit_predict TO 'examples/performance_test/data/performance_data_fit_predict.parquet' (FORMAT PARQUET);
.print '  - Saved: examples/performance_test/data/performance_data_fit_predict.parquet'

-- Save aggregate dataset
COPY performance_data_aggregate TO 'examples/performance_test/data/performance_data_aggregate.parquet' (FORMAT PARQUET);
.print '  - Saved: examples/performance_test/data/performance_data_aggregate.parquet'

.print ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
.print '============================================================================'
.print 'DATA GENERATION COMPLETE'
.print '============================================================================'
.print 'Files created:'
.print '  1. examples/performance_test/data/performance_data_fit_predict.parquet'
.print '     - For window function (fit-predict) tests'
.print '     - Contains NULL y values for prediction demonstration'
.print '     - Includes y_true for validation'
.print ''
.print '  2. examples/performance_test/data/performance_data_aggregate.parquet'
.print '     - For GROUP BY aggregate tests'
.print '     - No NULL values (required for aggregates)'
.print ''
.print 'Both datasets include:'
.print '  - group_id: Group identifier'
.print '  - obs_id: Observation sequence within group'
.print '  - x1-x8: Feature columns'
.print '  - y: Response variable'
.print '  - beta_0-beta_8: True coefficients for validation'
.print '============================================================================'
