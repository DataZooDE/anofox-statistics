-- Simplified OLS Validation Test
-- Tests basic OLS functionality against known data

-- Load input data
CREATE OR REPLACE TABLE ols_input AS
SELECT * FROM read_csv('test/data/ols_tests/input/simple_linear.csv');

-- Test using aggregate function (works directly on table data)
CREATE OR REPLACE TABLE ols_agg_result AS
SELECT
    (ols_fit_agg(y, x)).coefficient as slope,
    (ols_fit_agg(y, x)).r2 as r_squared
FROM ols_input;

-- Load expected R² from reference data
CREATE OR REPLACE TABLE ols_expected AS
SELECT r_squared as expected_r2
FROM read_json('test/data/ols_tests/expected/simple_linear.json',
    format='auto', maximum_object_size=10000000);

-- Compare R² values
CREATE OR REPLACE TABLE validation_result AS
SELECT
    r.r_squared as computed_r2,
    e.expected_r2,
    abs(r.r_squared - e.expected_r2) as error,
    CASE WHEN abs(r.r_squared - e.expected_r2) < 0.01 THEN 'PASS' ELSE 'FAIL' END as status
FROM ols_agg_result r, ols_expected e;

-- Show result
SELECT * FROM validation_result;

-- Return success only if passed
SELECT * FROM validation_result WHERE status = 'PASS';
