-- Phase 2 Validation: Dataset 2 (Realistic Multiple Regression - Housing Prices)
-- Expected R Results:
-- Intercept: 54.48 (SE=4.78, p<2e-16)
-- Size: 15.68 (SE=0.66, p<2e-16)
-- Bedrooms: 6.70 (SE=2.26, p=0.0047)
-- Age: -0.46 (SE=0.09, p=7.1e-06)
-- R²=0.9927, Residual SE=8.835

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

.mode box
.width 12 12 12 14 12 12 12

SELECT '=========================================';
SELECT 'DATASET 2: Realistic Multiple Regression';
SELECT '=========================================';
SELECT '';

-- Load data
CREATE TABLE realistic_data AS SELECT * FROM read_csv_auto('validation/data/realistic_housing.csv');
CREATE TABLE realistic_new AS SELECT * FROM read_csv_auto('validation/data/realistic_housing_new.csv');

SELECT 'First 10 rows of training data:';
SELECT * FROM realistic_data LIMIT 10;
SELECT '';

-- Extract arrays for manual testing
-- Note: Table functions require literal array values, not subqueries or column references
-- Run these queries to extract the arrays, then use them in separate test queries
SELECT '========== Extracting Arrays for Testing ==========';
SELECT 'Run these queries to get array literals:';
SELECT '';

SELECT 'Y training values (first 50 rows):';
SELECT list(price ORDER BY rowid)[1:50] as y_train FROM realistic_data;
SELECT '';

SELECT 'X training values (first 50 rows):';
SELECT list([size, bedrooms, age] ORDER BY rowid)[1:50] as x_train FROM realistic_data;
SELECT '';

SELECT 'X new values for prediction:';
SELECT list([size, bedrooms, age] ORDER BY rowid) as x_new FROM realistic_new;
SELECT '';

SELECT '========== Tests Skipped ==========';
SELECT 'Note: OLS inference, prediction intervals, and residual diagnostics tests';
SELECT 'require hardcoded array literals. Extract arrays using queries above,';
SELECT 'then call functions manually with those literal values.';
SELECT '';
SELECT 'Example for OLS Inference:';
SELECT 'SELECT * FROM ols_inference([...y values...], [[...x values...]], 0.95, true);';
