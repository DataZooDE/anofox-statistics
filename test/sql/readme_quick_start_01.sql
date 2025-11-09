-- DISABLED: This file uses the OLD API with individual array parameters
--
-- The old API used:
--   anofox_statistics_ols(y, x1, x2, x3, true)  -- Multiple individual arrays
--
-- The NEW API uses:
--   anofox_statistics_ols(y, [[x1, x2, x3]], MAP{'intercept': true})  -- 2D array + MAP options
--
-- Also references deprecated functions like ols_inference and uses non-existent sales_data table

SELECT 'readme_quick_start_01.sql - DISABLED - uses old API signature' as status;
