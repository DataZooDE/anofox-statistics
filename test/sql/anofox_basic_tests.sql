-- DISABLED: This test file is for the OLD table-based API that was removed
--
-- The old API used table-based functions:
--   - anofox_ols_fit(table, ['x1', 'x2'], 'y')
--   - anofox_grouped_ols_fit(table, ['group'], ['x1'], 'y')
--   - anofox_rolling_ols_fit(table, ['x1'], 'y', window_size)
--
-- The NEW API uses:
--   - Table functions with arrays: anofox_statistics_ols_fit([y_values], [[x_values]])
--   - Aggregates for grouping: SELECT ols_fit_agg(y, x) FROM table GROUP BY group_col
--   - Window functions for rolling: SELECT ols_fit_agg(y, x) OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
--
-- This file needs to be completely rewritten for the new API.
-- See test files like aggregate_basic_tests.sql for examples of the new API.

SELECT 'anofox_basic_tests.sql - DISABLED - needs rewrite for new API' as status;
