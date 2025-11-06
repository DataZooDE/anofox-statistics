-- Test normality of residuals (use literal array)
SELECT * FROM anofox_statistics_normality_test(
    [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.0, 0.1, -0.1, 0.2]::DOUBLE[],  -- residuals
    0.05                                                                 -- alpha
);

-- Note: To test residuals from a table, first extract to array using LIST()
-- Example: SELECT LIST(residual) FROM my_diagnostics_table
