-- Basic OLS fit (use positional parameters)
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1
    true                                   -- add_intercept
);
