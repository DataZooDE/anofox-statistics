-- Load the extension
LOAD 'anofox_statistics';

-- Simple OLS regression
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1: first predictor
    true                                   -- add_intercept
);

-- Multiple regression with 3 predictors
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1: first predictor
    [2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],  -- x2: second predictor
    [3.0, 4.0, 5.0, 6.0, 7.0]::DOUBLE[],  -- x3: third predictor
    true                                   -- add_intercept
);

-- Coefficient inference with p-values (also uses positional parameters)
SELECT * FROM ols_inference(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],                  -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],      -- x (matrix format)
    0.95,                                                  -- confidence_level
    true                                                   -- add_intercept
);

-- Per-group regression
SELECT
    category,
    ols_fit_agg(sales, price) as model
FROM sales_data
GROUP BY category;
