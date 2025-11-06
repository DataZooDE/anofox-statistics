LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Inference with confidence intervals (use positional parameters)
SELECT
    variable,
    ROUND(estimate, 4) as coefficient,
    ROUND(p_value, 4) as p_value,
    significant
FROM ols_inference(
    [2.1, 4.0, 6.1, 7.9, 10.2]::DOUBLE[],          -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x (matrix)
    0.95,                                            -- confidence_level
    true                                             -- add_intercept
);
