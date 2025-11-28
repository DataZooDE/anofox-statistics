LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Get statistical inference using fit with full_output
WITH model AS (
    SELECT * FROM anofox_statistics_ols_fit(
        [2.1, 4.0, 6.1, 7.9, 10.2]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95::DOUBLE}
    )
)
SELECT
    'x1' as variable,
    ROUND(coefficients[1], 4) as coefficient,
    ROUND(coefficient_p_values[1], 4) as p_value,
    coefficient_p_values[1] < 0.05 as significant
FROM model
UNION ALL
SELECT
    'intercept' as variable,
    ROUND(intercept, 4) as coefficient,
    ROUND(intercept_p_value, 4) as p_value,
    intercept_p_value < 0.05 as significant
FROM model;
