LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Test: Does advertising affect sales?
-- H₀: β_advertising = 0
-- H₁: β_advertising ≠ 0

WITH model AS (
    SELECT * FROM anofox_statistics_ols_fit(
        [100.0, 95.0, 92.0, 88.0, 85.0]::DOUBLE[],           -- y: sales
        [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0]]::DOUBLE[][],  -- x: price, advertising
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
    )
)
SELECT
    'advertising' as variable,
    coefficients[2] as effect,
    coefficient_p_values[2] as p_value,
    CASE
        WHEN coefficient_p_values[2] < 0.001 THEN 'Highly significant ***'
        WHEN coefficient_p_values[2] < 0.01 THEN 'Very significant **'
        WHEN coefficient_p_values[2] < 0.05 THEN 'Significant *'
        ELSE 'Not significant'
    END as significance_level
FROM model;
