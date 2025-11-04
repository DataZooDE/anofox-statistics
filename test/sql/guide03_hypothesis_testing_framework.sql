-- Test: Does advertising affect sales?
-- H₀: β_advertising = 0
-- H₁: β_advertising ≠ 0

SELECT
    variable,
    estimate as effect,
    p_value,
    CASE
        WHEN p_value < 0.001 THEN 'Highly significant ***'
        WHEN p_value < 0.01 THEN 'Very significant **'
        WHEN p_value < 0.05 THEN 'Significant *'
        ELSE 'Not significant'
    END as significance_level
FROM ols_inference(
    [100.0, 95.0, 92.0, 88.0, 85.0]::DOUBLE[],           -- y: sales
    [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0]]::DOUBLE[][],  -- x: price, advertising
    0.95,                                                  -- confidence_level
    true                                                   -- add_intercept
)
WHERE variable = 'advertising';
