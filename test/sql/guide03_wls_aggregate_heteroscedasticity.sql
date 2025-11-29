LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Statistics Guide: Weighted Least Squares - Handling Heteroscedasticity
-- Demonstrates when and how to use weights for non-constant variance

-- Scenario: Customer spending analysis where variance increases with income
-- High-income customers have more variable spending patterns
CREATE TEMP TABLE customer_spending AS
SELECT
    CASE
        WHEN i <= 15 THEN 'low_income'
        WHEN i <= 30 THEN 'medium_income'
        ELSE 'high_income'
    END as segment,
    i as customer_id,
    (20000 + i * 1000)::DOUBLE as annual_income,
    -- Spending with heteroscedastic errors (variance increases with income)
    (5000 + 0.3 * (20000 + i * 1000) + random() * (100 + i * 20))::DOUBLE as annual_spending,
    -- Weight by inverse variance (precision weighting)
    (1.0 / (1.0 + i * 0.1))::DOUBLE as precision_weight
FROM generate_series(1, 45) as t(i);

-- Compare OLS (ignores heteroscedasticity) vs WLS (accounts for it)
SELECT
    segment,
    'OLS (unweighted)' as method,
    result.coefficients[1] as income_propensity,
    result.intercept as base_spending,
    result.r2,
    NULL as weighted_mse
FROM (
    SELECT
        segment,
        anofox_stats_ols_fit_agg(
            annual_spending,
            [annual_income],
            {'intercept': true}
        ) as result
    FROM customer_spending
    GROUP BY segment
) sub
UNION ALL
SELECT
    segment,
    'WLS (precision weighted)' as method,
    result.coefficients[1] as income_propensity,
    result.intercept as base_spending,
    result.r2,
    result.weighted_mse
FROM (
    SELECT
        segment,
        anofox_stats_wls_fit_agg(
            annual_spending,
            [annual_income],
            precision_weight,
            {'intercept': true}
        ) as result
    FROM customer_spending
    GROUP BY segment
) sub
ORDER BY segment, method;

-- Interpretation note
SELECT
    'Statistical Note' as category,
    'WLS gives more weight to observations with lower variance (more reliable data points). This produces more efficient estimates when heteroscedasticity is present.' as explanation
UNION ALL
SELECT
    'When to use WLS',
    'Use WLS when: (1) variance changes systematically with predictors, (2) you have reliability measures for observations, or (3) you are combining data from sources with different precision.' as guidance;
