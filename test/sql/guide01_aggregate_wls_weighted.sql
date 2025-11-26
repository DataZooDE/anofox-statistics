LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Quick Start Example: Weighted Least Squares Aggregate
-- Demonstrates regression with observation weights (for heteroscedasticity)

-- Sample data: customer transactions with varying reliability
CREATE TEMP TABLE customer_transactions AS
SELECT
    'premium' as segment, 1 as month, 1000.0 as spend, 500.0 as income, 1.0 as reliability_weight
UNION ALL SELECT 'premium', 2, 1100.0, 510.0, 1.0
UNION ALL SELECT 'premium', 3, 1200.0, 520.0, 1.0
UNION ALL SELECT 'premium', 4, 1300.0, 530.0, 1.0
UNION ALL SELECT 'standard', 1, 300.0, 400.0, 0.8
UNION ALL SELECT 'standard', 2, 320.0, 410.0, 0.8
UNION ALL SELECT 'standard', 3, 340.0, 420.0, 0.8
UNION ALL SELECT 'standard', 4, 360.0, 430.0, 0.8
UNION ALL SELECT 'budget', 1, 100.0, 300.0, 0.5
UNION ALL SELECT 'budget', 2, 110.0, 305.0, 0.5
UNION ALL SELECT 'budget', 3, 120.0, 310.0, 0.5
UNION ALL SELECT 'budget', 4, 130.0, 315.0, 0.5;

-- WLS regression weighted by reliability
SELECT
    segment,
    result.coefficients[1] as income_sensitivity,
    result.intercept,
    result.r_squared,
    result.weighted_mse
FROM (
    SELECT
        segment,
        anofox_statistics_wls_fit_agg(
            spend,
            [income],
            reliability_weight,
            {'intercept': true}
        ) as result
    FROM customer_transactions
    GROUP BY segment
) sub;
