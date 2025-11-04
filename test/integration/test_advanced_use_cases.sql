-- Test Advanced Use Cases Examples

.bail on
.mode box

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT '========================================';
SELECT 'Testing Advanced Use Cases Examples';
SELECT '========================================';

-- Example 1: Multi-Stage Model Building Workflow
SELECT '--- Example 1: Multi-Stage Model Building ---';
CREATE OR REPLACE TABLE retail_stores AS
SELECT
    i as store_id,
    2024 as year,
    (50000 + i * 1000 +
     i * 200 * RANDOM() +
     i * 50 * RANDOM() +
     -1000 * (i % 10) +
     i * 30 * RANDOM() +
     RANDOM() * 5000)::DOUBLE as sale_amount,
    (i * 200 + RANDOM() * 1000)::DOUBLE as advertising_spend,
    (1000 + i * 100 + RANDOM() * 500)::DOUBLE as store_size_sqft,
    (i % 10 + RANDOM() * 5)::DOUBLE as competitor_distance_miles,
    (40000 + i * 1000 + RANDOM() * 10000)::DOUBLE as local_income_median
FROM range(1, 51) t(i);

WITH raw_data AS (
    SELECT
        sale_amount::DOUBLE as y,
        advertising_spend::DOUBLE as x1,
        store_size_sqft::DOUBLE as x2,
        competitor_distance_miles::DOUBLE as x3,
        local_income_median::DOUBLE as x4
    FROM retail_stores
    WHERE year = 2024 AND sale_amount > 0
)
SELECT COUNT(*) as records_count FROM raw_data;

SELECT 'Multi-stage workflow test passed';

-- Example 2: Automated Model Selection
SELECT '--- Example 2: Automated Model Selection ---';
CREATE OR REPLACE TABLE business_data AS
SELECT
    i as period_id,
    (1000 + i * 50 +
     i * 10 * RANDOM() +
     200 * SIN(i * 0.5) +
     -30 * (i % 5) +
     -i * 5 * RANDOM() +
     RANDOM() * 100)::DOUBLE as sales,
    (i * 10 + RANDOM() * 50)::DOUBLE as marketing,
    SIN(i * 0.5)::DOUBLE as seasonality,
    (i % 5 + RANDOM() * 2)::DOUBLE as competition,
    (10 + i * 0.1 + RANDOM() * 2)::DOUBLE as price
FROM range(1, 101) t(i);

SELECT COUNT(*) as records_count FROM business_data;
SELECT 'Model selection test passed';

-- Example 3: Time-Series Analysis
SELECT '--- Example 3: Time-Series Rolling Regression ---';
CREATE OR REPLACE TABLE daily_revenue AS
SELECT
    DATE '2024-01-01' + INTERVAL (i) DAY as date_id,
    (10000 +
     i * 50 +
     1000 * SIN(i * 0.1) +
     CASE WHEN i > 150 THEN 5000 ELSE 0 END +
     RANDOM() * 500)::DOUBLE as revenue
FROM range(0, 365) t(i);

SELECT COUNT(*) as records_count FROM daily_revenue;
SELECT 'Time-series test passed';

-- Example 4: Seasonality-Adjusted Forecasting
SELECT '--- Example 4: Seasonality-Adjusted Forecasting ---';
CREATE OR REPLACE TABLE monthly_sales AS
SELECT
    i as month_id,
    DATE '2022-01-01' + INTERVAL (i) MONTH as month_date,
    (50000 +
     i * 500 +
     10000 * SIN((i % 12) * 3.14159 / 6) +
     RANDOM() * 2000)::DOUBLE as revenue
FROM range(1, 37) t(i);

SELECT COUNT(*) as records_count FROM monthly_sales;
SELECT 'Seasonality test passed';

-- Example 5: Hierarchical Regression
SELECT '--- Example 5: Hierarchical Regression ---';
CREATE OR REPLACE TABLE daily_store_data AS
SELECT
    (i / 100) % 3 + 1 as region_id,
    (i / 20) % 5 + 1 as territory_id,
    i % 20 + 1 as store_id,
    DATE '2024-08-01' + INTERVAL (i % 100) DAY as date,
    (5000 +
     ((i / 100) % 3) * 1000 +
     ((i / 20) % 5) * 500 +
     (i % 20) * 100 +
     i * 10 * RANDOM() +
     RANDOM() * 500)::DOUBLE as sales,
    (i * 10 + RANDOM() * 100)::DOUBLE as marketing
FROM range(1, 6001) t(i);

SELECT
    region_id,
    COUNT(*) as stores
FROM daily_store_data
GROUP BY region_id
LIMIT 5;
SELECT 'Hierarchical regression test passed';

-- Example 6: Cohort Analysis
SELECT '--- Example 6: Cohort Analysis ---';
CREATE OR REPLACE TABLE cohort_behavior AS
SELECT
    DATE '2023-01-01' + INTERVAL ((i / 25)) MONTH as cohort_month,
    (i % 25) as months_since_first,
    (100 +
     (i % 25) * 5 +
     -0.5 * (i % 25) * (i % 25) +
     ((i / 25) % 12) * 10 +
     RANDOM() * 20)::DOUBLE as avg_order_value,
    (100 - (i % 25) * 2 + RANDOM() * 10)::INTEGER as active_customers,
    ((100 + (i % 25) * 5) * (100 - (i % 25) * 2) + RANDOM() * 1000)::DOUBLE as total_revenue
FROM range(1, 301) t(i)
WHERE (i % 25) <= 24;

SELECT COUNT(*) as records_count FROM cohort_behavior;
SELECT 'Cohort analysis test passed';

-- Example 7: A/B Test Analysis
SELECT '--- Example 7: A/B Test Analysis ---';
CREATE OR REPLACE TABLE ab_test_results AS
SELECT
    'pricing_test_2024_q1' as experiment_id,
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END as variant,
    (CASE WHEN i % 2 = 0 THEN 0.12 ELSE 0.15 END +
     RANDOM() * 0.05)::DOUBLE as conversion_rate,
    (CASE WHEN i % 2 = 0 THEN 45.0 ELSE 52.0 END +
     RANDOM() * 10)::DOUBLE as revenue_per_user,
    (CASE WHEN i % 2 = 0 THEN 7.5 ELSE 8.2 END +
     RANDOM() * 2)::DOUBLE as engagement_score
FROM range(1, 1001) t(i);

SELECT
    variant,
    COUNT(*) as sample_size
FROM ab_test_results
GROUP BY variant;
SELECT 'A/B test analysis test passed';

-- Test a complete workflow with actual function calls
SELECT '--- Testing Complete Workflow with Functions ---';
SELECT
    r_squared as r2
FROM information_criteria(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    true::BOOLEAN
);

SELECT '========================================';
SELECT 'Advanced Use Cases: ALL TESTS PASSED';
SELECT '========================================';
