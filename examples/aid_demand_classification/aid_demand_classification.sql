-- ============================================================================
-- AID (Automatic Identification of Demand) Examples
-- ============================================================================
-- Demonstrates demand pattern classification and anomaly detection for
-- inventory management and supply chain analysis.
--
-- Run: ./build/release/duckdb < examples/aid_demand_classification.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Example 1: Basic Demand Classification
-- ============================================================================
-- Classify a single demand series as regular or intermittent

SELECT '=== Example 1: Basic Demand Classification ===' AS section;

SELECT aid_agg(demand) as classification
FROM (VALUES (10), (12), (8), (15), (11), (9), (14), (10), (13), (11)) AS t(demand);

-- ============================================================================
-- Example 2: Intermittent Demand Detection
-- ============================================================================
-- High proportion of zeros indicates intermittent demand pattern

SELECT '=== Example 2: Intermittent Demand Detection ===' AS section;

SELECT
    result.demand_type,
    result.is_intermittent,
    result.distribution,
    ROUND(result.zero_proportion, 2) AS zero_proportion,
    result.n_observations
FROM (
    SELECT aid_agg(demand) AS result
    FROM (VALUES (0), (0), (5), (0), (8), (0), (3), (0), (0), (6)) AS t(demand)
);

-- ============================================================================
-- Example 3: Multi-SKU Classification with GROUP BY
-- ============================================================================
-- Classify demand patterns for multiple products

SELECT '=== Example 3: Multi-SKU Classification ===' AS section;

WITH sales AS (
    -- SKU001: Intermittent (many zeros)
    SELECT 'SKU001' as sku, val as demand, row_number() OVER () as period
    FROM (VALUES (0), (0), (5), (0), (8), (0), (3), (0), (0), (6)) AS t(val)
    UNION ALL
    -- SKU002: Regular (no zeros, consistent)
    SELECT 'SKU002' as sku, val as demand, row_number() OVER () as period
    FROM (VALUES (45), (48), (42), (50), (47), (44), (49), (46), (51), (43)) AS t(val)
    UNION ALL
    -- SKU003: New product (leading zeros)
    SELECT 'SKU003' as sku, val as demand, row_number() OVER () as period
    FROM (VALUES (0), (0), (0), (0), (5), (8), (12), (15), (18), (20)) AS t(val)
    UNION ALL
    -- SKU004: Obsolete product (trailing zeros)
    SELECT 'SKU004' as sku, val as demand, row_number() OVER () as period
    FROM (VALUES (25), (22), (18), (15), (10), (5), (0), (0), (0), (0)) AS t(val)
)
SELECT
    sku,
    result.demand_type,
    result.distribution,
    ROUND(result.mean, 1) AS mean,
    ROUND(result.zero_proportion, 2) AS zero_pct,
    result.is_new_product,
    result.is_obsolete_product,
    result.has_stockouts
FROM (
    SELECT sku, aid_agg(demand ORDER BY period) AS result
    FROM sales
    GROUP BY sku
) sub
ORDER BY sku;

-- ============================================================================
-- Example 4: Anomaly Detection Per Observation
-- ============================================================================
-- Get detailed anomaly flags for each row in the series

SELECT '=== Example 4: Per-Observation Anomaly Detection ===' AS section;

WITH demand_series AS (
    SELECT row_number() OVER () as period, demand
    FROM (VALUES (0), (0), (5), (0), (8), (0), (0)) AS t(demand)
),
anomalies AS (
    SELECT aid_anomaly_agg(demand ORDER BY period) AS anomaly_flags
    FROM demand_series
)
SELECT
    ds.period,
    ds.demand,
    f.stockout,
    f.new_product,
    f.obsolete_product,
    f.high_outlier,
    f.low_outlier
FROM demand_series ds, anomalies a, LATERAL UNNEST(a.anomaly_flags) WITH ORDINALITY AS t(f, ord)
WHERE ds.period = t.ord
ORDER BY ds.period;

-- ============================================================================
-- Example 5: Custom Threshold for Intermittent Classification
-- ============================================================================
-- Use stricter threshold (50% zeros required for intermittent)

SELECT '=== Example 5: Custom Intermittent Threshold ===' AS section;

WITH demand AS (
    SELECT val as demand FROM (VALUES (0), (5), (0), (8), (10), (0), (12), (7), (0), (9)) AS t(val)
)
SELECT
    'Default (30%)' AS threshold_setting,
    result.demand_type,
    ROUND(result.zero_proportion, 2) AS zero_proportion
FROM (SELECT aid_agg(demand) AS result FROM demand)
UNION ALL
SELECT
    'Custom (50%)' AS threshold_setting,
    result.demand_type,
    ROUND(result.zero_proportion, 2) AS zero_proportion
FROM (SELECT aid_agg(demand, {'intermittent_threshold': 0.5}) AS result FROM demand);

-- ============================================================================
-- Example 6: IQR vs Z-Score Outlier Detection
-- ============================================================================
-- Compare different outlier detection methods

SELECT '=== Example 6: Outlier Detection Methods ===' AS section;

WITH demand AS (
    -- Normal values with one extreme outlier
    SELECT val as demand
    FROM (VALUES (10), (11), (12), (10), (100), (11), (10), (12), (11), (10),
                 (10), (11), (12), (10), (11), (10), (12), (11), (10), (11)) AS t(val)
)
SELECT
    'Z-Score' AS method,
    result.high_outlier_count,
    result.low_outlier_count
FROM (SELECT aid_agg(demand, {'outlier_method': 'zscore'}) AS result FROM demand)
UNION ALL
SELECT
    'IQR' AS method,
    result.high_outlier_count,
    result.low_outlier_count
FROM (SELECT aid_agg(demand, {'outlier_method': 'iqr'}) AS result FROM demand);

-- ============================================================================
-- Example 7: Stockout Analysis
-- ============================================================================
-- Identify SKUs with stockout issues

SELECT '=== Example 7: Stockout Analysis ===' AS section;

WITH inventory AS (
    -- Product A: Has stockouts (zeros in middle of positive demand)
    SELECT 'Product_A' as product, val as demand, row_number() OVER () as week
    FROM (VALUES (50), (45), (0), (0), (52), (48), (0), (55), (47), (51)) AS t(val)
    UNION ALL
    -- Product B: No stockouts (continuous positive demand)
    SELECT 'Product_B' as product, val as demand, row_number() OVER () as week
    FROM (VALUES (30), (32), (28), (35), (31), (29), (33), (30), (34), (32)) AS t(val)
    UNION ALL
    -- Product C: Many stockouts
    SELECT 'Product_C' as product, val as demand, row_number() OVER () as week
    FROM (VALUES (20), (0), (18), (0), (0), (22), (0), (19), (0), (21)) AS t(val)
)
SELECT
    product,
    result.has_stockouts,
    result.stockout_count,
    ROUND(result.stockout_count::DOUBLE / result.n_observations * 100, 1) AS stockout_pct,
    result.demand_type
FROM (
    SELECT product, aid_agg(demand ORDER BY week) AS result
    FROM inventory
    GROUP BY product
) sub
ORDER BY result.stockout_count DESC;

-- ============================================================================
-- Example 8: Distribution Recommendation
-- ============================================================================
-- AID recommends appropriate statistical distribution for forecasting

SELECT '=== Example 8: Distribution Recommendations ===' AS section;

WITH products AS (
    -- Count data (integers, low values) -> Poisson family
    SELECT 'Low_Count_Data' as category, val as demand
    FROM (VALUES (2), (3), (1), (4), (2), (3), (2), (5), (3), (2)) AS t(val)
    UNION ALL
    -- Overdispersed count data -> Negative Binomial
    SELECT 'Overdispersed_Counts' as category, val as demand
    FROM (VALUES (0), (0), (5), (0), (12), (0), (0), (8), (0), (15)) AS t(val)
    UNION ALL
    -- Continuous positive data -> Gamma/Lognormal
    SELECT 'Continuous_Positive' as category, val as demand
    FROM (VALUES (10.5), (12.3), (8.7), (15.2), (11.8), (9.4), (14.1), (10.9), (13.5), (11.2)) AS t(val)
    UNION ALL
    -- Normal-like data -> Normal
    SELECT 'Normal_Like' as category, val as demand
    FROM (VALUES (100), (102), (98), (101), (99), (103), (97), (100), (101), (99)) AS t(val)
)
SELECT
    category,
    result.distribution AS recommended_distribution,
    result.demand_type,
    ROUND(result.mean, 2) AS mean,
    ROUND(result.variance, 2) AS variance
FROM (
    SELECT category, aid_agg(demand) AS result
    FROM products
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 9: Product Lifecycle Detection
-- ============================================================================
-- Identify new products (leading zeros) and obsolete products (trailing zeros)

SELECT '=== Example 9: Product Lifecycle Detection ===' AS section;

WITH lifecycle AS (
    SELECT 'NewProduct_2024' as product, val as demand, row_number() OVER () as month
    FROM (VALUES (0), (0), (0), (5), (12), (25), (40), (55), (70), (85), (95), (100)) AS t(val)
    UNION ALL
    SELECT 'MatureProduct' as product, val as demand, row_number() OVER () as month
    FROM (VALUES (80), (82), (78), (85), (81), (79), (83), (80), (84), (82), (81), (80)) AS t(val)
    UNION ALL
    SELECT 'EndOfLife_Legacy' as product, val as demand, row_number() OVER () as month
    FROM (VALUES (50), (42), (35), (28), (20), (12), (5), (0), (0), (0), (0), (0)) AS t(val)
)
SELECT
    product,
    CASE
        WHEN result.is_new_product AND NOT result.is_obsolete_product THEN 'Introduction Phase'
        WHEN result.is_obsolete_product AND NOT result.is_new_product THEN 'Decline Phase'
        WHEN result.is_new_product AND result.is_obsolete_product THEN 'Short Lifecycle'
        ELSE 'Mature/Stable'
    END AS lifecycle_stage,
    result.new_product_count AS intro_periods,
    result.obsolete_product_count AS decline_periods,
    result.n_observations AS total_periods
FROM (
    SELECT product, aid_agg(demand ORDER BY month) AS result
    FROM lifecycle
    GROUP BY product
) sub
ORDER BY product;

-- ============================================================================
-- Example 10: Comprehensive Demand Analysis Report
-- ============================================================================
-- Full analysis combining all AID features

SELECT '=== Example 10: Comprehensive Demand Analysis ===' AS section;

CREATE OR REPLACE TABLE demand_data AS
SELECT
    'SKU' || LPAD(CAST(sku_id AS VARCHAR), 3, '0') AS sku,
    CASE
        WHEN sku_id <= 3 THEN 'Electronics'
        WHEN sku_id <= 6 THEN 'Apparel'
        ELSE 'Food'
    END AS category,
    week,
    -- Generate different demand patterns
    CASE
        WHEN sku_id = 1 THEN GREATEST(0, 50 + CAST(FLOOR(RANDOM() * 20 - 10) AS INTEGER))
        WHEN sku_id = 2 THEN CASE WHEN RANDOM() < 0.4 THEN 0 ELSE CAST(FLOOR(RANDOM() * 30 + 10) AS INTEGER) END
        WHEN sku_id = 3 THEN CASE WHEN week <= 4 THEN 0 ELSE CAST(FLOOR(week * 5 + RANDOM() * 10) AS INTEGER) END
        WHEN sku_id = 4 THEN GREATEST(0, 100 + CAST(FLOOR(RANDOM() * 30 - 15) AS INTEGER))
        WHEN sku_id = 5 THEN CASE WHEN RANDOM() < 0.6 THEN 0 ELSE CAST(FLOOR(RANDOM() * 20 + 5) AS INTEGER) END
        WHEN sku_id = 6 THEN CASE WHEN week >= 9 THEN 0 ELSE GREATEST(0, 80 - week * 8 + CAST(FLOOR(RANDOM() * 10) AS INTEGER)) END
        WHEN sku_id = 7 THEN GREATEST(0, 200 + CAST(FLOOR(RANDOM() * 40 - 20) AS INTEGER))
        WHEN sku_id = 8 THEN CASE WHEN RANDOM() < 0.3 THEN 0 ELSE CAST(FLOOR(RANDOM() * 50 + 20) AS INTEGER) END
        ELSE GREATEST(0, 150 + CAST(FLOOR(RANDOM() * 50 - 25) AS INTEGER))
    END AS demand
FROM (SELECT * FROM range(1, 10) t1(sku_id), range(1, 13) t2(week));

SELECT
    sub.category,
    sub.sku,
    result.demand_type,
    result.distribution,
    ROUND(result.mean, 1) AS avg_demand,
    ROUND(result.zero_proportion * 100, 0) AS zero_pct,
    result.has_stockouts,
    result.stockout_count,
    CASE
        WHEN result.is_new_product THEN 'New'
        WHEN result.is_obsolete_product THEN 'EOL'
        ELSE 'Active'
    END AS status,
    result.high_outlier_count + result.low_outlier_count AS anomalies
FROM (
    SELECT sku, category, aid_agg(demand ORDER BY week) AS result
    FROM demand_data
    GROUP BY sku, category
) sub
ORDER BY sub.category, sub.sku;

-- Summary by category
SELECT
    category,
    COUNT(*) AS sku_count,
    SUM(CASE WHEN result.is_intermittent THEN 1 ELSE 0 END) AS intermittent_skus,
    SUM(CASE WHEN result.has_stockouts THEN 1 ELSE 0 END) AS skus_with_stockouts,
    SUM(CASE WHEN result.is_new_product THEN 1 ELSE 0 END) AS new_products,
    SUM(CASE WHEN result.is_obsolete_product THEN 1 ELSE 0 END) AS eol_products
FROM (
    SELECT category, aid_agg(demand ORDER BY week) AS result
    FROM demand_data
    GROUP BY sku, category
) sub
GROUP BY category
ORDER BY category;

-- Cleanup
DROP TABLE IF EXISTS demand_data;

SELECT '=== AID Examples Complete ===' AS section;
