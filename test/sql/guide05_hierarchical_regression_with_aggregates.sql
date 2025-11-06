LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample hierarchical store data
CREATE OR REPLACE TABLE daily_store_data AS
SELECT
    (i / 100) % 3 + 1 as region_id,
    (i / 20) % 5 + 1 as territory_id,
    i % 20 + 1 as store_id,
    DATE '2024-08-01' + INTERVAL (i % 100) DAY as date,
    (5000 +
     ((i / 100) % 3) * 1000 +                          -- region effect
     ((i / 20) % 5) * 500 +                            -- territory effect
     (i % 20) * 100 +                                  -- store effect
     i * 10 * RANDOM() +                               -- marketing effect
     RANDOM() * 500)::DOUBLE as sales,
    (i * 10 + RANDOM() * 100)::DOUBLE as marketing
FROM range(1, 6001) t(i);

-- Hierarchical sales analysis: Company → Region → Territory → Store
WITH store_level AS (
    SELECT
        region_id,
        territory_id,
        store_id,
        (ols_fit_agg(sales::DOUBLE, marketing::DOUBLE)).coefficient as store_roi,
        (ols_fit_agg(sales::DOUBLE, marketing::DOUBLE)).r2 as store_r2,
        COUNT(*) as store_observations
    FROM daily_store_data
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY region_id, territory_id, store_id
    HAVING COUNT(*) >= 30
),

territory_level AS (
    SELECT
        region_id,
        territory_id,
        AVG(store_roi) as avg_store_roi,
        COUNT(*) as num_stores,
        AVG(store_r2) as avg_r2,
        STDDEV(store_roi) as roi_variability,
        (ols_fit_agg(store_roi, store_r2)).coefficient as roi_predictability
    FROM store_level
    GROUP BY region_id, territory_id
),

region_level AS (
    SELECT
        region_id,
        AVG(avg_store_roi) as region_avg_roi,
        SUM(num_stores) as total_stores,
        STDDEV(avg_store_roi) as territory_variability,
        MIN(avg_store_roi) as worst_territory_roi,
        MAX(avg_store_roi) as best_territory_roi
    FROM territory_level
    GROUP BY region_id
),

-- Identify opportunities and risks
store_classification AS (
    SELECT
        sl.region_id,
        sl.territory_id,
        sl.store_id,
        sl.store_roi,
        sl.store_r2,
        tl.avg_store_roi as territory_avg,
        rl.region_avg_roi,
        CASE
            WHEN sl.store_roi > tl.avg_store_roi * 1.2 THEN 'Top Performer'
            WHEN sl.store_roi < tl.avg_store_roi * 0.8 THEN 'Underperformer'
            ELSE 'Average'
        END as performance_category,
        CASE
            WHEN sl.store_roi > rl.region_avg_roi AND sl.store_r2 > 0.7 THEN 'Best Practice - Replicate'
            WHEN sl.store_roi < rl.region_avg_roi AND sl.store_r2 > 0.7 THEN 'Consistent Low - Investigate'
            WHEN sl.store_roi > rl.region_avg_roi AND sl.store_r2 < 0.5 THEN 'High Variance - Monitor'
            ELSE 'Needs Analysis'
        END as action_recommendation
    FROM store_level sl
    JOIN territory_level tl ON sl.region_id = tl.region_id AND sl.territory_id = tl.territory_id
    JOIN region_level rl ON sl.region_id = rl.region_id
)

SELECT
    region_id,
    territory_id,
    store_id,
    ROUND(store_roi, 2) as store_marketing_roi,
    ROUND(store_r2, 3) as model_quality,
    ROUND(territory_avg, 2) as territory_benchmark,
    ROUND(region_avg_roi, 2) as region_benchmark,
    performance_category,
    action_recommendation
FROM store_classification
ORDER BY region_id, territory_id, store_roi DESC;
