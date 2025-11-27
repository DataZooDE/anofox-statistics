LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample daily sales data
CREATE OR REPLACE TABLE daily_sales AS
SELECT
    i as sale_id,
    product_id,
    season,
    price,
    promotion_flag,
    competitor_price,
    units_sold
FROM (
    SELECT
        i,
        'PROD-' || ((i % 3) + 1) as product_id,
        CASE (i % 4)
            WHEN 0 THEN 'Winter'
            WHEN 1 THEN 'Spring'
            WHEN 2 THEN 'Summer'
            ELSE 'Fall'
        END as season,
        (50 + RANDOM() * 50)::DOUBLE as price,
        (RANDOM() < 0.3)::INT::DOUBLE as promotion_flag,  -- 30% promotions
        (45 + RANDOM() * 55)::DOUBLE as competitor_price,
        units_sold
    FROM (
        SELECT
            i,
            (100 - base_price * 0.8 + promotion * 15 + (competitor - 50) * 0.5 + RANDOM() * 20)::DOUBLE as units_sold,
            base_price,
            promotion,
            competitor
        FROM (
            SELECT
                i,
                (50 + RANDOM() * 50) as base_price,
                (RANDOM() < 0.3)::INT * 1.0 as promotion,
                (45 + RANDOM() * 55) as competitor
            FROM range(1, 301) t(i)
        )
    )
);

-- Analyze price sensitivity for each product/season combination
SELECT
    product_id,
    season,
    ROUND((anofox_statistics_ols_fit_agg(units_sold, price)).coefficients[1], 2) as price_sensitivity,
    ROUND((anofox_statistics_ols_fit_agg(units_sold, price)).r2, 3) as forecast_accuracy,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.8 THEN 'High Confidence'
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as forecast_reliability,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.7 THEN 'Auto-Replenish'
        ELSE 'Manual Review'
    END as inventory_strategy,
    COUNT(*) as sample_size
FROM daily_sales
GROUP BY product_id, season
HAVING COUNT(*) >= 30  -- Minimum data for reliable estimates
ORDER BY forecast_accuracy DESC;
