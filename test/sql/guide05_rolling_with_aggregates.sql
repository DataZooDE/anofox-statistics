LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Advanced Use Case: Window Functions + GROUP BY with Aggregates
-- Combine rolling analysis with per-group regression

-- Sample data: Multiple products over time
CREATE TEMP TABLE product_time_series AS
SELECT
    CASE i % 3
        WHEN 0 THEN 'product_a'
        WHEN 1 THEN 'product_b'
        ELSE 'product_c'
    END as product_id,
    (i / 3)::INT as week,
    DATE '2024-01-01' + INTERVAL (i / 3) WEEK as week_start,
    (50 + (i / 3) * 2 + random() * 10)::DOUBLE as price,
    (100 + (i / 3) * 5 + random() * 20)::DOUBLE as units_sold
FROM generate_series(1, 90) as t(i);

-- Technique 1: Rolling window within each product (per-product adaptive models)
WITH rolling_models AS (
    SELECT
        product_id,
        week,
        week_start,
        price,
        units_sold,
        anofox_stats_ols_fit_agg(units_sold, [price], {'intercept': true}) OVER (
            PARTITION BY product_id
            ORDER BY week
            ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
        ) as rolling_model
    FROM product_time_series
)
SELECT
    product_id,
    week,
    rolling_model.coefficients[1] as price_elasticity,
    rolling_model.r2 as model_quality,
    rolling_model.n_obs as window_size,
    -- Detect significant elasticity changes
    LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week) as prev_elasticity,
    rolling_model.coefficients[1] - LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week) as elasticity_change,
    CASE
        WHEN ABS(rolling_model.coefficients[1] - LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week)) > 2
            THEN 'Significant change detected'
        ELSE 'Stable elasticity'
    END as change_indicator
FROM rolling_models
WHERE week >= 8  -- Need sufficient history
ORDER BY product_id, week
LIMIT 20;

-- Technique 2: Compare static vs adaptive models per product
WITH static_models AS (
    SELECT
        product_id,
        anofox_stats_ols_fit_agg(units_sold, [price], {'intercept': true}) as full_period_model
    FROM product_time_series
    GROUP BY product_id
),
adaptive_models AS (
    SELECT
        product_id,
        anofox_stats_rls_fit_agg(units_sold, [price], {'forgetting_factor': 0.92, 'intercept': true}) as adaptive_model
    FROM product_time_series
    GROUP BY product_id
)
SELECT
    sm.product_id,
    sm.full_period_model.coefficients[1] as static_elasticity,
    sm.full_period_model.r2 as static_r2,
    am.adaptive_model.coefficients[1] as adaptive_elasticity,
    am.adaptive_model.r2 as adaptive_r2,
    -- Performance comparison
    CASE
        WHEN am.adaptive_model.r2 > sm.full_period_model.r2 + 0.05
            THEN 'Adaptive model significantly better'
        WHEN am.adaptive_model.r2 > sm.full_period_model.r2
            THEN 'Adaptive model slightly better'
        ELSE 'Static model sufficient'
    END as model_comparison,
    -- Elasticity stability
    ABS(am.adaptive_model.coefficients[1] - sm.full_period_model.coefficients[1]) as elasticity_drift
FROM static_models sm
JOIN adaptive_models am USING (product_id)
ORDER BY sm.product_id;

-- Technique 3: Aggregate then window (summary metrics over time)
WITH weekly_summary AS (
    SELECT
        week,
        week_start,
        anofox_stats_ols_fit_agg(units_sold, [price], {'intercept': true}) as weekly_model
    FROM product_time_series
    GROUP BY week, week_start
)
SELECT
    week,
    weekly_model.r2 as weekly_r2,
    weekly_model.coefficients[1] as weekly_elasticity,
    -- Rolling average of R² (market-wide model quality trend)
    AVG(weekly_model.r2) OVER (
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) as rolling_avg_r2,
    -- Rolling average of elasticity
    AVG(weekly_model.coefficients[1]) OVER (
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) as rolling_avg_elasticity,
    -- Volatility of elasticity
    STDDEV(weekly_model.coefficients[1]) OVER (
        ORDER BY week
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) as elasticity_volatility,
    CASE
        WHEN STDDEV(weekly_model.coefficients[1]) OVER (
            ORDER BY week
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) > 1.0 THEN 'High volatility - unstable market'
        ELSE 'Stable market conditions'
    END as market_stability
FROM weekly_summary
WHERE week >= 5
ORDER BY week
LIMIT 20;

-- Technique 4: Cross-sectional comparison with time trends
WITH monthly_by_product AS (
    SELECT
        product_id,
        (week / 4)::INT as month,
        anofox_stats_ols_fit_agg(units_sold, [price], {'intercept': true}) as monthly_model
    FROM product_time_series
    GROUP BY product_id, month
)
SELECT
    month,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_a') as product_a_r2,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_b') as product_b_r2,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_c') as product_c_r2,
    AVG(monthly_model.r2) as avg_r2_across_products,
    -- Detect if one product is diverging from others
    MAX(monthly_model.r2) - MIN(monthly_model.r2) as r2_spread,
    CASE
        WHEN MAX(monthly_model.r2) - MIN(monthly_model.r2) > 0.2
            THEN 'High variation across products'
        ELSE 'Similar model quality'
    END as cross_product_assessment
FROM monthly_by_product
GROUP BY month
ORDER BY month;
