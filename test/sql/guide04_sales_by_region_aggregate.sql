-- Business Guide: Regional Sales Performance Analysis
-- Analyze how pricing and promotions affect sales across different regions

-- Sample data: Multi-region sales with pricing and promotion data
CREATE TEMP TABLE regional_sales AS
SELECT
    CASE i % 4
        WHEN 0 THEN 'north'
        WHEN 1 THEN 'south'
        WHEN 2 THEN 'east'
        ELSE 'west'
    END as region,
    i as week,
    (15.0 + random() * 5.0)::DOUBLE as price,
    (1000 + random() * 500)::DOUBLE as promo_spend,
    (500 + i * 10 - 50 * (15.0 + random() * 5.0) + 0.8 * (1000 + random() * 500) + random() * 200)::DOUBLE as units_sold
FROM generate_series(1, 80) as t(i);

-- Analyze price elasticity and promotion effectiveness per region
SELECT
    region,
    -- Key business metrics
    result.coefficients[1] as price_elasticity,
    result.coefficients[2] as promo_roi,
    result.intercept as baseline_demand,
    result.r2,
    result.n_obs as weeks_analyzed,
    -- Business interpretation
    CASE
        WHEN result.coefficients[1] < -30 THEN 'Highly price-sensitive'
        WHEN result.coefficients[1] < -20 THEN 'Moderately price-sensitive'
        ELSE 'Low price sensitivity'
    END as price_sensitivity_category,
    CASE
        WHEN result.coefficients[2] > 1.0 THEN 'Strong promotion response'
        WHEN result.coefficients[2] > 0.5 THEN 'Moderate promotion response'
        ELSE 'Weak promotion response'
    END as promo_effectiveness,
    -- Strategic recommendations
    CASE
        WHEN result.coefficients[1] < -30 AND result.coefficients[2] > 1.0
            THEN 'Focus on promotions over pricing'
        WHEN result.coefficients[1] > -20 AND result.coefficients[2] < 0.5
            THEN 'Consider premium pricing strategy'
        ELSE 'Balanced price and promotion strategy'
    END as strategy_recommendation
FROM (
    SELECT
        region,
        anofox_statistics_ols_agg(
            units_sold,
            [price, promo_spend],
            {'intercept': true}
        ) as result
    FROM regional_sales
    GROUP BY region
) sub
ORDER BY result.r2 DESC;

-- Calculate revenue impact of $1 price change
SELECT
    region,
    result.coefficients[1] as unit_change_per_dollar,
    AVG(price) as avg_price,
    result.coefficients[1] * AVG(price) as revenue_impact_per_dollar_increase,
    CASE
        WHEN result.coefficients[1] * AVG(price) < -1.0 THEN 'Price decrease would increase revenue'
        ELSE 'Current pricing may be optimal'
    END as pricing_insight
FROM (
    SELECT
        region,
        anofox_statistics_ols_agg(units_sold, [price], {'intercept': true}) as result
    FROM regional_sales
    GROUP BY region
) sub
JOIN (
    SELECT region, AVG(price) as avg_price
    FROM regional_sales
    GROUP BY region
) price_stats USING (region);
