-- Statistics Guide: Comprehensive OLS Aggregate Example
-- Demonstrates full statistical output and interpretation

-- Create sample data: advertising effectiveness study
CREATE TEMP TABLE advertising_data AS
SELECT
    CASE WHEN i <= 20 THEN 'campaign_a' ELSE 'campaign_b' END as campaign,
    i as week,
    (1000 + i * 50 + random() * 100)::DOUBLE as tv_spend,
    (500 + i * 25 + random() * 50)::DOUBLE as digital_spend,
    (10000 + i * 200 + 0.8 * (1000 + i * 50) + 1.2 * (500 + i * 25) + random() * 500)::DOUBLE as sales
FROM generate_series(1, 40) as t(i);

-- Run comprehensive OLS analysis per campaign
SELECT
    campaign,
    -- Coefficients (interpretation: change in sales per dollar spent)
    result.coefficients[1] as tv_roi,
    result.coefficients[2] as digital_roi,
    result.intercept as baseline_sales,
    -- Model fit
    result.r2 as r_squared,
    result.adj_r2 as adjusted_r_squared,
    result.n_obs as sample_size,
    result.n_features as num_predictors,
    -- Interpretation flags
    CASE
        WHEN result.r2 > 0.8 THEN 'Excellent fit'
        WHEN result.r2 > 0.6 THEN 'Good fit'
        WHEN result.r2 > 0.4 THEN 'Moderate fit'
        ELSE 'Poor fit'
    END as model_quality,
    CASE
        WHEN result.coefficients[1] > result.coefficients[2] THEN 'TV more effective'
        ELSE 'Digital more effective'
    END as channel_comparison
FROM (
    SELECT
        campaign,
        anofox_statistics_ols_agg(
            sales,
            [tv_spend, digital_spend],
            {'intercept': true}
        ) as result
    FROM advertising_data
    GROUP BY campaign
) sub;
