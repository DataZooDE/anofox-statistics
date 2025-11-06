LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample campaign data
CREATE OR REPLACE TABLE campaigns AS
SELECT
    i as campaign_id,
    spend,
    (spend * 2.3 + RANDOM() * 500)::DOUBLE as revenue  -- 2.3x ROI with noise
FROM (
    SELECT
        i,
        (1000 + RANDOM() * 4000)::DOUBLE as spend  -- $1k-5k spend per campaign
    FROM range(1, 31) t(i)
);

-- Calculate marketing ROI with statistical confidence using aggregate functions
SELECT
    'Marketing ROI' as metric,
    ROUND((ols_fit_agg(revenue, spend)).coefficient - 1, 2) as roi_multiplier,
    ROUND(((ols_fit_agg(revenue, spend)).coefficient - 1) * 100, 1) || '%' as roi_percentage,
    CASE
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05
             AND (ols_fit_agg(revenue, spend)).coefficient > 1.5 THEN 'Strong - Scale Up'
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05
             AND (ols_fit_agg(revenue, spend)).coefficient > 1.0 THEN 'Positive - Continue'
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05 THEN 'Negative - Stop Campaign'
        ELSE 'Inconclusive - Gather More Data'
    END as recommendation,
    ROUND((ols_fit_agg(revenue, spend)).p_value, 4) as p_value,
    ROUND((ols_fit_agg(revenue, spend)).r2, 3) as model_quality
FROM campaigns;
