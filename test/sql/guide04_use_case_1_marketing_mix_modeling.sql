LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample marketing campaign data
CREATE OR REPLACE TABLE weekly_campaigns AS
SELECT
    i as week_id,
    2024 as year,
    (50000 + i * 2.8 * tv + i * 4.2 * digital + i * 0.3 * print + i * 1.5 * radio + RANDOM() * 5000)::DOUBLE as revenue,
    tv::DOUBLE as tv_spend,
    digital::DOUBLE as digital_spend,
    print::DOUBLE as print_spend,
    radio::DOUBLE as radio_spend
FROM (
    SELECT
        i,
        (10 + RANDOM() * 5)::DOUBLE as tv,
        (8 + RANDOM() * 4)::DOUBLE as digital,
        (5 + RANDOM() * 3)::DOUBLE as print,
        (6 + RANDOM() * 3)::DOUBLE as radio
    FROM range(1, 21) t(i)
);

-- Analyze relationship between marketing spend and revenue using aggregate function
SELECT
    'tv' as channel,
    ROUND((anofox_stats_ols_fit_agg(revenue, [tv_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_stats_ols_fit_agg(revenue, [tv_spend], {'intercept': true})).r2, 3) as r2,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'digital' as channel,
    ROUND((anofox_stats_ols_fit_agg(revenue, [digital_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_stats_ols_fit_agg(revenue, [digital_spend], {'intercept': true})).r2, 3) as r2,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'print' as channel,
    ROUND((anofox_stats_ols_fit_agg(revenue, [print_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_stats_ols_fit_agg(revenue, [print_spend], {'intercept': true})).r2, 3) as r2,
    'Low Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'radio' as channel,
    ROUND((anofox_stats_ols_fit_agg(revenue, [radio_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_stats_ols_fit_agg(revenue, [radio_spend], {'intercept': true})).r2, 3) as r2,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
ORDER BY roi DESC;
