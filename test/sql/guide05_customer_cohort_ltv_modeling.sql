LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample cohort behavior data
CREATE OR REPLACE TABLE cohort_behavior AS
SELECT
    DATE '2023-01-01' + INTERVAL ((i / 25)) MONTH as cohort_month,
    (i % 25) as months_since_first,
    (100 +
     (i % 25) * 5 +                                    -- growth over time
     -0.5 * (i % 25) * (i % 25) +                     -- decay effect
     ((i / 25) % 12) * 10 +                           -- cohort variation
     RANDOM() * 20)::DOUBLE as avg_order_value,
    (100 - (i % 25) * 2 + RANDOM() * 10)::INTEGER as active_customers,
    ((100 + (i % 25) * 5) * (100 - (i % 25) * 2) + RANDOM() * 1000)::DOUBLE as total_revenue
FROM range(1, 301) t(i)
WHERE (i % 25) <= 24;

-- Cohort-based lifetime value analysis
WITH cohort_models_data AS (
    SELECT
        cohort_month,
        months_since_first,
        avg_order_value,
        active_customers,
        total_revenue
    FROM cohort_behavior
),

-- Model LTV curve for each cohort
cohort_models AS (
    SELECT
        cohort_month,
        (anofox_statistics_ols_fit_agg(
            avg_order_value::DOUBLE,
            [months_since_first::DOUBLE],
            {'intercept': true}
        )).coefficients[1] as ltv_slope,
        (anofox_statistics_ols_fit_agg(
            avg_order_value::DOUBLE,
            [months_since_first::DOUBLE],
            {'intercept': true}
        )).intercept as ltv_intercept,
        (anofox_statistics_ols_fit_agg(
            avg_order_value::DOUBLE,
            [months_since_first::DOUBLE],
            {'intercept': true}
        )).r2 as ltv_predictability,
        SUM(total_revenue) as cohort_total_revenue,
        AVG(active_customers) as avg_cohort_size
    FROM cohort_models_data
    GROUP BY cohort_month
),

-- Project future LTV
cohort_projections AS (
    SELECT
        cohort_month,
        ltv_intercept + ltv_slope * 36 as projected_36mo_value,
        ltv_predictability,
        cohort_total_revenue,
        avg_cohort_size,
        CASE
            WHEN ltv_slope > 0 AND ltv_predictability > 0.6 THEN 'Growing Cohort'
            WHEN ltv_slope < 0 AND ltv_predictability > 0.6 THEN 'Declining Cohort'
            ELSE 'Unstable Pattern'
        END as cohort_health,
        (ltv_intercept + ltv_slope * 36) * avg_cohort_size as projected_36mo_cohort_revenue
    FROM cohort_models
)

SELECT
    cohort_month,
    ROUND(cohort_total_revenue, 0) as revenue_to_date,
    ROUND(avg_cohort_size, 0) as cohort_size,
    ROUND(projected_36mo_value, 2) as projected_ltv_36mo,
    ROUND(ltv_predictability, 3) as model_r2,
    cohort_health,
    ROUND(projected_36mo_cohort_revenue, 0) as projected_cohort_revenue,
    CASE
        WHEN cohort_health = 'Growing Cohort' THEN 'Replicate Acquisition Strategy'
        WHEN cohort_health = 'Declining Cohort' THEN 'Improve Retention Programs'
        ELSE 'Monitor Closely'
    END as strategic_action
FROM cohort_projections
WHERE cohort_month >= '2023-01-01'
ORDER BY cohort_month DESC;
