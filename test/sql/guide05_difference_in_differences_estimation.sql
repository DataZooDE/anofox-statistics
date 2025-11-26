LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample weekly store sales data
CREATE OR REPLACE TABLE weekly_store_sales AS
SELECT
    store_id,
    DATE '2024-01-01' + INTERVAL (week_num * 7) DAY as week_date,
    (10000 +                                          -- Base sales
     CASE WHEN store_id <= 105 THEN 500 ELSE 0 END + -- Treatment group baseline
     CASE WHEN week_num >= 20 THEN 300 ELSE 0 END +  -- Time trend
     CASE WHEN store_id <= 105 AND week_num >= 20    -- Treatment effect (DID)
          THEN 1200 ELSE 0 END +
     RANDOM() * 500)::DOUBLE as sales                -- Random noise
FROM
    (SELECT unnest([101, 102, 103, 104, 105, 201, 202, 203, 204, 205]) as store_id) stores
    CROSS JOIN
    (SELECT i as week_num FROM range(0, 52) t(i)) weeks;

-- Difference-in-differences analysis using actual data
WITH store_data AS (
    SELECT
        store_id,
        week_date,
        sales::DOUBLE as sales,
        -- Treatment indicator (stores that got new layout)
        CASE WHEN store_id <= 105 THEN 1.0 ELSE 0.0 END as treatment_group,
        -- Post-intervention period (week 20 = approximately June 1)
        CASE WHEN week_date >= '2024-05-20' THEN 1.0 ELSE 0.0 END as post_period,
        -- Interaction term (DID estimator)
        (CASE WHEN store_id <= 105 THEN 1.0 ELSE 0.0 END) *
        (CASE WHEN week_date >= '2024-05-20' THEN 1.0 ELSE 0.0 END) as treatment_post
    FROM weekly_store_sales
),

-- Simple DID using treatment_post indicator
-- Coefficient on treatment_post = causal effect estimate
did_estimate AS (
    SELECT
        (anofox_statistics_ols_fit_agg(sales, treatment_post)).coefficients[1] as did_coefficient,
        (anofox_statistics_ols_fit_agg(sales, treatment_post)).std_error as std_error,
        (anofox_statistics_ols_fit_agg(sales, treatment_post)).r_squared as r_squared,
        COUNT(*) as n_obs
    FROM store_data
),

-- Compute significance
did_significance AS (
    SELECT
        did_coefficient as causal_effect,
        std_error,
        r_squared,
        did_coefficient / std_error as t_statistic,
        ABS(did_coefficient / std_error) > 1.96 as is_significant,
        did_coefficient - 1.96 * std_error as ci_lower,
        did_coefficient + 1.96 * std_error as ci_upper
    FROM did_estimate
),

-- Average treatment effects by period
descriptive_stats AS (
    SELECT
        CASE WHEN treatment_group = 1 THEN 'Treatment' ELSE 'Control' END as group_name,
        CASE WHEN post_period = 1 THEN 'Post' ELSE 'Pre' END as period,
        AVG(sales) as avg_sales,
        COUNT(*) as n_weeks
    FROM store_data
    GROUP BY treatment_group, post_period
),

-- Calculate parallel trends check (descriptive)
parallel_trends AS (
    SELECT
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Treatment' AND period = 'Pre') as treatment_pre,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Control' AND period = 'Pre') as control_pre,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Treatment' AND period = 'Post') as treatment_post,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Control' AND period = 'Post') as control_post
),

-- Manual DID calculation for verification
manual_did AS (
    SELECT
        treatment_post - treatment_pre as treatment_diff,
        control_post - control_pre as control_diff,
        (treatment_post - treatment_pre) - (control_post - control_pre) as did_manual
    FROM parallel_trends
)

-- Final results combining regression estimates and descriptive stats
SELECT
    'DID Regression Estimate' as analysis_type,
    'Causal Effect' as metric,
    ROUND(ds.causal_effect, 2)::VARCHAR as value,
    ROUND(ds.std_error, 2)::VARCHAR as std_error,
    CASE WHEN ds.is_significant THEN 'Yes' ELSE 'No' END as significant,
    '[' || ROUND(ds.ci_lower, 2)::VARCHAR || ', ' || ROUND(ds.ci_upper, 2)::VARCHAR || ']' as ci_95,
    CASE
        WHEN ds.is_significant AND ds.causal_effect > 0 THEN
            'New layout increased sales by $' || ROUND(ds.causal_effect, 0)::VARCHAR || ' per week (causal)'
        WHEN ds.is_significant AND ds.causal_effect < 0 THEN
            'New layout decreased sales by $' || ABS(ROUND(ds.causal_effect, 0))::VARCHAR || ' per week (causal)'
        ELSE 'No significant causal effect detected'
    END as interpretation
FROM did_significance ds

UNION ALL

SELECT
    'Manual DID Calculation',
    'Treatment Effect',
    ROUND(md.treatment_diff, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Treatment group changed by $' || ROUND(md.treatment_diff, 0)::VARCHAR
FROM manual_did md

UNION ALL

SELECT
    'Manual DID Calculation',
    'Control Effect',
    ROUND(md.control_diff, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Control group changed by $' || ROUND(md.control_diff, 0)::VARCHAR
FROM manual_did md

UNION ALL

SELECT
    'Manual DID Calculation',
    'DID (Manual)',
    ROUND(md.did_manual, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Manual DID = ' || ROUND(md.did_manual, 0)::VARCHAR || ' (should match regression estimate)'
FROM manual_did md

ORDER BY analysis_type DESC, metric;
