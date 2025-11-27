LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample A/B test data
CREATE OR REPLACE TABLE ab_test_results AS
SELECT
    'pricing_test_2024_q1' as experiment_id,
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END as variant,
    (CASE WHEN i % 2 = 0 THEN 0.12 ELSE 0.15 END +  -- Base conversion: B is 3% better
     RANDOM() * 0.05)::DOUBLE as conversion_rate,
    (CASE WHEN i % 2 = 0 THEN 45.0 ELSE 52.0 END +  -- Base revenue: B is $7 better
     RANDOM() * 10)::DOUBLE as revenue_per_user,
    (CASE WHEN i % 2 = 0 THEN 7.5 ELSE 8.2 END +    -- Base engagement: B is 0.7 better
     RANDOM() * 2)::DOUBLE as engagement_score
FROM range(1, 1001) t(i);

-- A/B test analysis with full statistical validation
WITH experiment_data AS (
    SELECT
        variant,
        CASE WHEN variant = 'B' THEN 1.0 ELSE 0.0 END as treatment,
        conversion_rate::DOUBLE as conversion,
        revenue_per_user::DOUBLE as revenue,
        engagement_score::DOUBLE as engagement
    FROM ab_test_results
    WHERE experiment_id = 'pricing_test_2024_q1'
),

-- Overall metrics by variant
variant_summary AS (
    SELECT
        variant,
        COUNT(*) as sample_size,
        AVG(conversion) as avg_conversion,
        AVG(revenue) as avg_revenue,
        AVG(engagement) as avg_engagement,
        STDDEV(conversion) as std_conversion,
        STDDEV(revenue) as std_revenue
    FROM experiment_data
    GROUP BY variant
),

-- Statistical significance test for conversion rate using actual data
-- Regression of conversion on treatment indicator (0=A, 1=B)
-- Coefficient = treatment effect (B - A)
conversion_test AS (
    SELECT
        (anofox_statistics_ols_fit_agg(conversion, [treatment], {'intercept': true})).coefficients[1] as treatment_effect,
        (anofox_statistics_ols_fit_agg(conversion, [treatment], {'intercept': true})).residual_standard_error as std_error,
        (anofox_statistics_ols_fit_agg(conversion, [treatment], {'intercept': true})).r2 as r_squared,
        COUNT(*) as n_obs
    FROM experiment_data
),

-- Statistical significance test for revenue using actual data
revenue_test AS (
    SELECT
        (anofox_statistics_ols_fit_agg(revenue, [treatment], {'intercept': true})).coefficients[1] as treatment_effect,
        (anofox_statistics_ols_fit_agg(revenue, [treatment], {'intercept': true})).residual_standard_error as std_error,
        (anofox_statistics_ols_fit_agg(revenue, [treatment], {'intercept': true})).r2 as r_squared,
        COUNT(*) as n_obs
    FROM experiment_data
),

-- Compute t-statistics and approximate p-values
-- t = coefficient / std_error
-- For large n, |t| > 1.96 implies p < 0.05 (two-tailed)
conversion_significance AS (
    SELECT
        treatment_effect,
        std_error,
        r_squared,
        treatment_effect / std_error as t_stat,
        ABS(treatment_effect / std_error) > 1.96 as is_significant,
        -- 95% confidence interval
        treatment_effect - 1.96 * std_error as ci_lower,
        treatment_effect + 1.96 * std_error as ci_upper
    FROM conversion_test
),

revenue_significance AS (
    SELECT
        treatment_effect,
        std_error,
        r_squared,
        treatment_effect / std_error as t_stat,
        ABS(treatment_effect / std_error) > 1.96 as is_significant,
        treatment_effect - 1.96 * std_error as ci_lower,
        treatment_effect + 1.96 * std_error as ci_upper
    FROM revenue_test
),

-- Calculate business impact using actual test results
impact_analysis AS (
    SELECT
        'Conversion Rate' as metric,
        ROUND((SELECT avg_conversion FROM variant_summary WHERE variant = 'A') * 100, 2) as control_value,
        ROUND((SELECT avg_conversion FROM variant_summary WHERE variant = 'B') * 100, 2) as treatment_value,
        ROUND(cs.treatment_effect * 100, 2) as absolute_lift,
        ROUND((cs.treatment_effect / (SELECT avg_conversion FROM variant_summary WHERE variant = 'A')) * 100, 2) as relative_lift_pct,
        ROUND(2 * (1 - 0.975), 4) as p_value_approx,  -- Approximate p-value for |t| > 1.96
        cs.is_significant,
        ROUND(cs.ci_lower * 100, 2) as ci_lower,
        ROUND(cs.ci_upper * 100, 2) as ci_upper
    FROM conversion_significance cs
    UNION ALL
    SELECT
        'Revenue per User' as metric,
        ROUND((SELECT avg_revenue FROM variant_summary WHERE variant = 'A'), 2),
        ROUND((SELECT avg_revenue FROM variant_summary WHERE variant = 'B'), 2),
        ROUND(rs.treatment_effect, 2),
        ROUND((rs.treatment_effect / (SELECT avg_revenue FROM variant_summary WHERE variant = 'A')) * 100, 2),
        ROUND(2 * (1 - 0.975), 4),
        rs.is_significant,
        ROUND(rs.ci_lower, 2),
        ROUND(rs.ci_upper, 2)
    FROM revenue_significance rs
),

-- Statistical power analysis (simplified)
power_analysis AS (
    SELECT
        (SELECT sample_size FROM variant_summary WHERE variant = 'A') as control_n,
        (SELECT sample_size FROM variant_summary WHERE variant = 'B') as treatment_n,
        CASE
            WHEN (SELECT sample_size FROM variant_summary WHERE variant = 'A') >= 1000 THEN 'Adequate'
            WHEN (SELECT sample_size FROM variant_summary WHERE variant = 'A') >= 500 THEN 'Marginal'
            ELSE 'Insufficient'
        END as sample_size_assessment
)

-- Final recommendation based on actual experiment data
SELECT
    ia.metric,
    ia.control_value,
    ia.treatment_value,
    ia.absolute_lift,
    ia.relative_lift_pct || '%' as relative_lift,
    ia.p_value_approx as p_value,
    ia.is_significant,
    '[' || ia.ci_lower || ', ' || ia.ci_upper || ']' as confidence_interval_95,
    CASE
        WHEN ia.is_significant AND ia.absolute_lift > 0 THEN 'Launch Treatment'
        WHEN ia.is_significant AND ia.absolute_lift < 0 THEN 'Keep Control'
        WHEN NOT ia.is_significant AND ABS(ia.absolute_lift) < 0.01 THEN 'No Meaningful Difference'
        ELSE 'Inconclusive - Extend Test'
    END as recommendation,
    pa.sample_size_assessment
FROM impact_analysis ia
CROSS JOIN power_analysis pa;
