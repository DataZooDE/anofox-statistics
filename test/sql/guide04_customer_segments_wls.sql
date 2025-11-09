LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Business Guide: Customer Lifetime Value by Segment (Weighted Analysis)
-- Weight analysis by customer value to focus on high-value relationships

-- Sample data: Customer cohorts with varying reliability and value
CREATE TEMP TABLE customer_cohorts AS
SELECT
    CASE
        WHEN i <= 30 THEN 'enterprise'
        WHEN i <= 60 THEN 'smb'
        ELSE 'startup'
    END as segment,
    i as customer_id,
    (i * 100)::DOUBLE as acquisition_cost,
    (1 + i / 20.0)::DOUBLE as tenure_months,
    CASE
        WHEN i <= 30 THEN (10000 + acquisition_cost * 0.8 + tenure_months * 1200 + random() * 1000)
        WHEN i <= 60 THEN (3000 + acquisition_cost * 0.6 + tenure_months * 500 + random() * 800)
        ELSE (800 + acquisition_cost * 0.4 + tenure_months * 200 + random() * 400)
    END::DOUBLE as lifetime_revenue,
    -- Weight by contract size (larger customers = more reliable data)
    CASE
        WHEN i <= 30 THEN 5.0  -- Enterprise: high weight
        WHEN i <= 60 THEN 2.0  -- SMB: medium weight
        ELSE 1.0               -- Startup: standard weight
    END as customer_value_weight
FROM generate_series(1, 90) as t(i);

-- Weighted analysis: Focus on high-value customer patterns
SELECT
    segment,
    -- Standard OLS (treats all customers equally)
    ols.coefficients[1] as ols_acq_cost_roi,
    ols.coefficients[2] as ols_tenure_value,
    ols.r2 as ols_r2,
    -- Weighted LS (emphasizes high-value customers)
    wls.coefficients[1] as wls_acq_cost_roi,
    wls.coefficients[2] as wls_tenure_value,
    wls.r2 as wls_r2,
    -- Business insights
    CASE
        WHEN wls.coefficients[1] > 1.0 THEN 'Positive ROI on acquisition'
        WHEN wls.coefficients[1] > 0.5 THEN 'Break-even on acquisition'
        ELSE 'Review acquisition strategy'
    END as acquisition_assessment,
    wls.coefficients[2] * 12 as annual_value_per_customer,
    wls.n_obs as customers_analyzed
FROM (
    SELECT
        segment,
        anofox_statistics_ols_agg(
            lifetime_revenue,
            [acquisition_cost, tenure_months],
            {'intercept': true}
        ) as ols,
        anofox_statistics_wls_agg(
            lifetime_revenue,
            [acquisition_cost, tenure_months],
            customer_value_weight,
            {'intercept': true}
        ) as wls
    FROM customer_cohorts
    GROUP BY segment
) sub
ORDER BY segment;

-- Calculate LTV:CAC ratio by segment
WITH ltv_analysis AS (
    SELECT
        segment,
        result.coefficients[1] as roi_per_dollar,
        result.coefficients[2] as monthly_value,
        result.intercept as base_value,
        cac.avg_cac
    FROM (
        SELECT
            segment,
            anofox_statistics_wls_agg(
                lifetime_revenue,
                [acquisition_cost, tenure_months],
                customer_value_weight,
                {'intercept': true}
            ) as result
        FROM customer_cohorts
        GROUP BY segment
    ) sub
    JOIN (
        SELECT segment, AVG(acquisition_cost) as avg_cac
        FROM customer_cohorts
        GROUP BY segment
    ) cac USING (segment)
)
SELECT
    segment,
    avg_cac,
    monthly_value * 24 as estimated_24mo_ltv,
    (monthly_value * 24) / NULLIF(avg_cac, 0) as ltv_cac_ratio,
    CASE
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 3.0 THEN 'Excellent (LTV > 3x CAC)'
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 2.0 THEN 'Good (LTV > 2x CAC)'
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 1.0 THEN 'Acceptable'
        ELSE 'Concerning - Review unit economics'
    END as segment_health
FROM ltv_analysis
ORDER BY ltv_cac_ratio DESC;
