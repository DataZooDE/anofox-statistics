-- Create sample monthly sales data by territory
CREATE OR REPLACE TABLE monthly_sales AS
SELECT
    territory_id,
    month_date,
    month_index,
    sales_amount
FROM (
    SELECT
        territory,
        DATE_TRUNC('month', CURRENT_DATE) - (i * INTERVAL '1 month') as month_date,
        i as month_index,
        CASE territory
            WHEN 'NORTH' THEN (600000 + i * 5000 + RANDOM() * 50000)::DOUBLE  -- Growing high performer
            WHEN 'SOUTH' THEN (550000 - i * 2000 + RANDOM() * 40000)::DOUBLE  -- Declining high performer
            WHEN 'EAST' THEN (300000 + i * 8000 + RANDOM() * 30000)::DOUBLE   -- Rising star
            ELSE (250000 - i * 1000 + RANDOM() * 25000)::DOUBLE                -- Declining low (WEST)
        END as sales_amount,
        territory as territory_id
    FROM
        (SELECT unnest(['NORTH', 'SOUTH', 'EAST', 'WEST']) as territory) territories
        CROSS JOIN range(0, 12) t(i)
);

-- Rolling 6-month sales trend by territory
WITH territory_trends AS (
    SELECT
        territory_id,
        month_date,
        sales_amount,
        (ols_fit_agg(sales_amount::DOUBLE, month_index::DOUBLE) OVER (
            PARTITION BY territory_id
            ORDER BY month_date
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        )).coefficient as trend_coefficient
    FROM monthly_sales
    WHERE month_date >= CURRENT_DATE - INTERVAL '12 months'
),
territory_classification AS (
    SELECT
        territory_id,
        ROUND(AVG(sales_amount), 0) as avg_sales,
        ROUND(AVG(trend_coefficient), 2) as avg_trend,
        CASE
            WHEN AVG(trend_coefficient) > 1000 THEN 'Growing'
            WHEN AVG(trend_coefficient) > -1000 THEN 'Stable'
            ELSE 'Declining'
        END as performance_status,
        CASE
            WHEN AVG(sales_amount) > 500000 AND AVG(trend_coefficient) > 0 THEN 'Star Territory'
            WHEN AVG(sales_amount) > 500000 AND AVG(trend_coefficient) < 0 THEN 'Cash Cow - Monitor'
            WHEN AVG(sales_amount) < 500000 AND AVG(trend_coefficient) > 0 THEN 'Rising Star'
            ELSE 'Needs Intervention'
        END as strategic_category
    FROM territory_trends
    GROUP BY territory_id
)
SELECT
    territory_id,
    avg_sales,
    avg_trend as monthly_growth,
    performance_status,
    strategic_category,
    CASE
        WHEN strategic_category = 'Star Territory' THEN 'Maintain & Expand'
        WHEN strategic_category = 'Rising Star' THEN 'Invest in Growth'
        WHEN strategic_category = 'Cash Cow - Monitor' THEN 'Investigate Decline'
        ELSE 'Urgent Action Needed'
    END as management_action
FROM territory_classification
ORDER BY avg_sales DESC;
