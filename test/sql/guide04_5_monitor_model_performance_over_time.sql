LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample monthly data
CREATE OR REPLACE TABLE monthly_data AS
SELECT
    DATE_TRUNC('month', CURRENT_DATE) - (i * INTERVAL '1 month') as month,
    marketing,
    (marketing * 2.5 + RANDOM() * 500)::DOUBLE as sales
FROM (
    SELECT
        i,
        (1000 + i * 50 + RANDOM() * 300)::DOUBLE as marketing  -- Increasing marketing spend
    FROM range(0, 24) t(i)
);

-- Track rolling 12-month ROI to detect relationship changes over time
SELECT
    month,
    ROUND((ols_fit_agg(sales, marketing) OVER (
        ORDER BY month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )).coefficient, 2) as rolling_12mo_roi,
    ROUND((ols_fit_agg(sales, marketing) OVER (
        ORDER BY month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )).r2, 3) as rolling_model_quality
FROM monthly_data
ORDER BY month DESC
LIMIT 12;  -- Show last 12 months
