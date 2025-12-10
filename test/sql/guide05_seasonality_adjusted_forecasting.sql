LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample monthly sales data with seasonality
CREATE OR REPLACE TABLE monthly_sales AS
SELECT
    i as month_id,
    DATE '2022-01-01' + INTERVAL (i) MONTH as month_date,
    (50000 +
     i * 500 +                                           -- trend
     10000 * SIN((i % 12) * 3.14159 / 6) +             -- seasonality
     RANDOM() * 2000)::DOUBLE as revenue
FROM range(1, 37) t(i);

-- Seasonal decomposition and forecasting
WITH monthly_data AS (
    SELECT
        month_id,
        revenue::DOUBLE as revenue,
        EXTRACT(MONTH FROM month_date) as month_num,
        ROW_NUMBER() OVER (ORDER BY month_date) as time_idx
    FROM monthly_sales
),

-- Fit overall trend using aggregate function
trend_model AS (
    SELECT
        anofox_stats_ols_fit_agg(revenue, [time_idx::DOUBLE], {'intercept': true}) as model
    FROM monthly_data
),

-- Calculate detrended values
detrended AS (
    SELECT
        md.month_id,
        md.revenue,
        md.month_num,
        md.time_idx,
        md.revenue - ((tm.model).intercept + (tm.model).coefficients[1] * md.time_idx) as detrended_revenue
    FROM monthly_data md
    CROSS JOIN trend_model tm
),

-- Calculate seasonal averages
seasonal_factors AS (
    SELECT
        month_num,
        AVG(detrended_revenue) as seasonal_component
    FROM detrended
    GROUP BY month_num
),

-- Forecast next 12 months
future_months AS (
    SELECT
        (SELECT MAX(time_idx) FROM monthly_data) + ROW_NUMBER() OVER (ORDER BY m.month_num) as future_idx,
        m.month_num
    FROM (SELECT UNNEST(GENERATE_SERIES(1, 12)) as month_num) m
),

forecasts AS (
    SELECT
        fm.future_idx as month_ahead,
        fm.month_num,
        -- Trend component
        (tm.model).intercept + (tm.model).coefficients[1] * fm.future_idx as trend_component,
        -- Seasonal component
        sf.seasonal_component,
        -- Combined forecast
        ((tm.model).intercept + (tm.model).coefficients[1] * fm.future_idx) + sf.seasonal_component as forecast
    FROM future_months fm
    CROSS JOIN trend_model tm
    LEFT JOIN seasonal_factors sf ON fm.month_num = sf.month_num
)

SELECT
    month_ahead,
    month_num,
    ROUND(trend_component, 0) as trend,
    ROUND(seasonal_component, 0) as seasonal,
    ROUND(forecast, 0) as total_forecast
FROM forecasts
ORDER BY month_ahead;
