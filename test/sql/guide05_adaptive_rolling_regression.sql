LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample daily revenue data
CREATE OR REPLACE TABLE daily_revenue AS
SELECT
    DATE '2024-01-01' + INTERVAL (i) DAY as date_id,
    (10000 +
     i * 50 +                              -- upward trend
     1000 * SIN(i * 0.1) +                 -- cyclical pattern
     CASE WHEN i > 150 THEN 5000 ELSE 0 END +  -- regime shift at day 150
     RANDOM() * 500)::DOUBLE as revenue
FROM range(0, 365) t(i);

-- Multi-window rolling regression to detect changes
WITH time_series AS (
    SELECT
        date_id,
        revenue::DOUBLE as y,
        ROW_NUMBER() OVER (ORDER BY date_id) as time_idx
    FROM daily_revenue
    WHERE date_id >= '2024-01-01'
),

-- 30-day rolling window
rolling_30 AS (
    SELECT
        date_id,
        y,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )).coefficients[1] as trend_30d,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )).r_squared as r2_30d
    FROM time_series
),

-- 90-day rolling window
rolling_90 AS (
    SELECT
        date_id,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        )).coefficients[1] as trend_90d,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        )).r_squared as r2_90d
    FROM time_series
),

-- Expanding window (all history)
expanding AS (
    SELECT
        date_id,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )).coefficients[1] as trend_expanding,
        (anofox_statistics_ols_fit_agg(y, [time_idx::DOUBLE], {'intercept': true}) OVER (
            ORDER BY time_idx
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )).r_squared as r2_expanding
    FROM time_series
)

-- Combine and detect regime changes
SELECT
    r30.date_id,
    r30.y as actual_revenue,
    ROUND(r30.trend_30d, 2) as short_term_trend,
    ROUND(r90.trend_90d, 2) as medium_term_trend,
    ROUND(exp.trend_expanding, 2) as long_term_trend,
    ROUND(r30.r2_30d, 3) as r2_short,
    ROUND(r90.r2_90d, 3) as r2_medium,
    ROUND(exp.r2_expanding, 3) as r2_long,
    -- Detect structural break
    CASE
        WHEN ABS(r30.trend_30d - r90.trend_90d) > 1000 THEN 'Regime Change'
        WHEN ABS(r30.trend_30d - exp.trend_expanding) > 500 THEN 'Trend Shift'
        ELSE 'Stable'
    END as regime_status,
    -- Forecast reliability
    CASE
        WHEN r30.r2_30d > 0.7 AND r90.r2_90d > 0.7 THEN 'High Confidence'
        WHEN r30.r2_30d > 0.5 AND r90.r2_90d > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as forecast_confidence
FROM rolling_30 r30
JOIN rolling_90 r90 ON r30.date_id = r90.date_id
JOIN expanding exp ON r30.date_id = exp.date_id
WHERE r30.date_id >= '2024-03-01'  -- Allow for window warmup
ORDER BY r30.date_id DESC
LIMIT 30;
