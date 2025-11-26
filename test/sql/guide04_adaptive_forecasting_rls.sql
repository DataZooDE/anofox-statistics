LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Business Guide: Adaptive Demand Forecasting with RLS
-- Real-time demand prediction that adapts to market changes

-- Sample data: Product demand with evolving seasonality and trends
CREATE TEMP TABLE demand_history AS
SELECT
    CASE
        WHEN i <= 40 THEN 'product_seasonal'
        ELSE 'product_trending'
    END as product_id,
    i as week,
    -- Cyclical pattern changes over time
    CASE
        WHEN i <= 20 THEN (1000 + 200 * SIN(i * 0.5) + i * 10 + random() * 50)
        WHEN i <= 40 THEN (1200 + 300 * SIN(i * 0.5) + i * 15 + random() * 80)  -- Pattern shift
        ELSE (1500 + 100 * SIN(i * 0.5) + i * 25 + random() * 60)                -- New trend
    END::DOUBLE as actual_demand,
    -- Predictors: lagged demand and trend
    LAG(CASE
        WHEN i <= 20 THEN (1000 + 200 * SIN(i * 0.5) + i * 10 + random() * 50)
        WHEN i <= 40 THEN (1200 + 300 * SIN(i * 0.5) + i * 15 + random() * 80)
        ELSE (1500 + 100 * SIN(i * 0.5) + i * 25 + random() * 60)
    END, 1) OVER (ORDER BY i) as lagged_demand,
    i::DOUBLE as time_trend
FROM generate_series(1, 80) as t(i);

-- Compare static OLS vs adaptive RLS forecasting
SELECT
    product_id,
    -- Static OLS model (fixed coefficients)
    ols.coefficients[1] as ols_lag_coef,
    ols.coefficients[2] as ols_trend_coef,
    ols.r_squared as ols_r2,
    -- Adaptive RLS model (adjusts to changes)
    rls_slow.coefficients[1] as rls_slow_lag_coef,
    rls_slow.coefficients[2] as rls_slow_trend_coef,
    rls_slow.r_squared as rls_slow_r2,
    rls_slow.forgetting_factor as ff_slow,
    -- Fast-adapting RLS
    rls_fast.coefficients[1] as rls_fast_lag_coef,
    rls_fast.coefficients[2] as rls_fast_trend_coef,
    rls_fast.r_squared as rls_fast_r2,
    rls_fast.forgetting_factor as ff_fast,
    -- Business insight
    CASE
        WHEN rls_fast.r_squared > ols.r_squared + 0.05 THEN 'RLS significantly better - demand pattern changing'
        WHEN rls_fast.r_squared > ols.r_squared THEN 'RLS slightly better - moderate changes'
        ELSE 'OLS sufficient - stable demand pattern'
    END as model_recommendation
FROM (
    SELECT
        product_id,
        anofox_statistics_ols_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'intercept': true}
        ) as ols,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.98, 'intercept': true}
        ) as rls_slow,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.92, 'intercept': true}
        ) as rls_fast
    FROM demand_history
    WHERE lagged_demand IS NOT NULL
    GROUP BY product_id
) sub;

-- Rolling window analysis: Track model adaptation
WITH recent_performance AS (
    SELECT
        product_id,
        week,
        actual_demand,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) OVER (
            PARTITION BY product_id
            ORDER BY week
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) as rolling_model
    FROM demand_history
    WHERE lagged_demand IS NOT NULL
)
SELECT
    product_id,
    week,
    rolling_model.coefficients[1] as adaptive_lag_coefficient,
    rolling_model.r_squared as rolling_r2,
    CASE
        WHEN rolling_model.r_squared > 0.8 THEN 'High forecast confidence'
        WHEN rolling_model.r_squared > 0.6 THEN 'Moderate forecast confidence'
        ELSE 'Low confidence - model recalibration needed'
    END as forecast_confidence
FROM recent_performance
WHERE week >= 60
ORDER BY product_id, week
LIMIT 15;

-- Business recommendation
SELECT
    'Forecasting Strategy' as topic,
    'Use RLS for products with evolving demand patterns, seasonal shifts, or trend changes. Lower forgetting factors (0.90-0.95) adapt faster but may be more volatile.' as guidance
UNION ALL
SELECT
    'When to Use',
    'RLS is ideal for: (1) Fast-moving consumer goods with changing preferences, (2) Tech products with rapid adoption curves, (3) Markets with frequent promotions or competitive dynamics.' as use_cases;
