-- Statistics Guide: Recursive Least Squares - Adaptive Online Learning
-- Demonstrates forgetting factor tuning for time-varying relationships

-- Scenario: Market beta estimation with regime changes
-- Stock's relationship to market changes over time
CREATE TEMP TABLE stock_market_data AS
SELECT
    CASE
        WHEN i <= 20 THEN 'regime_bull'
        WHEN i <= 40 THEN 'regime_bear'
        ELSE 'regime_recovery'
    END as market_regime,
    i as time_period,
    -- Market return (independent variable)
    (0.01 + random() * 0.02 - 0.01)::DOUBLE as market_return,
    -- Stock return (dependent variable) with changing beta
    CASE
        WHEN i <= 20 THEN (0.005 + 1.2 * (0.01 + random() * 0.02 - 0.01) + random() * 0.01)  -- Bull: high beta (1.2)
        WHEN i <= 40 THEN (0.001 + 0.8 * (0.01 + random() * 0.02 - 0.01) + random() * 0.015) -- Bear: low beta (0.8)
        ELSE (0.003 + 1.5 * (0.01 + random() * 0.02 - 0.01) + random() * 0.012)              -- Recovery: very high beta (1.5)
    END::DOUBLE as stock_return
FROM generate_series(1, 60) as t(i);

-- Compare different forgetting factors
SELECT
    'Forgetting Factor: 1.0 (OLS equivalent)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Averages all data equally - slow to adapt' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 1.0, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.98 (slow adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Gradual weight decay - moderate adaptation' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.98, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.95 (moderate adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Balanced - good for detecting regime changes' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.95, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.90 (fast adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Heavy decay - very responsive to recent changes' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.90, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub;

-- Per-regime analysis
SELECT
    market_regime,
    result.coefficients[1] as regime_beta,
    result.forgetting_factor,
    result.r2,
    result.n_obs
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_agg(
            stock_return,
            [market_return],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) as result
    FROM stock_market_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

-- Guidance
SELECT
    'Choosing Forgetting Factor' as topic,
    'λ close to 1.0: More stable, slower adaptation. λ < 0.95: More responsive, tracks changes quickly. Choose based on how fast you expect relationships to change.' as guidance;
