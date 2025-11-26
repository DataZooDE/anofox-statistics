LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Business Guide: Portfolio Risk Management with Ridge Regression
-- Handle correlated securities in portfolio beta estimation

-- Sample data: Portfolio holdings with correlated market factors
CREATE TEMP TABLE portfolio_holdings AS
SELECT
    CASE
        WHEN i <= 30 THEN 'stock_tech_a'
        WHEN i <= 60 THEN 'stock_tech_b'
        ELSE 'stock_finance'
    END as ticker,
    i as trading_day,
    (random() * 0.04 - 0.02)::DOUBLE as return,
    (random() * 0.03 - 0.015)::DOUBLE as market_return,
    (random() * 0.035 - 0.0175)::DOUBLE as tech_sector_return,  -- Correlated with market
    (random() * 0.025 - 0.0125)::DOUBLE as value_factor,
    (random() * 0.03 - 0.015)::DOUBLE as momentum_factor
FROM generate_series(1, 90) as t(i);

-- Compare OLS vs Ridge for beta estimation with correlated factors
SELECT
    ticker,
    -- OLS (may have unstable coefficients due to multicollinearity)
    ols.coefficients[1] as ols_market_beta,
    ols.coefficients[2] as ols_sector_beta,
    ols.coefficients[3] as ols_value_beta,
    ols.coefficients[4] as ols_momentum_beta,
    ols.r_squared as ols_r2,
    -- Ridge (stabilized coefficients)
    ridge.coefficients[1] as ridge_market_beta,
    ridge.coefficients[2] as ridge_sector_beta,
    ridge.coefficients[3] as ridge_value_beta,
    ridge.coefficients[4] as ridge_momentum_beta,
    ridge.r_squared as ridge_r2,
    ridge.lambda,
    -- Risk assessment
    CASE
        WHEN ridge.coefficients[1] > 1.2 THEN 'High systematic risk'
        WHEN ridge.coefficients[1] > 0.8 THEN 'Market-level risk'
        ELSE 'Defensive positioning'
    END as risk_profile
FROM (
    SELECT
        ticker,
        anofox_statistics_ols_fit_agg(
            return,
            [market_return, tech_sector_return, value_factor, momentum_factor],
            {'intercept': true}
        ) as ols,
        anofox_statistics_ridge_fit_agg(
            return,
            [market_return, tech_sector_return, value_factor, momentum_factor],
            {'lambda': 1.0, 'intercept': true}
        ) as ridge
    FROM portfolio_holdings
    GROUP BY ticker
) sub
ORDER BY ticker;

-- Calculate portfolio-level risk metrics
WITH stock_betas AS (
    SELECT
        ticker,
        result.coefficients[1] as market_beta,
        result.coefficients[2] as sector_beta,
        result.r_squared
    FROM (
        SELECT
            ticker,
            anofox_statistics_ridge_fit_agg(
                return,
                [market_return, tech_sector_return],
                {'lambda': 1.0, 'intercept': true}
            ) as result
        FROM portfolio_holdings
        GROUP BY ticker
    ) sub
)
SELECT
    ticker,
    market_beta,
    sector_beta,
    r2 as model_fit,
    -- Risk interpretation for portfolio management
    CASE
        WHEN market_beta > 1.5 THEN 'Aggressive - Consider hedging'
        WHEN market_beta > 1.0 THEN 'Growth-oriented'
        WHEN market_beta > 0.7 THEN 'Balanced'
        ELSE 'Defensive/Low volatility'
    END as investment_style,
    -- Expected volatility relative to market
    ROUND(market_beta * 100, 1) || '% of market volatility' as volatility_profile
FROM stock_betas
ORDER BY market_beta DESC;
