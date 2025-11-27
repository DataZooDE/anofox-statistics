LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample stock returns data
CREATE OR REPLACE TABLE daily_stock_returns AS
SELECT
    trade_date,
    stock_ticker,
    stock_return,
    market_return
FROM (
    SELECT
        CURRENT_DATE - (i % 504)::INT as trade_date,  -- 2 years of trading days
        ticker as stock_ticker,
        CASE
            WHEN ticker = 'TECH' THEN market_ret * 1.45 + (RANDOM() * 0.04 - 0.02)  -- High beta
            WHEN ticker = 'GROWTH' THEN market_ret * 1.18 + (RANDOM() * 0.03 - 0.015)  -- Above market
            WHEN ticker = 'INDEX' THEN market_ret * 0.98 + (RANDOM() * 0.02 - 0.01)  -- Market
            ELSE market_ret * 0.42 + (RANDOM() * 0.02 - 0.01)  -- Low beta (UTILITY)
        END::DOUBLE as stock_return,
        market_ret::DOUBLE as market_return
    FROM
        (SELECT unnest(['TECH', 'GROWTH', 'INDEX', 'UTILITY']) as ticker) tickers
        CROSS JOIN (
            SELECT
                i,
                (RANDOM() * 0.04 - 0.02)::DOUBLE as market_ret  -- Market returns: -2% to +2% daily
            FROM range(1, 505) t(i)
        ) dates
);

-- Calculate beta (market sensitivity) for each stock
SELECT
    stock_ticker,
    ROUND((anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1], 3) as beta,
    ROUND((anofox_statistics_ols_fit_agg(stock_return, market_return)).r2, 3) as r_squared,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 1.2 THEN 'High Risk'
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 0.8 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_category,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 1.0 THEN 'Aggressive'
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 0.5 THEN 'Moderate'
        ELSE 'Defensive'
    END as investor_suitability
FROM daily_stock_returns
WHERE trade_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY stock_ticker
ORDER BY beta DESC;
