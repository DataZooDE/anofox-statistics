-- Calculate beta for each stock (market sensitivity)
SELECT
    stock_id,
    (ols_fit_agg(stock_return, market_return)).coefficient as beta,
    (ols_fit_agg(stock_return, market_return)).r2 as correlation
FROM daily_returns
GROUP BY stock_id;
