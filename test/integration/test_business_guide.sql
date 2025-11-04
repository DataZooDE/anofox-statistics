-- Test Business Guide Examples
.bail on
.mode box

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT '========================================';
SELECT 'Testing Business Guide Examples';
SELECT '========================================';

-- Use Case 1: Marketing Mix Modeling
SELECT '--- Use Case 1: Marketing Mix ---';
CREATE OR REPLACE TABLE weekly_campaigns AS
SELECT
    i as week_id,
    2024 as year,
    (50000 + i * 2.8 * tv + i * 4.2 * digital + i * 0.3 * print + i * 1.5 * radio + RANDOM() * 5000)::DOUBLE as revenue,
    tv::DOUBLE as tv_spend,
    digital::DOUBLE as digital_spend,
    print::DOUBLE as print_spend,
    radio::DOUBLE as radio_spend
FROM (
    SELECT
        i,
        (10 + RANDOM() * 5)::DOUBLE as tv,
        (8 + RANDOM() * 4)::DOUBLE as digital,
        (5 + RANDOM() * 3)::DOUBLE as print,
        (6 + RANDOM() * 3)::DOUBLE as radio
    FROM range(1, 21) t(i)
);

SELECT channel, roi, r_squared
FROM (
    SELECT 'tv' as channel, ROUND(ols_coeff_agg(revenue, tv_spend), 2) as roi, ROUND((ols_fit_agg(revenue, tv_spend)).r2, 3) as r_squared FROM weekly_campaigns WHERE year = 2024
    UNION ALL SELECT 'digital', ROUND(ols_coeff_agg(revenue, digital_spend), 2), ROUND((ols_fit_agg(revenue, digital_spend)).r2, 3) FROM weekly_campaigns WHERE year = 2024
) ORDER BY roi DESC LIMIT 2;

-- Use Case 2: Price Elasticity
SELECT '--- Use Case 2: Price Elasticity ---';
CREATE OR REPLACE TABLE sales_transactions AS
SELECT i as transaction_id, CURRENT_DATE - (RANDOM() * 180)::INT as transaction_date, category, price, quantity
FROM (
    SELECT i, CASE (i % 4) WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Clothing' WHEN 2 THEN 'Groceries' ELSE 'Luxury Goods' END as category,
           CASE (i % 4) WHEN 0 THEN (100 + RANDOM() * 50)::DOUBLE WHEN 1 THEN (30 + RANDOM() * 20)::DOUBLE WHEN 2 THEN (5 + RANDOM() * 3)::DOUBLE ELSE (500 + RANDOM() * 200)::DOUBLE END as price,
           CASE (i % 4) WHEN 0 THEN (50 - (100 + RANDOM() * 50) * 0.25 + RANDOM() * 10)::DOUBLE WHEN 1 THEN (80 - (30 + RANDOM() * 20) * 0.8 + RANDOM() * 15)::DOUBLE 
                        WHEN 2 THEN (100 - (5 + RANDOM() * 3) * 0.5 + RANDOM() * 20)::DOUBLE ELSE (20 - (500 + RANDOM() * 200) * 0.01 + RANDOM() * 5)::DOUBLE END as quantity
    FROM range(1, 401) t(i)
);

SELECT category, elasticity, elasticity_type
FROM (
    SELECT category, ROUND((ols_fit_agg(quantity, price)).coefficient, 3) as elasticity,
           CASE WHEN ABS((ols_fit_agg(quantity, price)).coefficient) > 0.5 THEN 'Elastic' ELSE 'Inelastic' END as elasticity_type
    FROM sales_transactions WHERE transaction_date >= CURRENT_DATE - INTERVAL '6 months' GROUP BY category
) ORDER BY ABS(elasticity) DESC LIMIT 2;

-- Use Case 4: Portfolio Beta
SELECT '--- Use Case 4: Portfolio Beta ---';
CREATE OR REPLACE TABLE daily_stock_returns AS
SELECT trade_date, stock_ticker, stock_return, market_return
FROM (
    SELECT CURRENT_DATE - (i % 504)::INT as trade_date, ticker as stock_ticker,
           CASE WHEN ticker = 'TECH' THEN market_ret * 1.45 + (RANDOM() * 0.04 - 0.02) WHEN ticker = 'GROWTH' THEN market_ret * 1.18 + (RANDOM() * 0.03 - 0.015)
                WHEN ticker = 'INDEX' THEN market_ret * 0.98 + (RANDOM() * 0.02 - 0.01) ELSE market_ret * 0.42 + (RANDOM() * 0.02 - 0.01) END::DOUBLE as stock_return,
           market_ret::DOUBLE as market_return
    FROM (SELECT unnest(['TECH', 'GROWTH', 'INDEX', 'UTILITY']) as ticker) tickers
    CROSS JOIN (SELECT i, (RANDOM() * 0.04 - 0.02)::DOUBLE as market_ret FROM range(1, 505) t(i)) dates
);

SELECT stock_ticker, beta FROM (
    SELECT stock_ticker, ROUND((ols_fit_agg(stock_return, market_return)).coefficient, 3) as beta
    FROM daily_stock_returns GROUP BY stock_ticker
) ORDER BY beta DESC LIMIT 2;

SELECT '========================================';
SELECT 'Business Guide: ALL TESTS PASSED';
SELECT '========================================';
