LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Quick Start Example: Ridge Regression Aggregate
-- Demonstrates L2 regularization for handling multicollinearity

-- Sample data: stock returns with correlated factors
CREATE TEMP TABLE stock_returns AS
SELECT
    'tech_stock' as ticker,
    1 as period,
    0.05 as return,
    0.03 as market_return,
    0.04 as sector_return,  -- Highly correlated with market_return
    0.02 as momentum
UNION ALL SELECT 'tech_stock', 2, 0.08, 0.06, 0.07, 0.05
UNION ALL SELECT 'tech_stock', 3, -0.02, -0.01, -0.01, -0.03
UNION ALL SELECT 'tech_stock', 4, 0.12, 0.10, 0.11, 0.08
UNION ALL SELECT 'tech_stock', 5, 0.06, 0.04, 0.05, 0.03
UNION ALL SELECT 'finance_stock', 1, 0.04, 0.03, 0.02, 0.01
UNION ALL SELECT 'finance_stock', 2, 0.07, 0.06, 0.05, 0.04
UNION ALL SELECT 'finance_stock', 3, -0.01, -0.01, -0.02, -0.02
UNION ALL SELECT 'finance_stock', 4, 0.09, 0.10, 0.08, 0.06
UNION ALL SELECT 'finance_stock', 5, 0.05, 0.04, 0.03, 0.02;

-- Ridge regression with regularization parameter lambda=1.0
SELECT
    ticker,
    result.coefficients[1] as market_beta,
    result.coefficients[2] as sector_beta,
    result.coefficients[3] as momentum_factor,
    result.r2,
    result.lambda
FROM (
    SELECT
        ticker,
        anofox_stats_ridge_fit_agg(
            return,
            [market_return, sector_return, momentum],
            {'lambda': 1.0, 'intercept': true}
        ) as result
    FROM stock_returns
    GROUP BY ticker
) sub;
