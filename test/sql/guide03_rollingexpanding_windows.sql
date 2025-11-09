LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Generate time-series data
CREATE TEMP TABLE time_series AS
SELECT
    DATE '2023-01-01' + INTERVAL (i) DAY as date,
    (100 + i * 2 + random() * 10)::DOUBLE as sales,
    (50 + i * 0.5 + random() * 5)::DOUBLE as price
FROM generate_series(1, 100) t(i);

SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as rolling_elasticity
FROM time_series;
