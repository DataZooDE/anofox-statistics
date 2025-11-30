LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Generate time-series data
CREATE TEMP TABLE data AS
SELECT
    i as time,
    (10 + i * 0.5 + random() * 3)::DOUBLE as y,
    (5 + i * 0.3)::DOUBLE as x
FROM generate_series(1, 100) t(i);

SELECT *, (anofox_stats_ols_fit_agg(y, [x], {'intercept': true}) OVER (
    ORDER BY time ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
)).coefficients[1] as rolling_coef FROM data;
