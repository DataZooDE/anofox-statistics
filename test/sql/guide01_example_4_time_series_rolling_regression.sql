LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create time series
CREATE TABLE time_series AS
SELECT
    i as time_idx,
    (i * 1.5 + RANDOM() * 0.3)::DOUBLE as value
FROM range(1, 51) t(i);

-- Rolling 10-period regression
SELECT
    time_idx,
    value,
    ols_coeff_agg(value, time_idx) OVER (
        ORDER BY time_idx
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as rolling_trend
FROM time_series
WHERE time_idx >= 10;
