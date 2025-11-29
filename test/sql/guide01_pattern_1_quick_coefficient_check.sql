LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Generate sample data for pattern demonstration
CREATE TEMP TABLE data AS
SELECT
    i::DOUBLE as y,
    (i * 2.5 + 10)::DOUBLE as x
FROM generate_series(1, 20) t(i);

SELECT (anofox_stats_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as slope FROM data;
