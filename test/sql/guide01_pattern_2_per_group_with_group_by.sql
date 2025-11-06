LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT category, ols_fit_agg(y, x) as model
FROM data GROUP BY category;
