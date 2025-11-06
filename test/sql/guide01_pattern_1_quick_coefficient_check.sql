LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT ols_coeff_agg(y, x) as slope FROM data;
