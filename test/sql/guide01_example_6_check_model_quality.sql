LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Extract model quality metrics using aggregate fit function
WITH data AS (
    SELECT UNNEST([2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 15.9]::DOUBLE[]) as y,
           UNNEST([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][]) as x
)
SELECT
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).n_obs as n_obs,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).r2 as r2,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).adj_r2 as adj_r2,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).aic as aic,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).aicc as aicc,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).bic as bic,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).log_likelihood as log_likelihood
FROM data;
