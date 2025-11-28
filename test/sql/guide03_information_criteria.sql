LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Compare two models using information criteria (AIC/BIC from aggregate function)
-- Model 1: Price only
WITH data AS (
    SELECT
        unnest([100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[]) as sales,
        unnest([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]::DOUBLE[]) as price,
        unnest([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]::DOUBLE[]) as advertising
),
model1 AS (
    SELECT
        'Model 1 (price only)' as model,
        (anofox_statistics_ols_fit_agg(sales, [price], {'intercept': true})).aic as aic,
        (anofox_statistics_ols_fit_agg(sales, [price], {'intercept': true})).bic as bic,
        (anofox_statistics_ols_fit_agg(sales, [price], {'intercept': true})).r2 as r2
    FROM data
),
model2 AS (
    SELECT
        'Model 2 (price + ads)' as model,
        (anofox_statistics_ols_fit_agg(sales, [price, advertising], {'intercept': true})).aic as aic,
        (anofox_statistics_ols_fit_agg(sales, [price, advertising], {'intercept': true})).bic as bic,
        (anofox_statistics_ols_fit_agg(sales, [price, advertising], {'intercept': true})).r2 as r2
    FROM data
)
SELECT * FROM model1
UNION ALL
SELECT * FROM model2;
