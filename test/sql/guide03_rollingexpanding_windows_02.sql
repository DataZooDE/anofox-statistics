LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_elasticity
FROM time_series;
