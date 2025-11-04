SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as rolling_elasticity
FROM time_series;
