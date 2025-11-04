SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_elasticity
FROM time_series;
