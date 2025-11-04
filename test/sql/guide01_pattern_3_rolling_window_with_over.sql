SELECT *, ols_coeff_agg(y, x) OVER (
    ORDER BY time ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
) as rolling_coef FROM data;
