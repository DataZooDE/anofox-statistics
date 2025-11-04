-- Rolling regression for adaptive forecasting
SELECT
    date,
    value,
    ols_coeff_agg(value, time_index) OVER (
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as trend_coefficient
FROM time_series_data;
