-- Create sample quarterly revenue data
CREATE OR REPLACE TABLE quarterly_financials AS
SELECT
    i as quarter_id,
    (40000000 + i * 500000 + RANDOM() * 2000000)::DOUBLE as revenue  -- Growing from $40M with trend
FROM range(1, 21) t(i);

-- Time-series revenue forecasting using table function with literal arrays
-- For real tables: Run `SELECT LIST(revenue) FROM quarterly_financials` first,
-- then paste the result as a literal array below
WITH forecast AS (
    SELECT * FROM ols_predict_interval(
        [40572849.0, 41233491.0, 42455123.0, 42789456.0, 43891234.0, 43234567.0, 44567890.0, 44123456.0, 45678901.0, 45234567.0, 46789012.0, 46345678.0, 47890123.0, 47456789.0, 48901234.0, 48567890.0, 49912345.0, 49678901.0, 50923456.0, 50789012.0]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0]]::DOUBLE[][],
        [[21.0], [22.0], [23.0], [24.0]]::DOUBLE[][],  -- Next 4 quarters
        0.90,
        'prediction',
        true
    )
)
SELECT
    observation_id as quarter_offset,
    20 + observation_id as quarter_id,
    ROUND(predicted / 1000000, 2) as forecast_revenue_millions,
    ROUND(ci_lower / 1000000, 2) as conservative_estimate_millions,
    ROUND(ci_upper / 1000000, 2) as optimistic_estimate_millions,
    ROUND((ci_upper - ci_lower) / predicted * 100, 1) as uncertainty_pct
FROM forecast;
