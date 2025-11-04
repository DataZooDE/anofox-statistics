-- Analyze effect of marketing channels on sales
SELECT
    week,
    ols_fit_agg_array(
        revenue,
        [tv_spend, digital_spend, print_spend]::DOUBLE[]
    ) as marketing_model
FROM campaigns
GROUP BY week;
