LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Efficient processing of large datasets
-- Strategy 1: Partition by time/category
WITH monthly_partitions AS (
    SELECT
        DATE_TRUNC('month', date) as month,
        category,
        LIST(sales::DOUBLE) as y_values,
        LIST(marketing::DOUBLE) as x_values,
        COUNT(*) as n_obs
    FROM large_sales_table
    WHERE date >= '2023-01-01'
    GROUP BY DATE_TRUNC('month', date), category
),

monthly_models AS (
    SELECT
        month,
        category,
        -- Use aggregate functions directly on arrays
        ols_coeff_agg(UNNEST(y_values), UNNEST(x_values)) as coefficient,
        (SELECT (ols_fit_agg(UNNEST(y_values), UNNEST(x_values))).r2) as r2,
        n_obs
    FROM monthly_partitions
    WHERE n_obs >= 30
    GROUP BY month, category, y_values, x_values, n_obs
)

SELECT
    month,
    category,
    ROUND(coefficient, 2) as coefficient,
    ROUND(r2, 3) as r2,
    n_obs
FROM monthly_models
ORDER BY month DESC, category;

-- Strategy 2: Sample for exploration, full data for final model
WITH sample_data AS (
    SELECT *
    FROM large_table
    USING SAMPLE 10 PERCENT  -- Quick exploration
),
sample_model AS (
    SELECT
        ols_coeff_agg(y, x1) as coeff_x1,
        ols_coeff_agg(y, x2) as coeff_x2,
        (ols_fit_agg(y, x1)).r2 as r2_x1,
        (ols_fit_agg(y, x2)).r2 as r2_x2
    FROM sample_data
)
-- Validate on sample, then run on full data if promising
SELECT * FROM sample_model;
