LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample large dataset for testing
CREATE TEMP TABLE large_sales_table AS
SELECT
    DATE '2023-01-01' + INTERVAL (i) DAY as date,
    CASE (i % 3)
        WHEN 0 THEN 'electronics'
        WHEN 1 THEN 'clothing'
        ELSE 'furniture'
    END as category,
    (1000 + i * 10 + random() * 200)::DOUBLE as sales,
    (500 + i * 5 + random() * 100)::DOUBLE as marketing
FROM generate_series(1, 365) as t(i);

CREATE TEMP TABLE large_table AS
SELECT
    (100 + i * 2 + random() * 10)::DOUBLE as y,
    (50 + i * 1.5 + random() * 5)::DOUBLE as x1,
    (30 + i * 0.8 + random() * 3)::DOUBLE as x2
FROM generate_series(1, 1000) as t(i);

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
        n_obs,
        -- Use aggregate functions with UNNEST in subquery
        (SELECT (anofox_stats_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] FROM (
            SELECT UNNEST(y_values) as y, UNNEST(x_values) as x
        )) as coefficient,
        (SELECT (anofox_stats_ols_fit_agg(y, [x], {'intercept': true})).r2 FROM (
            SELECT UNNEST(y_values) as y, UNNEST(x_values) as x
        )) as r2
    FROM monthly_partitions
    WHERE n_obs >= 30
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
        (anofox_stats_ols_fit_agg(y, [x1], {'intercept': true})).coefficients[1] as coeff_x1,
        (anofox_stats_ols_fit_agg(y, [x2], {'intercept': true})).coefficients[1] as coeff_x2,
        (anofox_stats_ols_fit_agg(y, [x1], {'intercept': true})).r2 as r2_x1,
        (anofox_stats_ols_fit_agg(y, [x2], {'intercept': true})).r2 as r2_x2
    FROM sample_data
)
-- Validate on sample, then run on full data if promising
SELECT * FROM sample_model;
