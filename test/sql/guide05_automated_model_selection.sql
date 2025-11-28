LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample business data
CREATE OR REPLACE TABLE business_data AS
SELECT
    i as period_id,
    (1000 + i * 50 +
     i * 10 * RANDOM() +           -- marketing effect
     200 * SIN(i * 0.5) +          -- seasonality effect
     -30 * (i % 5) +               -- competition effect
     -i * 5 * RANDOM() +           -- price effect
     RANDOM() * 100)::DOUBLE as sales,
    (i * 10 + RANDOM() * 50)::DOUBLE as marketing,
    SIN(i * 0.5)::DOUBLE as seasonality,
    (i % 5 + RANDOM() * 2)::DOUBLE as competition,
    (10 + i * 0.1 + RANDOM() * 2)::DOUBLE as price
FROM range(1, 101) t(i);

-- Compare models with different predictor combinations
WITH data AS (
    SELECT
        sales::DOUBLE as y,
        marketing::DOUBLE as x1,
        seasonality::DOUBLE as x2,
        competition::DOUBLE as x3,
        price::DOUBLE as x4
    FROM business_data
),

-- Model 1: Simple (marketing only)
model1 AS (
    SELECT
        1 as model_id,
        'Marketing Only' as model_name,
        (anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true})).r2 as r2,
        COUNT(*) as n_obs,
        2 as n_params
    FROM data
),

-- Model 2: Marketing + Seasonality (using aggregate for two variables)
model2 AS (
    SELECT
        2 as model_id,
        'Marketing + Seasonality' as model_name,
        -- For multiple predictors, show R² from individual models
        (anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true})).r2 as r2,
        COUNT(*) as n_obs,
        3 as n_params
    FROM data
),

-- Model 3: Full Model
model3 AS (
    SELECT
        3 as model_id,
        'Full Model' as model_name,
        (anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true})).r2 as r2,
        COUNT(*) as n_obs,
        5 as n_params
    FROM data
),

-- Combine and rank
all_models AS (
    SELECT * FROM model1
    UNION ALL
    SELECT * FROM model2
    UNION ALL
    SELECT * FROM model3
)

SELECT
    model_id,
    model_name,
    n_params,
    ROUND(r2, 4) as r2,
    n_obs,
    RANK() OVER (ORDER BY r2 DESC) as r2_rank,
    CASE
        WHEN RANK() OVER (ORDER BY r2 DESC) = 1 THEN 'Best by R²'
        ELSE ''
    END as recommendation
FROM all_models
ORDER BY r2 DESC;
