-- Create materialized model table
CREATE TABLE IF NOT EXISTS model_results_cache AS
WITH source_data AS (
    SELECT
        product_id,
        date,
        sales::DOUBLE as y,
        price::DOUBLE as x1,
        advertising::DOUBLE as x2,
        competition::DOUBLE as x3
    FROM daily_product_data
    WHERE date >= CURRENT_DATE - INTERVAL '365 days'
),

product_models AS (
    SELECT
        product_id,
        ols_fit_agg(y, x1) as model_x1,
        ols_fit_agg(y, x2) as model_x2,
        ols_fit_agg(y, x3) as model_x3,
        COUNT(*) as data_points,
        MIN(date) as training_start,
        MAX(date) as training_end,
        CURRENT_TIMESTAMP as model_update_time
    FROM source_data
    GROUP BY product_id
    HAVING COUNT(*) >= 30
)

SELECT
    product_id,
    (model_x1).coefficient as price_coefficient,
    (model_x2).coefficient as advertising_coefficient,
    (model_x3).coefficient as competition_coefficient,
    (model_x1).r2 as r2_price,
    (model_x2).r2 as r2_advertising,
    (model_x3).r2 as r2_competition,
    data_points,
    training_start,
    training_end,
    model_update_time
FROM product_models;

-- Query cached models
SELECT
    product_id,
    price_coefficient as price_elasticity,
    advertising_coefficient as advertising_effectiveness,
    competition_coefficient as competition_impact,
    r2_price as model_quality_price,
    data_points,
    model_update_time
FROM model_results_cache
WHERE r2_price > 0.7
ORDER BY r2_price DESC;
