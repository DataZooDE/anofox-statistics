LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Incremental model update procedure
CREATE OR REPLACE PROCEDURE refresh_product_models(lookback_days INT DEFAULT 365)
AS
BEGIN
    -- Archive old models
    INSERT INTO model_results_archive
    SELECT *, CURRENT_TIMESTAMP as archived_at
    FROM model_results_cache;

    -- Delete old cache
    DELETE FROM model_results_cache;

    -- Rebuild with fresh data
    INSERT INTO model_results_cache
    WITH source_data AS (
        SELECT
            product_id,
            sales::DOUBLE as y,
            price::DOUBLE as x1,
            advertising::DOUBLE as x2,
            competition::DOUBLE as x3
        FROM daily_product_data
        WHERE date >= CURRENT_DATE - INTERVAL lookback_days DAYS
    ),
    product_models AS (
        SELECT
            product_id,
            ols_fit_agg(y, x1) as model_x1,
            ols_fit_agg(y, x2) as model_x2,
            ols_fit_agg(y, x3) as model_x3,
            COUNT(*) as data_points,
            CURRENT_DATE - INTERVAL lookback_days DAYS as training_start,
            CURRENT_DATE as training_end,
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

    -- Log refresh
    INSERT INTO model_refresh_log VALUES (
        CURRENT_TIMESTAMP,
        lookback_days,
        (SELECT COUNT(*) FROM model_results_cache)
    );
END;

-- Schedule: Run daily
-- CALL refresh_product_models(365);
