LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Example: Incremental model refresh using parameterized queries
-- NOTE: DuckDB doesn't support stored procedures, but the same functionality
-- can be achieved using parameter variables and prepared statements

-- Set parameters (in practice, these would be passed from your application)
CREATE TEMP TABLE params AS SELECT 365 as lookback_days;

-- Create sample schema tables for demonstration
CREATE OR REPLACE TABLE model_results_archive (
    product_id INT,
    price_coefficient DOUBLE,
    advertising_coefficient DOUBLE,
    competition_coefficient DOUBLE,
    r2_price DOUBLE,
    r2_advertising DOUBLE,
    r2_competition DOUBLE,
    data_points INT,
    training_start DATE,
    training_end DATE,
    model_update_time TIMESTAMP,
    archived_at TIMESTAMP
);

CREATE OR REPLACE TABLE model_results_cache (
    product_id INT,
    price_coefficient DOUBLE,
    advertising_coefficient DOUBLE,
    competition_coefficient DOUBLE,
    r2_price DOUBLE,
    r2_advertising DOUBLE,
    r2_competition DOUBLE,
    data_points INT,
    training_start DATE,
    training_end DATE,
    model_update_time TIMESTAMP
);

CREATE OR REPLACE TABLE model_refresh_log (
    refresh_time TIMESTAMP,
    lookback_days INT,
    models_refreshed INT
);

-- Create sample daily product data
CREATE OR REPLACE TABLE daily_product_data AS
SELECT
    (i % 10) + 1 as product_id,
    CURRENT_DATE - ((RANDOM() * 365)::INT) as date,
    (1000 + RANDOM() * 1000)::INT as sales,
    (50 + RANDOM() * 50)::DOUBLE as price,
    (500 + RANDOM() * 500)::DOUBLE as advertising,
    (10 + RANDOM() * 20)::DOUBLE as competition
FROM range(1, 1001) t(i);

-- ==============================================================================
-- Model Refresh Logic (equivalent to stored procedure functionality)
-- ==============================================================================

-- Step 1: Archive old models
INSERT INTO model_results_archive
SELECT *, CURRENT_TIMESTAMP as archived_at
FROM model_results_cache;

-- Step 2: Delete old cache
DELETE FROM model_results_cache;

-- Step 3: Rebuild with fresh data
INSERT INTO model_results_cache
WITH lookback_param AS (
    SELECT lookback_days FROM params
),
source_data AS (
    SELECT
        product_id,
        sales::DOUBLE as y,
        price::DOUBLE as x1,
        advertising::DOUBLE as x2,
        competition::DOUBLE as x3,
        (SELECT lookback_days FROM lookback_param) as lookback
    FROM daily_product_data
    CROSS JOIN lookback_param
    WHERE date >= CURRENT_DATE - (lookback_param.lookback_days || ' DAYS')::INTERVAL
),
product_models AS (
    SELECT
        product_id,
        ols_fit_agg(y, x1) as model_x1,
        ols_fit_agg(y, x2) as model_x2,
        ols_fit_agg(y, x3) as model_x3,
        COUNT(*) as data_points,
        CURRENT_DATE - (MAX(lookback) || ' DAYS')::INTERVAL as training_start,
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

-- Step 4: Log refresh
INSERT INTO model_refresh_log
SELECT
    CURRENT_TIMESTAMP as refresh_time,
    lookback_days,
    (SELECT COUNT(*) FROM model_results_cache) as models_refreshed
FROM params;

-- Display results
SELECT 'Refresh completed at: ' || CURRENT_TIMESTAMP::VARCHAR as status;
SELECT * FROM model_refresh_log ORDER BY refresh_time DESC LIMIT 1;
SELECT COUNT(*) as models_in_cache FROM model_results_cache;

-- NOTE: To schedule this daily, use an external scheduler (cron, Airflow, etc.)
-- and execute this script with: duckdb < guide05_automated_model_refresh.sql
