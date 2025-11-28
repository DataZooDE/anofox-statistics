LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample manufacturing batch data
CREATE OR REPLACE TABLE production_batches AS
SELECT
    i as batch_id,
    CURRENT_DATE - ((RANDOM() * 90)::INT) as batch_date,
    temp::DOUBLE as temperature,
    pressure::DOUBLE as pressure,
    humidity::DOUBLE as humidity,
    speed::DOUBLE as line_speed,
    -- Defects increase with high temp, high speed, decrease with optimal pressure
    (2.0 + temp * 0.05 + speed * 0.03 - pressure * 0.02 + humidity * 0.01 + RANDOM() * 1.5)::DOUBLE as defect_rate
FROM (
    SELECT
        i,
        (180 + RANDOM() * 40) as temp,  -- 180-220°F
        (25 + RANDOM() * 10) as pressure,  -- 25-35 PSI
        (40 + RANDOM() * 30) as humidity,  -- 40-70%
        (50 + RANDOM() * 30) as speed  -- 50-80 units/min
    FROM range(1, 101) t(i)
);

-- Analyze impact of each process parameter on defect rates
WITH recent_batches AS (
    SELECT * FROM production_batches
    WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
),
parameter_impacts AS (
    SELECT
        'Temperature' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [temperature], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [temperature], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Pressure' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [pressure], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [pressure], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Humidity' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [humidity], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [humidity], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Line Speed' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [line_speed], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [line_speed], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
),
impacts_materialized AS (
    SELECT * FROM parameter_impacts
)
SELECT
    variable,
    impact_on_defects,
    model_fit,
    CASE
        WHEN impact_on_defects > 0 THEN 'Increases Defects'
        WHEN impact_on_defects < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN ABS(impact_on_defects) > 0.05 AND impact_on_defects > 0 THEN 'Critical - Reduce'
        WHEN ABS(impact_on_defects) > 0.05 AND impact_on_defects < 0 THEN 'Beneficial - Increase'
        ELSE 'Low Impact'
    END as action_recommendation
FROM impacts_materialized
ORDER BY ABS(impact_on_defects) DESC;
