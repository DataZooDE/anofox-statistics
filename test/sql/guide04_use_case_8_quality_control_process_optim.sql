-- Create sample manufacturing batch data
CREATE OR REPLACE TABLE production_batches AS
SELECT
    i as batch_id,
    CURRENT_DATE - ((RANDOM() * 90)::INT) as batch_date,
    temp::DOUBLE as temperature,
    pressure::DOUBLE as pressure,
    humidity::DOUBLE as humidity,
    speed::DOUBLE as line_speed,
    defect_rate::DOUBLE
FROM (
    SELECT
        i,
        (180 + RANDOM() * 40) as temp,  -- 180-220°F
        (25 + RANDOM() * 10) as pressure,  -- 25-35 PSI
        (40 + RANDOM() * 30) as humidity,  -- 40-70%
        (50 + RANDOM() * 30) as speed,  -- 50-80 units/min
        -- Defects increase with high temp, high speed, decrease with optimal pressure
        (2.0 + temp * 0.05 + speed * 0.03 - pressure * 0.02 + humidity * 0.01 + RANDOM() * 1.5) as defect_rate
    FROM range(1, 101) t(i)
);

-- Analyze impact of each process parameter on defect rates
SELECT
    'Temperature' as variable,
    ROUND((ols_fit_agg(defect_rate, temperature)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, temperature)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, temperature)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, temperature)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, temperature)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, temperature)).significant
             AND (ols_fit_agg(defect_rate, temperature)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, temperature)).significant
             AND (ols_fit_agg(defect_rate, temperature)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Pressure' as variable,
    ROUND((ols_fit_agg(defect_rate, pressure)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, pressure)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, pressure)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, pressure)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, pressure)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, pressure)).significant
             AND (ols_fit_agg(defect_rate, pressure)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, pressure)).significant
             AND (ols_fit_agg(defect_rate, pressure)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Humidity' as variable,
    ROUND((ols_fit_agg(defect_rate, humidity)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, humidity)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, humidity)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, humidity)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, humidity)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, humidity)).significant
             AND (ols_fit_agg(defect_rate, humidity)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, humidity)).significant
             AND (ols_fit_agg(defect_rate, humidity)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Line Speed' as variable,
    ROUND((ols_fit_agg(defect_rate, line_speed)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, line_speed)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, line_speed)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, line_speed)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, line_speed)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, line_speed)).significant
             AND (ols_fit_agg(defect_rate, line_speed)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, line_speed)).significant
             AND (ols_fit_agg(defect_rate, line_speed)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
ORDER BY ABS(impact_on_defects) DESC;
