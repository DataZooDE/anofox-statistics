LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Quick Start Example: Recursive Least Squares Aggregate
-- Demonstrates adaptive regression for changing relationships (online learning)

-- Sample data: sensor readings with evolving calibration
CREATE TEMP TABLE sensor_readings AS
SELECT
    'sensor_a' as sensor_id,
    1 as time_index,
    100.0 as raw_reading,
    98.0 as true_value
UNION ALL SELECT 'sensor_a', 2, 105.0, 103.0
UNION ALL SELECT 'sensor_a', 3, 110.0, 107.0
UNION ALL SELECT 'sensor_a', 4, 115.0, 112.0  -- Drift starts here
UNION ALL SELECT 'sensor_a', 5, 120.0, 116.5
UNION ALL SELECT 'sensor_a', 6, 125.0, 121.0
UNION ALL SELECT 'sensor_b', 1, 200.0, 202.0
UNION ALL SELECT 'sensor_b', 2, 205.0, 207.5
UNION ALL SELECT 'sensor_b', 3, 210.0, 213.0
UNION ALL SELECT 'sensor_b', 4, 215.0, 218.5
UNION ALL SELECT 'sensor_b', 5, 220.0, 224.0
UNION ALL SELECT 'sensor_b', 6, 225.0, 229.5;

-- RLS with forgetting_factor=0.95 (emphasizes recent observations)
SELECT
    sensor_id,
    result.coefficients[1] as calibration_slope,
    result.intercept as calibration_offset,
    result.r2,
    result.forgetting_factor,
    result.n_obs
FROM (
    SELECT
        sensor_id,
        anofox_statistics_rls_agg(
            true_value,
            [raw_reading],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) as result
    FROM sensor_readings
    GROUP BY sensor_id
) sub;
