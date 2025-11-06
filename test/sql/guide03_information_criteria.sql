LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Compare two models (using literal arrays)
WITH model1 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],  -- y: sales
        [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]::DOUBLE[][],  -- x: price only
        true                                                -- add_intercept
    )
),
model2 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],  -- y: sales
        [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0], [15.0, 10.0]]::DOUBLE[][],  -- x: price + advertising
        true                                                -- add_intercept
    )
)
SELECT
    'Model 1 (price only)' as model,
    aic, bic, r_squared FROM model1
UNION ALL
SELECT
    'Model 2 (price + ads)',
    aic, bic, r_squared FROM model2;
