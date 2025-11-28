LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Generate sample data with multiple categories
CREATE TEMP TABLE data AS
SELECT
    CASE
        WHEN i <= 10 THEN 'A'
        WHEN i <= 20 THEN 'B'
        ELSE 'C'
    END as category,
    (i + random() * 5)::DOUBLE as y,
    (i * 1.5 + 5)::DOUBLE as x
FROM generate_series(1, 30) t(i);

SELECT category, anofox_statistics_ols_fit_agg(y, [x], {'intercept': true}) as model
FROM data GROUP BY category;
