-- W1: Aggregate dispatch over many GROUP BY groups (scaled: 10K groups / 1M rows).
-- Exercises the aggregate-function dispatch path: one OLS fit per group.
-- The harness loads the extension; this file must contain no LOAD statement.
.timer on
.mode markdown

WITH test_data AS (
    SELECT
        i % 10000 AS group_id,
        random() * 100 AS x1,
        random() * 50  AS x2,
        random() * 25  AS x3,
        random() * 100 AS y
    FROM generate_series(1, 1000000) t(i)
)
SELECT
    COUNT(*)            AS n_groups,
    COUNT(r2)           AS n_fitted,
    ROUND(AVG(r2), 6)   AS mean_r2
FROM (
    SELECT
        group_id,
        (anofox_stats_ols_fit_agg(y, [x1, x2, x3], {'intercept': true})).r_squared AS r2
    FROM test_data
    GROUP BY group_id
) t;
