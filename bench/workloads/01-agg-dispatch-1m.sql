-- W1-full: Aggregate dispatch at official scale (1M groups / 100M rows).
-- Identical shape to 01-agg-dispatch.sql but full-scale. Documented ~8 GB RAM,
-- ~160-210 s. Local-only, run via `bash scripts/bench.sh --full`; NOT default/CI.
-- The harness loads the extension; this file must contain no LOAD statement.
.timer on
.mode markdown

WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        random() * 100 AS x1,
        random() * 50  AS x2,
        random() * 25  AS x3,
        random() * 100 AS y
    FROM generate_series(1, 100000000) t(i)
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
