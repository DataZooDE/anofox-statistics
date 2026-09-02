-- W3: FFI-marshalling micro-benchmark (small groups / high call count).
-- {'compute_inference': true} forces the 5-array libc::malloc inference block
-- (std_errors/t_values/p_values/ci_lower/ci_upper) on every group invocation —
-- the exact allocation pattern Plan 02 refactors. ~500 groups × ~100 rows keeps
-- the fit cheap so per-call FFI marshalling/allocation dominates the timing.
-- The harness loads the extension; this file must contain no LOAD statement.
.timer on
.mode markdown

WITH test_data AS (
    SELECT
        i % 500        AS group_id,
        random() * 100 AS x1,
        random() * 100 AS y
    FROM generate_series(1, 50000) t(i)
)
SELECT
    COUNT(*)             AS n_groups,
    COUNT(se1)           AS n_with_inference,
    ROUND(AVG(se1), 6)   AS mean_first_std_error
FROM (
    SELECT
        group_id,
        (ols_fit_agg(y, [x1], {'intercept': true, 'compute_inference': true})).std_errors[1] AS se1
    FROM test_data
    GROUP BY group_id
) t;
