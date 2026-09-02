-- W2: Fit/predict path (scaled: 10K groups / 1M rows).
-- Exercises fit-once-per-group + predict-all-rows via the predict aggregate:
-- ~80% training rows (y not null) fit the model, all rows get a prediction.
-- This drives the fit -> predict -> FFI-marshalling path.
--
-- NOTE (deviation): Plan 04-01 specified the rolling `fit_predict ... OVER (...)`
-- window shape, but that path hits a pre-existing INTERNAL error — the expanding
-- frame fits on degenerate sub-(n_features+1) frames at each partition start
-- (an extension robustness gap slated for the ERGO milestone, unrelated to
-- Phase 4 measurement). `predict_agg` is the sibling analog
-- (examples/performance_1m_groups/benchmark_ols_predict_agg.sql) and exercises
-- the same fit/predict marshalling robustly. The harness loads the extension;
-- this file must contain no LOAD statement.
.timer on
.mode markdown

WITH test_data AS (
    SELECT
        i % 10000 AS group_id,
        i / 10000 AS row_num,
        random() * 100 AS x1,
        random() * 50  AS x2,
        random() * 25  AS x3,
        CASE WHEN (i % 100) < 80 THEN random() * 100 ELSE NULL END AS y
    FROM generate_series(1, 1000000) t(i)
)
SELECT
    COUNT(*)                                              AS total_predictions,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END)   AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT
        group_id,
        UNNEST(ols_fit_predict_agg(y, [x1, x2, x3])) AS pred
    FROM test_data
    GROUP BY group_id
) t;
