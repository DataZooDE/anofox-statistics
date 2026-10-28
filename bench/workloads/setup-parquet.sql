-- Writes the Parquet inputs of workloads 04-06 to bench/data/ (git-ignored).
-- Deterministic: setseed() + random(). Run from the repo root; no LOAD needed.
SELECT setseed(0.42);

-- 500k series x 20 rows: the per-series regression panel of the _by macros.
COPY (
    SELECT 'S' || lpad((i // 20)::VARCHAR, 6, '0') AS series, i % 20 AS t,
           random() AS x1, random() AS x2,
           CASE WHEN i % 20 < 16 THEN random() + (i // 20) % 7 * 0.1 ELSE NULL END AS y
    FROM range(10000000) r(i)
) TO 'bench/data/panel.parquet' (FORMAT parquet);

-- 20M rows in 500 VARCHAR segments with two groups and a VARCHAR category.
COPY (
    SELECT 'segment_' || (i % 500) AS segment, (i // 500) % 2 AS grp,
           random() + ((i // 500) % 2) * 0.05 AS v,
           'cat_' || ((i * 7919) % 10) AS cat
    FROM range(20000000) r(i)
) TO 'bench/data/segments.parquet' (FORMAT parquet);

-- 30 days x 1M rows, Hive-partitioned by day: the drift-monitor input.
COPY (
    SELECT i // 1000000 AS day, random() + (i // 1000000) * 0.01 AS amount,
           'merchant_' || ((i * 31) % 50) AS merchant
    FROM range(30000000) r(i)
) TO 'bench/data/events' (FORMAT parquet, PARTITION_BY (day), OVERWRITE_OR_IGNORE);
