-- Drift monitor over a Hive-partitioned Parquet dataset (30 days x 1M rows): every day
-- against day 0, numeric (KS, Wasserstein, PSI) and categorical (Jensen-Shannon).
-- Needs bench/data (scripts/bench.sh runs setup-parquet.sql). No LOAD here.
CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet('bench/data/events/*/*.parquet', hive_partitioning = true);
.timer on
.print @@ drift monitor: 29 days vs day 0, 4 statistics
SELECT count(*), sum((psi).drift::INT) AS drifted_days FROM (
    SELECT d.day,
           ks_test_agg(e.amount, (e.day = d.day)::INT) AS ks,
           wasserstein_agg(e.amount, (e.day = d.day)::INT) AS w,
           psi_agg(e.amount, (e.day = d.day)::INT) AS psi,
           jensen_shannon_agg(e.merchant, (e.day = d.day)::INT) AS js
    FROM (SELECT DISTINCT day FROM events WHERE day > 0) d
    JOIN events e ON e.day = 0 OR e.day = d.day
    GROUP BY d.day);
