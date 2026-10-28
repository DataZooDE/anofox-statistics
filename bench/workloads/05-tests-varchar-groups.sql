-- Hypothesis and drift tests per VARCHAR segment (500 segments, 20M rows from Parquet).
-- Exercises GROUP BY on VARCHAR labels and VARCHAR categorical inputs (dictionary paths).
-- Needs bench/data (scripts/bench.sh runs setup-parquet.sql). No LOAD here.
CREATE OR REPLACE VIEW seg AS SELECT * FROM 'bench/data/segments.parquet';
.timer on
.print @@ t_test_agg per segment
SELECT count(*), round(avg((r).p_value), 4) FROM (SELECT segment, t_test_agg(v, grp) r FROM seg GROUP BY segment);
.print @@ ks_test_agg per segment
SELECT count(*), round(avg((r).p_value), 4) FROM (SELECT segment, ks_test_agg(v, grp) r FROM seg GROUP BY segment);
.print @@ psi_agg on VARCHAR categories per segment
SELECT count(*), round(avg((r).psi), 6) FROM (SELECT segment, psi_agg(cat, grp) r FROM seg GROUP BY segment);
.print @@ pearson_agg per segment
SELECT count(*), round(avg((r).r), 4) FROM (SELECT segment, pearson_agg(v, grp::DOUBLE) r FROM seg GROUP BY segment);
