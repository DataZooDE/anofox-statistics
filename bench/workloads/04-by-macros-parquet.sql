-- _by table macros over a Parquet panel (500k series x 20 rows): GROUP BY a VARCHAR
-- series id, the path DuckDB 2.0's hash group partitioning and async Parquet I/O touch.
-- Needs bench/data (scripts/bench.sh runs setup-parquet.sql). No LOAD here.
CREATE OR REPLACE VIEW panel AS SELECT * FROM 'bench/data/panel.parquet';
.timer on
.print @@ ols_fit_predict_by, 500k series, 10M rows
SELECT count(*), round(avg(yhat), 4) FROM ols_fit_predict_by('panel', series, y, [x1, x2]);
.print @@ ols_tidy_by, 500k series
SELECT count(*), round(avg(estimate), 4) FROM ols_tidy_by('panel', series, y, [x1, x2]);
.print @@ ols_fit_agg GROUP BY series (Parquet scan)
SELECT count(*), round(avg((f).r_squared), 4) FROM (SELECT series, ols_fit_agg(y, [x1, x2]) f FROM panel GROUP BY series);
