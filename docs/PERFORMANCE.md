# Performance: DuckDB v1.5.6 vs v2.0

What DuckDB 2.0 changes for anofox_statistics workloads without any extension change
(#269). The same extension release (v0.14.0, commit b3ccb00) is installed from
get.erpl.io into the official v1.5.6 and v2.0.0 CLIs. Both run the workloads in
[`bench/workloads/`](../bench/workloads/) on the same machine.

## Results

Median of 9 runs, wall-clock seconds, default threads (20). Machine: Intel Core
i9-13900H (20 threads), 64 GB RAM, NVMe SSD. Generated with:

```bash
python3 scripts/bench_compare.py --cli v1.5.6=<duckdb 1.5.6> --cli v2.0.0=<duckdb 2.0.0> \
    --install-from https://get.erpl.io --runs 9
```

| Workload | Statement | v1.5.6 | v2.0.0 | v2.0 / v1.5.6 |
|---|---|---|---|---|
| 01 aggregate dispatch | `ols_fit_agg` GROUP BY, 10K groups / 1M rows (generated) | 0.180 | 0.135 | 0.75 |
| 02 fit + predict | `predict_agg`, 10K groups / 1M rows (generated) | 0.714 | 0.621 | 0.87 |
| 03 FFI micro | `compute_inference`, 500 groups / 50K rows | 0.009 | 0.012 | (too small to compare) |
| 04 `_by` on Parquet | `ols_fit_predict_by`, 500K series / 10M rows | 7.079 | 3.505 | **0.50** |
| 04 `_by` on Parquet | `ols_tidy_by`, 500K series | 1.099 | 1.188 | 1.08 |
| 04 `_by` on Parquet | `ols_fit_agg` GROUP BY series | 0.673 | 0.668 | 0.99 |
| 05 tests per segment | `t_test_agg`, 500 VARCHAR segments / 20M rows | 0.274 | 0.239 | 0.87 |
| 05 tests per segment | `ks_test_agg` | 0.224 | 0.226 | 1.01 |
| 05 tests per segment | `psi_agg` on VARCHAR categories | 0.168 | 0.141 | 0.84 |
| 05 tests per segment | `pearson_agg` | 0.298 | 0.305 | 1.02 |
| 06 drift monitor | KS + Wasserstein + PSI + JS, 29 days vs day 0, 30M rows | 1.084 | 0.984 | 0.91 |

## Reading the numbers

- **`*_fit_predict_by` is 2× faster on 2.0.** Its cost is almost all in DuckDB's window
  operator: each series is fitted by the `*_fit_predict` window aggregate over
  `PARTITION BY series`. That operator takes 97 s CPU on v1.5.6 and 46 s on v2.0 for 10M
  rows; the rest of the plan (Parquet scan, the final `ORDER BY`) is under 2 s.
- **Even on 2.0 the window path is the hotspot.** The same 500K fits as a plain
  `GROUP BY` (`ols_fit_agg`, last row of workload 04) take 0.67 s wall-clock, against
  3.5 s for `ols_fit_predict_by`. Fitting once per partition with the batched window
  callback (#268), or a GROUP BY fit joined back to the rows, is the largest available
  speed-up for the `_fit_predict` family.
- **The aggregate-bound workloads are flat (±15%).** For `ols_fit_agg`, the tests and
  the drift statistics, the time goes to the extension's own per-group work (buffering,
  canonical sorting #226, marshalling, the fit). DuckDB 2.0's faster grouping and Parquet
  I/O shave a little (t-test, PSI, the drift monitor: 9–16%), but per-group finalize
  overhead dominates. That is the target of #265 (finalize-local state).
- **Workload 03** runs in about 10 ms, too short for a stable comparison. It is meant for
  before/after measurements of FFI changes on a single build.

## Reproducing

The inputs of workloads 04–06 (about 1 GB of Parquet) are generated on the first run
into `bench/data/`. `--install-from` keeps the installed extensions in
`bench/results/extensions/`, so your `~/.duckdb` is untouched. See
[`bench/README.md`](../bench/README.md) for the single-build harness (`scripts/bench.sh`).
