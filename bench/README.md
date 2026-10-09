# Benchmark harness

A repeatable benchmark suite over representative extension workloads. It measures
the paths that matter for this extension — aggregate dispatch, fit/predict, and
FFI marshalling — and writes timings to a diffable file so optimization work can
be proven with before/after numbers.

## Run it

One documented command, from the repo root:

```bash
bash scripts/bench.sh          # default: scaled workloads, fast (~1 s total)
bash scripts/bench.sh --full   # additionally run the 1M-group official variant
```

**Precondition:** a local release build must exist —
`build/release/duckdb` and
`build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension`.
Build it with `make release` if missing. The harness loads the extension from
that local path (`-unsigned` + explicit `LOAD`), so timings reflect the current
working tree, never an autoloaded community build.

## Workloads

| Workload | File | Path exercised | Default scale |
|----------|------|----------------|---------------|
| **Aggregate dispatch** | `workloads/01-agg-dispatch.sql` | Aggregate-function dispatch — one OLS fit per `GROUP BY` group | 10K groups / 1M rows |
| **Fit + predict** | `workloads/02-fit-predict.sql` | Fit-once-per-group + predict-all-rows (`predict_agg`) — the fit → predict → marshalling path | 10K groups / 1M rows |
| **FFI micro-bench (inference)** | `workloads/03-ffi-micro.sql` | FFI marshalling micro-bench — `compute_inference: true` forces the 5-array `libc::malloc` inference block on every group invocation | 500 groups / 50K rows |
| **Aggregate dispatch, full scale** | `workloads/01-agg-dispatch-1m.sql` | Same as aggregate dispatch at full scale (`--full` only) | 1M groups / 100M rows |

The FFI micro-bench is the workload most sensitive to FFI allocation changes: its
tiny fits make per-call marshalling/allocation dominate the timing.

Each workload SQL file starts with `.timer on` and contains **no** `LOAD`
statement — the harness loads the extension. Data is generated with `random()`;
no external files are read.

## Results & diffing

Each run writes a timestamped markdown file:

```
bench/results/bench-<YYYYMMDD-HHMMSS>.md
```

It captures the full process output — the query results and the per-statement
`Run Time (s)` lines DuckDB's `.timer` prints. To compare two runs (e.g. before
and after an optimization), diff the two files:

```bash
diff bench/results/bench-<before>.md bench/results/bench-<after>.md
# or, for a committed baseline:
git diff --no-index bench/results/bench-<before>.md bench/results/bench-<after>.md
```

Result files are git-ignored (`bench/.gitignore`) — they are local run
artifacts, not source. Copy a specific run out of `results/` if you want to keep
it as a recorded baseline.

## `--full` scale (local only)

`--full` adds the full-scale aggregate-dispatch workload (1M groups / 100M rows). Expect **~8 GB RAM** and
**~160–210 s**. It is opt-in and intended for producing full-scale before/after
numbers on a workstation — it is **never** part of the default run.

## CI perf tracking (optional, not enforced)

The benchmark is intentionally **not** wired into a gating CI job. Benchmark timings are
machine- and load-dependent, so a naive wall-clock threshold in CI is flaky.

If added later, a reasonable shape would be: a **non-gating** scheduled job that
runs `bash scripts/bench.sh` (default scale only — never `--full`, which would
exceed CI memory/time), uploads the `bench/results/*.md` file as an artifact, and
optionally posts a trend comment. Keep it advisory; do not fail the build on
timing alone.
