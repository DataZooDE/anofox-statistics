---
phase: 04-benchmarking-performance
plan: 01
subsystem: testing
tags: [benchmark, duckdb, bash, sql, perf, timer]

requires:
  - phase: 03-wasm-support
    provides: green native + WASM suites; a buildable release extension
provides:
  - scripts/bench.sh — one-command benchmark harness loading the local release extension by explicit path
  - three representative workload SQL files (aggregate dispatch, fit/predict, FFI micro-bench) + a --full 1M-group variant
  - bench/README.md documenting run command, workloads, results diffing, and optional (non-enforced) CI perf tracking
  - a diffable bench/results/bench-<timestamp>.md produced on each run (git-ignored)
affects: [04-02-ffi-refactor, 04-03-profiling-hotspots]

actuals:
  tokens: 9000
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "SQL benchmark scripts driven by a bash wrapper; extension loaded by explicit local path (-unsigned + LOAD), not autoload"
    - "Per-workload failure captured (PIPESTATUS) and reported without aborting the whole run"

key-files:
  created:
    - scripts/bench.sh
    - bench/workloads/01-agg-dispatch.sql
    - bench/workloads/02-fit-predict.sql
    - bench/workloads/03-ffi-micro.sql
    - bench/workloads/01-agg-dispatch-1m.sql
    - bench/README.md
    - bench/.gitignore
  modified: []

key-decisions:
  - "W2 uses predict_agg instead of the rolling fit_predict window (the window path trips a pre-existing INTERNAL error on degenerate small frames)"
  - "Results files are git-ignored local artifacts; the harness path (script + workloads + docs) is what is committed"
  - "Default scale 10K groups / 1M rows for fast iteration; --full runs the 1M-group official variant (~8 GB, ~160-210 s)"

patterns-established:
  - "Benchmark harness: bash scripts/bench.sh -> build/release/duckdb -unsigned -cmd LOAD -f workload.sql -> tee to bench/results/"

requirements-completed: [PERF-01, PERF-02]

coverage:
  - id: D1
    description: "One documented command (bash scripts/bench.sh) runs three workloads against the local build and reports timings"
    requirement: "PERF-01"
    verification:
      - kind: integration
        ref: "bash scripts/bench.sh && grep -q 'Run Time' bench/results/bench-*.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "Runs reproduce locally with documented scope + --full official variant; optional CI perf tracking noted not enforced"
    requirement: "PERF-02"
    verification:
      - kind: integration
        ref: "test -f bench/README.md && grep -qi 'scripts/bench.sh' bench/README.md && grep -qi -- '--full' bench/README.md"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-08-31
status: complete
---

# Phase 4 (Plan 01): Benchmark Harness Summary

**One-command bash+SQL benchmark harness that loads the local release extension by explicit path and times three representative workloads (aggregate dispatch, fit/predict, FFI marshalling) into a diffable results file.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-08-31T20:43:00+02:00
- **Tasks:** 2
- **Files modified:** 7 created

## Accomplishments
- `scripts/bench.sh`: one documented command runs the suite end-to-end against `build/release/`, loading the extension by explicit local path (`-unsigned` + `LOAD`), never the autoloaded community build. Verified: exit 0, results file with per-statement `Run Time (s)` lines for all three workloads (W1 0.19s, W2 0.54s, W3 0.008s).
- Three workload SQL files exercise the named paths — W1 aggregate dispatch (10K groups/1M rows), W2 fit/predict (`predict_agg`), W3 FFI marshalling micro-bench (`compute_inference: true` forces the 5-array `libc::malloc` inference block Plan 02 refactors) — plus a `--full` 1M-group variant.
- `bench/README.md` documents the run command, workloads and scales, results diffing, `--full` resource expectations, and an optional (non-enforced) CI perf-tracking note.

## Task Commits
1. **Task 1: benchmark harness + workloads** — `c274b59` (perf)
2. **Task 2: bench/README.md** — `9e0fe45` (docs)

## Files Created/Modified
- `scripts/bench.sh` — the harness (extension load, run_workload, --full flag, per-workload failure capture)
- `bench/workloads/01-agg-dispatch.sql`, `02-fit-predict.sql`, `03-ffi-micro.sql`, `01-agg-dispatch-1m.sql` — the workloads
- `bench/README.md` — harness documentation
- `bench/.gitignore` — ignores `results/` (local run artifacts)

## Decisions Made
- W2 substituted `predict_agg` for the rolling `fit_predict` window (see deviation below).
- Result files are git-ignored; only the harness/workloads/docs are versioned.

## Deviations from Plan

### 1. W2 workload shape — rolling window → predict_agg
- **Found during:** Task 1 (running the harness to validate the tracer)
- **Issue:** The plan specified the exact rolling `anofox_stats_ols_fit_predict(...) OVER (... ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)` shape (copied from the analog `benchmark_ols.sql`). That path throws a DuckDB INTERNAL error ("Attempted to access index 0 within vector of size 0") because the expanding frame fits on degenerate sub-(n_features+1) frames at each partition start. Confirmed the crash is pre-existing (the analog shape crashes at small scale too), not introduced here.
- **Fix:** W2 now uses the sibling analog `anofox_stats_ols_predict_agg` (fit-once-per-group + predict-all-rows), which exercises the same fit → predict → FFI-marshalling path robustly. Documented inline in the SQL file.
- **Verification:** `bash scripts/bench.sh` exits 0 with all three workloads timed.
- **Committed in:** `c274b59`

---

**Total deviations:** 1 (workload shape substitution for a pre-existing extension bug).
**Impact on plan:** None to phase goal — the tracer measures the fit/predict path as intended. The rolling-window INTERNAL error is a genuine robustness gap flagged for the ERGO milestone (Phase 5), out of scope for Phase 4.

## Issues Encountered
- Discovered a latent extension bug: `fit_predict` window crashes on degenerate small frames. Routed around for benchmarking; flagged for ERGO/Phase 5 (this is exactly the kind of input-validation gap ERGO-01/02 target).

## Next Phase Readiness
- The "before" measurement capability is in place. Plan 02 (FFI refactor) and Plan 03 (hotspots) can now produce before/after numbers from `scripts/bench.sh` (W3 is the allocation-sensitive workload).
- `perf` is not installed on the dev box — Plan 03 will need it (`sudo pacman -S perf`), which is its human-action checkpoint.

---
*Phase: 04-benchmarking-performance*
*Completed: 2026-08-31*
