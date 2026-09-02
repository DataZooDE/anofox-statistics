---
phase: 04-benchmarking-performance
verified: 2026-08-31T20:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: Benchmarking & Performance Verification Report

**Phase Goal:** The extension has a repeatable benchmark harness that measures representative workloads, and the surfaced hotspots plus the FFI allocation pattern are optimized with before/after numbers proving the improvement — behavior unchanged.

**Verified:** 2026-08-31T20:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `bash scripts/bench.sh` runs end-to-end and writes a results file with `Run Time` lines for all three default workloads | ✓ VERIFIED | Ran live: exit 0, `bench/results/bench-20260831-215736.md` contains 3 `Run Time (s)` lines (W1 0.189 s, W2 0.488 s, W3 0.011 s) |
| 2 | The harness loads the LOCAL build extension by explicit path, never the autoloaded community version | ✓ VERIFIED | `bench.sh:98`: `"$DUCKDB" -unsigned -cmd "LOAD '$EXT';"` — `$EXT` is the absolute release build path; no bare `LOAD 'anofox_statistics'` in any workload SQL |
| 3 | Each run writes a diffable timestamped results file under `bench/results/` | ✓ VERIFIED | Confirmed: 4 files (`bench-20260831-{204229,204342,212847,215736}.md`) present; `bench/.gitignore` excludes `results/`; `OUTFILE="$RESULTS_DIR/bench-$TIMESTAMP.md"` in script |
| 4 | A `--full` flag runs the 1M-group variant; default runs the scaled 10K-group variant | ✓ VERIFIED | `bench.sh:119-121`: `if [ "$RUN_FULL" -eq 1 ]; then run_workload "W1-full" …/01-agg-dispatch-1m.sql; fi`. Flag parsing at lines 28-30. |
| 5 | `bench/README.md` documents run command, workloads, `--full`, and optional CI perf tracking | ✓ VERIFIED | README contains: run command (`bash scripts/bench.sh`), `--full` flag, workload table (W1/W2/W3/W1-full), results diffing, `--full` resource note (~8 GB, ~160-210 s), and "CI perf tracking (optional — not enforced this phase)" section |
| 6 | `FfiVec<T>` RAII wrapper backed by `libc::malloc` exists; no `Box`/`Vec` on the FfiVec allocation path; `into_raw()` pointer freeable by `libc::free` | ✓ VERIFIED | `types.rs:103-226`: `FfiVec<T>` allocates via `libc::malloc` (line 136), frees via `libc::free` in `Drop` (line 185), `into_raw` uses `std::mem::forget` (line 175). `grep -nE "Box::into_raw\|Vec::into_raw_parts"` returns empty. Unit test `ffi_vec_ptr_is_freeable_by_libc` passes (live: `cargo test -p anofox_stats_ffi` — 6/6 ok) |
| 7 | `alloc_inference_arrays!` macro exists and is applied at 6 strict inference sites; 7 genuinely-divergent sites are documented hand-written | ✓ VERIFIED | `lib.rs:52-97`: macro definition. `grep -c "alloc_inference_arrays!("` = 6. `grep -c "NOTE: hand-written"` = 7. Total = 13 (matches original "13 blocks" scope). Hand-written sites include safety comments explaining differing OOM semantics (GLM z_values, lenient OOM; ALM field names) |
| 8 | `anofox_free_result_core` and `anofox_free_result_inference` bodies are byte-identical to base commit `b52c720` | ✓ VERIFIED | Compared line-by-line with `git show b52c720:crates/anofox-stats-ffi/src/lib.rs`: both function bodies are character-identical. `git diff b52c720 HEAD -- crates/anofox-stats-ffi/src/lib.rs` shows zero diff lines touching these function bodies |
| 9 | Behavior unchanged: `cargo test` (workspace) and `make test` (test/sql) pass green | ✓ VERIFIED | Live runs: `cargo test` — 289 core + 6 ffi = 295 passed, 0 failed. `make test` — "All tests passed (1 skipped test, 2421 assertions in 99 test cases)" |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/bench.sh` | One-command harness loading local build | ✓ VERIFIED | Exists, substantive (130 lines), wired — invoked directly and run successfully |
| `bench/workloads/01-agg-dispatch.sql` | W1 aggregate dispatch | ✓ VERIFIED | Exists; `.timer on` present; no LOAD statement; W1 aggregate dispatch query |
| `bench/workloads/02-fit-predict.sql` | W2 fit/predict | ✓ VERIFIED | Exists; `.timer on` present; no LOAD statement; predict_agg query |
| `bench/workloads/03-ffi-micro.sql` | W3 FFI micro-bench with `compute_inference: true` | ✓ VERIFIED | Exists; `.timer on` present; `compute_inference: true` in query |
| `bench/workloads/01-agg-dispatch-1m.sql` | W1-full 1M-group variant | ✓ VERIFIED | Exists; `.timer on` present; 1M-group scale |
| `bench/README.md` | Harness documentation | ✓ VERIFIED | Exists, substantive — run command, workload table, diffing, `--full` resource note, CI perf tracking note |
| `bench/PROFILING.md` | Profiling methodology + top-3 hotspots | ✓ VERIFIED | Exists, substantive — methodology, operator attribution tables, top-3 dispositions each with rationale + before/after |
| `crates/anofox-stats-ffi/src/types.rs` | `FfiVec<T>` + `DataArray::to_vec` fast path | ✓ VERIFIED | `FfiVec<T>` at lines 103-226; bulk-copy fast path at `to_vec` lines 88-90 |
| `crates/anofox-stats-ffi/src/lib.rs` | `alloc_inference_arrays!` macro + 6 converted sites | ✓ VERIFIED | Macro at lines 52-97; 6 macro invocations + 7 documented hand-written sites |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `bench.sh` | `build/release/duckdb` | `-unsigned -cmd "LOAD '$EXT';"` | ✓ WIRED | Line 98; `$EXT` is the absolute release extension path; fail-fast if not present (lines 44-53) |
| `bench.sh` stdout | `bench/results/bench-<timestamp>.md` | `tee -a "$OUTFILE"` | ✓ WIRED | Lines 73, 93, 98, 110 — `.timer` output captured via process pipe, not DuckDB `.output` |
| `FfiVec<T>::into_raw()` | `FitResultInference` fields | `alloc_inference_arrays!` macro | ✓ WIRED | Macro lines 69-78: all 5 `into_raw()` calls write into `*$out_inference` fields after all 5 allocations succeed |
| `DataArray::to_vec` fast path | Extension fit calls | Existing caller sites in lib.rs | ✓ WIRED | Fast path is inside the only `to_vec` implementation used by every fit function |
| `scripts/bench.sh` (W3 workload) | `DataArray::to_vec` + inference allocation path | `compute_inference: true` in SQL | ✓ WIRED | W3 forces the inference path on every group; confirmed by live run output showing `n_with_inference: 500` |

### Data-Flow Trace (Level 4)

This phase produces benchmarking infrastructure and a refactoring, not a new data pipeline. Data-flow tracing is not applicable: bench results are written to files (not rendered in a UI), and the FFI refactor is a correctness/safety change on an existing data-flow path.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| bench.sh exits 0 and writes results file with `Run Time` for all 3 workloads | `bash scripts/bench.sh` (live) | Exit 0; 3 `Run Time (s)` lines; file written to `bench/results/bench-20260831-215736.md` | ✓ PASS |
| `ffi_vec_ptr_is_freeable_by_libc` unit test passes | `cargo test -p anofox_stats_ffi` (live) | 6/6 tests passed (incl. `ffi_vec_ptr_is_freeable_by_libc`, `ffi_vec_alloc_zero_is_null`, `ffi_vec_drop_frees`) | ✓ PASS |
| Workspace cargo tests pass (behavior unchanged) | `cargo test` (live) | 289 core + 6 ffi = 295 passed, 0 failed | ✓ PASS |
| Full test/sql suite passes (behavior unchanged) | `make test` (live) | 2421 assertions in 99 test cases, 0 failures | ✓ PASS |
| `--help` shows only usage block (WR-02 fix) | `bash scripts/bench.sh --help` (live) | Prints only the Usage: block (5 lines), not all `#` lines | ✓ PASS |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| PERF-01 | Repeatable benchmark harness measures aggregate dispatch, fit/predict, FFI marshalling and reports timings | ✓ SATISFIED | `bash scripts/bench.sh` runs end-to-end (live verified); results file has `Run Time (s)` for W1/W2/W3 |
| PERF-02 | Benchmark runs are reproducible locally and documented; optional CI perf tracking noted | ✓ SATISFIED | `bench/README.md` documents run command, workloads, scales, results diffing, `--full` expectations, and "CI perf tracking (optional — not enforced this phase)" |
| PERF-03 | Top hotspots each optimized or documented as inherent, with before/after numbers | ✓ SATISFIED | `bench/PROFILING.md`: (1) HASH_GROUP_BY — inherent, ~66% W1, not extension-controllable; (2) FFI 5-array inference — leak-safety optimized (Plan 02), count inherent to ABI; (3) DataArray::to_vec — optimized (bulk-copy fast path, ~3-4%, A/B-controlled, 0.0845 s → 0.0813 s) |
| PERF-04 | FFI `libc::malloc`/`free` pattern refactored with RAII wrapper/codegen macro; tests green | ✓ SATISFIED | `FfiVec<T>` (libc::malloc-backed) + `alloc_inference_arrays!` macro at 6 strict sites; 7 genuinely-divergent sites documented hand-written; `anofox_free_*` byte-identical; `make test` 2421 assertions green (live verified) |

**Honest scoping note on PERF-04:** The plan premised "13 byte-identical blocks" for macro conversion, but inspection found three structural variants: 6 strict linear-model sites (OLS/Huber/RANSAC/RLS/WLS/Theil-Sen — code-identical modulo comments), 6 GLM sites (map `z_values` onto `t_values`, lenient OOM path), and 1 ALM site (different field names). The 6 strict sites are all converted; the 7 divergent sites are left hand-written with safety comments explaining the different OOM semantics. This is a PASS for PERF-04: the RAII wrapper and macro exist, the dominant strict allocation pattern is covered, the free contract is preserved, and all suites are green. Converting the divergent sites under the strict macro would alter their (different) OOM behavior — the conservative decision is correct.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No `TBD`, `FIXME`, or `XXX` markers in any Phase 4 file. No placeholder return values. No orphaned code. Code review findings (CR-01, WR-01, WR-02) were all resolved in commit `cabc622` before this verification.

### Code Review Resolutions Confirmed

| Finding | Severity | Fix | Status |
|---------|----------|-----|--------|
| CR-01: dangling `*out_core.coefficients` on OOM path | BLOCKER | `$oom_cleanup` block in macro sets `*out_core = FitResultCore::default()` at all 6 sites | ✓ CONFIRMED — visible at `lib.rs:270, 451, 652, 840, 977, 1329` |
| WR-01: `debug_assert_eq!` compiled out in release | WARNING | Promoted to `assert_eq!` in `copy_from_slice` | ✓ CONFIRMED — `types.rs:158`: `assert_eq!` |
| WR-02: `--help` printed all `#` lines | WARNING | `sed` range prints only Usage: block | ✓ CONFIRMED — live: `bash scripts/bench.sh --help` outputs only usage block |
| IN-01/IN-02: `FfiVec !Send+!Sync`, zero-len null ptr flow | INFO | Acknowledged; no change needed | ✓ CONFIRMED — behavior documented in docstring and covered by `ffi_vec_alloc_zero_is_null` test |

### Human Verification Required

None. All must-haves are verifiable programmatically and have been verified by live command execution.

---

## Gaps Summary

No gaps. All 9 observable truths are VERIFIED by live command execution. All 4 requirements (PERF-01 through PERF-04) are satisfied. The one honest deviation from plan (6 macro sites instead of the premised 13) is correct — the 7 remaining sites have structurally different OOM semantics and are documented with safety comments, not silently skipped.

---

_Verified: 2026-08-31T20:00:00Z_
_Verifier: Claude (gsd-verifier)_
