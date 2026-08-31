---
phase: 04-benchmarking-performance
plan: 02
subsystem: infra
tags: [rust, ffi, raii, macro, libc, abi, perf]

requires:
  - phase: 04-benchmarking-performance
    provides: bench.sh harness (W3 is the allocation-heavy before/after workload)
provides:
  - FfiVec<T> — a libc::malloc-backed RAII wrapper in types.rs (frees via libc::free on Drop; into_raw hands ownership to C++)
  - alloc_inference_arrays! macro in lib.rs marshalling the 5-array inference block via FfiVec
  - 6 strict linear-model inference sites (OLS/Huber/RANSAC/RLS/WLS/Theil-Sen) refactored to the macro
affects: [04-03-profiling-hotspots]

actuals:
  tokens: 14000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "libc::malloc-backed RAII wrapper (FfiVec) for FFI result arrays — NOT Box/Vec (musl/WASM ABI safety)"
    - "macro_rules! codegen for the repeated 5-array inference marshalling; allocate-all-then-into_raw for OOM safety"

key-files:
  created: []
  modified:
    - crates/anofox-stats-ffi/src/types.rs
    - crates/anofox-stats-ffi/src/lib.rs

key-decisions:
  - "FfiVec allocates with libc::malloc / frees with libc::free (one-way ABI decision, confirmed at the Task-1 checkpoint) — never Box/Vec"
  - "The '13 byte-identical blocks' premise was inaccurate: 6 strict sites converted; 6 GLM (z_values, lenient OOM) + 1 ALM (standard_errors/conf_int) sites left hand-written with a safety comment to avoid a behavior change"
  - "anofox_free_* functions kept byte-identical (verified via git diff) — Rust-side-only refactor"

patterns-established:
  - "FfiVec<T>::alloc(n) -> copy_from_slice -> into_raw() feeds C++-owned raw pointers freed by anofox_free_* via C free()"

requirements-completed: [PERF-04]

coverage:
  - id: D1
    description: "FfiVec<T> RAII wrapper (libc::malloc-backed) exists; into_raw pointer freeable by libc::free"
    requirement: "PERF-04"
    verification:
      - kind: unit
        ref: "cargo test -p anofox_stats_ffi ffi_vec_ptr_is_freeable_by_libc (+ alloc_zero, drop_frees)"
        status: pass
    human_judgment: false
  - id: D2
    description: "alloc_inference_arrays! macro replaces the 6 strict 5-array inference blocks; behavior unchanged"
    requirement: "PERF-04"
    verification:
      - kind: integration
        ref: "make test — 2421 assertions, 99 test cases, all pass (behavior oracle)"
        status: pass
      - kind: unit
        ref: "cargo test (workspace) — 289 core + 6 ffi pass"
        status: pass
    human_judgment: false
  - id: D3
    description: "anofox_free_* free contract byte-identical; no Box/Vec on the FfiVec allocation path"
    requirement: "PERF-04"
    verification:
      - kind: other
        ref: "git diff b52c720 shows no change to anofox_free_result_inference/core; grep finds no Box::into_raw/Vec::into_raw_parts in types.rs"
        status: pass
    human_judgment: false

duration: 40min
completed: 2026-08-31
status: complete
---

# Phase 4 (Plan 02): FFI Allocation Refactor Summary

**libc::malloc-backed `FfiVec<T>` RAII wrapper + `alloc_inference_arrays!` macro replacing the 6 strict 5-array inference-allocation blocks, with the C++ `anofox_free_*` contract byte-identical and the full test/sql + cargo suites green.**

## Performance
- **Duration:** ~40 min (incl. a full DuckDB rebuild to run the behavior oracle)
- **Completed:** 2026-08-31T21:26:00+02:00
- **Tasks:** 3 (1 decision checkpoint + 2 code)
- **Files modified:** 2

## Accomplishments
- **`FfiVec<T>`** (types.rs): owns a `libc::malloc` buffer, frees via `libc::free` on `Drop`, or relinquishes ownership with `into_raw()` to the C++ side (which frees with C `free()`). Deliberately not `Box`/`Vec`-backed — Rust's global allocator can differ from libc malloc on musl (WASM/CI), which would break the free contract. Three unit tests, incl. `ffi_vec_ptr_is_freeable_by_libc` which would be UB (ASan/valgrind-flagged) if the impl ever switched to Box/Vec.
- **`alloc_inference_arrays!`** (lib.rs): marshals the 5-array inference block (std_errors/t_values/p_values/ci_lower/ci_upper) through `FfiVec`, allocating all five before any `into_raw()` so a mid-sequence OOM frees cleanly with no dangling pointer written into `FitResultInference`. Applied to the **6 strict** linear-model sites (OLS, Huber, RANSAC, RLS, WLS, Theil-Sen).
- **Behavior oracle green:** `make test` — 2421 assertions across 99 test cases pass; `cargo test` (workspace) — 289 core + 6 ffi pass. Results unchanged from baseline.
- **Contracts preserved:** `anofox_free_*` bodies byte-identical (empty git diff); no `Box`/`Vec` on the FfiVec allocation path.

## Task Commits
1. **Task 1: DECISION checkpoint (libc::malloc allocator)** — resolved via AskUserQuestion ("Confirm libc::malloc"); no code commit (gate).
2. **Task 2: FfiVec<T> RAII wrapper + unit tests** — `a7ec67e` (perf)
3. **Task 3: alloc_inference_arrays! macro over 6 strict sites** — `7324009` (perf)

## Files Created/Modified
- `crates/anofox-stats-ffi/src/types.rs` — `FfiVec<T>` + 3 unit tests
- `crates/anofox-stats-ffi/src/lib.rs` — the macro + 6 converted sites + safety comments on the 7 divergent sites

## Decisions Made
- **libc::malloc allocator** confirmed at the blocking-human checkpoint (one-way ABI door).
- **Scope of conversion:** 6 of 13 sites. See deviation below.

## Deviations from Plan

### 1. "13 byte-identical inference blocks" was inaccurate — converted 6, documented 7
- **Found during:** Task 3 (surveying the 13 sites before writing the macro)
- **Issue:** The plan/research premised 13 byte-identical blocks. In fact there are **three variants**: (A) 6 strict linear-model sites — check all 5 allocations, free-all + coefficients + `return false` on OOM, `t_values` source, real `f_statistic`; (B) 6 GLM sites — map `z_values` onto the `t_values` field, lenient OOM (`if n>0 && !std_err_ptr.is_null()`, no return), `f_statistic: NAN`; (C) 1 ALM site — `standard_errors` / `conf_int_lower/upper` field names, lenient OOM. The 6 variant-A blocks are code-identical (differences are only inline comments).
- **Fix:** Converted the 6 strict variant-A sites with `alloc_inference_arrays!` (the exact pattern the FfiVec+macro was designed for — "the bulk fit-result marshalling pattern" per CONTEXT.md). Left the 6 GLM + 1 ALM sites hand-written, each with a short safety comment noting the different inference contract — converting them under the strict macro would alter their (untested but real) OOM semantics. CONTEXT.md explicitly permits leaving genuinely-divergent sites documented rather than forcing an unfaithful conversion.
- **Verification:** `make test` (2421 assertions) + `cargo test` (289+6) green — success-path behavior unchanged for both converted and untouched sites.
- **Committed in:** `7324009`

### 2. Package name in verify commands
- The plan's verify used `-p anofox-stats-ffi`; the actual Cargo package is `anofox_stats_ffi` (underscores). Used the correct name.

---
**Total deviations:** 2 (scope-of-conversion honesty + package-name correction). No behavior change; no scope creep.

## Issues Encountered
- The local `build/release` (Aug 5) was stale and the `datazoo-banner` submodule was uninitialized; had to `git submodule update --init datazoo-banner` and run a full `make release` (~20–40 min) before `make test` could serve as the behavior oracle. The rebuild + full suite then passed.

## User Setup Required
- **Leak check deferred (noted/optional):** `valgrind` is not installed (needs `sudo pacman -S valgrind`). The FfiVec RAII `Drop` + `ffi_vec_ptr_is_freeable_by_libc` unit test provide the leak-safety guarantee; a valgrind/ASan pass over W3 is a nice-to-have not run here.

## Next Phase Readiness
- Plan 03 (profiling) can proceed: the extension is freshly rebuilt (release, with the refactor) and `bench/workloads/03-ffi-micro.sql` exercises the refactored allocation path. Plan 03 needs `perf` installed (`sudo pacman -S perf`) — its human-action checkpoint.

---
*Phase: 04-benchmarking-performance*
*Completed: 2026-08-31*
