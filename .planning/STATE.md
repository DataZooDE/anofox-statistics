---
gsd_state_version: 1.0
milestone: v0.3.0
milestone_name: Performance & Polish
current_phase: 5
current_phase_name: API Ergonomics
status: executing
stopped_at: Completed 05-02-PLAN.md
last_updated: "2026-09-01T21:43:52.574Z"
last_activity: 2026-09-01
last_activity_desc: Phase 5 execution started
state_head: 1492032fc1fd6aefa0729f69b9e36038fac8796b
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 6
  completed_plans: 5
  percent: 33
---

# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-31)

**Core value:** Users can run rigorous statistical models in plain SQL wherever DuckDB runs, including the browser (DuckDB-Wasm) — this milestone makes that faster and easier to use.
**Current focus:** Phase 5 — API Ergonomics

## Current Position

Phase: 5 (API Ergonomics) — EXECUTING
Plan: 3 of 3
Status: Ready to execute
Last activity: 2026-09-01 — Phase 5 execution started

## Accumulated Context

### Decisions

- Breaking API changes are ALLOWED this milestone (early-dev); docs/tests updated to match
- No new statistical models; no named-parameters work (ERGOX-01 deferred to a dedicated milestone)
- No external domain research phase for v0.3.0
- Benchmark suite (PERF-01/02) comes first — it is the measurement foundation for PERF-03 (hotspots) and PERF-04 (FFI/alloc refactor), which need before/after numbers
- Docs work (Phase 6) comes last, after ERGO renames land, so SQL examples are validated against the final API
- [Phase 4] FFI result arrays are `libc::malloc`-backed (`FfiVec`), never Box/Vec — C++ frees with C `free()` (musl/WASM ABI)
- [Phase 4] Benchmark harness = `scripts/bench.sh`; hotspots dominated by DuckDB `HASH_GROUP_BY` dispatch (inherent), not extension code
- [Phase 5]: Scaling min_obs NULL guard (fit_intercept ? n_features+1 : n_features) replaces fixed < 2 threshold in OlsAggFinalize — degenerate frames return NULL per ERGO-01
- [Phase 5]: All 7 non-OLS window fit_predict files already had the correct guard; only ols_aggregate.cpp needed fixing
- [Phase 5]: FunctionException not in embedded DuckDB — InternalException used for numerical FFI failures (SingularMatrix, ConvergenceFailure, Internal, AllocationFailure)
- [Phase 5]: [Phase 5]: All 11 FFI !success sites wired to ThrowFromFfiError; GROUP BY aggregate finalize now throws instead of silently NULLing
- [Phase 5]: [Phase 5]: Unknown MAP option keys rejected at bind via InvalidInputException in RegressionMapOptions and all 10 test-option parsers; intercept alias preserved

### Blockers

- ⚠️ [Phase 4→5] `anofox_stats_ols_fit_predict(...) OVER (...)` rolling window throws a DuckDB INTERNAL error on degenerate sub-`(n_features+1)` frames at partition start — pre-existing input-validation gap; prime ERGO-01/02 target for Phase 5

### Todos

- (none)

## Session Continuity

Last session: 2026-09-01T21:43:52.542Z
Stopped at: Completed 05-02-PLAN.md
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 4`

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 05 P01 | 25 | 2 tasks | 2 files |
| Phase 05 P02 | 18 | 3 tasks | 14 files |
