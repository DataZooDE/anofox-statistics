---
gsd_state_version: 1.0
milestone: v0.3.0
milestone_name: Performance & Polish
current_phase: 6
current_phase_name: Docs Refresh & SQL Validation
status: executing
stopped_at: Completed 06-03-PLAN.md (README restructure + harness passes)
last_updated: "2026-09-02T08:26:31.957Z"
last_activity: 2026-09-02
last_activity_desc: Phase 6 execution resumed (wave continue)
state_head: 930c05056b9a9ef308874441a1ab69d1f53b0c16
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 10
  completed_plans: 9
  percent: 67
---

# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-31)

**Core value:** Users can run rigorous statistical models in plain SQL wherever DuckDB runs, including the browser (DuckDB-Wasm) — this milestone makes that faster and easier to use.
**Current focus:** Phase 6 — Docs Refresh & SQL Validation

## Current Position

Phase: 6 (Docs Refresh & SQL Validation) — EXECUTING
Plan: 1 of ?
Status: Executing Phase 6
Last activity: 2026-09-02 — Phase 6 execution resumed (wave continue)

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
- [Phase 5]: [Phase 5][ERGO-03]: anofox_stats_ prefix dropped from all SQL registrations; theilsen->theil_sen; no deprecated aliases; docs/API_CONVENTIONS.md is the authoritative naming reference for Phase 6
- [Phase 6]: Python harness validate_docs_sql.py: per-file concatenated DuckDB session with .bail on; skip marker is 'sql skip' info-string
- [Phase 6]: 5 blocks in API_CONVENTIONS.md skip-marked: 2 migration Before examples + 3 syntax-illustration blocks with FROM tbl
- [Phase 6]: Skip-mark API_REFERENCE.md signature blocks — they are API type sketches, not runnable examples
- [Phase 6]: ols_fit_agg/rls_fit_agg OVER window crashes DuckDB INTERNAL Error — skip-mark, not a doc error
- [Phase 6]: README Quick Start uses (ols_fit([y_vals], [[x_col_vals]])).coefficients subquery inside predict() — cross-join stored struct approach had NULL issue in DuckDB optimizer for small datasets
- [Phase 6]: ols_fit scalar X is column-major: each inner array is all observations for one feature (not row-major)
- [Phase 6]: README Installation/telemetry sql blocks marked 'sql skip' — not runnable in local harness; GROUP BY illustrative example also skip-marked (references undefined sales_data)

### Blockers

- ⚠️ [Phase 4→5] `anofox_stats_ols_fit_predict(...) OVER (...)` rolling window throws a DuckDB INTERNAL error on degenerate sub-`(n_features+1)` frames at partition start — pre-existing input-validation gap; prime ERGO-01/02 target for Phase 5

### Todos

- (none)

## Session Continuity

Last session: 2026-09-02T08:26:31.911Z
Stopped at: Completed 06-03-PLAN.md (README restructure + harness passes)
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 4`

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 05 P01 | 25 | 2 tasks | 2 files |
| Phase 05 P02 | 18 | 3 tasks | 14 files |
| Phase 05 P03 | 35 | 3 tasks | 161 files |
| Phase 06 P01 | 4 | 2 tasks | 2 files |
| Phase 06-docs-refresh-sql-validation P02 | 45 | 3 tasks | 5 files |
| Phase 06-docs-refresh-sql-validation P03 | 25 | 2 tasks | 1 files |
