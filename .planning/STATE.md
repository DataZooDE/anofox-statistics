---
gsd_state_version: 1.0
milestone: v0.3.0
milestone_name: Performance & Polish
current_phase: 5
current_phase_name: API Ergonomics
status: planning
stopped_at: Phase 04 complete, ready to plan Phase 5
last_updated: "2026-08-31T20:02:05.979Z"
last_activity: 2026-08-31
last_activity_desc: Phase 04 complete, transitioned to Phase 5
state_head: 866497affe284497557308128f27f9079736252b
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 33
---

# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-31)

**Core value:** Users can run rigorous statistical models in plain SQL wherever DuckDB runs, including the browser (DuckDB-Wasm) — this milestone makes that faster and easier to use.
**Current focus:** Milestone v0.3.0 — Performance & Polish (Phase 4 complete; ready to plan Phase 5 — API Ergonomics)

## Current Position

Phase: 5 — API Ergonomics
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-31 — Phase 04 complete, transitioned to Phase 5

## Accumulated Context

### Decisions

- Breaking API changes are ALLOWED this milestone (early-dev); docs/tests updated to match
- No new statistical models; no named-parameters work (ERGOX-01 deferred to a dedicated milestone)
- No external domain research phase for v0.3.0
- Benchmark suite (PERF-01/02) comes first — it is the measurement foundation for PERF-03 (hotspots) and PERF-04 (FFI/alloc refactor), which need before/after numbers
- Docs work (Phase 6) comes last, after ERGO renames land, so SQL examples are validated against the final API
- [Phase 4] FFI result arrays are `libc::malloc`-backed (`FfiVec`), never Box/Vec — C++ frees with C `free()` (musl/WASM ABI)
- [Phase 4] Benchmark harness = `scripts/bench.sh`; hotspots dominated by DuckDB `HASH_GROUP_BY` dispatch (inherent), not extension code

### Blockers

- ⚠️ [Phase 4→5] `anofox_stats_ols_fit_predict(...) OVER (...)` rolling window throws a DuckDB INTERNAL error on degenerate sub-`(n_features+1)` frames at partition start — pre-existing input-validation gap; prime ERGO-01/02 target for Phase 5

### Todos

- (none)

## Session Continuity

Last session: 2026-08-31
Stopped at: Phase 04 complete, ready to plan Phase 5
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 4`
