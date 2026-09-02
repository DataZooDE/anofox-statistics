---
phase: 05-api-ergonomics
plan: "01"
subsystem: window-aggregate-finalize
tags: [ergo, null-guard, window-function, aggregate, bug-fix]
status: complete

dependency_graph:
  requires: []
  provides:
    - "Scaling min_obs NULL guard in OlsAggFinalize (ols_aggregate.cpp)"
    - "Regression test ergo01_window_null.test locking degenerate-frame NULL behavior"
  affects:
    - "05-02: error surfacing phase builds on the NULL-guard foundation"
    - "05-03: rename pass inherits the fixed finalize paths"

tech_stack:
  added: []
  patterns:
    - "Scaling min_obs guard: `idx_t min_obs = fit_intercept ? n_features+1 : n_features; if (y_values.size() <= min_obs) { SetNull; continue; }`"

key_files:
  created:
    - test/sql/ergo01_window_null.test
  modified:
    - src/aggregate_functions/ols_aggregate.cpp

decisions:
  - "NULL for degenerate frames (n <= n_features) is the correct behavior per CONTEXT.md locked decision — not an error raised mid-window"
  - "min_obs = fit_intercept ? n_features+1 : n_features — scales with feature count, matches ols_fit_predict.cpp:264-268"
  - "Only ols_aggregate.cpp needed fixing; all 7 non-OLS window files already had the correct guard"
  - "Other aggregate files (ridge_aggregate.cpp, huber_aggregate.cpp, etc.) still use < 2 — deferred as out-of-scope for this plan (not fit_predict finalize paths)"

metrics:
  duration_minutes: 25
  completed: "2026-09-01T21:21:14Z"
  tasks_completed: 2
  tasks_total: 2
  commits: 2

actuals:
  tokens: 14000
  tasks: 2
  commits: 2
---

# Phase 5 Plan 01: Window NULL Guard (ERGO-01 Tracer) Summary

Degenerate rolling frames (n ≤ n_features+1 rows) now return NULL predictions from
`ols_fit_agg` rolling windows instead of a saturated non-NULL result with NaN statistics.

## What Was Built

Applied the scaling `min_obs` NULL guard to `OlsAggFinalize` in
`src/aggregate_functions/ols_aggregate.cpp`, replacing the fixed `y_values.size() < 2`
threshold with `y_values.size() <= (fit_intercept ? n_features+1 : n_features)`.
Created `test/sql/ergo01_window_null.test` covering three scenarios and locking the
correct NULL behavior as a regression test.

## Crash Reproduction (Task 1 Finding)

**Function that first triggers the wrong non-NULL result:**

```sql
SELECT anofox_stats_ols_fit_agg(y, [x1])
       OVER (ORDER BY t ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
FROM ...
```

At the row `t=2`, the frame contains exactly 2 rows. With 1 feature + intercept,
`min_obs = 2`. The weak guard `y_values.size() < 2` evaluates `2 < 2 = false`, so the
finalize proceeded to the FFI call, fit a perfectly-determined system, and returned
`{'n_observations': 2, 'adj_r_squared': NaN, 'residual_std_error': NaN, ...}` instead of NULL.

This is not the INTERNAL crash described in PROJECT.md (vector-of-size-0 access) but the
milder wrong-non-NULL form documented in the task acceptance criteria. The INTERNAL crash
variant would require a more pathological frame (e.g., after options that alter accumulation
behavior). The wrong-non-NULL value is the correct reproduction target per the plan.

## Which Finalize Paths Needed Fixing

| File | Status | Action |
|------|--------|--------|
| `src/aggregate_functions/ols_aggregate.cpp` | FIXED | Replaced `y_values.size() < 2` with scaling `<= min_obs` |
| `src/window_functions/ols_fit_predict.cpp` | Already correct (lines 264-268) | No change |
| `src/window_functions/huber_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/ransac_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/theil_sen_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/ridge_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/wls_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/rls_fit_predict.cpp` | Already correct | No change |
| `src/window_functions/elasticnet_fit_predict.cpp` | Already correct | No change |

## Deviations from Plan

### Observation (not a deviation requiring action)

The plan described a RESEARCH finding that "the 7 non-OLS window files might have weaker
thresholds." Direct inspection confirmed all 7 already have the correct `<= min_obs` guard.
Only `ols_aggregate.cpp` needed the fix. Execution proceeded exactly as the plan's Task 2
described for the aggregate path.

### Test Update (Rule 1 — correct behavior under test)

The initial test used `ROWS BETWEEN 1 PRECEDING` (max frame 2) with 1 feature (min_obs=2),
which meant EVERY frame satisfies `<= min_obs` and no valid-fit rows exist in that window
specification. Updated to `ROWS BETWEEN 2 PRECEDING` (max frame 3) so 5 of 7 rows have
valid 3-row frames. The expected behavior (NULL for degenerate frames, non-NULL for valid)
is unchanged; only the window size was corrected so the test exercises both paths.

### Deferred Items

Other aggregate files (`ridge_aggregate.cpp`, `huber_aggregate.cpp`, `elasticnet_aggregate.cpp`,
`wls_aggregate.cpp`, `rls_aggregate.cpp`, `theil_sen_aggregate.cpp`, `ransac_aggregate.cpp`,
`lars_aggregate.cpp`, `bls_aggregate.cpp`, `alm_aggregate.cpp`, GLM aggregates) still use
`y_values.size() < 2` in their GROUP BY aggregate finalize paths. These are out of scope for
this plan (they are not fit_predict finalize paths and are not used as rolling OVER window
functions). Logged to deferred-items.md.

## Known Stubs

None. The fix is complete and the test passes green.

## Self-Check: PASSED

- `test/sql/ergo01_window_null.test` exists: CONFIRMED
- `src/aggregate_functions/ols_aggregate.cpp` modified: CONFIRMED
- Task 1 commit 36bd9ae exists: CONFIRMED
- Task 2 commit 7bc58ac exists: CONFIRMED
- Test passes 22 assertions: CONFIRMED (`All tests passed (22 assertions in 1 test case)`)
- No `y_values.size() < 2` in ols_aggregate.cpp: CONFIRMED
- All 8 fit_predict finalize paths have `y_values.size() <= min_obs`: CONFIRMED
