---
phase: 05-api-ergonomics
fixed_at: 2026-09-02T09:10:00Z
review_path: .planning/phases/05-api-ergonomics/05-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-09-02
**Source review:** `.planning/phases/05-api-ergonomics/05-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (WR-01, WR-02, IN-01, IN-02; IN-03 excluded per instruction)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: `BlsAggFinalize` degenerate-frame guard does not scale with feature count

**Files modified:** `src/aggregate_functions/bls_aggregate.cpp`
**Commit:** 16c63d8
**Applied fix:** Replaced the fixed `y_values.size() < 2` guard with the feature-scaled
two-phase guard mirroring OLS (ols_aggregate.cpp:267-275):
- Split `!state.initialized` into its own null-return branch.
- Added `idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;`
- Guard on `state.y_values.size() <= min_obs` returns NULL for degenerate frames.

This means saturated window frames now return NULL directly in C++ without hitting the
Rust FFI `InsufficientData` path.

### WR-02: `BlsAggFinalize` silently swallows FFI errors as NULL

**Files modified:** `src/aggregate_functions/bls_aggregate.cpp`
**Commit:** 16c63d8
**Applied fix:**
- Added `#include "../include/error_dispatch.hpp"` to the file header.
- Replaced the silent `FlatVector::SetNull` + `continue` on FFI failure with
  `ThrowFromFfiError("bls_fit_agg", error)`, matching the OLS aggregate pattern.
- Degenerate frames (filtered by the WR-01 guard) still return NULL; genuine errors
  (singular matrix, convergence failure, invalid input) now throw a typed exception
  per the error_dispatch.hpp taxonomy (InternalException for numerical failures,
  InvalidInputException for user data / shape problems).

**Note:** Both WR-01 and WR-02 were applied in a single atomic commit because they
interact: after WR-01 the Rust `InsufficientData` path is no longer reachable for
degenerate frames, so WR-02's ThrowFromFfiError no longer fires on saturated windows.

### IN-01: Stale code comments referencing the old `anofox_stats_` prefix

**Files modified:**
- `src/aggregate_functions/ols_aggregate.cpp` (2 comments, lines 387, 396)
- `src/aggregate_functions/wls_aggregate.cpp` (2 comments, lines 404, 413)
- `src/aggregate_functions/ridge_aggregate.cpp` (2 comments, lines 391, 401)
- `src/aggregate_functions/elasticnet_aggregate.cpp` (2 comments, lines 364, 374)
- `src/aggregate_functions/rls_aggregate.cpp` (2 comments, lines 333, 342)
- `src/aggregate_functions/residuals_diagnostics_aggregate.cpp` (2 comments, lines 322, 332)

**Commit:** dda627f
**Applied fix:** Removed `anofox_stats_` prefix from all 12 inline comments inside
`Register*Function()` bodies. The `#include "../include/anofox_stats_ffi.h"` lines
in the same files are NOT stale — that is the actual FFI header filename and was
left unchanged.

### IN-02: `is_nnls` field in `BlsAggregateBindData` is dead code

**Files modified:** `src/aggregate_functions/bls_aggregate.cpp`
**Commit:** dda627f
**Applied fix:** Verified via grep that `is_nnls` was set in `NnlsAggBind` (line 375)
but never read by `BlsAggUpdate`, `BlsAggCombine`, or `BlsAggFinalize`. Removed:
- The field declaration `bool is_nnls = false;` from `BlsAggregateBindData`
- The copy line `result->is_nnls = is_nnls;` in `Copy()`
- The comparison `&& is_nnls == other.is_nnls` in `Equals()`
- The assignment `result->is_nnls = true;` in `NnlsAggBind`

Added explanatory comment to `NnlsAggBind` clarifying that NNLS routing is achieved
via absent bounds (`lower_bounds=None`, `upper_bounds=None` → `BlsRegressor::nnls()`
in the Rust core), so no explicit discriminant is needed at the C++ layer.

## Skipped Issues

None — all in-scope findings were fixed.

## Deferred (not in scope)

**IN-03:** Guide `.sql` files referencing `residual_standard_error` instead of
`residual_std_error` — deferred to Phase 6 doc-SQL validation as chartered.
The guide `.sql` files are not executed by the `duckdb_sqllogictest` runner
(only `.test` files are), so the test suite remains green.

## Verification

Build and test suites ran in the **main checkout** (workflow.use_worktrees=false).

- `make release`: succeeded (all targets built, 0 errors)
- `make test`: **103 passed / 1 skipped** (2473 assertions) — matches Phase 5 baseline
- `cargo test --workspace`: **295 passed** (289 unit + 6 integration, 0 failed) — matches baseline

---

_Fixed: 2026-09-02_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
