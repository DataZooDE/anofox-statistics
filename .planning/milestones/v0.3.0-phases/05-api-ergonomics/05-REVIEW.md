---
phase: 05-api-ergonomics
reviewed: 2026-09-02T00:00:00Z
depth: deep
files_reviewed: 9
files_reviewed_list:
  - src/aggregate_functions/ols_aggregate.cpp
  - src/aggregate_functions/bls_aggregate.cpp
  - src/include/error_dispatch.hpp
  - src/include/map_options_parser.hpp
  - src/include/map_options_parser.cpp
  - CMakeLists.txt
  - crates/anofox-stats-ffi/src/lib.rs
  - crates/anofox-stats-core/src/models/bls.rs
  - crates/anofox-stats-core/src/types.rs
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: resolved
# WR-01, WR-02, IN-01, IN-02 fixed in commits 16c63d8 and dda627f (2026-09-02)
# IN-03 deferred to Phase 6 doc-SQL validation (guide .sql files not executed by test runner)
---

# Phase 5: Code Review Report

**Reviewed:** 2026-09-02
**Depth:** deep (cross-file call-chain tracing)
**Files Reviewed:** 9 core files + supporting aggregates scanned
**Status:** issues_found

## Summary

Phase 5 delivers three clean deliverables: the OLS aggregate window NULL guard (ERGO-01),
structured FFI error dispatch (ERGO-01/02), and the cross-family rename with unknown-option
rejection (ERGO-02/03). The primary deliverable correctness is sound:

- The `size() <= min_obs` guard in `OlsAggFinalize` (line 272) is mathematically correct.
  With `fit_intercept=true` and `n_features=k`, `min_obs=k+1` matches the saturated-system
  boundary; the `<=` ensures a frame of exactly `min_obs` rows returns NULL. This matches
  the test at `ergo01_window_null.test` (22 assertions).
- `ThrowFromFfiError` is correctly wired at all 11 FFI throw sites. The `InternalException`
  substitution for the absent `FunctionException` is sound and well-documented.
- The `intercept` alias (`key == "intercept" || key == "fit_intercept"` at
  `map_options_parser.cpp:641`) survives the unknown-option rejection — existing tests
  using `{'intercept': true}` continue to bind.
- The CMakeLists.txt WASM guard is correct: gating `Rust_CARGO_TARGET` override on
  `WASM_LOADABLE_EXTENSIONS` prevents native builds from picking up a WASM-format static
  library when the emscripten Rust target is merely installed locally.
- The BLS/NNLS rename is clean. The old alias-registration blocks (which used
  local-scope `AggregateFunction` objects) are removed without lifetime issues.
- NNLS behavioral correctness: `NnlsAggBind` leaves both bounds absent
  (`has_lower_bound=false`, `has_upper_bound=false`). `BlsAggFinalize` passes
  `lower_bounds=nullptr/len=0` to `anofox_bls_fit`. The Rust FFI converts
  `lower_bounds_len=0` to `lower_bounds=None`, which `fit_bls` in `bls.rs:148`
  routes to `BlsRegressor::nnls()` (non-negativity constraints). Behaviorally correct
  even though the `is_nnls` flag in `BlsAggregateBindData` is never read.

Two warnings and three info items follow.

---

## Warnings

### WR-01: `BlsAggFinalize` degenerate-frame guard does not scale with feature count

**File:** `src/aggregate_functions/bls_aggregate.cpp:260`
**Issue:** The finalize guard for BLS (and by inheritance, NNLS since it shares
`BlsAggFinalize`) uses the fixed constant `size() < 2` rather than the
feature-scaled `size() <= min_obs` guard introduced for OLS in this phase.
For a 1-feature BLS fit with intercept, `min_obs=2`; a 2-row frame satisfies
`2 < 2 == false` and proceeds to the Rust FFI. The Rust core
(`bls.rs:97-105`) performs its own `min_obs` check and will return a
`StatsError::InsufficientData`, which `BlsAggFinalize` silently converts to
NULL at line 308 rather than propagating an error. The visible result: a 2-row
BLS window frame returns NULL (correct end result, wrong mechanism — the
degenerate fit attempt happens inside Rust before being caught). For GROUP BY
aggregation this is benign; when `bls_fit_agg` is used as a window aggregate
with `OVER()`, the Rust error path is hit on every saturated frame.

```cpp
// Current (line 260):
if (!state.initialized || state.y_values.size() < 2) {

// Fix — mirrors the OLS guard added in this phase:
if (!state.initialized) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
if (state.y_values.size() <= min_obs) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
```

This applies equally to the other aggregate families that still use `< 2`
(ridge, wls, alm, elasticnet, huber, ransac, lars, rls, poisson, binomial,
negbinom, gamma, logistic, theil_sen) when used as window aggregates with `OVER()`.
The BLS/NNLS case is highlighted here because it is newly registered and
explicitly tested in this phase.

---

### WR-02: `BlsAggFinalize` silently swallows FFI errors as NULL

**File:** `src/aggregate_functions/bls_aggregate.cpp:307-310`
**Issue:** When `anofox_bls_fit` returns `false` (e.g., singular matrix,
convergence failure, or numerical error), the finalize sets a NULL result and
continues rather than calling `ThrowFromFfiError`. This contradicts the
ERGO-01 goal (clear, actionable error messages) and creates a user-experience
asymmetry with OLS — `ols_fit_agg` throws a typed `InvalidInputException` or
`InternalException` on the same errors while `bls_fit_agg` silently NULLs.
`error_dispatch.hpp` is not even included in `bls_aggregate.cpp`.

```cpp
// Current (lines 307-310):
if (!success) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}

// Fix — add #include "../include/error_dispatch.hpp" to the file header,
// then replace the silent NULL with:
if (!success) {
    ThrowFromFfiError("bls_fit_agg", error);
}
```

Note: this fix should be applied at the same time as WR-01 since the Rust
`InsufficientData` error that currently reaches this path would also be
promoted from NULL to a typed `InvalidInputException`.

---

## Info

### IN-01: Stale code comments referencing the old `anofox_stats_` prefix

**Files:**
- `src/aggregate_functions/ols_aggregate.cpp:387,396`
- `src/aggregate_functions/wls_aggregate.cpp:404,413`
- `src/aggregate_functions/ridge_aggregate.cpp:391,401`
- `src/aggregate_functions/elasticnet_aggregate.cpp:364,374`
- `src/aggregate_functions/rls_aggregate.cpp:333,342`
- `src/aggregate_functions/residuals_diagnostics_aggregate.cpp:322`

**Issue:** Eleven inline code comments inside `Register*Function()` bodies
describe overloads with the old `anofox_stats_` prefix (e.g.
`// Basic version: anofox_stats_ols_fit_agg(y, x) - uses defaults`). The
registration strings themselves were correctly renamed; only the comments are
stale. They mislead future contributors into thinking the old names are valid.

**Fix:** Update each comment to use the new unprefixed name. Example for
`ols_aggregate.cpp:387`:
```cpp
// Basic version: ols_fit_agg(y, x) - uses defaults
```

---

### IN-02: `is_nnls` field in `BlsAggregateBindData` is dead code

**File:** `src/aggregate_functions/bls_aggregate.cpp:56`
**Issue:** `BlsAggregateBindData::is_nnls` is set to `true` by `NnlsAggBind`
(line 375) but never read: `BlsAggUpdate` does not copy it to state, and
`BlsAggFinalize` does not inspect it. The NNLS behavior is achieved correctly
via the absent-bounds path in the Rust FFI, making this field a no-op. It
creates a false impression that the C++ layer discriminates BLS from NNLS.

**Fix:** Remove `is_nnls` from `BlsAggregateBindData` (declaration, `Copy()`,
and `Equals()`). Add a brief comment to `NnlsAggBind` explaining that the
absence of bounds is what routes to `BlsRegressor::nnls()` in the Rust core:

```cpp
// NNLS is implemented via anofox_bls_fit with no bounds.
// In the Rust core, lower_bounds=None AND upper_bounds=None routes to
// BlsRegressor::nnls() which enforces lower bound of 0 for all coefficients.
// No explicit is_nnls flag is needed at the C++ layer.
```

---

### IN-03: Guide `.sql` files reference non-existent struct field `residual_standard_error`

**Files:**
- `test/sql/guide01_pattern_4_full_statistical_workflow.sql:21`
- `test/sql/guide04_marketing_campaign_roi.sql:27`
- `test/sql/guide05_comprehensive_ab_test_evaluation.sql:48,58`
- `test/sql/guide05_difference_in_differences_estimation.sql:40`
- `test/sql/guide05_complete_statistical_pipeline.sql:53`
- (and 4 additional guide files)

**Issue:** Ten guide SQL files reference `.residual_standard_error` but the
OLS result struct defines the field as `residual_std_error`
(`ols_aggregate.cpp:82`). The `.sql` guide files are not executed by the
`duckdb_sqllogictest` runner (only `.test` files are), so the suite stays
green. However, users copying examples from these files would get a DuckDB
struct-field-not-found error.

**Fix:** Global replace `residual_standard_error` → `residual_std_error` in
all `.sql` guide files. Alternatively, defer to Phase 6 doc-SQL validation
which is chartered to check SQL examples against the final API.

---

_Reviewed: 2026-09-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
