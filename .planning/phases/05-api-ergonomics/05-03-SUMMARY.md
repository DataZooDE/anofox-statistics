---
phase: 05-api-ergonomics
plan: "03"
subsystem: api
tags: [duckdb-extension, sql-functions, naming-convention, test-suite, c++, rust]

requires:
  - phase: 05-01
    provides: window null-guard fixes (ERGO-01) against which rename lands
  - phase: 05-02
    provides: bind-time option validation (ERGO-02) against which rename lands

provides:
  - Unprefixed public SQL API: every function registered as {model}_{verb}[_agg|_predict], no anofox_stats_ prefix
  - theil_sen rename: theilsen fixed to theil_sen everywhere (registration, macros, tests)
  - ergo03_naming.test: smoke test proving renamed names resolve and old prefixed names fail
  - docs/API_CONVENTIONS.md: written naming convention (names, option keys, return fields, z_values exception, v0.3.0 breaking changes)
  - Full test suite (make test + cargo test --workspace) green against the renamed API

affects:
  - 06-docs (Phase 6 doc-SQL validation checks examples against docs/API_CONVENTIONS.md)

actuals:
  tokens: 42000
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "All SQL functions registered under single unprefixed canonical name — alias blocks deleted entirely"
    - "docs/API_CONVENTIONS.md as the authoritative reference for Phase 6 doc-SQL validation"

key-files:
  created:
    - test/sql/ergo03_naming.test
    - docs/API_CONVENTIONS.md
  modified:
    - src/aggregate_functions/bls_aggregate.cpp
    - CMakeLists.txt
    - test/sql/* (158 files — prefix strip, .r2->.r_squared, theilsen->theil_sen)

key-decisions:
  - "ols_predict_agg (deprecated alias) is dropped; canonical name is ols_fit_predict_agg — test updated accordingly"
  - "t_test_agg signature is (DOUBLE, INTEGER, ...) — ergo03 smoke test corrected to use INTEGER group_id"
  - "vif takes a single DOUBLE[][] argument, not (DOUBLE[], DOUBLE[][]) — ergo03 smoke test corrected"
  - "CMakeLists.txt WASM cargo target override scoped to WASM_LOADABLE_EXTENSIONS only to avoid mis-linking native builds"

patterns-established:
  - "Rename map: every anofox_stats_ prefix dropped; theilsen -> theil_sen; no deprecated aliases retained"
  - "ergo03_naming.test pattern: positive + negative smoke tests for any future bulk rename"

requirements-completed: [ERGO-03]

coverage:
  - id: D1
    description: "All SQL functions registered under unprefixed {model}_{verb}[_agg|_predict] names with alias blocks deleted"
    requirement: ERGO-03
    verification:
      - kind: integration
        ref: "grep -rn '\"anofox_stats_' src/ | grep -v '#include' | grep -v comment | wc -l == 0"
        status: pass
      - kind: integration
        ref: "make release (build succeeds after registration rename)"
        status: pass
    human_judgment: false
  - id: D2
    description: "theilsen renamed to theil_sen in all registrations, macros, and tests"
    requirement: ERGO-03
    verification:
      - kind: integration
        ref: "grep -rn '\\btheilsen\\b' src/ test/sql/ | wc -l == 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "Full test/sql suite rewritten to renamed API with .r_squared fields — make test green"
    requirement: ERGO-03
    verification:
      - kind: integration
        ref: "make test: 103 passed, 1 skipped (quack), 0 failed"
        status: pass
    human_judgment: false
  - id: D4
    description: "ergo03_naming.test: positive + negative smoke test proving the rename"
    requirement: ERGO-03
    verification:
      - kind: integration
        ref: "test/sql/ergo03_naming.test (included in make test run above)"
        status: pass
    human_judgment: false
  - id: D5
    description: "docs/API_CONVENTIONS.md documenting naming convention, option keys, return fields, z_values exception, v0.3.0 breaking changes"
    requirement: ERGO-03
    verification:
      - kind: other
        ref: "grep -qi 'z_values' docs/API_CONVENTIONS.md && grep -qiE 'theil_sen|no deprecated alias|breaking' docs/API_CONVENTIONS.md"
        status: pass
    human_judgment: false
  - id: D6
    description: "cargo test --workspace green (Rust side unaffected by SQL rename)"
    requirement: ERGO-03
    verification:
      - kind: unit
        ref: "cargo test --workspace: 289 unit + 6 ffi tests passed"
        status: pass
    human_judgment: false

duration: ~35min
completed: "2026-09-02"
status: complete
---

# Phase 5 Plan 03: API Naming Convention (ERGO-03) Summary

**Dropped anofox_stats_ prefix from all SQL registrations, fixed theilsen→theil_sen, deleted all alias blocks, and shipped docs/API_CONVENTIONS.md with the written convention; make test (103 tests) and cargo test --workspace (295 tests) fully green against the renamed API**

## Performance

- **Duration:** ~35 min (continuation of interrupted session)
- **Completed:** 2026-09-02
- **Tasks:** 3 (Task 1 pre-committed; Tasks 2+3 completed and committed in this session)
- **Files modified:** 161 (158 test/sql files + bls_aggregate.cpp + CMakeLists.txt + docs/API_CONVENTIONS.md)

## Accomplishments

- Every SQL function now resolves under one unprefixed name; the `anofox_stats_` prefix is fully purged from all registration strings and SQL function references
- `theilsen` fixed to `theil_sen` everywhere — registration, macro entry, test files
- 158 test/sql files rewritten: prefix stripped, `.r2` -> `.r_squared`, stale function names corrected
- `test/sql/ergo03_naming.test` added: positive checks that renamed names resolve across OLS, Theil-Sen, GLM Poisson, t-test, VIF, BLS, NNLS families; negative checks that `anofox_stats_ols_fit`, `anofox_stats_theil_sen_fit`, `anofox_stats_t_test_agg` all fail with "does not exist"
- `docs/API_CONVENTIONS.md` written: function naming pattern, snake_case option keys with accepted aliases, standard return-struct field set, documented GLM/AFT z_values and ALM exceptions, v0.3.0 breaking-change section
- `bls_aggregate.cpp` completed: `bls_fit_agg` + `nnls_fit_agg` registered under unprefixed names
- `CMakeLists.txt` fixed: Rust cargo WASM target override scoped to `WASM_LOADABLE_EXTENSIONS` build only, preventing native `make release` from linking a WASM static lib

## Task Commits

1. **Task 1: Rename all registrations** - `819c5bf` (feat — pre-committed by prior session)
2. **Task 2: Reconcile test suite + registrations** - `9efbfdc` (fix)
3. **Task 3: docs/API_CONVENTIONS.md + cargo green** - `bb9195b` (docs)

## Resolved Ambiguities (Previously "(check file)" entries)

| Stale/Ambiguous | Resolved to | Location |
|---|---|---|
| `ols_predict_agg` (deprecated alias) | `ols_fit_predict_agg` (canonical) | test/sql/predict_agg/test_ols_predict_agg.test |
| `t_test_agg(DECIMAL, DECIMAL)` wrong signature | `t_test_agg(DOUBLE, INTEGER)` — group_id is INTEGER | ergo03_naming.test |
| `vif(y, x[])` two-arg call | `vif(DOUBLE[][])` — single feature matrix arg | ergo03_naming.test |

## Files Created/Modified

- `test/sql/ergo03_naming.test` — ERGO-03 smoke test (new)
- `docs/API_CONVENTIONS.md` — full naming convention (new)
- `src/aggregate_functions/bls_aggregate.cpp` — completed bls_fit_agg/nnls_fit_agg registrations
- `CMakeLists.txt` — WASM cargo override scoped correctly
- `test/sql/*` — 158 files rewritten (prefix, .r2, theilsen, stale names)

## Decisions Made

- `ols_predict_agg` deprecated alias was deleted in Task 1; the canonical name is `ols_fit_predict_agg`. The test file `test_ols_predict_agg.test` was updated to use the canonical name rather than restoring the alias — consistent with the "no deprecated aliases" locked decision.
- `t_test_agg` signature is `(DOUBLE, INTEGER, [MAP])` — group_id is an integer label, not a second measurement. The ergo03 smoke test was corrected to pass integer group IDs.
- `vif` takes a single `DOUBLE[][]` feature matrix. The ergo03 smoke test was corrected from a two-argument form to the actual one-argument form.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ols_predict_agg test used wrong function name**
- **Found during:** Task 2 (make test — first run)
- **Issue:** `test_ols_predict_agg.test` called `ols_predict_agg` which was dropped as an alias; canonical name is `ols_fit_predict_agg`
- **Fix:** Updated test file to use `ols_fit_predict_agg` throughout
- **Files modified:** `test/sql/predict_agg/test_ols_predict_agg.test`
- **Committed in:** `9efbfdc`

**2. [Rule 1 - Bug] ergo03_naming.test used wrong t_test_agg signature**
- **Found during:** Task 2 (make test — second run, after fix 1)
- **Issue:** `t_test_agg(x, y)` with two DECIMAL values; actual signature is `(DOUBLE, INTEGER)` — second arg is group_id
- **Fix:** Updated test to pass integer group labels: `t_test_agg(x, g) FROM (VALUES (1.0, 0), ...) t(x, g)`
- **Files modified:** `test/sql/ergo03_naming.test`
- **Committed in:** `9efbfdc`

**3. [Rule 1 - Bug] ergo03_naming.test used wrong vif signature**
- **Found during:** Task 2 (make test — third run, after fix 2)
- **Issue:** `vif(y_vec, x_matrix)` two-argument call; actual signature is `vif(DOUBLE[][])` single feature matrix
- **Fix:** Updated test to `vif([[col1], [col2]])` form
- **Files modified:** `test/sql/ergo03_naming.test`
- **Committed in:** `9efbfdc`

**4. [Rule 1 - Bug] ergo03_naming.test had stale comment with bare `theilsen`**
- **Found during:** Pre-commit acceptance check (grep -rn '\\btheilsen\\b')
- **Issue:** Comment line `# Theil-Sen family (theilsen -> theil_sen verified)` matched the acceptance criterion grep
- **Fix:** Reworded comment to `# Theil-Sen family (theil_sen rename verified)`
- **Files modified:** `test/sql/ergo03_naming.test`
- **Committed in:** `9efbfdc`

---

**Total deviations:** 4 auto-fixed (all Rule 1 — bugs in the ergo03 smoke test and one stale alias reference)
**Impact on plan:** All fixes are test-correctness corrections. The rename itself landed exactly as planned.

## Issues Encountered

None beyond the four auto-fixed test bugs above.

## Next Phase Readiness

- Phase 6 (docs) can now check SQL examples against `docs/API_CONVENTIONS.md` using the final unprefixed API
- ERGO-03 requirement fully satisfied; all three ERGO requirements (01, 02, 03) complete
- No deprecated aliases — Phase 6 does not need to handle dual-name resolution

## Self-Check

- `grep -rn '"anofox_stats_' src/ | grep -v '#include' | grep -v '//': 0 matches` - PASS
- `grep -rn 'anofox_stats_' test/sql/ | grep -v ergo03_naming: 0 matches` - PASS
- `grep -rn '\.r2\b' test/sql/: 0 matches` - PASS
- `grep -rn '\btheilsen\b' src/ test/sql/: 0 matches` - PASS
- `docs/API_CONVENTIONS.md exists`: PASS
- `test/sql/ergo03_naming.test exists`: PASS
- `make test: 103 passed, 1 skipped, 0 failed`: PASS
- `cargo test --workspace: 295 passed, 0 failed`: PASS
- Commits 819c5bf, 9efbfdc, bb9195b verified in git log: PASS

## Self-Check: PASSED

---
*Phase: 05-api-ergonomics*
*Completed: 2026-09-02*
