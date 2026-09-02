---
phase: 05-api-ergonomics
plan: "02"
subsystem: error-surfacing, option-validation
tags: [ergo, error-dispatch, bind-validation, invalid-input, function-exception]
status: complete

dependency_graph:
  requires:
    - "05-01: Window NULL guard (degenerate-frame NULL path locked)"
  provides:
    - "ThrowFromFfiError(fn, err) dispatch helper in src/include/error_dispatch.hpp"
    - "All 11 FFI !success throw sites wired to typed, function-named exceptions"
    - "Unknown MAP option keys rejected at bind in RegressionMapOptions + 10 test parsers"
    - "ergo01_clear_errors.test and ergo02_unknown_option.test regression gates"
  affects:
    - "05-03: rename pass inherits the function-named error messages (fn prefix is the post-rename SQL name)"

tech_stack:
  added: []
  patterns:
    - "ThrowFromFfiError dispatch: switch on AnofoxError.code — InternalException for numerical codes (SINGULAR_MATRIX/CONVERGENCE/INTERNAL/ALLOCATION); InvalidInputException by default"
    - "Unknown-key rejection: add else { throw InvalidInputException(\"unknown option '%s'; valid keys: ...\", key.c_str()) } as final branch in each ParseFromValue"

key_files:
  created:
    - src/include/error_dispatch.hpp
    - test/sql/ergo01_clear_errors.test
    - test/sql/ergo02_unknown_option.test
  modified:
    - src/table_functions/ols_fit.cpp
    - src/table_functions/ridge_fit.cpp
    - src/table_functions/elasticnet_fit.cpp
    - src/table_functions/wls_fit.cpp
    - src/table_functions/huber_fit.cpp
    - src/table_functions/ransac_fit.cpp
    - src/table_functions/theil_sen_fit.cpp
    - src/table_functions/predict.cpp
    - src/scalar_functions/vif.cpp
    - src/scalar_functions/aic_bic.cpp
    - src/aggregate_functions/ols_aggregate.cpp
    - src/include/map_options_parser.cpp
    - test/sql/regression/test_glm_fit_agg.test

decisions:
  - "FunctionException does not exist in the embedded DuckDB build (not in exception.hpp); InternalException (ExceptionType::INTERNAL) is used for the numerical failure codes instead. This deviates from CONTEXT.md taxonomy wording but matches the intent — non-user-data failures use a distinct exception class from InvalidInputException."
  - "GROUP BY aggregate finalize (OlsAggFinalize) now throws ThrowFromFfiError on genuine FFI failure post-guard; the degenerate-frame NULL paths (min_obs guard from Plan 01) remain NULL as locked."
  - "Pre-existing test test_glm_fit_agg.test TEST 11 used 'lower_bounds'/'upper_bounds' (plural, list-valued) which were silently ignored; fixed to 'lower_bound'/'upper_bound' (singular, scalar) matching the valid C++ parser keys."

metrics:
  duration_minutes: 18
  completed: "2026-09-01T21:42:35Z"
  tasks_completed: 3
  tasks_total: 3
  commits: 4

actuals:
  tokens: 28000
  tasks: 3
  commits: 4
---

# Phase 5 Plan 02: Error Surfacing + Early Validation (ERGO-01/02) Summary

ThrowFromFfiError dispatch helper wired at all 11 FFI error sites; unknown MAP
option keys rejected at bind time across regression and all 10 test-family parsers.

## What Was Built

### Task 1: ThrowFromFfiError dispatch helper (ERGO-01)

Created `src/include/error_dispatch.hpp` with `ThrowFromFfiError(fn_name, err)`:
- Switches on `AnofoxError.code`
- `SINGULAR_MATRIX`, `CONVERGENCE_FAILURE`, `INTERNAL`, `ALLOCATION_FAILURE` → `InternalException`
- All other codes (InsufficientData, DimensionMismatch, InvalidInput, NoValidData, etc.) → `InvalidInputException`
- Message format: `"<fn_name>: <error.message>"` — always names the function

Wired at all 11 FFI error sites:

| Site | File | Old Exception | New Call |
|------|------|---------------|----------|
| ols_fit | table_functions/ols_fit.cpp:178 | InvalidInputException("OLS fit failed: ") | ThrowFromFfiError("ols_fit", error) |
| ridge_fit | table_functions/ridge_fit.cpp:187 | InvalidInputException("Ridge fit failed: ") | ThrowFromFfiError("ridge_fit", error) |
| elasticnet_fit | table_functions/elasticnet_fit.cpp:174 | InvalidInputException("Elastic Net fit failed: ") | ThrowFromFfiError("elasticnet_fit", error) |
| wls_fit | table_functions/wls_fit.cpp:187 | InvalidInputException("WLS fit failed: ") | ThrowFromFfiError("wls_fit", error) |
| huber_fit | table_functions/huber_fit.cpp:181 | InvalidInputException("Huber fit failed: ") | ThrowFromFfiError("huber_fit", error) |
| ransac_fit | table_functions/ransac_fit.cpp:211 | InvalidInputException("RANSAC fit failed: ") | ThrowFromFfiError("ransac_fit", error) |
| theil_sen_fit | table_functions/theil_sen_fit.cpp:189 | InvalidInputException("Theil-Sen fit failed: ") | ThrowFromFfiError("theil_sen_fit", error) |
| predict | table_functions/predict.cpp:90 | InvalidInputException("Predict failed: ") | ThrowFromFfiError("predict", error) |
| vif | scalar_functions/vif.cpp:71 | InvalidInputException("VIF computation failed: ") | ThrowFromFfiError("vif", error) |
| aic | scalar_functions/aic_bic.cpp:54 | InvalidInputException("AIC computation failed: ") | ThrowFromFfiError("aic", error) |
| bic | scalar_functions/aic_bic.cpp:104 | InvalidInputException("BIC computation failed: ") | ThrowFromFfiError("bic", error) |
| ols_fit_agg (GROUP BY finalize) | aggregate_functions/ols_aggregate.cpp:307 | FlatVector::SetNull (silent NULL) | ThrowFromFfiError("ols_fit_agg", error) |

### Task 2: Unknown MAP key rejection at bind (ERGO-02)

In `src/include/map_options_parser.cpp`:
- Replaced `// Unknown keys are silently ignored for forward compatibility` in `RegressionMapOptions::ParseFromValue` with `throw InvalidInputException("unknown option '%s'; valid keys: ...", key.c_str())`
- The `"intercept"` alias branch (`if (key == "intercept" || key == "fit_intercept")`) at line 641 is preserved intact — 315 test files depend on it

Applied unknown-key rejection to 10 test-option parsers in the same file:
TTestMapOptions, MannWhitneyMapOptions, WilcoxonMapOptions, BrunnerMunzelMapOptions,
CorrelationMapOptions, KendallMapOptions, ChiSquareMapOptions, FisherExactMapOptions,
EnergyDistanceMapOptions, MmdMapOptions, TostMapOptions, YuenMapOptions, PermutationMapOptions.

### Task 3: Full-suite reconciliation

- `make test`: 102/102 tests pass (1 skipped: `require quack`)
- `cargo test --workspace`: 6/6 tests pass
- Pre-existing test `test_glm_fit_agg.test TEST 11` fixed: `'lower_bounds'` (plural, silently ignored) → `'lower_bound'` (singular, valid)

## Deviations from Plan

### [Rule 1 - Bug] FunctionException does not exist in this DuckDB build

- **Found during:** Task 1 build
- **Issue:** CONTEXT.md specifies "FunctionException" for numerical failures, but `class FunctionException` does not exist in the embedded DuckDB `exception.hpp`. The compiler reported: `'FunctionException' was not declared in this scope; did you mean 'ConnectionException'?`
- **Fix:** Used `InternalException` (ExceptionType::INTERNAL) instead — it is the closest available type and already used in this codebase (`macros/fit_predict_macros.cpp`) for non-user-caused failures. The intent is preserved: numerical failures throw a distinct exception class from `InvalidInputException`.
- **Files modified:** `src/include/error_dispatch.hpp` — comment updated to document the substitution
- **Commit:** 33a638f

### [Rule 1 - Bug] Test case used wrong predict signature

- **Found during:** Task 1 test run
- **Issue:** `ergo01_clear_errors.test` initially called `predict([[1.0, 2.0, 3.0]], [1.0], 0.0)` expecting a dimension-mismatch error, but this is valid (1 feature, 1 coefficient — they match). The test unexpectedly succeeded.
- **Fix:** Changed to `predict([[1.0, 2.0, 3.0]], [1.0, 2.0, 3.0], 0.0)` — 1 feature column but 3 coefficients, which triggers the actual mismatch.
- **Commit:** 33a638f (same commit)

### [Rule 1 - Bug] Pre-existing test used silently-ignored plural option key

- **Found during:** Task 3 full suite run
- **Issue:** `test_glm_fit_agg.test TEST 11` used `{'lower_bounds': [0.0, 0.0], 'upper_bounds': [10.0, 10.0]}` — plural keys with list values. These were silently ignored before this plan (the parser only accepts `lower_bound`/`upper_bound` as scalar doubles). Now that unknown-key rejection is active, the test properly fails.
- **Fix:** Updated to `{'lower_bound': 0.0, 'upper_bound': 10.0}` — valid scalar keys. Test intent (verify BLS accepts bound options) unchanged.
- **Files modified:** `test/sql/regression/test_glm_fit_agg.test`
- **Commit:** 1492032

## Known Stubs

None. All error surfaces are wired to real behavior.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The changes are purely exception-routing and validation additions within existing paths.

## Self-Check: PASSED

- `src/include/error_dispatch.hpp` exists: CONFIRMED
- `grep -rn 'fit failed:' src/` returns 0 matches: CONFIRMED
- All 11 ThrowFromFfiError sites wired: CONFIRMED (12 matches in grep output including 2 for aic/bic)
- `grep -n 'silently ignored' src/include/map_options_parser.cpp` returns 0 matches: CONFIRMED
- `ergo01_clear_errors.test` passes (10 assertions): CONFIRMED
- `ergo02_unknown_option.test` passes (8 assertions): CONFIRMED
- `make test`: 102/102 passed: CONFIRMED
- `cargo test --workspace`: 6/6 passed: CONFIRMED
- Commits exist: 33a638f, 2a2a546, 1492032: CONFIRMED
