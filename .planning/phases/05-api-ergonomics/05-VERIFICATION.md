---
phase: 05-api-ergonomics
verified: 2026-09-02T00:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 5: API Ergonomics Verification Report

**Phase Goal:** Fit/predict/test functions fail fast with clear, actionable messages for invalid input, and signatures, option-map keys, and return-struct field names follow one documented convention consistent across model families, with breaking renames reflected in the test suite (test/sql + cargo test green).

**Verified:** 2026-09-02
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Degenerate rolling-window frames (n ≤ n_features+1) return NULL instead of an INTERNAL crash or wrong non-NULL result (ERGO-01 SC-1) | VERIFIED | `ols_aggregate.cpp:271-274` — scaling `min_obs = state.fit_intercept ? state.n_features+1 : state.n_features`; `y_values.size() <= min_obs` → SetNull. `ergo01_window_null.test` asserts 2 NULL rows then 5 non-NULL rows in Case A, 3 NULL + 4 non-NULL in Case B. No `y_values.size() < 2` guard survives in ols_aggregate.cpp. |
| 2  | The scaling min_obs guard is present in all 8 fit_predict finalize paths (ERGO-01 SC-1) | VERIFIED | `grep -n 'y_values.size() <= min_obs'` returns at least one match in each of: `ols_aggregate.cpp`, `ols_fit_predict.cpp`, `huber_fit_predict.cpp`, `ransac_fit_predict.cpp`, `theil_sen_fit_predict.cpp`, `ridge_fit_predict.cpp`, `wls_fit_predict.cpp`, `rls_fit_predict.cpp`, `elasticnet_fit_predict.cpp` — all 8 confirmed at their exact guard lines. |
| 3  | Invalid input surfaces a typed, function-named DuckDB exception carrying real Rust error detail, not a generic "fit failed" wrapper (ERGO-01 SC-1) | VERIFIED | `src/include/error_dispatch.hpp` exists, is `#pragma once`, defines `ThrowFromFfiError` inside `namespace duckdb`, switches on `err.code` throwing `InternalException` for 4 numerical codes and `InvalidInputException` by default. `grep -rn 'fit failed:' src/table_functions/ src/scalar_functions/` returns 0 matches. `ergo01_clear_errors.test` asserts messages prefix with function name (`ols_fit: Dimension mismatch`, `ols_fit: Insufficient data`). **Documented deviation:** CONTEXT.md specified `FunctionException` for numerical codes but that class does not exist in the embedded DuckDB build; `InternalException` (ExceptionType::INTERNAL) is used instead. The intent is preserved — two distinct exception classes for user-data vs numerical errors. This is fully documented in the header comment and SUMMARY-02. |
| 4  | `ThrowFromFfiError` is wired at all 11 FFI error sites (10 table/scalar + 1 aggregate GROUP BY finalize) (ERGO-01 SC-1) | VERIFIED | `grep -rn 'ThrowFromFfiError' src/` returns 12 lines: the definition in `error_dispatch.hpp` plus 1 call each in `ols_fit.cpp`, `ridge_fit.cpp`, `elasticnet_fit.cpp`, `wls_fit.cpp`, `huber_fit.cpp`, `ransac_fit.cpp`, `theil_sen_fit.cpp`, `predict.cpp`, `vif.cpp`, `ols_aggregate.cpp`, and 2 calls in `aic_bic.cpp` (one for AIC, one for BIC). All 11 files confirmed. |
| 5  | Unknown MAP option keys are rejected at bind time with a message naming the offending key and listing valid keys (ERGO-02 SC-2) | VERIFIED | `grep -n 'silently ignored' src/include/map_options_parser.cpp` returns 0 matches. `grep -n 'unknown option' src/include/map_options_parser.cpp` returns 14 throw sites covering `RegressionMapOptions::ParseFromValue` (line 799) and 10+ test-family option parsers. `ergo02_unknown_option.test` asserts `{'unknow_key': 1}` raises "unknown option 'unknow_key'" and `{'compute_inferense': true}` raises "unknown option 'compute_inferense'". |
| 6  | The `intercept` alias and every currently-accepted key still bind successfully (ERGO-02 SC-2, no regression) | VERIFIED | `grep -n '"intercept"' src/include/map_options_parser.cpp` shows line 641: `if (key == "intercept" \|\| key == "fit_intercept")` — alias branch is before the unknown-key throw and untouched. `ergo02_unknown_option.test` asserts `{'intercept': true}`, `{'fit_intercept': true}`, and `{'compute_inference': true}` all bind and return non-NULL results. |
| 7  | Every SQL function is registered under one unprefixed `{model}_{verb}[_agg]` name; `anofox_stats_` prefix fully dropped; all alias blocks deleted (ERGO-03 SC-3) | VERIFIED | `grep -rn '"anofox_stats_' src/ \| grep -v '#include' \| grep -v '//'` returns 0 matches. `grep -rn 'alias_of = "anofox_stats_' src/` returns 0 matches. Confirmed: commits `819c5bf` (feat — rename registrations) and `9efbfdc` (fix — test suite reconciliation) landed the rename across all families. |
| 8  | `theilsen` → `theil_sen` rename is applied everywhere (registrations, macros, test files) (ERGO-03 SC-3) | VERIFIED | `grep -rn '\btheilsen\b' src/ test/sql/` returns 0 matches. `grep -n 'theil_sen' src/macros/fit_predict_macros.cpp` confirms `theil_sen_fit_predict_by` in macro registration. `ergo03_naming.test` positively asserts `theil_sen_fit` and `theil_sen_fit_agg` resolve, and the comment on line 22 uses the corrected wording "theil_sen rename verified". |
| 9  | The naming convention is documented in `docs/API_CONVENTIONS.md` covering function names, snake_case option keys, the standard return-field set, and the GLM/AFT `z_values` per-family exception; the test suite (make test 103 + cargo 295) is green against the renamed API (ERGO-03 SC-3, SC-4) | VERIFIED | `docs/API_CONVENTIONS.md` exists (1274 lines). Contains: (a) `{model}_{verb}[_{suffix}]` naming pattern, (b) snake_case option keys with `intercept`/`lambda` accepted aliases documented, (c) standard return-field set including `r_squared`, `t_values`, `p_values`, (d) per-family exceptions — GLM `z_values`, AFT `z_values`, ALM omits `r_squared`, (e) v0.3.0 breaking-changes section listing dropped prefix, `theilsen`→`theil_sen`, no deprecated aliases. `grep -qi 'z_values'` and `grep -qiE 'theil_sen\|no deprecated alias\|breaking'` both pass. SUMMARY-03 self-check reports `make test: 103 passed, 1 skipped, 0 failed` and `cargo test --workspace: 295 passed, 0 failed`. `grep -rn '\.r2\b' test/sql/` and `grep -rn 'anofox_stats_' test/sql/ \| grep -v ergo03_naming` both return 0 matches. |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `test/sql/ergo01_window_null.test` | Degenerate-frame NULL regression test | VERIFIED | Exists. Header `# name: test/sql/ergo01_window_null.test`, `require anofox_statistics`. 3 cases: 1-feature (Case A), 2-feature (Case B), ols_fit_predict window function (Case C). 22 assertions. |
| `src/aggregate_functions/ols_aggregate.cpp` | Scaling min_obs guard replacing fixed < 2 | VERIFIED | Line 271-274: `min_obs = state.fit_intercept ? state.n_features+1 : state.n_features; if (state.y_values.size() <= min_obs) { FlatVector::SetNull(result, result_idx, true); continue; }`. No `y_values.size() < 2` found. |
| `src/include/error_dispatch.hpp` | ThrowFromFfiError dispatch helper | VERIFIED | Exists, 45 lines, `#pragma once`, `namespace duckdb`, switches on `AnofoxError.code`, throws `InternalException` for 4 numerical codes, `InvalidInputException("%s", msg.c_str())` for all others. |
| `test/sql/ergo01_clear_errors.test` | Typed error assertions for invalid input | VERIFIED | Exists. Tests dimension-mismatch (ols_fit, ridge_fit, predict), insufficient-rows (ols_fit, wls_fit, elasticnet_fit), and aggregate GROUP BY NULL path. All assert function-prefixed messages. |
| `test/sql/ergo02_unknown_option.test` | Unknown-key rejection + alias positive controls | VERIFIED | Exists. Two unknown-key `statement error` blocks + TTestMapOptions unknown key + positive controls for `intercept`, `fit_intercept`, `compute_inference`, `alpha`. |
| `docs/API_CONVENTIONS.md` | Written naming convention for Phase 6 validation | VERIFIED | Exists. Documents all 6 required elements: naming pattern, verbs/suffixes, snake_case option keys + aliases, standard return-field set, per-family exceptions (GLM/AFT z_values, ALM), v0.3.0 breaking changes. |
| `test/sql/ergo03_naming.test` | Positive + negative smoke test for the rename | VERIFIED | Exists. Positive: `ols_fit`, `ols_fit_agg`, `theil_sen_fit`, `theil_sen_fit_agg`, `poisson_fit_agg`, `t_test_agg`, `vif`, `bls_fit_agg`, `nnls_fit_agg` all resolve. Negative: `anofox_stats_ols_fit`, `anofox_stats_theil_sen_fit`, `anofox_stats_t_test_agg` all fail with `statement error`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `OVER (ROWS BETWEEN N PRECEDING...)` rolling frame | `FlatVector::SetNull(result, result_idx, true)` in OlsAggFinalize | `y_values.size() <= min_obs` scaling guard at `ols_aggregate.cpp:271-274` | WIRED | Guard present; NULL path confirmed at correct location before the FFI call. |
| Rust `AnofoxError.code` | `InternalException` or `InvalidInputException` | `ThrowFromFfiError` in `error_dispatch.hpp:30-42`, called at all 11 `!success` sites | WIRED | 12 grep matches (1 definition + 11 call sites); `ols_fit.cpp:178` includes `"../include/error_dispatch.hpp"` and calls `ThrowFromFfiError("ols_fit", error)`. |
| MAP literal unknown key | `throw InvalidInputException("unknown option '%s'; valid keys: ...", key.c_str())` | `RegressionMapOptions::ParseFromValue` final else at `map_options_parser.cpp:799` | WIRED | `silently ignored` comment absent; `unknown option` throw present at line 799 and 13 additional parsers. |
| Registration string `"anofox_stats_ols_fit_agg"` | `"ols_fit_agg"` (no prefix) | Prefix drop + alias block deletion across `src/{aggregate,table,window,scalar}_functions/*.cpp` | WIRED | `grep -rn '"anofox_stats_' src/ \| grep -v '#include' \| grep -v '//'` returns 0 matches; `alias_of = "anofox_stats_"` returns 0 matches. |

---

### Data-Flow Trace (Level 4)

Not applicable to this phase — no UI rendering or display layer. Error messages, exception types, and NULL return values are tested by the SQL test suite which exercises the full data path end-to-end.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `ols_fit_agg` uses `y_values.size() <= min_obs` guard | `grep -n 'y_values.size() <= min_obs' src/aggregate_functions/ols_aggregate.cpp` | Line 272 match | PASS |
| No `y_values.size() < 2` weak guard survives in ols_aggregate.cpp | `grep -n 'y_values.size() < 2' src/aggregate_functions/ols_aggregate.cpp` | 0 matches | PASS |
| All 8 window/aggregate fit_predict paths have scaling guard | `grep -rn 'y_values.size() <= min_obs' src/aggregate_functions/ols_aggregate.cpp src/window_functions/*fit_predict*.cpp` | 9 matches across 9 files | PASS |
| No generic "fit failed" wrappers remain | `grep -rn 'fit failed:' src/table_functions/ src/scalar_functions/` (excluding comments) | 0 matches | PASS |
| No `silently ignored` comment in map_options_parser.cpp | `grep -n 'silently ignored' src/include/map_options_parser.cpp` | 0 matches | PASS |
| No `anofox_stats_` registration strings in src/ | `grep -rn '"anofox_stats_' src/ \| grep -v '#include' \| grep -v '//'` | 0 matches | PASS |
| No bare `theilsen` anywhere | `grep -rn '\btheilsen\b' src/ test/sql/` | 0 matches | PASS |
| No `.r2` field references in tests | `grep -rn '\.r2\b' test/sql/` | 0 matches | PASS |
| `docs/API_CONVENTIONS.md` contains z_values and breaking-change content | `grep -qi 'z_values' docs/API_CONVENTIONS.md && grep -qiE 'theil_sen\|no deprecated alias\|breaking' docs/API_CONVENTIONS.md` | Both pass | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ERGO-01 | 05-01, 05-02 | Fit/predict/test functions return clear, actionable error messages for invalid input instead of panics or opaque errors | SATISFIED | Scaling min_obs NULL guard in all 8 finalize paths (Plan 01); `ThrowFromFfiError` at all 11 FFI error sites replacing generic wrappers (Plan 02); `ergo01_window_null.test` + `ergo01_clear_errors.test` green |
| ERGO-02 | 05-02 | Inputs validated early with specific messages naming offending argument and expected shape | SATISFIED | Unknown MAP option keys rejected at bind in `RegressionMapOptions::ParseFromValue` and 10+ test-family parsers; `ergo02_unknown_option.test` asserts key-naming and valid-key-listing messages at bind time |
| ERGO-03 | 05-03 | Function signatures, option-map keys, and return-struct field names follow one documented convention consistent across model families | SATISFIED | `anofox_stats_` prefix dropped from all registrations; alias blocks deleted; `theilsen`→`theil_sen`; `.r2`→`.r_squared`; `docs/API_CONVENTIONS.md` written; `ergo03_naming.test` proves rename; `make test` 103 passed + `cargo test --workspace` 295 passed |

All three requirement IDs are accounted for and satisfied. Traceability table in REQUIREMENTS.md marks all three "Complete" for Phase 5.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/include/error_dispatch.hpp` | 15 | Comment documents `FunctionException` was the intended class per CONTEXT.md but `InternalException` was used instead | INFO | Not a stub — this is a documented deviation. `FunctionException` does not exist in the embedded DuckDB build (`exception.hpp`). `InternalException` achieves the same intent (distinct class for numerical vs user-data errors). The deviation is documented in both the header comment and SUMMARY-02. CONTEXT.md taxonomy intent is preserved. |

No TBD/FIXME/XXX markers found in phase-modified files. No placeholder or stub patterns found.

---

### Human Verification Required

Two items from `05-VALIDATION.md`'s Manual-Only Verifications section are noted for completeness:

**1. Error message quality**

**Test:** Trigger a dimension-mismatch error (e.g. `SELECT ols_fit([1.0,2.0,3.0,4.0], [[1.0,2.0,3.0]])`) and an insufficient-rows error; read the messages.
**Expected:** Messages name the function, describe the problem clearly, and would be actionable to a user encountering them for the first time.
**Why human:** Message quality (clarity, actionability) is a judgment call beyond string-match assertions.

**2. API_CONVENTIONS.md accuracy against shipped convention**

**Test:** Spot-check the written convention in `docs/API_CONVENTIONS.md` against 3 functions per family (e.g. OLS, GLM Poisson, t-test) by calling them and inspecting result struct fields.
**Expected:** Every field name and exception description in the doc matches what the extension actually returns.
**Why human:** Doc/API consistency is validated end-to-end in Phase 6's doc-SQL harness; the manual check here is a pre-Phase-6 sanity pass.

These are judgment-quality checks. All automated success criteria for ERGO-01/02/03 are confirmed.

---

## Gaps Summary

No gaps. All 9 must-have truths are VERIFIED against the actual codebase.

The only notable deviation from the plan — `InternalException` substituted for the non-existent `FunctionException` — is fully documented in `error_dispatch.hpp` and SUMMARY-02, preserves the taxonomy's intent (two distinct exception classes), and does not constitute a gap in goal achievement.

---

## Overall Verdict

**Phase 5: API Ergonomics — PASSED**

All four ROADMAP success criteria are satisfied:

1. Invalid input returns clear, actionable typed exceptions via `ThrowFromFfiError` (ERGO-01).
2. Unknown option keys rejected at bind with key-naming message (ERGO-02); degenerate frames return NULL not errors.
3. One documented naming convention in `docs/API_CONVENTIONS.md` covering all families with z_values exception recorded (ERGO-03).
4. Breaking renames reflected in the test suite: `make test` 103 passed, `cargo test --workspace` 295 passed, `ergo03_naming.test` proves rename is observable (ERGO-03 + SC-4).

---

_Verified: 2026-09-02_
_Verifier: Claude (gsd-verifier)_
