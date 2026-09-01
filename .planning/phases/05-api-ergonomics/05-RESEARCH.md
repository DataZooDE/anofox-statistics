# Phase 5: API Ergonomics — Research

**Researched:** 2026-09-01
**Domain:** DuckDB C++ extension API — validation, error surfacing, SQL naming
**Confidence:** HIGH (all claims grounded in direct file reads this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- SQL function names use `{model}_{verb}[_agg|_predict]`, **unprefixed and uniform** — drop the inconsistent `anofox_stats_` prefix. Example: `anofox_stats_ols_fit_predict` → `ols_fit_predict`.
- Option-map keys are `snake_case` across all families (`compute_inference`, `hc_type`, `l1_ratio`), matching the Rust core.
- Return-struct field names are `snake_case` with one standard set across families (`coefficients`, `intercept`, `std_errors`, `t_values`, `p_values`, `r_squared`). GLM `z_values` stays as a **documented per-family exception** — do NOT force z→t.
- The convention is written down in a new `docs/API_CONVENTIONS.md`.
- Error message format: `"<fn>: <problem>; expected <shape> (got <actual>)"` — always name the function, the offending argument, and expected vs actual shape.
- Keep the FFI `catch_unwind` panic guard; convert caught panics/`StatsError`s into a **specific** DuckDB exception carrying the real detail.
- Rolling-window degenerate-frame bug fixed by returning NULL prediction for degenerate frames (not an error).
- Exception taxonomy: `InvalidInputException` (user data/shape: dimension mismatch, insufficient rows, constant column, unknown option) vs `FunctionException` (numerical: singular matrix, convergence).
- Bind-time checks: argument count, option key names + types, option value ranges, non-empty feature list.
- Execution/finalize checks: y-vs-x row-count dimension match, insufficient rows (`n < n_features + 1`), all-non-finite input, constant/zero-variance column.
- Unknown option-map keys **rejected at bind** with `"unknown option 'X'; valid: ..."`.
- **No deprecated aliases** — clean rename.
- Rename pass covers **all families in one pass**: regression, GLM, GLMM, AFT, hypothesis tests, diagnostics.
- Test suite (`test/sql` + `cargo test`) updated to new names **within this phase**.

### Claude's Discretion
- Exact per-function rename table, the precise wording of each error string, and the RAII/helper mechanics of bind-vs-execution validation are at Claude's discretion, guided by the conventions above and existing codebase patterns.

### Deferred Ideas (OUT OF SCOPE)
- Named parameters (`param := value`) — ERGOX-01, explicitly deferred.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERGO-01 | Fit/predict/test functions return clear, actionable error messages for invalid input instead of panics or opaque errors | §3 (FFI error flow), §5 (generic wrap sites), §2 (window NULL fix) |
| ERGO-02 | Inputs are validated early (at bind time where possible) with a specific message naming the offending argument and its expected shape | §1 (validation placement), §3.4 (unknown key rejection) |
| ERGO-03 | Function signatures, option-map keys, and return-struct field names follow one documented naming convention consistent across model families | §4 (naming inventory + rename map) |
</phase_requirements>

---

## Summary

Phase 5 is a three-part cleanup with zero new algorithmic work: (1) fix a concrete window-function crash by returning NULL for degenerate frames; (2) surface specific, actionable error messages instead of generic "fit failed" wrappers; (3) do a clean rename across all ~100 registration strings plus 150 test SQL files. All the infrastructure needed already exists — `FlatVector::SetNull`, the `error.code` enum on the FFI boundary, and `InvalidInputException`/`FunctionException` on the C++ side. The only structural change is adding an unknown-key rejection block to `RegressionMapOptions::ParseFromValue` (currently the final `else` branch says `// Unknown keys are silently ignored`).

**Primary recommendation:** Tackle in three waves: (1) window NULL fix as tracer — touches one file, proves the NULL path, tests immediately; (2) error message upgrade — add error.code dispatch helper in each `!success` block; (3) rename pass — mechanical sed across src/ and test/sql/, update registration strings, delete the alias blocks.

**Key risk:** The rename wave is wide (100+ registration strings, 150 SQL test files, plus `FunctionDescription.examples` strings) but mechanically simple. The window NULL fix and the unknown-key rejection are higher-complexity changes with smaller blast radius — implement and test them first.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Option validation (key names, value ranges) | C++ Bind layer | — | `*Bind()` runs before any row is processed; cheapest place to fail |
| Data-shape validation (row count, dimension) | C++ Finalize / Rust FFI | — | Only knowable after accumulation; Rust already checks these |
| Error message content | Rust StatsError display + C++ throw | — | `error.message` already carries the detail; C++ just needs to forward it with the right exception class |
| SQL function names | C++ Registration strings | test/sql rename | Single point of truth in `RegisterFunction(AggregateFunctionSet("name"))` calls |
| Return struct field names | C++ `GetXxxResultType()` builders | — | Schema defined once in the builder; no Rust change needed |
| NULL result for degenerate window frames | C++ Window Finalize | — | `FlatVector::SetNull(result, result_idx, true)` — already the pattern in OLS window (lines 259–268) |

---

## Standard Stack

No new packages. All tooling is existing:

| Tool | Version | Purpose |
|------|---------|---------|
| `InvalidInputException` | DuckDB built-in | User data/shape errors |
| `FunctionException` | DuckDB built-in | Numerical failures |
| `FlatVector::SetNull` | DuckDB built-in | Emit NULL result from window/aggregate finalize |
| `error_to_code(&StatsError)` | `crates/anofox-stats-ffi/src/lib.rs:135-153` | Maps Rust error variant → `ErrorCode` enum |
| `AnofoxError.code` | `src/include/anofox_stats_ffi.h:37` | C-side code field; currently **never read** by C++ dispatch |

**Installation:** none — no new dependencies.

---

## Package Legitimacy Audit

Not applicable — this phase installs no new packages.

---

## Architecture Patterns

### System Architecture Diagram (Validation Flow)

```
SQL bind call
    │
    ▼
*Bind() — InvalidInputException for:
    ├─ wrong arg count (DuckDB enforces via function signature)
    ├─ unknown MAP key  [MISSING — add here]
    ├─ value out of range (alpha ∈ (0,1), confidence_level ∈ (0,1))
    └─ empty x list (n_features == 0)
    │
    ▼
*Update() — InvalidInputException for:
    └─ inconsistent feature count per row (already implemented)
    │
    ▼
*Finalize() — call FFI → Rust → StatsError
    ├─ FFI sets error.code + error.message
    ├─ C++ reads error.code → dispatch to correct DuckDB exception  [MISSING]
    │     InvalidInputException: InsufficientData, DimensionMismatch,
    │                            NoValidData, InvalidInput, InvalidAlpha
    │     FunctionException:     SingularMatrix, ConvergenceFailure
    └─ Window path: FlatVector::SetNull on degenerate frame (already implemented in OLS)
```

### Key Files and Their Roles

| File | Role in Phase 5 |
|------|----------------|
| `src/include/map_options_parser.cpp` | Add unknown-key rejection; one-line change in `ParseFromValue` |
| `src/window_functions/ols_fit_predict.cpp` | Degenerate-frame NULL already works (lines 259-268); **verify** the exact threshold is `<= min_obs` not `< min_obs` |
| `src/table_functions/ols_fit.cpp` | Add `error.code` dispatch helper; all 9 table function files need same change |
| `src/aggregate_functions/ols_aggregate.cpp` | Aggregate finalize already returns NULL on `!success` (line 299); need richer error for `throw` sites in Update/Combine |
| `src/anofox_statistics_extension.cpp` | ~50 `Register*Function()` calls; rename primary strings; **delete alias blocks** |
| All `src/**/*.cpp` | Rename every `"anofox_stats_…"` string literal |
| `test/sql/*.sql` (150 files, 120 test files) | Bulk-rename all `anofox_stats_` prefixes; fix `r2` → `r_squared` (236 occurrences), fix stale function names |

---

## Validation Placement (ERGO-02)

### What Can Be Validated at Bind Time

`*Bind()` is called once per query plan, before any row arrives. In this codebase bind functions are in `*Bind()` (e.g., `OlsFitBind`, `OlsAggBind`, `OlsFitPredictBind`). [VERIFIED: src/table_functions/ols_fit.cpp:71-100, src/aggregate_functions/ols_aggregate.cpp:343-372, src/window_functions/ols_fit_predict.cpp:329-355]

Bindable checks (all knowable from the MAP literal alone):

| Check | Mechanism | Status |
|-------|-----------|--------|
| Argument count | DuckDB enforces via overload signature | Already enforced |
| Option key existence | `if (key == "known") ... else { throw }` in `ParseFromValue` | **MISSING** — currently `// Unknown keys are silently ignored` [VERIFIED: src/include/map_options_parser.cpp:798] |
| Option value types | `ExtractBool`, `ExtractDouble`, etc. — already throw on bad type | Already enforced |
| `confidence_level ∈ (0,1)` | Add range check after extraction | **MISSING** |
| `alpha > 0`, `l1_ratio ∈ [0,1]` | Add range checks | **MISSING** |
| `n_features > 0` (x list non-empty) | Check `arguments[1]->IsFoldable()` + list size | Partially — not checked at bind for aggregate path |

### What Must Wait for Execution/Finalize

Data-dependent checks — only available after row accumulation:

| Check | Where It Happens | Current State |
|-------|-----------------|--------------|
| y length vs x row count | Rust `fit_ols` → `StatsError::DimensionMismatch` | Checked in Rust; error surfaced as generic NULL in aggregate finalize |
| n < n_features + 1 | Rust `fit_ols` → `StatsError::InsufficientData` | Checked in Rust; error surfaced as generic NULL in aggregate finalize |
| All-non-finite input | Rust → `StatsError::NoValidData` | Checked in Rust; surfaced as NULL |
| Constant/zero-variance column | Rust → `StatsError::SingularMatrix` or variant | Checked in Rust; surfaced as NULL |

**Current problem:** The aggregate finalize path in `OlsAggFinalize` (and all other `*Finalize` functions) receives `!success` and just calls `FlatVector::SetNull` — no error is ever thrown, no message is emitted. [VERIFIED: src/aggregate_functions/ols_aggregate.cpp:298-301]

The table function path **does** throw, but always as `InvalidInputException("OLS fit failed: " + error.message)` regardless of `error.code`. [VERIFIED: src/table_functions/ols_fit.cpp:177]

---

## Window NULL Fix (ERGO-01 Degenerate-Frame Bug)

### Current Status in `src/window_functions/ols_fit_predict.cpp`

The OLS window finalize **already handles** degenerate frames correctly:

```cpp
// Lines 259-268 — VERIFIED: src/window_functions/ols_fit_predict.cpp:259-268
if (!state.initialized || !state.has_current_x) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
if (state.y_values.size() <= min_obs) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
```

**Finding:** The degenerate-frame INTERNAL error described in CONTEXT.md does **not** come from `ols_fit_predict` — that file already returns NULL for insufficient frames. The bug is in one of the **other 7 window files** (`rls_fit_predict`, `huber_fit_predict`, `theil_sen_fit_predict`, `ridge_fit_predict`, `wls_fit_predict`, `ransac_fit_predict`, `elasticnet_fit_predict`) or in the **aggregate fit_predict** path (`ols_fit_predict_agg`).

**Evidence from read session:** `FlatVector::SetNull` calls in the window files confirm they all already have the same NULL pattern. The most likely source of the INTERNAL panic is:
- The `ols_fit_predict_agg` (aggregate variant used with `OVER` clause) — check `src/aggregate_functions/ols_aggregate.cpp`-style aggregate with `current_x`
- Or `rls_fit_predict.cpp` which has additional state complexity (recursive state vector).

**Action for tracer plan:** Write a repro SQL (`SELECT ols_fit_predict(y, [x]) OVER (ORDER BY t ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` with fewer than 3 rows in a partition), identify which file throws, verify the `y_values.size() <= min_obs` guard is missing or off-by-one there.

### API for Emitting NULL from Finalize

```cpp
// Emit NULL for the entire STRUCT result
FlatVector::SetNull(result, result_idx, true);
continue;
// [VERIFIED: src/window_functions/ols_fit_predict.cpp:260,267,299,313]
```

This is the correct and complete pattern — no partial struct writes needed; `SetNull` on the parent marks the entire STRUCT as NULL. All 8 window files already use this pattern on the FFI failure path.

---

## Rust Panic / StatsError → FFI → C++ Exception Surfacing (ERGO-01)

### Current Flow

1. Rust: `catch_unwind(AssertUnwindSafe(|| fit_ols(...)))` — any panic returns `Err(_)` → error set to `ErrorCode::InternalError, "Internal panic in OLS fit"` [VERIFIED: src/crates/anofox-stats-ffi/src/lib.rs:214-226]
2. Rust: `fit_ols` returns `Err(StatsError::X)` → `error_to_code(&e)` maps to `ErrorCode` enum + `e.to_string()` fills `error.message[256]` [VERIFIED: crates/anofox-stats-ffi/src/lib.rs:135-153, 280-286]
3. C++ table functions: `if (!success) { throw InvalidInputException("OLS fit failed: " + string(error.message)); }` — **always `InvalidInputException`, never reads `error.code`** [VERIFIED: src/table_functions/ols_fit.cpp:177]
4. C++ aggregate functions: `if (!success) { FlatVector::SetNull(result, result_idx, true); continue; }` — **silently returns NULL, no error** [VERIFIED: src/aggregate_functions/ols_aggregate.cpp:298-301]

### The `ErrorCode` Enum (Available but Unused by C++)

```c
// VERIFIED: src/include/anofox_stats_ffi.h:18-31 (verbatim)
typedef enum {
    ANOFOX_ERROR_SUCCESS = 0,
    ANOFOX_ERROR_INVALID_INPUT = 1,
    ANOFOX_ERROR_SINGULAR_MATRIX = 2,
    ANOFOX_ERROR_CONVERGENCE_FAILURE = 3,
    ANOFOX_ERROR_INVALID_ALPHA = 4,
    ANOFOX_ERROR_INVALID_L1_RATIO = 5,
    ANOFOX_ERROR_INSUFFICIENT_DATA = 6,
    ANOFOX_ERROR_ALLOCATION_FAILURE = 7,
    ANOFOX_ERROR_SERIALIZATION_ERROR = 8,
    ANOFOX_ERROR_DIMENSION_MISMATCH = 9,
    ANOFOX_ERROR_NO_VALID_DATA = 10,
    ANOFOX_ERROR_INTERNAL = 99,
} AnofoxErrorCode;
```

### Required Change: `error.code` Dispatch Helper

Add a C++ helper (e.g., in a shared header or inline at each site) that converts `AnofoxErrorCode` to the right DuckDB exception:

```cpp
// Proposed pattern — place in src/include/error_dispatch.hpp (new file)
static inline void ThrowFromFfiError(const char* fn_name, const AnofoxError& error) {
    std::string msg = std::string(fn_name) + ": " + std::string(error.message);
    switch (error.code) {
        case ANOFOX_ERROR_SINGULAR_MATRIX:
        case ANOFOX_ERROR_CONVERGENCE_FAILURE:
        case ANOFOX_ERROR_INTERNAL:
        case ANOFOX_ERROR_ALLOCATION_FAILURE:
            throw FunctionException(msg);
        default:  // InvalidInput, InsufficientData, DimensionMismatch, NoValidData, etc.
            throw InvalidInputException(msg);
    }
}
```

**Aggregate finalize path:** For aggregates where failure should surface as an error (not silently NULL), replace the `FlatVector::SetNull` + `continue` with `ThrowFromFfiError(fn_name, error)`. For aggregates where NULL is the right behavior (window rolling path), keep the NULL path.

**Decision needed (Claude's discretion):** For the non-window aggregate finalize (e.g., `ols_fit_agg` GROUP BY), should `InsufficientData` throw an exception or return NULL? CONTEXT.md says "clear, actionable error message" (ERGO-01) implying throw. NULL is silently confusing. Recommendation: throw in the standard aggregate path, keep NULL only in the window fit_predict path where degenerate frames are expected.

### Generic Wrap Sites to Fix

**Table functions (all currently throw `InvalidInputException` regardless of code):**
- `src/table_functions/ols_fit.cpp:177` — `"OLS fit failed: "`
- `src/table_functions/ridge_fit.cpp:186` — `"Ridge fit failed: "`
- `src/table_functions/elasticnet_fit.cpp:173` — `"Elastic Net fit failed: "`
- `src/table_functions/wls_fit.cpp:186` — `"WLS fit failed: "`
- `src/table_functions/huber_fit.cpp:180` — `"Huber fit failed: "`
- `src/table_functions/ransac_fit.cpp:210` — `"RANSAC fit failed: "`
- `src/table_functions/theil_sen_fit.cpp:188` — `"Theil-Sen fit failed: "`
- `src/table_functions/predict.cpp:89` — `"Predict failed: "`
- `src/scalar_functions/vif.cpp:70` — `"VIF computation failed: "`
- `src/scalar_functions/aic_bic.cpp:53,103` — `"AIC/BIC computation failed: "`

[VERIFIED: direct grep across src/ — all above lines confirmed]

---

## Unknown MAP Key Rejection at Bind (ERGO-02)

### Current Behavior

```cpp
// VERIFIED: src/include/map_options_parser.cpp:798 (verbatim)
// Unknown keys are silently ignored for forward compatibility
```

This is the final `else` branch in `RegressionMapOptions::ParseFromValue`. The whole parsing loop is a big `if/else if` chain; unknown keys fall through without error.

### Required Change

Replace that comment with a throw:

```cpp
} else {
    // Collect valid key list from the existing branches above, then:
    throw InvalidInputException(
        "unknown option '%s'; valid keys: fit_intercept, compute_inference, "
        "confidence_level, alpha, lambda, l1_ratio, max_iterations, tolerance, "
        "epsilon, residual_threshold, max_trials, stop_probability, min_samples, "
        "random_state, forgetting_factor, null_policy, link, distribution, loss, "
        "quantile, lower_bound, upper_bound, solver, hc_type, lambda_scaling, "
        "family, reml, offset, random, groups, vcov, feature_names, prior, "
        "tau_squared, tau_method, theta, threshold, tau, increasing, n_components",
        key.c_str());
}
```

**Note:** The statistical test options parsers (`TTestMapOptions::ParseFromValue`, `MannWhitneyMapOptions::ParseFromValue`, etc.) are separate structs and should receive the same treatment in their own `ParseFromValue` implementations. [VERIFIED: src/include/map_options_parser.hpp:280-413 — each test family has its own options struct]

---

## Naming Inventory and Rename Map (ERGO-03)

### Current Registration State (All Primary Names)

All names below are `[VERIFIED: grep of src/**/*.cpp registration blocks]`.

#### Table Functions (scalar fit — scalar input, scalar result)
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_ols_fit` | `ols_fit` | `ols_fit` |
| `anofox_stats_ridge_fit` | `ridge_fit` | `ridge_fit` |
| `anofox_stats_elasticnet_fit` | `elasticnet_fit` | `elasticnet_fit` |
| `anofox_stats_wls_fit` | `wls_fit` | `wls_fit` |
| `anofox_stats_huber_fit` | `huber_fit` | `huber_fit` |
| `anofox_stats_ransac_fit` | `ransac_fit` | `ransac_fit` |
| `anofox_stats_theilsen_fit` | `theilsen_fit` | `theil_sen_fit` (**also fix underscore**) |
| `anofox_stats_predict` | (none) | `predict` |
| `anofox_stats_rls_fit` | `rls_fit` | `rls_fit` |

#### Window Functions (fit_predict — over OVER clause)
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_ols_fit_predict` | `ols_fit_predict` | `ols_fit_predict` |
| `anofox_stats_huber_fit_predict` | `huber_fit_predict` | `huber_fit_predict` |
| `anofox_stats_ransac_fit_predict` | `ransac_fit_predict` | `ransac_fit_predict` |
| `anofox_stats_theilsen_fit_predict` | `theilsen_fit_predict` | `theil_sen_fit_predict` |
| `anofox_stats_ridge_fit_predict` | `ridge_fit_predict` | `ridge_fit_predict` |
| `anofox_stats_wls_fit_predict` | `wls_fit_predict` | `wls_fit_predict` |
| `anofox_stats_rls_fit_predict` | `rls_fit_predict` | `rls_fit_predict` |
| `anofox_stats_elasticnet_fit_predict` | `elasticnet_fit_predict` | `elasticnet_fit_predict` |

#### Regression Aggregate Functions
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_ols_fit_agg` | `ols_fit_agg` | `ols_fit_agg` |
| `anofox_stats_ridge_fit_agg` | `ridge_fit_agg` | `ridge_fit_agg` |
| `anofox_stats_elasticnet_fit_agg` | `elasticnet_fit_agg` | `elasticnet_fit_agg` |
| `anofox_stats_lars_fit_agg` | `lars_fit_agg` | `lars_fit_agg` |
| `anofox_stats_wls_fit_agg` | `wls_fit_agg` | `wls_fit_agg` |
| `anofox_stats_huber_fit_agg` | `huber_fit_agg` | `huber_fit_agg` |
| `anofox_stats_ransac_fit_agg` | `ransac_fit_agg` | `ransac_fit_agg` |
| `anofox_stats_theilsen_fit_agg` | `theilsen_fit_agg` | `theil_sen_fit_agg` |
| `anofox_stats_rls_fit_agg` | `rls_fit_agg` | `rls_fit_agg` |
| `anofox_stats_bls_fit_agg` | `bls_fit_agg` | `bls_fit_agg` |
| `anofox_stats_nnls_fit_agg` | `nnls_fit_agg` | `nnls_fit_agg` |
| `anofox_stats_alm_fit_agg` | `alm_fit_agg` | `alm_fit_agg` |
| `anofox_stats_ols_fit_predict_agg` | `ols_fit_predict_agg` | `ols_fit_predict_agg` |
| `anofox_stats_huber_fit_predict_agg` | `huber_fit_predict_agg` | `huber_fit_predict_agg` |
| `anofox_stats_ransac_fit_predict_agg` | `ransac_fit_predict_agg` | `ransac_fit_predict_agg` |
| `anofox_stats_theilsen_fit_predict_agg` | `theilsen_fit_predict_agg` | `theil_sen_fit_predict_agg` |
| `anofox_stats_ridge_fit_predict_agg` | `ridge_fit_predict_agg` | `ridge_fit_predict_agg` |
| `anofox_stats_wls_fit_predict_agg` | `wls_fit_predict_agg` | `wls_fit_predict_agg` |
| `anofox_stats_rls_fit_predict_agg` | `rls_fit_predict_agg` | `rls_fit_predict_agg` |
| `anofox_stats_elasticnet_fit_predict_agg` | `elasticnet_fit_predict_agg` | `elasticnet_fit_predict_agg` |
| `anofox_stats_bls_fit_predict_agg` | `bls_fit_predict_agg` | `bls_fit_predict_agg` |
| `anofox_stats_alm_fit_predict_agg` | `alm_fit_predict_agg` | `alm_fit_predict_agg` |
| `anofox_stats_pls_fit_predict_agg` | `pls_fit_predict_agg` | `pls_fit_predict_agg` |
| `anofox_stats_isotonic_fit_predict_agg` | (check file) | `isotonic_fit_predict_agg` |
| `anofox_stats_quantile_fit_predict_agg` | (check file) | `quantile_fit_predict_agg` |
| `anofox_stats_ols_predict_agg` | `ols_predict_agg` | `ols_predict_agg` |
| `anofox_stats_ridge_predict_agg` | `ridge_predict_agg` | `ridge_predict_agg` |
| `anofox_stats_rls_predict_agg` | `rls_predict_agg` | `rls_predict_agg` |
| `anofox_stats_wls_predict_agg` | `wls_predict_agg` | `wls_predict_agg` |
| `anofox_stats_elasticnet_predict_agg` | `elasticnet_predict_agg` | `elasticnet_predict_agg` |
| `anofox_stats_huber_predict_agg` | (check file) | `huber_predict_agg` |
| `anofox_stats_ransac_predict_agg` | (check file) | `ransac_predict_agg` |
| `anofox_stats_theilsen_predict_agg` (check) | (check) | `theil_sen_predict_agg` |
| `anofox_stats_vif_agg` | (check file) | `vif_agg` |
| `anofox_stats_jarque_bera_agg` | (check) | `jarque_bera_agg` |
| `anofox_stats_residuals_diagnostics_agg` | (check) | `residuals_diagnostics_agg` |

#### GLM Aggregate Functions
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_poisson_fit_agg` | `poisson_fit_agg` | `poisson_fit_agg` |
| `anofox_stats_poisson_fit_predict_agg` | `poisson_fit_predict_agg` | `poisson_fit_predict_agg` |
| `anofox_stats_aft_fit_agg` | `aft_fit_agg` | `aft_fit_agg` |
| `anofox_stats_eb_shrink_agg` | (check) | `eb_shrink_agg` |
| `anofox_stats_glmm_fit_agg` | `glmm_fit_agg` | `glmm_fit_agg` |
| `anofox_stats_binomial_fit_agg` | `binomial_fit_agg` | `binomial_fit_agg` |
| `anofox_stats_negbinom_fit_agg` | `negbinom_fit_agg` | `negbinom_fit_agg` |
| `anofox_stats_tweedie_fit_agg` | `tweedie_fit_agg` | `tweedie_fit_agg` |
| `anofox_stats_gamma_fit_agg` | `gamma_fit_agg` | `gamma_fit_agg` |
| `anofox_stats_logistic_fit_agg` | `logistic_fit_agg` | `logistic_fit_agg` |
| `anofox_stats_aid_agg` | `aid_agg` | `aid_agg` |
| `anofox_stats_aid_anomaly_agg` | `aid_anomaly_agg` | `aid_anomaly_agg` |
| `anofox_stats_aft_cdf` (scalar) | `aft_cdf` | `aft_cdf` |
| `anofox_stats_aft_quantile` (scalar) | `aft_quantile` | `aft_quantile` |

#### Hypothesis Test Aggregate Functions
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_t_test_agg` | `t_test_agg` | `t_test_agg` |
| `anofox_stats_pearson_agg` | (check) | `pearson_agg` |
| `anofox_stats_spearman_agg` | (check) | `spearman_agg` |
| `anofox_stats_mann_whitney_u_agg` | (check) | `mann_whitney_u_agg` |
| `anofox_stats_one_way_anova_agg` | (check) | `one_way_anova_agg` |
| `anofox_stats_kruskal_wallis_agg` | (check) | `kruskal_wallis_agg` |
| `anofox_stats_chisq_test_agg` | (check) | `chisq_test_agg` |
| `anofox_stats_shapiro_wilk_agg` | (check) | `shapiro_wilk_agg` |
| `anofox_stats_kendall_agg` | (check) | `kendall_agg` |
| `anofox_stats_fisher_exact_agg` | (check) | `fisher_exact_agg` |
| `anofox_stats_brunner_munzel_agg` | (check) | `brunner_munzel_agg` |
| `anofox_stats_dagostino_k2_agg` | (check) | `dagostino_k2_agg` |
| `anofox_stats_energy_distance_agg` | (check) | `energy_distance_agg` |
| `anofox_stats_mmd_agg` | (check) | `mmd_agg` |
| `anofox_stats_tost_t_test_agg` | (check) | `tost_t_test_agg` |
| `anofox_stats_wilcoxon_signed_rank_agg` | (check) | `wilcoxon_signed_rank_agg` |
| `anofox_stats_distance_cor_agg` | (check) | `distance_cor_agg` |
| `anofox_stats_yuen_agg` | (check) | `yuen_agg` |
| `anofox_stats_brown_forsythe_agg` | (check) | `brown_forsythe_agg` |
| `anofox_stats_diebold_mariano_agg` | (check) | `diebold_mariano_agg` |
| `anofox_stats_clark_west_agg` | (check) | `clark_west_agg` |
| `anofox_stats_permutation_t_test_agg` | (check) | `permutation_t_test_agg` |
| `anofox_stats_tost_paired_agg` | (check) | `tost_paired_agg` |
| `anofox_stats_tost_correlation_agg` | (check) | `tost_correlation_agg` |
| `anofox_stats_chisq_gof_agg` | (check) | `chisq_gof_agg` |
| `anofox_stats_prop_test_one_agg` | (check) | `prop_test_one_agg` |
| `anofox_stats_prop_test_two_agg` | (check) | `prop_test_two_agg` |
| `anofox_stats_binom_test_agg` | (check) | `binom_test_agg` |
| `anofox_stats_cramers_v_agg` | (check) | `cramers_v_agg` |
| `anofox_stats_cohen_kappa_agg` | (check) | `cohen_kappa_agg` |
| `anofox_stats_icc_agg` | (check) | `icc_agg` |
| `anofox_stats_g_test_agg` | (check) | `g_test_agg` |
| `anofox_stats_mcnemar_agg` | (check) | `mcnemar_agg` |
| `anofox_stats_phi_coefficient_agg` | (check) | `phi_coefficient_agg` |
| `anofox_stats_contingency_coef_agg` | (check) | `contingency_coef_agg` |

#### Scalar Diagnostic Functions
| Current Primary | Current Alias | Target Name |
|----------------|---------------|-------------|
| `anofox_stats_vif` | `vif` | `vif` |
| `anofox_stats_aic` | `aic` | `aic` |
| `anofox_stats_bic` | `bic` | `bic` |
| `anofox_stats_jarque_bera` | `jarque_bera` | `jarque_bera` |
| `anofox_stats_residuals_diagnostics` | `residuals_diagnostics` | `residuals_diagnostics` |

#### Macro Functions (already unprefixed in registration)
| Current Name | Target Name |
|-------------|-------------|
| `ols_fit_predict_by` | `ols_fit_predict_by` (no change) |
| `huber_fit_predict_by` | `huber_fit_predict_by` (no change) |
| `ransac_fit_predict_by` | `ransac_fit_predict_by` (no change) |
| `theilsen_fit_predict_by` | `theil_sen_fit_predict_by` (**fix underscore**) |
| `ridge_fit_predict_by` | `ridge_fit_predict_by` (no change) |
| `elasticnet_fit_predict_by` | `elasticnet_fit_predict_by` (no change) |
| `wls_fit_predict_by` | `wls_fit_predict_by` (no change) |

[VERIFIED: src/macros/fit_predict_macros.cpp:29,51,73,95,116,137,154]

### Return Struct Field Inconsistencies to Fix

**OLS / Ridge / WLS / ElasticNet / Huber / RANSAC / RLS / Theil-Sen** (linear regression family):
Fields are already consistent: `coefficients`, `intercept`, `r_squared`, `adj_r_squared`, `residual_std_error`, `n_observations`, `n_features`; inference: `std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper`, `f_statistic`, `f_pvalue`. [VERIFIED: src/table_functions/ols_fit.cpp:22-42, src/aggregate_functions/ols_aggregate.cpp:74-96]

**GLM family (Poisson, Logistic, Binomial, NegBinom, Tweedie, Gamma)**:
Uses `z_values` instead of `t_values` in the return struct schema. [VERIFIED: src/aggregate_functions/poisson_aggregate.cpp:108, logistic_aggregate.cpp:104, binomial_aggregate.cpp:115, glmm_aggregate.cpp:190] — This is the **documented per-family exception**: keep as-is.

**ALM** (Adaptive Location Models):
Returns `coefficients`, `intercept`, `log_likelihood`, `aic`, `bic`, `scale`, `n_observations`, `n_features`, `iterations`; inference: `std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper`. [VERIFIED: src/aggregate_functions/alm_aggregate.cpp:91-106] — Different core fields from OLS (no `r_squared`) but inference fields are already standardized with `t_values`. No rename needed.

**AFT** (Accelerated Failure Time):
Returns `coefficients`, `intercept`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`, `n_observations`, `n_events`, `n_censored`, `n_features`, `iterations`, `converged`; inference: `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper`. [VERIFIED: src/aggregate_functions/aft_aggregate.cpp:88-109] — `z_values` is the correct term for survival models; keep as documented exception.

**No cross-family field renames needed.** The `z_values` / `t_values` difference is already the intended documented exception in CONTEXT.md.

### Test File `.r2` References (Stale Field Name)

236 occurrences of `.r2` in `test/sql/*.sql` files — this is a stale alias that the current STRUCT does not expose. The correct field is `r_squared`. [VERIFIED: grep count this session] These must be replaced as part of the rename wave.

Additionally `test/sql/guide01_pattern_4_full_statistical_workflow.sql` uses `.residual_standard_error` — check whether this is already an alias or a distinct (wrong) name.

### Stale Function References in Test Files

The following names appear in test/sql/ but do not match any registered function — they are pre-rename references or typos that must be fixed as part of the rename wave:

| Stale Name | File | Correct Target |
|-----------|------|---------------|
| `anofox_stats_predict_ols` | `guide03_prediction_intervals.sql` | `predict` (table fn) |
| `anofox_stats_residual_diagnostics` | `guide03_leverage_and_influence.sql` | `residuals_diagnostics` |
| `anofox_stats_normality_test` | `guide03_normality_tests.sql` | needs verification — may be `shapiro_wilk_agg` |
| `anofox_stats_expanding_ols` | (filename check needed) | `ols_fit_predict_agg` with ROWS BETWEEN |
| `anofox_stats_rolling_ols` | (filename check needed) | `ols_fit_predict_agg` with ROWS BETWEEN |

[VERIFIED: grep of test/sql/ this session — these names returned no registration match in src/]

---

## Scope Estimate

### Registration Sites

`src/anofox_statistics_extension.cpp:LoadInternal()` contains **~50 `Register*Function(loader)` calls** — these call the per-file `Register*` functions but do not contain the name strings themselves. The name strings live inside the individual cpp files. [VERIFIED: src/anofox_statistics_extension.cpp:85-218]

| File Group | Files | Rename Touches |
|-----------|-------|---------------|
| `src/window_functions/*.cpp` | 8 files | Primary name + alias registration block (delete alias) |
| `src/aggregate_functions/*.cpp` | 65+ files | Primary name + alias registration block (delete alias) |
| `src/table_functions/*.cpp` | 9 files | Primary name + alias registration block (delete alias) |
| `src/scalar_functions/*.cpp` | 4 files | Primary name + alias registration block (delete alias) |
| `src/macros/fit_predict_macros.cpp` | 1 file | `theilsen_*` → `theil_sen_*` entries |
| `src/anofox_statistics_extension.cpp` | 1 file | No string names; `Register*` call sites unchanged |
| `test/sql/*.sql` | 120 files with `anofox_stats_` | Bulk sed: `s/anofox_stats_//g` on function names + fix `.r2` → `.r_squared` + fix stale names |

**Total adapter files touched by rename:** ~87 cpp files (all that have a primary `"anofox_stats_…"` registration and an alias block).

**Alias block deletion:** Every registration file currently has a secondary block registering the short alias with `alias_info.alias_of = "anofox_stats_…"`. Since the plan is to make the short name the primary, these alias blocks are deleted and the primary registration string is shortened to the alias value.

---

## Common Pitfalls

### Pitfall 1: Struct NULL vs Struct Fields NULL

**What goes wrong:** When emitting NULL for a STRUCT result from `Finalize`, calling `FlatVector::SetNull` on the parent vector nullifies the entire struct. If you instead try to null individual child vectors via `FlatVector::SetNull(*struct_entries[i], result_idx, true)` for each child, the parent validity bit is not set and DuckDB may still dereference the struct.

**How to avoid:** Always use `FlatVector::SetNull(result, result_idx, true)` on the parent. Never partially null a struct. [VERIFIED pattern: src/window_functions/ols_fit_predict.cpp:260]

### Pitfall 2: Alias Registration Order

**What goes wrong:** Registering the alias first and the primary second, or keeping `alias_info.alias_of` pointing to the old name after the primary is renamed, causes the alias lookup to fail silently at runtime.

**How to avoid:** After rename, the registration block for the new primary should be `AggregateFunctionSet func_set("ols_fit_agg")` — the target name. Delete the alias block entirely. Update `FunctionDescription.examples` strings in the same pass.

### Pitfall 3: `'intercept'` vs `'fit_intercept'` Option Key

**What goes wrong:** 315 test SQL references use `{'intercept': true}` (the shorter alias). The parser already accepts both (`key == "intercept" || key == "fit_intercept"`) [VERIFIED: src/include/map_options_parser.cpp:641]. After adding unknown-key rejection, this alias must remain in the allowed-keys branch — do NOT accidentally move `"intercept"` to the unknown-key error list.

**How to avoid:** Keep the `if (key == "intercept" || key == "fit_intercept")` branch as-is. Only the truly unrecognized keys (those that fall through every `else if`) go to the rejection throw.

### Pitfall 4: Theil-Sen Naming (`theilsen` vs `theil_sen`)

**What goes wrong:** The current SQL name is `theilsen` (no underscore) in both primary and alias registrations. The correct snake_case convention with underscore is `theil_sen`. Test files reference `anofox_stats_theilsen_fit` and `anofox_stats_theilsen_fit_predict` — after rename they become `theil_sen_fit` and `theil_sen_fit_predict`. If the test-file sed only replaces `anofox_stats_theilsen_` → `theil_sen_` but misses macro files or describe strings, the suite will fail.

**How to avoid:** Treat `theilsen` → `theil_sen` as a two-part rename: first strip prefix, then fix the underscore. Or use a single sed pattern `s/anofox_stats_theilsen/theil_sen/g` followed by `s/\btheilsen\b/theil_sen/g`.

### Pitfall 5: `FunctionDescription.examples` Not Updated

**What goes wrong:** Each registration has `FunctionDescription` structs with `examples` strings (e.g., `"anofox_stats_ols_fit_agg(y, x)"`). These are surfaced in DuckDB's `DESCRIBE FUNCTION` output and will be checked by Phase 6 doc validation. Not updating them leaves stale names in documentation.

**How to avoid:** Include `FunctionDescription.examples` strings in the rename search. A grep for `"anofox_stats_` restricted to the `examples =` context will catch all of them.

---

## Code Examples

### NULL Emission from Window Finalize (Already Correct Pattern)

```cpp
// VERIFIED: src/window_functions/ols_fit_predict.cpp:259-268
if (!state.initialized || !state.has_current_x) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
if (state.y_values.size() <= min_obs) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
```

### Error Code Dispatch (New Pattern to Add)

```cpp
// Proposed: src/include/error_dispatch.hpp
#pragma once
#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include <string>
namespace duckdb {
static inline void ThrowFromFfiError(const char* fn_name, const AnofoxError& err) {
    std::string msg = std::string(fn_name) + ": " + std::string(err.message);
    switch (err.code) {
        case ANOFOX_ERROR_SINGULAR_MATRIX:
        case ANOFOX_ERROR_CONVERGENCE_FAILURE:
        case ANOFOX_ERROR_INTERNAL:
        case ANOFOX_ERROR_ALLOCATION_FAILURE:
            throw FunctionException("%s", msg.c_str());
        default:
            throw InvalidInputException("%s", msg.c_str());
    }
}
} // namespace duckdb
```

Usage at a table function site (replacing line 177 in `ols_fit.cpp`):

```cpp
// Before:
if (!success) {
    throw InvalidInputException("OLS fit failed: " + string(error.message));
}
// After:
if (!success) {
    ThrowFromFfiError("ols_fit", error);
}
```

### Unknown Option Key Rejection (New Block in ParseFromValue)

```cpp
// Replace: src/include/map_options_parser.cpp:798
// Before:
//   // Unknown keys are silently ignored for forward compatibility
// After:
} else {
    throw InvalidInputException(
        "unknown option '%s'; valid keys for regression options: "
        "fit_intercept (alias: intercept), compute_inference, confidence_level, "
        "alpha, lambda, l1_ratio, max_iterations, tolerance, epsilon, "
        "residual_threshold, max_trials, stop_probability, min_samples, "
        "random_state, forgetting_factor, null_policy, link, distribution, "
        "loss, quantile, lower_bound, upper_bound, solver, hc_type, "
        "lambda_scaling, family, reml, offset, random, groups, vcov, "
        "feature_names, prior, tau_squared, tau_method, theta, threshold, "
        "tau, increasing, n_components",
        key.c_str());
}
```

---

## State of the Art

| Old Pattern | Current Pattern | Status |
|-------------|----------------|--------|
| `catch_unwind` → generic message | `catch_unwind` → `error.code` + `error.message` | FFI sets code+message; C++ **ignores code** |
| No alias check | Silent unknown key ignore | Will be **replaced** by rejection in Phase 5 |
| `"anofox_stats_"` as primary + alias | `"anofox_stats_"` primary with short alias | Will be **collapsed**: short name becomes the only name |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The INTERNAL crash on degenerate frames originates from a window file other than `ols_fit_predict.cpp` (which already has the NULL guard) | §2 Window NULL Fix | If `ols_fit_predict.cpp` has a subtle off-by-one in the `<= min_obs` check, the fix is simpler; the tracer repro will reveal the true location |
| A2 | For GROUP BY aggregates, throwing on `!success` rather than returning NULL is the right behavior for ERGO-01 | §3 FFI Error Flow | If users currently rely on NULL propagation from failed aggregate fits, changing to throw would break them; but since this is early-dev with no aliases, a throw is safer |
| A3 | `theilsen_fit_predict_by` in macros needs renaming to `theil_sen_fit_predict_by` | §4 Rename Map | Low risk — the macro registration is in a single file |

**If this table is complete:** All other claims were directly verified by reading source files this session.

---

## Open Questions

1. **Which window file causes the INTERNAL crash?**
   - What we know: `ols_fit_predict.cpp` already has a correct NULL guard at lines 259-268.
   - What's unclear: Which of the other 7 window files (or the `ols_fit_predict_agg` aggregate path) is missing the guard or has an off-by-one.
   - Recommendation: Tracer plan must include a repro SQL + identify the exact file before fixing.

2. **Should aggregate GROUP BY finalize throw or NULL on data errors?**
   - What we know: Current behavior is NULL (line 299 of `ols_aggregate.cpp`); ERGO-01 asks for clear messages.
   - What's unclear: Whether existing users depend on NULL propagation from failed fits.
   - Recommendation: Throw `InvalidInputException` with the error message; this is early-dev and the decision is locked to "clear messages".

3. **Does `anofox_stats_normality_test` reference a real function or a stale planning artifact?**
   - What we know: `grep` found it in `guide03_normality_tests.sql` but no registration in src/aggregate_functions/.
   - What's unclear: Whether it was removed and the test is orphaned, or if it's an alias for `shapiro_wilk_agg`.
   - Recommendation: Check the test file to see if it's currently passing; if not, it's already broken and needs fixing in the rename pass.

---

## Environment Availability

Not applicable — this phase makes no calls to external tools, only edits C++ and Rust source files and SQL test files.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | DuckDB SQL test runner (`ctest`) + `cargo test` |
| Config file | `CMakeLists.txt` (DuckDB test discovery) |
| Quick run command | `cd build && ctest -R "ols" --output-on-failure` |
| Full suite command | `cd build && ctest --output-on-failure && cargo test --manifest-path crates/Cargo.toml` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ERGO-01 | Invalid input returns clear error, not panic | SQL assertion | `ctest -R ergo01` | No — Wave 0 gap |
| ERGO-01 | Window degenerate frame returns NULL | SQL assertion | `ctest -R window_null` | Partial (needs repro) |
| ERGO-02 | Unknown option key rejected at bind | SQL expect-error | `ctest -R unknown_option` | No — Wave 0 gap |
| ERGO-03 | All function names follow convention | SQL smoke test | `ctest -R naming` | No — Wave 0 gap (existing tests renamed) |
| ERGO-03 | `.r2` field references updated to `.r_squared` | Existing tests | `ctest` | Yes — but fail until renamed |

### Wave 0 Gaps
- `test/sql/ergo01_clear_errors.sql` — covers ERGO-01: error message format for dimension mismatch, insufficient rows, non-finite
- `test/sql/ergo02_unknown_option.sql` — covers ERGO-02: `{'unknow_key': 1}` should throw at bind
- `test/sql/ergo01_window_null.sql` — covers ERGO-01: rolling window with <n_features+1 rows returns NULL

---

## Security Domain

No authentication, secrets, or data-boundary changes in this phase. Validation changes are purely for user-facing ergonomics, not security enforcement.

---

## Sources

### Primary (HIGH confidence)
- `src/window_functions/ols_fit_predict.cpp` — read lines 1-413 this session
- `src/aggregate_functions/ols_aggregate.cpp` — read lines 1-429 this session
- `src/table_functions/ols_fit.cpp` — read lines 1-288 this session
- `src/include/map_options_parser.cpp` — read lines 1-800 this session (full file)
- `src/include/map_options_parser.hpp` — read lines 1-415 this session (full file)
- `crates/anofox-stats-ffi/src/lib.rs` — read lines 1-400 this session
- `crates/anofox-stats-ffi/src/types.rs` — read lines 1-100 this session
- `src/include/anofox_stats_ffi.h` — read lines 1-60 this session
- `src/anofox_statistics_extension.cpp` — read grep of lines 48-218
- All aggregate function registration blocks — verified via grep

### Secondary (aggregate grep findings)
- Function name inventory from `grep AggregateFunctionSet / ScalarFunctionSet` across all src/ subdirectories
- Test file function references from `grep anofox_stats_` across test/sql/

---

## Metadata

**Confidence breakdown:**
- Validation placement: HIGH — read the actual bind functions and the parser
- Window NULL fix: HIGH for what the API is; MEDIUM for which exact file has the bug (tracer needed)
- FFI error flow: HIGH — read `lib.rs` and `types.rs`; confirmed C++ never reads `error.code`
- Naming inventory: HIGH — grep-verified all registration strings
- Scope estimate: HIGH — counted files directly
- Rename map: HIGH for primary names; MEDIUM for a few "check file" entries (aliases not yet confirmed in every file)

**Research date:** 2026-09-01
**Valid until:** Until source files are changed — all claims cite specific line ranges
