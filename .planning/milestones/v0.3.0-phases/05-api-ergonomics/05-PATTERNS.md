# Phase 5: API Ergonomics — Pattern Map

**Mapped:** 2026-09-01
**Files analyzed:** 7 work-item targets + shared header (new file)
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/include/error_dispatch.hpp` (new) | utility/header | request-response | `src/include/anofox_stats_ffi.h` (structure) + `src/table_functions/ols_fit.cpp:177` (call site) | partial — new file, pattern assembled from FFI header + existing throw sites |
| `src/window_functions/*_fit_predict.cpp` (7 files minus ols) | aggregate/window | streaming | `src/window_functions/ols_fit_predict.cpp:259-268` | exact |
| `src/table_functions/ols_fit.cpp` (+ 9 peers) | table function | request-response | itself — existing `!success` throw at line 177 | self — replace |
| `src/aggregate_functions/ols_aggregate.cpp` (+ all agg peers) | aggregate | CRUD | itself — existing `!success` NULL at line 298-300 | self — replace |
| `src/include/map_options_parser.cpp` | utility | request-response | itself — existing `ParseFromValue` at line 798 | self — replace |
| `src/aggregate_functions/ols_aggregate.cpp` (registration block) | registration | — | itself lines 377-425 | exact — collapse alias |
| `src/window_functions/ols_fit_predict.cpp` (registration block) | registration | — | itself lines 360-410 | exact — collapse alias |

---

## Pattern Assignments

### Work Item 1 — Window NULL path (ERGO-01 degenerate-frame fix)

**Analog (already correct):** `src/window_functions/ols_fit_predict.cpp` lines 258-268

**Exact template to copy into the other 7 window files** (`huber_fit_predict.cpp`, `ransac_fit_predict.cpp`, `theil_sen_fit_predict.cpp`, `ridge_fit_predict.cpp`, `wls_fit_predict.cpp`, `rls_fit_predict.cpp`, `elasticnet_fit_predict.cpp`) and into the aggregate-path `ols_fit_predict_agg` finalize:

```cpp
// src/window_functions/ols_fit_predict.cpp:258-268  (VERIFIED — already correct here)
// ──────────────────────────────────────────────────────────────────────────────
// Need minimum data to fit
idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
if (state.y_values.size() <= min_obs) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}
```

Context: immediately after the `!state.initialized || !state.has_current_x` guard (lines 259-261) and immediately before the FFI call. Both guards must be present.

**What the other 7 files currently do:** They have `!state.initialized || !state.has_current_x` and `!success` guards but **the `<= min_obs` check is missing or uses a weaker threshold** (e.g. `huber_fit_predict.cpp:259` uses `<= min_obs`, verified present; the aggregate-path `ols_aggregate.cpp:263` uses `y_values.size() < 2` — a fixed constant that does not scale with `n_features`).

**Aggregate-path analog to fix** (`src/aggregate_functions/ols_aggregate.cpp:261-266`):

```cpp
// CURRENT (weak — does not scale with n_features):
if (!state.initialized || state.y_values.size() < 2) {
    FlatVector::SetNull(result, result_idx, true);
    continue;
}

// REPLACE WITH (mirroring ols_fit_predict.cpp:258-268):
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

**Pitfall:** Use `FlatVector::SetNull(result, result_idx, true)` on the **parent** STRUCT vector, never on individual child vectors. Verified correct in `ols_fit_predict.cpp:260`.

---

### Work Item 2 — `ThrowFromFfiError` dispatch helper (ERGO-01 error surfacing)

**New file:** `src/include/error_dispatch.hpp`

**Pattern assembled from two sources:**

**Source A — `AnofoxErrorCode` enum** (`src/include/anofox_stats_ffi.h:18-31`, verbatim):

```c
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

**Source B — current generic throw site** (`src/table_functions/ols_fit.cpp:176-178`, verbatim):

```cpp
if (!success) {
    throw InvalidInputException("OLS fit failed: " + string(error.message));
}
```

**New header to write** — mirrors the `#pragma once` + `duckdb.hpp` + `anofox_stats_ffi.h` include pattern already in every table function file:

```cpp
// src/include/error_dispatch.hpp  (NEW FILE)
#pragma once
#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include <string>

namespace duckdb {

static inline void ThrowFromFfiError(const char *fn_name, const AnofoxError &err) {
    std::string msg = std::string(fn_name) + ": " + std::string(err.message);
    switch (err.code) {
        case ANOFOX_ERROR_SINGULAR_MATRIX:
        case ANOFOX_ERROR_CONVERGENCE_FAILURE:
        case ANOFOX_ERROR_INTERNAL:
        case ANOFOX_ERROR_ALLOCATION_FAILURE:
            throw FunctionException("%s", msg.c_str());
        default:
            // InsufficientData, DimensionMismatch, InvalidInput, NoValidData,
            // InvalidAlpha, InvalidL1Ratio, SerializationError → user data problem
            throw InvalidInputException("%s", msg.c_str());
    }
}

} // namespace duckdb
```

**Call-site replacement pattern** — apply at every `!success` throw site listed in RESEARCH.md §3 (10 files):

```cpp
// BEFORE (e.g. src/table_functions/ols_fit.cpp:176-178):
if (!success) {
    throw InvalidInputException("OLS fit failed: " + string(error.message));
}

// AFTER:
if (!success) {
    ThrowFromFfiError("ols_fit", error);
}
```

**All 10 call sites requiring this replacement** (file : old line : old string):
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

**Aggregate finalize sites** (`ols_aggregate.cpp:298-300` and all agg peers):
For GROUP BY (non-window) finalize, replace the silent `FlatVector::SetNull` + `continue` with `ThrowFromFfiError`. For window finalize keep the NULL path (degenerate frames are expected — see Work Item 1).

---

### Work Item 3 — Unknown MAP key rejection at bind (ERGO-02)

**Analog:** `src/include/map_options_parser.cpp:637-799`

**Current silent-ignore else at line 798** (verbatim):

```cpp
// src/include/map_options_parser.cpp:798
// Unknown keys are silently ignored for forward compatibility
```

This is the last statement inside the `VisitOptionEntries` lambda in `RegressionMapOptions::ParseFromValue`. The full if/else-if chain starts at line 641.

**Alias pitfall — keep this branch as-is** (`map_options_parser.cpp:641`):

```cpp
if (key == "intercept" || key == "fit_intercept") {
    result.fit_intercept = ExtractBool(val);
} else if (key == "compute_inference" || key == "inference") {
```

The `"intercept"` alias must NOT be added to the unknown-key rejection list. 315 test files use it.

**Replace line 798 with:**

```cpp
} else {
    throw InvalidInputException(
        "unknown option '%s'; valid keys: fit_intercept (alias: intercept), "
        "compute_inference (alias: inference), confidence_level (alias: confidence), "
        "alpha, lambda, l1_ratio, max_iterations (alias: max_iter), "
        "tolerance (alias: tol), epsilon, residual_threshold, max_trials, "
        "stop_probability, min_samples, random_state, forgetting_factor, "
        "null_policy, link, distribution, loss, quantile, lower_bound, "
        "upper_bound, solver, hc_type, lambda_scaling, family, reml, "
        "offset, random, groups (alias: crossed), vcov (alias: vcov_type), "
        "feature_names, prior, tau_squared (alias: tau2), "
        "tau_method (alias: shrinkage), theta (alias: nb_theta, dispersion), "
        "threshold, tau, increasing, n_components",
        key.c_str());
}
```

**Separate test-option parsers needing the same treatment** (each has its own `ParseFromValue` with its own silent-ignore else — apply the same pattern with its own valid-key list):
- `TTestMapOptions::ParseFromValue` — valid keys: `alpha`, `alternative`, `mu`
- `MannWhitneyMapOptions::ParseFromValue` — verify in `map_options_parser.hpp:280-413`
- All other statistical test option structs listed in `src/include/map_options_parser.hpp:280-413`

---

### Work Item 4 — Rename pass: collapse primary + alias into single registration (ERGO-03)

**Analog (full registration block):** `src/aggregate_functions/ols_aggregate.cpp:377-425` (verbatim):

```cpp
// src/aggregate_functions/ols_aggregate.cpp:377-425
void RegisterOlsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_ols_fit_agg");   // ← PRIMARY string to rename

    auto basic_func = AggregateFunction(
        "anofox_stats_ols_fit_agg",                              // ← rename here too
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,
        AggregateFunction::StateSize<OlsAggregateState>, OlsAggInitialize,
        OlsAggUpdate, OlsAggCombine, OlsAggFinalize,
        nullptr, OlsAggBind, OlsAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction("anofox_stats_ols_fit_agg",  // ← rename here too
                                      ...);
    func_set.AddFunction(map_func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.examples = {"anofox_stats_ols_fit_agg(y, x, {'fit_intercept': true})"}; // ← rename in example string
    ...
    loader.RegisterFunction(std::move(info));

    // Register short alias  ← DELETE THIS ENTIRE BLOCK
    {
        AggregateFunctionSet alias_set("ols_fit_agg");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_ols_fit_agg";
        loader.RegisterFunction(std::move(alias_info));
    }
}
```

**After rename, the block becomes:**

```cpp
void RegisterOlsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("ols_fit_agg");   // short name is now the ONLY name

    auto basic_func = AggregateFunction(
        "ols_fit_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,
        AggregateFunction::StateSize<OlsAggregateState>, OlsAggInitialize,
        OlsAggUpdate, OlsAggCombine, OlsAggFinalize,
        nullptr, OlsAggBind, OlsAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction("ols_fit_agg", ...);
    func_set.AddFunction(map_func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.examples = {"ols_fit_agg(y, x, {'fit_intercept': true})"};  // updated example
    ...
    loader.RegisterFunction(std::move(info));
    // alias block deleted
}
```

**Window function registration analog** (`src/window_functions/ols_fit_predict.cpp:360-410`) — same structure; the alias block at lines 401-410 is deleted, primary strings at lines 362, 369, 376, 384, 392 become `"ols_fit_predict"`.

**Theil-sen special case** — current registration uses `"anofox_stats_theilsen_fit_agg"` (no underscore). Target is `"theil_sen_fit_agg"` (with underscore). This is a two-part change: strip prefix AND fix the underscore. Apply the same registration collapse pattern.

**`FunctionDescription.examples` strings** must be updated in the same pass — they appear inside the registration block and are surfaced by `DESCRIBE FUNCTION`. Grep: `grep -rn '"anofox_stats_' src/ | grep 'examples'`.

**Macro file** (`src/macros/fit_predict_macros.cpp:95`) — `"theilsen_fit_predict_by"` → `"theil_sen_fit_predict_by"`. Same string-replacement; no alias block to delete.

---

## Shared Patterns

### NULL emission from window/aggregate finalize
**Source:** `src/window_functions/ols_fit_predict.cpp:260, 267, 299, 313`
**Apply to:** All 7 other `*_fit_predict.cpp` window files + aggregate-path finalize with `FlatVector::SetNull` on the **parent** STRUCT vector.

```cpp
FlatVector::SetNull(result, result_idx, true);
continue;
```

### InvalidInputException / FunctionException throw
**Source:** `src/table_functions/ols_fit.cpp:177` (existing call site), replaced by `ThrowFromFfiError`
**Apply to:** All 10 `!success` throw sites in table functions and scalar functions after including `error_dispatch.hpp`.

### MAP options bind pattern
**Source:** `src/table_functions/ols_fit.cpp:71-99` (full `OlsFitBind`)
**Apply to:** All `*Bind()` functions — unknown-key rejection happens inside `ParseFromValue` (called transitively via `RegressionMapOptions::ParseFromExpression`), not at each bind site individually. No per-bind-function change needed once `ParseFromValue` is fixed.

### Registration string scope
**Source:** Each `src/**/*.cpp` file contains its own name strings — `src/anofox_statistics_extension.cpp` only calls `Register*Function(loader)`, it does **not** contain the string literals. Rename targets are the individual `src/aggregate_functions/`, `src/window_functions/`, `src/table_functions/`, `src/scalar_functions/` files.

---

## No Analog Found

None. All four work items have direct analogs in the existing codebase.

---

## Metadata

**Analog search scope:** `src/window_functions/`, `src/aggregate_functions/`, `src/table_functions/`, `src/scalar_functions/`, `src/include/`, `src/macros/`
**Key files read:** `ols_fit_predict.cpp` (full), `ols_aggregate.cpp` (finalize + registration), `ols_fit.cpp` (bind + throw site), `map_options_parser.cpp:635-802`, `anofox_stats_ffi.h:15-52`
**Pattern extraction date:** 2026-09-01
