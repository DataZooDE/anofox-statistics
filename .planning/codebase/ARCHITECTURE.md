<!-- refreshed: 2026-08-11 -->
# Architecture

**Analysis Date:** 2026-08-11

## System Overview

Anofox Statistics is a DuckDB extension providing statistical analysis capabilities. It consists of three layers: the DuckDB C++ wrapper layer, the C FFI boundary (Rust), and the core Rust statistics library. Data flows from DuckDB SQL queries through the C++ function adapters into the Rust FFI, then into the core statistical algorithms.

```text
┌──────────────────────────────────────────────────────────────────┐
│                         DuckDB SQL Layer                          │
│  Queries invoke: ols_fit_agg(), poisson_fit_agg(), t_test_agg()   │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    C++ Wrapper Layer (Extension)                  │
│  `src/` - DuckDB function registration and binding                │
│  ├─ aggregate_functions/ (45+ aggregate adapters)                │
│  ├─ table_functions/ (scalar fit functions)                      │
│  ├─ window_functions/ (rolling/expanding fit+predict)            │
│  ├─ scalar_functions/ (diagnostics: vif, aic, bic)               │
│  └─ macros/ (SQL table macros for fit_predict_by)                │
└────────────┬─────────────────────────────────────────────────────┘
             │ Calls via anofox_stats_ffi.h
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   C FFI Layer (anofox-stats-ffi)                  │
│  `crates/anofox-stats-ffi/src/` - FFI bindings                    │
│  ├─ lib.rs: 50+ FFI function definitions (anofox_ols_fit, ...)    │
│  └─ types.rs: C-compatible type definitions (DataArray, Options)  │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│              Rust Core Library (anofox-stats-core)                │
│  `crates/anofox-stats-core/` - Pure Rust statistics               │
│  ├─ models/: fit_ols(), fit_ridge(), fit_glmm(), fit_aft(), ...   │
│  ├─ diagnostics/: compute_vif(), jarque_bera(), ...               │
│  └─ External deps: anofox-regression, anofox-tests, faer, statrs  │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Extension Entry | Load extension, register all functions, initialize telemetry | `src/anofox_statistics_extension.cpp` |
| Aggregate Functions | Accumulate rows, fit models per group, serialize results as STRUCT | `src/aggregate_functions/*.cpp` (45+ files) |
| Table Functions (Scalar Fit) | Single-shot model fitting from arrays (for per-row or simple use cases) | `src/table_functions/*.cpp` (8 files: ols_fit, ridge_fit, ...) |
| Window Functions | Fit+predict per window partition (rolling regression, expanding windows) | `src/window_functions/*.cpp` (8 files: ols_fit_predict, ...) |
| Scalar Diagnostics | Compute statistics on residuals/models (VIF, AIC, BIC, Jarque-Bera) | `src/scalar_functions/*.cpp` (4 files) |
| SQL Table Macros | Programmatically generate per-group fit_predict queries | `src/macros/fit_predict_macros.cpp` |
| FFI Boundary | Convert C++ DuckDB types to Rust types, call core library, error handling | `crates/anofox-stats-ffi/src/lib.rs` & `types.rs` |
| Core Statistics | Pure Rust regression, GLM, survival models, hypothesis tests | External: `anofox-regression`, `anofox-tests` crates |
| Telemetry | PostHog integration for anonymized usage tracking (optional, disableable) | `posthog-telemetry/` (separate C++ module) |

## Pattern Overview

**Overall:** DuckDB extension with language-boundary FFI to maximize type safety and leverage existing Rust statistical libraries.

**Key Characteristics:**
- **Layered architecture**: C++ wraps Rust, Rust FFI wraps core Rust logic
- **Function polymorphism**: Same model (OLS, Ridge, etc.) offered as scalar table function, aggregate, window function, and fit_predict macro
- **Positional arguments only**: All functions use positional parameters (not `:=` named args) for consistency
- **MAP options pattern**: Options passed as DuckDB MAP type and parsed at bind time
- **Result serialization**: All functions return STRUCT with fixed schema (coefficients, intercept, inference stats, metadata)
- **Per-group analysis via GROUP BY**: No specialized grouping logic; uses DuckDB's native aggregation
- **Window function support**: fit_predict aggregates work with OVER clauses for rolling/expanding windows

## Layers

**DuckDB C++ Extension Layer:**
- Purpose: Provide SQL function interface, handle DuckDB memory/vectorization, parse options
- Location: `src/`
- Contains: Function registration, bind data structures, state machines for aggregation
- Depends on: DuckDB headers, anofox_stats_ffi.h, posthog-telemetry
- Used by: DuckDB query planner/executor

**C FFI Boundary:**
- Purpose: Convert C++ DuckDB types (DataChunk, Vector, etc.) to Rust types, call core functions, propagate errors
- Location: `crates/anofox-stats-ffi/src/lib.rs`, `types.rs`
- Contains: 50+ C-compatible functions, type converters, error codes
- Depends on: anofox-regression, anofox-tests, statrs crates
- Used by: C++ aggregate_functions, table_functions, etc.

**Rust Core Statistics (External):**
- Purpose: Pure Rust implementations of regression, GLM, survival models, hypothesis tests
- Location: External crates `anofox-regression` (v0.5.13), `anofox-tests` (v0.4.2)
- Contains: fit_ols, fit_ridge, fit_glmm, fit_aft, t_test, mann_whitney_u, ...
- Depends on: faer (linear algebra), statrs (distributions)
- Used by: FFI layer

## Data Flow

### Primary Request Path: OLS Aggregate Query

1. User runs:
   ```sql
   SELECT category, ols_fit_agg(y, [x1, x2], {'intercept': true}) as result
   FROM data GROUP BY category
   ```

2. DuckDB calls `RegisterOlsAggregateFunction()` → binds `ols_fit_agg` → sets up aggregation state (`src/aggregate_functions/ols_aggregate.cpp:AftAggregateState`)

3. For each row, `OlsAggUpdate()` appends y, x values to state vectors

4. On finalize: `OlsAggFinalize()` calls `anofox_ols_fit()` FFI function

5. FFI function (`crates/anofox-stats-ffi/src/lib.rs:anofox_ols_fit()`) converts:
   - `Vec<f64>` ← `DataArray::to_vec()`
   - `OlsOptions` ← `OlsOptionsFFI` struct
   - Calls `fit_ols()` from anofox-regression crate

6. Core function returns `FitResult { coefficients, intercept, r_squared, ... }`

7. FFI wraps result in `FitResultCore` struct

8. C++ finalize wraps in DuckDB STRUCT and returns

9. Result serialized as SQL column with schema: `{coefficients: DOUBLE[], intercept: DOUBLE, r_squared: DOUBLE, ...}`

### Window Function Path: Rolling Regression

1. User runs:
   ```sql
   SELECT *, 
     ols_fit_predict_agg(y, [x]) OVER (
       PARTITION BY group 
       ORDER BY time ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
     ) as model
   FROM data
   ```

2. DuckDB creates window frame for each partition

3. For each frame, identical aggregate flow as above, but executes multiple times per group (once per window frame)

4. Returns one result row per input row with predictions

### Table Macro Path: Per-Group Fit+Predict

1. User calls:
   ```sql
   SELECT * FROM ols_fit_predict_by('my_table', category, y, [x1, x2])
   ```

2. Macro expansion (`src/macros/fit_predict_macros.cpp`) generates SQL:
   ```sql
   SELECT *, 
     (_pred[_rn]).yhat, (_pred[_rn]).yhat_lower, (_pred[_rn]).yhat_upper
   FROM (SELECT *,
     ROW_NUMBER() OVER (PARTITION BY category) AS _rn,
     ols_fit_predict_agg(y, [x1, x2]) OVER (PARTITION BY category) AS _pred
     FROM my_table
   ) sub
   ```

3. Internally uses window functions to fit once per group, predict for all rows in group

**State Management:**
- Aggregate state: Accumulates in vectors (`time_values`, `x_columns`, etc.), destroyed at finalize
- Bind state: Parsed from MAP options at query bind time, cloned for each partition (FunctionData subclass)
- Thread safety: DuckDB handles per-thread state isolation via UnifiedVectorFormat

## Key Abstractions

**Aggregate State Pattern:**
- Purpose: Accumulate observations for group, defer computation to finalize
- Examples: `src/aggregate_functions/ols_aggregate.cpp:OlsAggregateState`, `AftAggregateState`
- Pattern: Struct with `vector<double>` members for y/x/weight/event, plus bind data copy

**BindData Pattern:**
- Purpose: Store parsed options and hold cloned state across partitions
- Examples: `OlsFitBindData`, `AftAggregateBindData`
- Pattern: Derives `FunctionData`, implements `Copy()` and `Equals()` virtual methods

**FFI Options Converters:**
- Purpose: Convert C++ enum types (SolverType, HcType) to Rust enums, vice versa
- Examples: `convert_solver_ffi()`, `convert_hc_type_ffi()`, `error_to_code()`
- Pattern: Simple match on enum variants, no logic

**Result Type Builders:**
- Purpose: Construct return STRUCT schema based on options (e.g., inference vs. no inference)
- Examples: `GetOlsResultType(bool compute_inference)`, `GetAftAftResultType(bool compute_inference)`
- Pattern: Build `child_list_t<LogicalType>` conditionally, return `LogicalType::STRUCT(...)`

## Entry Points

**Extension Load:**
- Location: `src/anofox_statistics_extension.cpp:LoadInternal()`
- Triggers: When extension is first `LOAD`ed in DuckDB session
- Responsibilities: Register all 50+ functions (scalar, aggregate, window, table macros), initialize telemetry if enabled

**Per-Query Execution:**
- For aggregates: `AggregateFunction::bind()` → `update()` → `combine()` → `finalize()`
- For scalars: `ScalarFunction::bind()` → `execute()`
- For windows: Same as aggregate but called per window frame

**Telemetry (optional):**
- Location: `src/anofox_statistics_extension.cpp:RegisterTelemetryOptions()`, `PostHogTelemetry::Instance()`
- Triggers: On extension load and per function call (if telemetry enabled)
- Responsibilities: Send usage events to PostHog (disabled on MinGW due to OpenSSL dependency)

## Architectural Constraints

- **Threading:** Single-threaded per partition within DuckDB's vectorized execution model; DuckDB handles thread pooling across partitions
- **Global state:** `PostHogTelemetry::Instance()` singleton (thread-safe); no other module-level mutable state
- **Circular imports:** Header-only abstract classes (FunctionData) prevent cycles; FFI headers included by C++ source files only
- **FFI ABI stability:** C data types (struct, enum, pointer) must match exact binary layout; checked by cbindgen.toml
- **Memory ownership:** DuckDB allocates state structs; FFI functions receive pointers and read/write via unsafe blocks
- **Option parsing:** All options parsed at bind time (compile-time check), not execution time (for performance)
- **Return types:** Always STRUCT for aggregates/fit functions, enabling nested field access in SQL (e.g., `result.coefficients[1]`)

## Anti-Patterns

### Deferred Validation (Option Parsing)

**What happens:** Options parsed from MAP at bind time, used at finalize. If validation was skipped at bind, errors occur late.

**Why it's wrong:** User gets query error after aggregation completes, wasting CPU time and giving poor feedback.

**Do this instead:** Validate all options in bind function (`src/table_functions/ols_fit.cpp:OlsFitBind()`) and throw `InvalidInputException` immediately. DuckDB stops query before execution starts.

### Uninitialized State Check Bypass

**What happens:** Aggregate state is lazily initialized on first row update, but doesn't check `initialized` flag consistently.

**Why it's wrong:** Race condition or undefined behavior if combine/finalize called before any update.

**Do this instead:** Always check `if (!initialized)` before using state vectors, or always initialize in `*Initialize()` callback even if empty.

## Error Handling

**Strategy:** Multi-level error propagation using C error codes + message strings.

**Patterns:**
- **C++ level:** Throw `InvalidInputException` for validation errors; `FunctionException` for computation errors
- **FFI level:** Return `bool` false on error; populate `AnofoxError` struct with code and message
- **Computation level:** Core library returns `Result<T, StatsError>`; FFI converts to `ErrorCode` enum (at `crates/anofox-stats-ffi/src/lib.rs:error_to_code()`)
- **DuckDB level:** C++ catch wraps Rust FFI errors and throws DuckDB exception

Example (`src/aggregate_functions/ols_aggregate.cpp:OlsAggFinalize()`):
```cpp
if (!status) {
    throw FunctionException("OLS fit failed: %s", error.message);
}
```

## Cross-Cutting Concerns

**Logging:** No structured logging; errors propagated via exceptions. Telemetry captures function calls for analytics.

**Validation:** 
- Bind-time: Options type-checked by MAP parser
- Update-time: Vector dimension consistency checked on each row
- Finalize-time: Sufficient sample size (n >= p+1 for regression) checked in FFI

**Authentication:** None; extension assumes trusted database connection.

**Telemetry (Optional):**
- On load: `PostHogTelemetry::Instance().CaptureExtensionLoad("anofox_statistics", version)`
- Per call: `PostHogTelemetry::Instance().RecordFunctionCall("ols_fit")`
- Disableable via env var `DATAZOO_DISABLE_TELEMETRY=1` or SQL `SET anofox_telemetry_enabled = false`

---

*Architecture analysis: 2026-08-11*
