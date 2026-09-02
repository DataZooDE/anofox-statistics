# Phase 5: API Ergonomics - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the public SQL surface fail fast and read consistently. Three deliverables:
(ERGO-01) fit/predict/test functions return clear, actionable error messages for
invalid input instead of panics or opaque errors; (ERGO-02) inputs are validated
as early as possible — option/shape checks at bind time, data-dependent checks at
execution — with messages naming the offending argument and its expected shape;
(ERGO-03) function names, option-map keys, and return-struct field names follow one
documented naming convention consistent across all model families, with breaking
renames reflected in the test suite.

Out of scope: named parameters (`param := value`, ERGOX-01 — deferred milestone);
new statistical models; rewriting Rust core numerics.

</domain>

<decisions>
## Implementation Decisions

### Naming Convention (ERGO-03)
- SQL function names use `{model}_{verb}[_agg|_predict]`, **unprefixed and uniform** — drop the inconsistent `anofox_stats_` prefix that only some functions currently carry (e.g. standardize `anofox_stats_ols_fit_predict` → `ols_fit_predict`).
- Option-map keys are `snake_case` across all families (`compute_inference`, `hc_type`, `l1_ratio`), matching the Rust core.
- Return-struct field names are `snake_case` with one standard set across families (`coefficients`, `intercept`, `std_errors`, `t_values`, `p_values`, `r_squared`). GLM `z_values` stays as a **documented per-family exception** — do NOT force z→t (preserves the Phase-4 decision that GLM maps z→t with lenient OOM and ALM uses different field names).
- The convention is written down in a new `docs/API_CONVENTIONS.md`, which Phase 6's doc-SQL validation will check examples against.

### Error Messages (ERGO-01)
- Message format: `"<fn>: <problem>; expected <shape> (got <actual>)"` — always name the function, the offending argument, and expected vs actual shape.
- Keep the FFI `catch_unwind` panic guard, but convert caught panics/`StatsError`s into a **specific** DuckDB exception carrying the real detail — replace the generic "fit failed" wrapper.
- The rolling-window degenerate-frame bug (`ols_fit_predict(...) OVER (...)` throwing `INTERNAL: access index 0 within vector of size 0` when a frame has fewer than `n_features+1` rows at partition start) is fixed by returning a **NULL prediction** for degenerate frames — standard rolling-regression behavior — documented, not an error raised mid-window.
- Exception taxonomy: map `StatsError` variants to `InvalidInputException` (user data/shape problems: dimension mismatch, insufficient rows, constant column, unknown option) vs `FunctionException` (numerical failures: singular matrix, convergence).

### Validation Placement (ERGO-02)
- Bind-time checks (knowable before execution): argument count, option key names + types, option value ranges (e.g. `alpha ∈ (0,1)`, `l1_ratio ∈ [0,1]`), non-empty feature list.
- Execution/finalize checks (data-dependent): y-vs-x row-count dimension match, insufficient rows (`n < n_features + 1`), all-non-finite input, constant / zero-variance column — each with a clear message.
- Unknown option-map keys are **rejected at bind** with `"unknown option 'X'; valid: ..."` to catch typos (change from current silent-ignore).
- A constant / zero-variance column produces a clear error naming the column position; never silently dropped.

### Renames & Migration Scope
- **No deprecated aliases** — early-dev, breaking renames are allowed (locked project decision); do a clean rename.
- Rename pass covers **all families in one pass**: regression, GLM, GLMM, AFT, hypothesis tests, diagnostics.
- The test suite (`test/sql` + `cargo test`) is updated to the new names **within this phase** — success criterion 4 requires both green against the new names.
- Renames land in Phase 5; Phase 6 then restructures docs and validates every SQL example against the final, renamed API (existing roadmap order).

### Claude's Discretion
- Exact per-function rename table, the precise wording of each error string, and the RAII/helper mechanics of bind-vs-execution validation are at Claude's discretion, guided by the conventions above and existing codebase patterns (`StatsError` variants, `InvalidInputException`/`FunctionException`, MAP-option bind parsing).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/anofox-stats-core/src/errors.rs` — structured `StatsError` variants already exist and cover most cases: `EmptyInput { field }`, `DimensionMismatch { y_len, x_rows }`, `InsufficientData { rows, cols }`, `NoValidData`, `SingularMatrix`, `CholeskyFailed`, `ConvergenceFailure`. ERGO work mostly surfaces these clearly rather than inventing new ones.
- FFI boundary already converts `StatsError` → `ErrorCode` via `error_to_code()` and populates `AnofoxError { code, message: [c_char;256] }` (`crates/anofox-stats-ffi/src/lib.rs`).
- FFI already wraps Rust calls in `std::panic::catch_unwind(AssertUnwindSafe(...))`.
- C++ bind-time option parsing from DuckDB `MAP` already exists (per ARCHITECTURE.md "MAP options pattern; options parsed at bind time").

### Established Patterns
- Positional arguments only (no `:=`) — signatures stay positional this milestone.
- Options passed as DuckDB `MAP`, parsed and validated at bind (`*Bind()` functions, e.g. `src/table_functions/ols_fit.cpp:OlsFitBind()`).
- Results returned as fixed-schema `STRUCT`; result type built conditionally via `GetOlsResultType(bool compute_inference)` style builders.
- C++ throws `InvalidInputException` for validation, `FunctionException` for computation failures; Rust returns `StatsResult<T>`.
- snake_case throughout Rust; `Options`/`Result` struct suffixes.

### Integration Points
- SQL function registration in `src/anofox_statistics_extension.cpp:LoadInternal()` — rename touches every `Register*Function()` call site + the string names.
- Adapters across `src/aggregate_functions/` (45+), `src/table_functions/` (8), `src/window_functions/` (8), `src/scalar_functions/` (4), `src/macros/fit_predict_macros.cpp`.
- Test suite: `test/sql/*` (SQL assertions reference function names, option keys, struct fields) + `cargo test` in both crates.
- Window degenerate-frame fix lives in the window aggregate finalize path (`src/window_functions/*fit_predict*`).

</code_context>

<specifics>
## Specific Ideas

- Prime concrete target for ERGO-01/02: the rolling `ols_fit_predict(...) OVER (...)` INTERNAL error on sub-`(n_features+1)` frames — carried forward from Phase 4 benchmark findings (see PROJECT.md / STATE.md blocker). Fix = NULL prediction for degenerate frames.
- Error-string shape to standardize on: `"<fn>: <problem>; expected <shape> (got <actual>)"`.

</specifics>

<deferred>
## Deferred Ideas

- Named parameters (`param := value`) — ERGOX-01, explicitly deferred to a dedicated milestone (Out of Scope in REQUIREMENTS.md).

</deferred>
