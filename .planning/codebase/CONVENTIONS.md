# Coding Conventions

**Analysis Date:** 2026-08-11

## Naming Patterns

**Files:**
- Rust modules use snake_case: `ols.rs`, `lm_dynamic.rs`, `vif.rs`
- Module files named `mod.rs` organize public exports and documentation
- FFI wrapper files use suffixes indicating C interoperability (no special naming, but `ffi/src/lib.rs` and `ffi/src/types.rs` are conventions)

**Functions:**
- Public functions use snake_case: `fit_ols()`, `compute_vif()`, `diebold_mariano()`
- FFI functions use explicit naming with C-compatible signatures prefixed with crate name: `anofox_ols_fit()`, `anofox_free_result_core()`
- Internal helper functions (converters, error handlers) follow pattern: `convert_solver()`, `convert_ic()`, `convert_error()`

**Variables:**
- Descriptive names with snake_case: `n_observations`, `valid_indices`, `is_constant_column`
- Mathematical variables use conventional abbreviations: `y` (response), `x` (features), `r_squared` (coefficient), `df` (degrees of freedom)
- Configuration/options structs end with `Options`: `OlsOptions`, `TTestOptions`, `DieboldMarianoOptions`
- Result structs end with `Result` or `Inference`: `FitResult`, `FitResultInference`, `TestResult`

**Types:**
- Result type aliases use `Result<T>`: `StatsResult<T>` (defined as `Result<T, StatsError>` in `src/errors.rs`)
- FFI struct suffixes: plain struct name for core types (e.g., `FitResultCore`), `FFI` suffix for C-compatible versions (e.g., `OlsOptionsFFI`, `SolverTypeFFI`)
- Enum variants use PascalCase: `TTestKind::Welch`, `Alternative::TwoSided`, `ConditionSeverity::Severe`

## Code Style

**Formatting:**
- Rust default formatting is used (no explicit `rustfmt.toml` found in repo)
- Follows Rust standard: 4-space indentation, no tabs
- Line length not explicitly constrained (typical Rust range ~100-120)

**Linting:**
- Cargo.toml includes `[patch.crates-io]` to patch `argmin` and `argmin-math` for stable Rust compatibility, indicating strict clippy adherence
- FFI code uses `#[no_mangle]` and `unsafe extern "C"` for C boundary functions (located in `crates/anofox-stats-ffi/src/lib.rs`)
- Edition: 2021 (workspace default in `Cargo.toml`)

## Import Organization

**Order:**
1. Standard library imports (`use std::*`)
2. External crate imports (`use thiserror::*`, `use anofox_regression::*`)
3. Internal crate imports (`use crate::errors::*`, `use crate::models::*`)
4. Module-level `use` for private functions in same module

**Path Aliases:**
- Workspace dependencies defined in `[workspace.dependencies]` in root `Cargo.toml`
- Workspace members (`anofox-stats-core`, `anofox-stats-ffi`) share versions and dependencies through workspace inheritance
- No alias syntax observed; full qualified imports used (e.g., `use anofox_regression::prelude::*`)

**Module Pattern:**
- Public re-exports follow barrel pattern: `pub mod diagnostics;` in `lib.rs` followed by module-specific exports
- `mod.rs` files centralize exports: `pub use aft::{...}`, `pub use alm::{...}` in `crates/anofox-stats-core/src/models/mod.rs`

## Error Handling

**Patterns:**
- Custom error type `StatsError` defined in `src/errors.rs` with `#[derive(Error, Debug)]` from `thiserror` crate
- Specific error variants for each error class:
  - Input validation: `InvalidAlpha(f64)`, `InvalidL1Ratio(f64)`, `EmptyInput { field: &'static str }`
  - Dimension mismatches: `DimensionMismatch { y_len, x_rows }`, `DimensionMismatchMsg(String)`
  - Data issues: `InsufficientData { rows, cols }`, `NoValidData`
  - Numerical failures: `SingularMatrix`, `CholeskyFailed`, `QrFailed`
  - Convergence: `ConvergenceFailure { iterations, tolerance }`
- All public functions return `StatsResult<T>` where `StatsResult<T> = Result<T, StatsError>`
- Error mapping from upstream libraries implemented via `From` trait (e.g., `impl From<anofox_regression::solvers::RegressionError> for StatsError`)
- FFI boundary converts `StatsError` to `ErrorCode` enum in `error_to_code()` helper

**Example patterns:**
```rust
// Input validation with explicit errors
if y.is_empty() {
    return Err(StatsError::EmptyInput { field: "y" });
}

// Dimension checking
if col.len() != n_obs {
    return Err(StatsError::DimensionMismatch {
        y_len: n_obs,
        x_rows: col.len(),
    });
}

// NaN/Infinity filtering with error on complete failure
let valid_indices: Vec<usize> = (0..n_obs)
    .filter(|&i| {
        !y[i].is_nan() && !y[i].is_infinite()
            && x.iter().all(|col| !col[i].is_nan() && !col[i].is_infinite())
    })
    .collect();
if valid_indices.is_empty() {
    return Err(StatsError::NoValidData);
}
```

## Logging

**Framework:** No logging framework detected. Code uses standard Rust error propagation via `Result` types.

**Patterns:**
- Errors are propagated as `StatsError` with descriptive messages
- FFI layer converts errors to `AnofoxError` struct with error code and message field (`message: [c_char; 256]`)
- No debug logging, println!, or log crate usage observed

## Comments

**When to Comment:**
- Module-level documentation comments (lines starting with `//!`) explain overall purpose at module tops
- Function documentation comments (`///`) used for all public functions, including:
  - Summary line describing function purpose
  - `# Arguments` section with parameter descriptions
  - `# Returns` section with output description
  - Common addition: `# Safety` section for `unsafe extern "C"` FFI functions

**JSDoc/TSDoc:**
- Not applicable (Rust uses `///` doc comments)
- Doc comments converted to rustdoc by `cargo doc`
- Example from `src/errors.rs`:
```rust
/// Map upstream regression errors onto this crate's error type.
///
/// Call sites previously did `.map_err(|e| StatsError::RegressError(format!("{:?}", e)))`,
/// which collapsed every upstream variant into a `Debug` string and lost the
/// structure. Cases with a faithful local counterpart are translated; the rest keep
/// the string form.
impl From<anofox_regression::solvers::RegressionError> for StatsError {
```

## Function Design

**Size:** No explicit maximum observed; functions range from 10-150 lines depending on algorithm complexity. Utility functions like `filter_nan()` are brief (1-3 lines).

**Parameters:**
- Prefer `&[f64]` for input arrays (read-only slice reference)
- Options passed as references to dedicated options structs: `&OlsOptions`, `&TTestOptions`
- Use of separate Options structs for optional/configuration parameters (rather than function-level `Option<T>` parameters)
- Example from `src/models/ols.rs`:
```rust
pub fn fit_ols(y: &[f64], x: &[Vec<f64>], options: &OlsOptions) -> StatsResult<FitResult>
```

**Return Values:**
- Always return `StatsResult<T>` for fallible operations
- Return `Vec<f64>` for coefficient/diagnostic arrays
- Return structured result types (FitResult, TestResult, AnovaResult) containing all output fields
- Never return bare `Option<T>`; use `Result<T, E>` instead

## Module Design

**Exports:**
- Selective re-exports via `pub use` in `mod.rs` files
- Example from `crates/anofox-stats-core/src/models/mod.rs`:
```rust
pub use aft::{fit_aft, AftFitResult, AftInference, AftOptions, AftResult};
pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use diagnostics::{compute_aic, compute_bic, compute_residuals, compute_vif, jarque_bera};
```

**Barrel Files:**
- `lib.rs` uses barrel pattern to expose entire module hierarchy:
```rust
pub mod diagnostics;
pub mod errors;
pub mod models;
pub mod tests;
pub mod types;

pub use errors::{StatsError, StatsResult};
pub use types::*;
```

**Visibility:**
- Private functions (without `pub`) for internal utilities like `convert_solver()`, `filter_nan()`, `convert_error()`
- Public functions/types prefixed with `pub` for FFI and library users
- FFI layer has separate private converters (`fn convert_solver_ffi()`, `fn error_to_code()`)

## NULL/NaN Handling

**Patterns:**
- Invalid floating point values are treated as NULL:
  - NaN: `x.is_nan()` check
  - Infinity: `x.is_infinite()` check
- NULL values in FFI context use bit-packed validity mask in `DataArray.validity` field
- Filtering occurs upfront:
```rust
let valid_indices: Vec<usize> = (0..n_obs)
    .filter(|&i| {
        !y[i].is_nan() && !y[i].is_infinite()
            && x.iter().all(|col| !col[i].is_nan() && !col[i].is_infinite())
    })
    .collect();
```
- Missing data completely excluded from computation (complete-case analysis)

## FFI Conventions

**Safety Invariants:**
- All FFI functions use `unsafe extern "C"` signature
- Caller responsibility documented in `# Safety` section:
  - "X must be a valid pointer to Y"
  - "X must point to N valid Z structs"
  - Array bounds and lifetime requirements
- Validation gates placed at FFI boundary to check preconditions
- Panic catching via `std::panic::catch_unwind(std::panic::AssertUnwindSafe(...))` wraps Rust calls in FFI

**Memory Management:**
- FFI functions allocate output arrays via `libc::malloc()`
- Caller responsible for freeing via `anofox_free_result_core()` and `anofox_free_result_inference()` functions
- Example from `src/lib.rs`:
```rust
#[no_mangle]
pub unsafe extern "C" fn anofox_free_result_core(result: *mut FitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
    }
}
```

---

*Convention analysis: 2026-08-11*
