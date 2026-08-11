# Testing Patterns

**Analysis Date:** 2026-08-11

## Test Framework

**Runner:**
- Cargo test framework (built-in to Rust)
- No explicit `Cargo.toml` config section for test runner (uses defaults)
- Run tests with `cargo test` in any crate directory

**Assertion Library:**
- Standard Rust assertions (`assert!`, `assert_eq!`, `assert_ne!`)
- `approx` crate v0.5 for floating-point comparisons (listed in `[dev-dependencies]` of `crates/anofox-stats-core/Cargo.toml`)

**Run Commands:**
```bash
# Run all tests in workspace
cargo test

# Run tests in specific crate
cargo test -p anofox-stats-core
cargo test -p anofox_stats_ffi

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Watch mode (requires cargo-watch)
cargo watch -x test
```

## Test File Organization

**Location:**
- Tests are co-located with source code in `src/tests/` directory
- Structure: `crates/anofox-stats-core/src/tests/` contains subdirectory per test category
- Each statistical test domain has its own module: `categorical.rs`, `correlation.rs`, `distributional.rs`, `equivalence.rs`, `forecast.rs`, `modern.rs`, `nonparametric.rs`, `parametric.rs`, `resampling.rs`

**Naming:**
- Test files named by domain: `parametric.rs` (contains parametric tests like t-tests, ANOVA)
- Modules exported and organized in `mod.rs` file
- No separate `tests/` directory at crate root; all tests within `src/tests/`

**Structure:**
```
crates/anofox-stats-core/src/
├── tests/
│   ├── mod.rs              # Module organization, shared test utilities
│   ├── categorical.rs      # Chi-square, Fisher's exact, proportions
│   ├── parametric.rs       # t-tests, ANOVA, Yuen test
│   ├── nonparametric.rs    # Mann-Whitney, Kruskal-Wallis, etc.
│   ├── correlation.rs      # Correlation tests
│   ├── distributional.rs   # Goodness-of-fit tests
│   ├── equivalence.rs      # TOST (Two One-Sided Tests)
│   ├── forecast.rs         # Diebold-Mariano, Clark-West
│   ├── modern.rs           # Modern hypothesis tests
│   └── resampling.rs       # Bootstrap, permutation tests
```

## Test Structure

**Suite Organization:**
Tests are organized as public functions returning `StatsResult<TestResult>` or similar. Tests are **not unit tests** in the traditional sense; rather, they are statistical test **implementations** that wrap the `anofox-tests` crate.

Example from `src/tests/parametric.rs`:
```rust
/// Two-sample t-test
///
/// Performs Welch's t-test (default), Student's t-test, or paired t-test.
///
/// # Arguments
/// * `group1` - First sample data
/// * `group2` - Second sample data
/// * `options` - Test options
///
/// # Returns
/// Test result with t-statistic, p-value, df, and CI
pub fn t_test(group1: &[f64], group2: &[f64], options: &TTestOptions) -> StatsResult<TestResult> {
    let g1 = filter_nan(group1);
    let g2 = filter_nan(group2);

    if g1.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "t-test requires at least 2 observations in group 1".into(),
        ));
    }
    if g2.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "t-test requires at least 2 observations in group 2".into(),
        ));
    }

    let result = lib_t_test(
        &g1,
        &g2,
        options.kind,
        options.alternative,
        options.mu,
        options.confidence_level,
    )
    .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        effect_size: f64::NAN,
        ci_lower: result.conf_int.as_ref().map(|ci| ci.lower).unwrap_or(f64::NAN),
        ci_upper: result.conf_int.as_ref().map(|ci| ci.upper).unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: g1.len() + g2.len(),
        n1: g1.len(),
        n2: g2.len(),
        alternative: options.alternative,
        method: format!("{:?} t-test", options.kind),
    })
}
```

**Patterns:**

1. **Options Pattern**: Each statistical test has an associated `Options` struct with reasonable defaults:
```rust
#[derive(Debug, Clone)]
pub struct TTestOptions {
    pub alternative: Alternative,
    pub kind: TTestKind,
    pub confidence_level: Option<f64>,
    pub mu: f64,
}

impl Default for TTestOptions {
    fn default() -> Self {
        Self {
            alternative: Alternative::TwoSided,
            kind: TTestKind::Welch,
            confidence_level: Some(0.95),
            mu: 0.0,
        }
    }
}
```

2. **Input Validation**: Early validation at function entry with informative error messages:
```rust
if g1.len() < 2 {
    return Err(StatsError::InsufficientDataMsg(
        "t-test requires at least 2 observations in group 1".into(),
    ));
}
```

3. **NaN Filtering**: Always filter NaN values before computation:
```rust
let g1 = filter_nan(group1);
let g2 = filter_nan(group2);
```

4. **Error Wrapping**: Wrap upstream library errors via helper functions:
```rust
let result = lib_t_test(...).map_err(convert_error)?;
```

5. **Result Mapping**: Convert library result types to uniform `TestResult` struct:
```rust
Ok(TestResult {
    statistic: result.statistic,
    p_value: result.p_value,
    // ... populate all fields
})
```

## Mocking

**Framework:** Not applicable for this codebase.

**Approach:** No mocking detected. Tests work with real data:
- Tests use actual floating-point arrays passed as function arguments
- Tests call real underlying implementations (anofox-tests, anofox-regression)
- No stub or mock objects
- Error conditions tested via explicit input validation (e.g., empty arrays, dimension mismatches)

**What to Mock:**
- Not needed. The statistical library design passes data directly; no external dependencies requiring mocking.

**What NOT to Mock:**
- Upstream statistical libraries (anofox-tests, anofox-regression) should NOT be mocked; they are critical to correctness and tested via integration.

## Fixtures and Factories

**Test Data:**
No explicit fixture factory pattern observed. Tests use inline data passed as arguments:

Example from domain patterns - options struct acts as configuration fixture:
```rust
let options = TTestOptions {
    alternative: Alternative::TwoSided,
    kind: TTestKind::Welch,
    confidence_level: Some(0.95),
    mu: 0.0,
};
```

Shared helper from `src/tests/mod.rs`:
```rust
/// Filter NaN values from a slice
fn filter_nan(data: &[f64]) -> Vec<f64> {
    data.iter().copied().filter(|x| !x.is_nan()).collect()
}
```

**Location:**
- Utility functions (`filter_nan`, `convert_error`) in `crates/anofox-stats-core/src/tests/mod.rs`
- Options structs (configuration fixtures) defined in each test module where used
- Result mapping helpers (generic `TestResult`, `AnovaResult`, `CorrelationResult`) defined in `mod.rs`

## Coverage

**Requirements:** None enforced. No coverage tooling or threshold configured in repo.

**View Coverage:**
```bash
# Generate coverage report (requires cargo-tarpaulin or similar)
cargo tarpaulin

# With output formatting
cargo tarpaulin --out Html
```

**Current State:**
- Core algorithms and error paths are well-tested via integration with upstream libraries
- FFI boundary testing relies on downstream usage in C++ DuckDB extension
- No explicit unit test suite found (tests are statistical test implementations, not unit tests)

## Test Types

**Unit Tests:**
- Not explicitly present as standalone `#[test]` functions
- Statistical test functions themselves ARE the tests (public functions that implement tests)
- Testing is integration-based: functions call upstream libraries and verify result structure

**Integration Tests:**
- Implicit via test implementations in `src/tests/` modules
- Each statistical test function validates:
  - Input constraints (array lengths, value ranges)
  - Error propagation from upstream libraries
  - Result structure correctness
  - Edge cases (empty data, NaN handling, dimension mismatches)

**E2E Tests:**
- Not found in this repository
- DuckDB extension integration tests likely exist in downstream duckdb repository

## Common Patterns

**Async Testing:**
Not applicable (no async code in this codebase).

**Error Testing:**
Error conditions tested via argument validation patterns:

```rust
// Test insufficient data
if g1.len() < 2 {
    return Err(StatsError::InsufficientDataMsg(
        "t-test requires at least 2 observations in group 1".into(),
    ));
}

// Test dimension mismatch
if observed.len() != expected.len() {
    return Err(StatsError::DimensionMismatchMsg(
        "Observed and expected must have same length".into(),
    ));
}

// Test empty input
if table.is_empty() {
    return Err(StatsError::InvalidInput("Empty contingency table".into()));
}
```

**NaN/Infinity Handling Tests:**
Implicit via `filter_nan` and validation logic:

```rust
let valid_indices: Vec<usize> = (0..n_obs)
    .filter(|&i| {
        !y[i].is_nan() && !y[i].is_infinite()
            && x.iter()
                .all(|col| !col[i].is_nan() && !col[i].is_infinite())
    })
    .collect();

if valid_indices.is_empty() {
    return Err(StatsError::NoValidData);
}
```

## Floating-Point Comparison

**Strategy:** Uses `approx` crate (v0.5) for approximate equality:

From `Cargo.toml` dev-dependencies:
```toml
[dev-dependencies]
approx = "0.5"
```

**Pattern:**
```rust
// Example pattern (not shown in provided snippets, but standard for numerical code)
use approx::assert_abs_diff_eq;

assert_abs_diff_eq!(result.r_squared, expected_r_squared, epsilon = 1e-10);
```

## CI/CD Testing

**CI Pipeline:** Tests run as part of GitHub Actions workflows:

From `.github/workflows/ExtensionTemplate.yml`:
```yaml
- name: Test
  run: |
    make test
```

Standard testing likely via `make test` target in Makefile.

## No Direct Test Functions

**Important Note:** This codebase does NOT contain traditional Rust unit tests with `#[test]` attributes. Instead:

1. The **statistical test functions** in `src/tests/` modules ARE the tests (e.g., `t_test()`, `chisq_test()`, `diebold_mariano()`)
2. These functions are **library APIs** that implement statistical hypothesis tests
3. **Testing happens at the DuckDB extension layer** (downstream) where these functions are called
4. **Error testing** is implicit via input validation and early returns

This is a **statistical testing library**, not a test suite for other code. The organization reflects this: `src/tests/` is the public API for hypothesis testing, not internal test fixtures.

---

*Testing analysis: 2026-08-11*
