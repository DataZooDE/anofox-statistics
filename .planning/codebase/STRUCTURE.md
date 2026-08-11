# Codebase Structure

**Analysis Date:** 2026-08-11

## Directory Layout

```
anofox-statistics/
├── src/                              # C++ DuckDB extension code
│   ├── anofox_statistics_extension.cpp  # Entry point: function registration
│   ├── include/
│   │   ├── anofox_statistics_extension.hpp  # Forward declarations
│   │   ├── anofox_stats_ffi.h          # Generated C FFI header
│   │   ├── map_options_parser.hpp      # Parse MAP options to structs
│   │   ├── glm_prior_options.hpp       # Bayesian priors for GLM
│   │   └── ffi_enum_converters.hpp     # Convert between C++/Rust enums
│   ├── aggregate_functions/            # 45+ aggregate function implementations
│   │   ├── ols_aggregate.cpp           # OLS (Group BY support)
│   │   ├── poisson_aggregate.cpp       # Poisson GLM
│   │   ├── aft_aggregate.cpp           # Survival models (AFT)
│   │   ├── glmm_aggregate.cpp          # Mixed effects models
│   │   ├── shapiro_wilk_aggregate.cpp  # Normality test
│   │   ├── t_test_aggregate.cpp        # Parametric t-test
│   │   └── ... (40+ more)
│   ├── table_functions/                # Scalar fit functions (single call)
│   │   ├── ols_fit.cpp                 # Array input → model result struct
│   │   ├── ridge_fit.cpp
│   │   ├── wls_fit.cpp
│   │   ├── huber_fit.cpp
│   │   ├── ransac_fit.cpp
│   │   ├── theil_sen_fit.cpp
│   │   ├── elasticnet_fit.cpp
│   │   ├── rls_fit.cpp
│   │   └── predict.cpp                 # Generic prediction from model struct
│   ├── window_functions/                # Rolling/expanding fit+predict
│   │   ├── ols_fit_predict.cpp         # Window aggregate for OLS fit+predict
│   │   ├── ridge_fit_predict.cpp
│   │   ├── wls_fit_predict.cpp
│   │   ├── huber_fit_predict.cpp
│   │   ├── ransac_fit_predict.cpp
│   │   ├── theil_sen_fit_predict.cpp
│   │   ├── elasticnet_fit_predict.cpp
│   │   └── rls_fit_predict.cpp
│   ├── scalar_functions/                # Diagnostics and info criteria
│   │   ├── vif.cpp                     # Variance Inflation Factor
│   │   ├── aic_bic.cpp                 # Information criteria
│   │   ├── jarque_bera.cpp             # Normality test
│   │   └── residuals_diagnostics.cpp   # Residual analysis
│   └── macros/
│       └── fit_predict_macros.cpp      # SQL table macros: ols_fit_predict_by, etc.
│
├── crates/                             # Rust workspace
│   ├── anofox-stats-core/              # Pure Rust statistics (internal)
│   │   ├── src/
│   │   ├── Cargo.toml
│   │   └── [Does not exist in this repo - external dependency]
│   └── anofox-stats-ffi/               # C FFI boundary (Rust ↔ C++)
│       ├── src/
│       │   ├── lib.rs                  # 50+ FFI extern "C" functions
│       │   └── types.rs                # C-compatible type definitions
│       ├── Cargo.toml
│       └── cbindgen.toml               # Config to generate anofox_stats_ffi.h
│
├── posthog-telemetry/                  # Optional telemetry (PostHog integration)
│   ├── src/telemetry.cpp
│   ├── include/telemetry.hpp
│   └── test/
│
├── test/                               # Test suite (SQL-based)
│   ├── sql/                            # Integration tests
│   │   ├── aggregate_basic_tests.sql   # OLS, WLS, Ridge, RLS aggregate tests
│   │   ├── aggregate_integration_tests.sql
│   │   ├── guide01_example_*.sql       # Quick-start guide examples
│   │   └── ... (40+ more)
│   ├── integration/                    # End-to-end guide validation
│   │   ├── test_all_guide_examples.sql
│   │   ├── test_technical_guide.sql
│   │   └── test_business_guide.sql
│   ├── data/                           # Test data files
│   └── rank_deficiency_*.sql           # Edge case tests
│
├── validation/                         # R-based validation scripts (legacy)
│   ├── data/
│   ├── generators/
│   └── legacy/
│
├── examples/                           # Marimo notebooks and example queries
│   ├── aid_demand_classification/
│   └── performance_1m_groups/
│
├── docs/                               # Generated API documentation
│   ├── API_REFERENCE.md
│   └── api/
│
├── guides/                             # User guides (Markdown)
│   ├── 01_quick_start.md
│   ├── 02_technical_guide.md
│   ├── 03_business_guide.md
│   ├── 04_advanced_use_cases.md
│   └── templates/
│
├── scripts/                            # Build and utility scripts
│   ├── build.sh
│   └── ... (CI/CD helpers)
│
├── extension-ci-tools/                 # Shared CI/CD tooling (git submodule)
│   ├── makefiles/
│   ├── scripts/
│   ├── docker/
│   └── toolchains/
│
├── duckdb/                             # DuckDB source code (git submodule)
│   ├── src/
│   ├── extension/
│   ├── test/
│   └── ... (full DuckDB repo)
│
├── Cargo.toml                          # Rust workspace root
├── Cargo.lock                          # Locked dependency versions
├── CMakeLists.txt                      # Build configuration (C++ → Rust via Corrosion)
├── Makefile                            # Quick build targets
├── extension_config.cmake              # DuckDB extension configuration
├── vcpkg.json                          # C++ dependencies (OpenSSL for telemetry)
└── README.md                           # Project overview
```

## Directory Purposes

**`src/`:**
- Purpose: All C++ DuckDB extension code
- Contains: Function adapters, state machines, option parsing
- Key files: `anofox_statistics_extension.cpp` (entry point), 45+ aggregate implementations

**`src/include/`:**
- Purpose: Header files and generated FFI headers
- Key files: `anofox_statistics_extension.hpp` (function declarations), `anofox_stats_ffi.h` (generated from Rust, defines C data types)

**`src/aggregate_functions/`:**
- Purpose: Aggregate function implementations (e.g., `SELECT ... GROUP BY`)
- Contains: 45 files implementing regression, GLM, and hypothesis test aggregates
- Pattern: Each file follows template:
  - `*State` struct: Accumulates row data
  - `*BindData` struct: Holds parsed options
  - `*Initialize()`, `*Update()`, `*Combine()`, `*Finalize()` callbacks

**`src/table_functions/`:**
- Purpose: Scalar fit functions taking arrays as input (not rows)
- Contains: 8 files for array-based regression fitting (OLS, Ridge, Huber, etc.)
- Pattern: Accepts `DOUBLE[]` for y and `DOUBLE[][]` for X; returns STRUCT result

**`src/window_functions/`:**
- Purpose: Rolling and expanding window regression (fit on window frame)
- Contains: 8 files for fit+predict with `OVER` clause support
- Pattern: Combines aggregate state machine with prediction in one function

**`src/scalar_functions/`:**
- Purpose: Diagnostic and statistical utility functions
- Contains: VIF, AIC/BIC, Jarque-Bera, residuals diagnostics
- Pattern: Scalar functions; no aggregation state

**`src/macros/`:**
- Purpose: SQL table macros for per-group workflows
- Contains: `fit_predict_macros.cpp` which programmatically registers 14 SQL macros
- Pattern: Generates SQL DDL at extension load time

**`crates/anofox-stats-ffi/`:**
- Purpose: C FFI boundary between C++ and Rust
- Contains: `lib.rs` (50+ extern "C" functions), `types.rs` (C data types)
- Key responsibility: Convert DuckDB Vector/DataChunk to Rust Vec, call core library, propagate errors

**`crates/anofox-stats-core/`:**
- Purpose: Pure Rust statistics implementations (external dependency)
- NOT in repo; imported via `anofox-regression` and `anofox-tests` crates
- Contains: fit_ols, fit_ridge, fit_glmm, fit_aft, hypothesis tests

**`test/`:**
- Purpose: SQL-based integration tests
- Contains: 50+ .sql files validating aggregate and window functions
- Pattern: Each test includes R validation baseline and tolerance thresholds
- Run via: `duckdb -init test/sql/aggregate_basic_tests.sql`

**`docs/`:**
- Purpose: Generated API documentation (auto-built from comments)
- Contains: API_REFERENCE.md with complete function signatures

**`guides/`:**
- Purpose: User documentation with examples
- Contains: 4 markdown guides (quick start, technical, business, advanced)
- Validated via: `test/integration/test_all_guide_examples.sql`

## Key File Locations

**Entry Points:**
- `src/anofox_statistics_extension.cpp` - Extension load and function registration
- `crates/anofox-stats-ffi/src/lib.rs` - FFI function implementations (called by C++)
- `duckdb/src/main.cpp` (DuckDB submodule) - DuckDB main entry

**Configuration:**
- `CMakeLists.txt` - C++ build (Corrosion handles Rust compilation)
- `Cargo.toml` - Rust workspace dependencies
- `extension_config.cmake` - DuckDB extension metadata (version, name)
- `vcpkg.json` - C++ deps (OpenSSL for PostHog telemetry)

**Core Logic:**
- Regression models: External `anofox-regression` crate (fit_ols, fit_ridge, ...)
- Statistical tests: External `anofox-tests` crate (t_test, mann_whitney_u, ...)
- Linear algebra: External `faer` crate (matrix decomposition)
- Distributions: External `statrs` crate (normal, t, chi-square, ...)

**Testing:**
- `test/sql/aggregate_basic_tests.sql` - Primary regression test suite
- `test/integration/test_all_guide_examples.sql` - Guide validation
- `validation/legacy/` - R baseline scripts (for historical reference)

## Naming Conventions

**Files:**
- Aggregate: `{method}_aggregate.cpp` (e.g., `ols_aggregate.cpp`, `poisson_aggregate.cpp`)
- Scalar fit: `{method}_fit.cpp` (e.g., `ridge_fit.cpp`, `huber_fit.cpp`)
- Window fit+predict: `{method}_fit_predict.cpp` (e.g., `ols_fit_predict.cpp`)
- Aggregate fit+predict: `{method}_fit_predict_aggregate.cpp` (e.g., `bls_fit_predict_aggregate.cpp`)
- Test: `{pattern}_test{s}.sql` (e.g., `aggregate_basic_tests.sql`)

**Functions (C++):**
- Registration: `Register{MethodName}AggregateFunction()` (e.g., `RegisterOlsAggregateFunction()`)
- Aggregate callbacks: `{Method}AggInitialize`, `{Method}AggUpdate`, `{Method}AggFinalize`
- Bind function: `{Method}Bind` (e.g., `OlsFitBind`)
- Struct types: `{Method}State`, `{Method}BindData` (e.g., `OlsAggregateState`)

**Functions (Rust FFI):**
- All functions: `anofox_{method}_{operation}` (e.g., `anofox_ols_fit`, `anofox_ridge_fit`)
- Converter functions: `convert_{type}_ffi` (e.g., `convert_solver_ffi`, `error_to_code`)

**SQL Functions:**
- Aggregate: `{method}_fit_agg` or `anofox_stats_{method}_fit_agg` (e.g., `ols_fit_agg`)
- Scalar: `{method}_fit` or `anofox_stats_{method}_fit` (e.g., `ols_fit`)
- Window: `{method}_fit_predict` (same as scalar when used with OVER clause)
- Macro: `{method}_fit_predict_by` (e.g., `ols_fit_predict_by`)

## Where to Add New Code

**New Regression Algorithm:**
1. **Core implementation:** Add to `anofox-regression` crate (external)
2. **FFI wrapper:** Add `anofox_new_method_fit()` function to `crates/anofox-stats-ffi/src/lib.rs` (option converters, error handling)
3. **C++ aggregate:** Create `src/aggregate_functions/new_method_aggregate.cpp` (follow `ols_aggregate.cpp` pattern)
4. **C++ scalar:** Create `src/table_functions/new_method_fit.cpp` (follow `ols_fit.cpp` pattern)
5. **C++ window:** Create `src/window_functions/new_method_fit_predict.cpp` (follow `ols_fit_predict.cpp` pattern)
6. **Registration:** Add declarations to `src/include/anofox_statistics_extension.hpp`, register in `src/anofox_statistics_extension.cpp:LoadInternal()`
7. **SQL macro:** Add entry to `fit_predict_table_macros[]` array in `src/macros/fit_predict_macros.cpp`
8. **Tests:** Add SQL tests to `test/sql/aggregate_integration_tests.sql`

**New Hypothesis Test:**
1. **Core implementation:** Add to `anofox-tests` crate (external)
2. **FFI wrapper:** Add `anofox_new_test()` function to `crates/anofox-stats-ffi/src/lib.rs`
3. **C++ aggregate:** Create `src/aggregate_functions/new_test_aggregate.cpp` (no scalar/window variants needed)
4. **Registration:** Add to `src/anofox_statistics_extension.hpp` and `LoadInternal()`
5. **Tests:** Add SQL tests to `test/sql/aggregate_integration_tests.sql`

**New Diagnostic Function:**
1. **Core implementation:** Add to statistics crate or implement in FFI if simple
2. **C++ scalar:** Create `src/scalar_functions/new_diagnostic.cpp` (e.g., `residuals_diagnostics.cpp`)
3. **Registration:** Add to `src/anofox_statistics_extension.hpp` and `LoadInternal()`
4. **Tests:** Add to `test/sql/`

**New Guide/Documentation:**
1. **Guide markdown:** Add to `guides/{number}_{topic}.md`
2. **Guide validation test:** Add to `test/integration/test_all_guide_examples.sql`
3. **Update README:** Link new guide from `README.md`

**Configuration or Build Changes:**
- CMake: `CMakeLists.txt` (C++ build config, Corrosion setup)
- Cargo: `Cargo.toml` (Rust deps, workspace config)
- vcpkg: `vcpkg.json` (C++ deps like OpenSSL)
- DuckDB config: `extension_config.cmake` (version, load logic)

## Special Directories

**`duckdb/`:**
- Purpose: DuckDB source code (git submodule at specific commit/branch)
- Generated: No
- Committed: Yes (tracked as submodule)
- Note: Full DuckDB repo; headers used at compile time, extension compiles against DuckDB API

**`build/`:**
- Purpose: CMake build output
- Generated: Yes (created by `cmake --build`)
- Committed: No (.gitignore)
- Contents: Object files, libraries, final extension `.so`/`.dll`

**`target/`:**
- Purpose: Cargo build output (Rust)
- Generated: Yes (created by `cargo build`)
- Committed: No (.gitignore)
- Contents: Rust artifacts (.rlib, .a, binaries)

**`duckdb_unittest_tempdir/`:**
- Purpose: Test temporary directories (DuckDB integration tests)
- Generated: Yes (created by test runner)
- Committed: No
- Note: Can be safely deleted

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents (this directory)
- Generated: Yes (by `/gsd-map-codebase`)
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, STACK.md, INTEGRATIONS.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

---

*Structure analysis: 2026-08-11*
