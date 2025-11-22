# libanostat - Core Statistical Regression Library

A standalone C++ library providing statistical regression algorithms independent from DuckDB.

## Architecture

This library follows the separation of concerns pattern established by `anofox-forecast/anofox-time`:

- **Core Library (`libanostat/`)**: Pure C++ statistical algorithms with only Eigen3 dependency
- **Extension Layer (`src/`)**: DuckDB bindings and integration
- **Bridge Layer (`src/bridge/`)**: Type conversion between DuckDB and Eigen

## Features

- Ordinary Least Squares (OLS) regression
- Ridge regression (L2 regularization)
- Elastic Net regression (L1 + L2 regularization)
- Weighted Least Squares (WLS)
- Recursive Least Squares (RLS)
- Statistical inference (p-values, confidence intervals, prediction intervals)
- Model diagnostics (VIF, leverage, Cook's D, residual analysis)
- Statistical distributions (Student's t, chi-squared)

## Building Standalone

```bash
cd libanostat
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build
ctest --test-dir build --output-on-failure
```

## Dependencies

- **Eigen3**: Linear algebra operations
- **Catch2**: Unit testing framework (test-only)

## Design Principles

1. **Zero DuckDB Dependencies**: Library code has no knowledge of DuckDB
2. **Header-Only Where Appropriate**: Small utilities and distributions
3. **Simple and Clear**: No over-engineering, straightforward interfaces
4. **Testable**: C++ unit tests independent from SQL integration tests
5. **Reusable**: Can be used in Python bindings, standalone apps, etc.

## Testing Strategy

- **Library Tests** (`libanostat/tests/`): Algorithm correctness using Catch2
- **Extension Tests** (`test/sql/`): SQL integration tests using DuckDB .test files

This separation enables clear identification of bugs: statistical algorithm vs DuckDB integration.

## Status

This library is being extracted from the monolithic anofox_statistics extension.
See refactoring plan in root directory documentation.
