# Technology Stack

**Analysis Date:** 2026-08-11

## Languages

**Primary:**
- Rust 2021 edition - FFI boundary and core statistical computation
- C++ 17 - DuckDB extension implementation, aggregate/table/scalar/window functions
- C - FFI interface via `libc`

**Secondary:**
- Python 3.10+ - Example notebooks and testing utilities

## Runtime

**Environment:**
- DuckDB (v1.5.5 stable, v1.4.5 LTS) - Primary host for this extension
- Rust toolchain - Build-time only for compiling FFI library

**Package Manager:**
- Cargo - Rust dependency management
- CMake - Build orchestration (v3.20+)
- Corrosion v0.5.2 - Rust-CMake integration
- cbindgen 0.27 - C header generation from Rust FFI

**Lockfile:**
- `Cargo.lock` - Present; pinned to specific Rust dependency versions
- No Python lockfile (examples use loose pinning with `>=`)

## Frameworks

**Core:**
- DuckDB Extension API - Extension framework for table functions, aggregate functions, scalar functions, window functions
- Corrosion (v0.5.2) - Git integration for building Rust crates within CMake
- anofox-regression (0.5.13) - Upstream statistical regression library
- anofox-statistics (0.4.2) - Upstream statistical test implementations

**Testing:**
- DuckDB's native extension test harness - Integrated via `LOAD_TESTS` in `extension_config.cmake`
- Cargo test runner - For Rust unit tests in `crates/anofox-stats-core` and `crates/anofox-stats-ffi`
- Python pytest (examples) - For example notebook validation

**Build/Dev:**
- CMake (3.20+) - Primary build system
- Cargo (Rust 2021 edition) - Rust compilation
- GitHub Actions - CI/CD orchestration
- duckdb/extension-ci-tools (v1.5-variegata, v1.4-andium) - Distributed multi-architecture build matrix
- DuckDB stable and LTS branch builds - Matrix includes Linux (amd64, arm64), macOS (amd64, arm64), WASM
- Corrosion - Integrates Rust compilation into CMake workflow

## Key Dependencies

**Critical (Workspace-level):**
- `anofox-regression` (0.5.13) - Provides optimization algorithms for regression fitting (via argmin)
- `anofox-tests` (0.4.2) - Statistical test implementations
- `faer` (0.23) - Dense linear algebra with SIMD optimizations (rayon feature disabled; DuckDB parallelizes)
- `statrs` (0.18) - Statistical distributions and probability functions
- `thiserror` (2.0) - Structured error handling
- `libc` (0.2) - C interoperability layer

**Infrastructure:**
- `argmin` (0.11.0 from git) - Nonlinear optimization algorithms; patched custom fork for stable Rust compatibility
- `argmin-math` (0.5.1 from git) - Math traits for argmin; same patched fork
- `rand` (0.8.5) - CSPRNG for statistical sampling (via anofox-regression)
- `approx` (0.5.1) - Floating-point comparison for unit tests

**Build-Time Only:**
- `cbindgen` (0.27) - Generates C header file from Rust FFI crate

## Configuration

**Environment:**
- Platform-specific Rust targets via CMake detection:
  - Linux: `x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-gnu`
  - macOS: `x86_64-apple-darwin`, `aarch64-apple-darwin`
  - Windows: `x86_64-pc-windows-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-gnu`, `aarch64-pc-windows-msvc` (MinGW/MSVC)
  - WASM: `wasm32-unknown-emscripten`

**Build:**
- `CMakeLists.txt` (v3.20+) - Orchestrates extension build, Rust compilation via Corrosion, telemetry compilation
- `extension_config.cmake` - Registers extension with DuckDB's extension system
- `Cargo.toml` (workspace) - Defines Rust crates and workspace dependencies
- `crates/anofox-stats-core/Cargo.toml` - Core statistical library
- `crates/anofox-stats-ffi/Cargo.toml` - C FFI boundary (generates `anofox_stats_ffi-static` library)

**Release Profile:**
- LTO: Enabled
- Codegen units: 1 (for better optimization)
- Optimization level: 3 (aggressive)

## Platform Requirements

**Development:**
- CMake 3.20+
- Rust toolchain (1.70+; uses 2021 edition)
- C++ compiler (C++17 support required)
- OpenSSL development headers (for telemetry; optional on MinGW where telemetry is disabled)
- cbindgen for generating C headers from Rust FFI

**Production:**
- DuckDB v1.5.5 (stable release) or v1.4.5 (LTS)
- No external database connections required
- No cloud storage integrations required (extension manages data in-process)

**Deployment:**
- Published as precompiled extension binaries for:
  - Linux x86_64 (glibc and musl)
  - Linux ARM64 (aarch64-gnu)
  - macOS x86_64 and ARM64 (Apple Silicon)
  - WASM (DuckDB-Wasm target)
- Windows builds excluded (see CI/CD constraints below)
- Extension repository: `https://github.com/DataZooDE/anofox-statistics`

---

*Stack analysis: 2026-08-11*
