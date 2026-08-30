# Requirements: Anofox Statistics — v0.2.0 (WASM Support)

**Defined:** 2026-08-30
**Core Value:** Users can run rigorous statistical models in plain SQL wherever DuckDB runs, including the browser (DuckDB-Wasm).

## v0.2.0 Requirements

Requirements for the WASM Support milestone. Each maps to a roadmap phase.

### Build (WASM)

- [ ] **WASM-01**: The Rust FFI crate (`anofox_stats_ffi`) compiles for the `wasm32-unknown-emscripten` target
- [ ] **WASM-02**: The Rust FFI static archive is linked into the final `.wasm` so no `anofox_*` FFI symbols remain unresolved imports
- [ ] **WASM-03**: Telemetry (raw HTTP + OpenSSL) is compiled out on Emscripten so it breaks neither the WASM build nor the load path
- [ ] **WASM-04**: The extension builds green for all WASM archs (wasm_mvp, wasm_eh, wasm_threads) for both shipped DuckDB versions (v1.5.5, v1.4.5 LTS) in CI

### Runtime / Load (LOAD)

- [ ] **LOAD-01**: The extension `LOAD`s without error in DuckDB-Wasm under Node
- [ ] **LOAD-02**: Core statistical functions (representative regression, aggregate, and test functions) return correct results under DuckDB-Wasm

### Testing & CI (TEST)

- [ ] **TEST-01**: A Node-based harness loads the built `.wasm` via `@duckdb/duckdb-wasm` and executes the `test/sql/*.test` suite (or a WASM-appropriate subset)
- [ ] **TEST-02**: The WASM harness runs in CI on every push/PR and fails the build on any WASM load or runtime error
- [ ] **TEST-03**: Running the WASM tests locally is documented (README / CONTRIBUTING)

## Future Requirements

Deferred to a later milestone. Tracked but not in this roadmap.

### WASM ergonomics

- **WASMX-01**: Publish/verify the extension entry in the DuckDB community-extensions catalog descriptor for WASM
- **WASMX-02**: Size/perf tuning of the `.wasm` artifact (LTO, dead-code, thread flags)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Telemetry on WASM | Raw HTTP/socket + OpenSSL has no viable WASM path; compiled out |
| Windows distribution | Never shipped; mingw/MSVC CI legs are excluded for unrelated reasons |
| Bespoke browser/JS UI | Milestone verifies the extension in DuckDB-Wasm, not a web frontend |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| WASM-01 | Phase 1 | Implemented (CI-gate pending) |
| WASM-02 | Phase 1 | Implemented (CI-gate pending) |
| WASM-03 | Phase 1 | Implemented (CI-gate pending) |
| WASM-04 | Phase 1 | Implemented (CI-gate pending) |
| LOAD-01 | Phase 2 | Implemented (CI-gate pending) |
| LOAD-02 | Phase 2 | Implemented (CI-gate pending) |
| TEST-01 | Phase 3 | Implemented (CI-gate pending) |
| TEST-02 | Phase 3 | Implemented (CI-gate pending) |
| TEST-03 | Phase 3 | Implemented (CI-gate pending) |

**Coverage:**
- v0.2.0 requirements: 9 total
- Mapped to phases: 9 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-30*
*Last updated: 2026-08-30 after roadmap creation (3 phases, full coverage)*
