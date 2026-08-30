# Roadmap: Anofox Statistics — v0.2.0 (WASM Support)

## Overview

This milestone takes the native statistics extension and makes it work in
DuckDB-Wasm. The journey has three legs: first make the WASM artifact build
correctly and green in CI (Rust-for-emscripten compilation plus the already-applied
`LINKED_LIBS` and telemetry-compile-out fixes); then prove the built `.wasm`
actually loads and computes correct results in DuckDB-Wasm under Node; finally
lock that verification in with an automated Node harness that runs the SQL suite,
gates every push/PR in CI, and is documented for local use.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: WASM Build Green** - Rust compiles for emscripten and the extension builds for all WASM archs in CI *(implemented; CI-gate verification pending push)*
- [ ] **Phase 2: Load & Runtime Correctness** - The built `.wasm` loads in DuckDB-Wasm and core functions return correct results
- [ ] **Phase 3: Automated Harness & CI Gate** - Node harness runs the SQL suite, gates CI on every push/PR, and is documented

## Phase Details

### Phase 1: WASM Build Green
**Goal**: The extension compiles and links cleanly into a `.wasm` artifact for every WASM arch on both shipped DuckDB versions, with no unresolved FFI symbols and no telemetry/OpenSSL dependency in the WASM code path.
**Depends on**: Nothing (first phase)
**Requirements**: WASM-01, WASM-02, WASM-03, WASM-04
**Success Criteria** (what must be TRUE):
  1. The Rust FFI crate (`anofox_stats_ffi`) and its dependency graph (faer, statrs, argmin fork, getrandom) compile for `wasm32-unknown-emscripten`.
  2. The final `.wasm` contains the Rust FFI archive with zero unresolved `anofox_*` imports (`LINKED_LIBS` fix confirmed effective).
  3. No telemetry / raw-HTTP / OpenSSL symbols are compiled into the Emscripten build (guard `if(NOT MINGW AND NOT EMSCRIPTEN)` and `!wasm32` openssl exclusion confirmed).
  4. The CI matrix builds green for wasm_mvp, wasm_eh, and wasm_threads for both DuckDB v1.5.5 and v1.4.5 LTS.
**Plans**: TBD

Notes:
- The main technical risk lives here: whether the Rust dependency graph actually compiles for `wasm32-unknown-emscripten` (WASM-01). Sequence this first; expect possible feature/config adjustments (e.g. getrandom backend, faer/argmin features).
- Two fixes are already applied in the working tree — this phase confirms them in CI, it does not re-discover them:
  - `LINKED_LIBS "$<TARGET_FILE:anofox_stats_ffi-static>"` in `extension_config.cmake` (#103) → WASM-02.
  - Telemetry compiled out on Emscripten in `CMakeLists.txt`, `openssl` made `"!wasm32"` in `vcpkg.json` → WASM-03.
- WASM archs are already in the build matrix via `.github/workflows/MainDistributionPipeline.yml` → `_extension_distribution.yml` (only Windows archs excluded) → WASM-04.

### Phase 2: Load & Runtime Correctness
**Goal**: The `.wasm` produced by Phase 1 loads without error in DuckDB-Wasm under Node, and representative statistical functions return correct results — proving the extension is functional in-browser-runtime, not just buildable.
**Depends on**: Phase 1
**Requirements**: LOAD-01, LOAD-02
**Success Criteria** (what must be TRUE):
  1. A locally-built `.wasm` `LOAD`s in DuckDB-Wasm under Node with no load-time error or unresolved symbol.
  2. A representative regression function (e.g. `ols_fit_agg`) returns coefficients matching the native result within tolerance.
  3. A representative aggregate/test function (e.g. a hypothesis test) returns correct statistics under DuckDB-Wasm.
**Plans**: TBD

Notes:
- Verification is manual/smoke-level here (interactive Node load + spot-checked results). Phase 3 turns this into an automated, CI-gated harness over the full SQL suite.
- Uses `@duckdb/duckdb-wasm` in Node, which has a real filesystem so it installs/loads like production — this catches load/runtime failures that the compile+link CI matrix (Phase 1) cannot.

### Phase 3: Automated Harness & CI Gate
**Goal**: WASM verification is a repeatable, automated gate — a Node harness loads the built `.wasm` via `@duckdb/duckdb-wasm` and runs the SQL test suite, wired into CI to fail the build on any WASM load or runtime error, with local usage documented.
**Depends on**: Phase 2
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. A Node harness (e.g. under `test/wasm/`) loads the built `.wasm` via `@duckdb/duckdb-wasm` and executes the `test/sql/*.test` suite (or a WASM-appropriate subset).
  2. A CI job runs the harness on every push/PR and fails the build on any WASM load or runtime error.
  3. Running the WASM tests locally is documented in README/CONTRIBUTING and the documented steps work.
**Plans**: TBD

Notes:
- This is net-new work (the query.farm "Testing DuckDB WASM Extensions" approach): a `test/wasm/` Node project installing `@duckdb/duckdb-wasm`, a runner over `test/sql`, a new CI job wiring it in, and local docs.
- Builds directly on the manual load/run recipe validated in Phase 2.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. WASM Build Green | 1/1 | Implemented (CI-gate pending) | 2026-08-30 |
| 2. Load & Runtime Correctness | 0/TBD | Not started | - |
| 3. Automated Harness & CI Gate | 0/TBD | Not started | - |
