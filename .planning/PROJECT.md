# Anofox Statistics

## What This Is

A DuckDB extension providing statistical analysis directly in SQL — regression
(OLS, Ridge, Elastic Net, WLS, Huber, RANSAC, Theil-Sen, RLS), GLMs (Poisson,
logistic, gamma, negative binomial, Tweedie), mixed-effects (GLMM), survival
(AFT), 40+ hypothesis tests, and model diagnostics (VIF, AIC/BIC, Jarque-Bera),
exposed as aggregate, table, scalar, and window functions. Built as a C++
DuckDB extension over a Rust C-FFI core (`anofox-stats-ffi` → `anofox-stats-core`).
It ships to the DuckDB community-extensions catalog for native platforms and
DuckDB-Wasm.

## Core Value

Users can run rigorous statistical models on their data in-process, in plain
SQL, wherever DuckDB runs — **including the browser (DuckDB-Wasm)**.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Native statistical function suite (regression, GLM, GLMM, AFT, tests, diagnostics) — v0.1.0
- ✓ Native distribution for linux/osx (amd64+arm64) via extension-ci-tools — v0.1.0
- ✓ Rust FFI static archive linked into WASM builds (`LINKED_LIBS`) — #103
- ✓ Extension builds green for all WASM archs (wasm_mvp/eh/threads) on v1.5.5 + v1.4.5 — v0.2.0
- ✓ Extension loads without error in DuckDB-Wasm — v0.2.0
- ✓ Statistical functions return correct results under DuckDB-Wasm (full SQL suite, 2095/2095) — v0.2.0
- ✓ Automated Node WASM harness runs the SQL suite and gates CI + status badge — v0.2.0
- ✓ Repeatable benchmark suite over representative workloads (`scripts/bench.sh` + 3 workloads, diffable results) — Phase 4
- ✓ Hotspots profiled and optimized-or-documented-as-inherent, with before/after numbers (`bench/PROFILING.md`) — Phase 4
- ✓ FFI manual malloc/free pattern refactored (`FfiVec` RAII + `alloc_inference_arrays!` macro; C `free` contract preserved; suites green) — Phase 4
- ✓ Clear, actionable typed error messages + early bind-time input validation (`ThrowFromFfiError`; unknown-option rejection; degenerate-frame NULL guards) — v0.3.0
- ✓ Consistent unprefixed signatures / snake_case option keys / return-struct fields across function families (`docs/API_CONVENTIONS.md`; breaking rename, no aliases) — v0.3.0
- ✓ README restructured to anofox-forecast form (emoji sections, ToC, ⚡/🎨 Key Features, Quick Start, structured API ref) — v0.3.0
- ✓ Every documented SQL example (README + guides + docs) validated against the built extension in CI (`scripts/validate_docs_sql.py` + `DocsSqlValidation.yml`) — v0.3.0

### Active

<!-- Next milestone: TBD. v0.3.0 (Performance & Polish) shipped. -->

(None — v0.3.0 shipped. Candidate next-milestone items in tech-debt: named parameters `param := value` (ERGOX-01), scalar `ols_fit` insufficient-rows fast-fail, the skip-marked `ols_fit_agg OVER()` window edge.)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Windows distribution (amd64 / mingw) — never shipped; mingw 404s on a deleted MSYS2 libtool package and MSVC fmt is flaky (see CI comments)
- PostHog telemetry on WASM — raw HTTP/socket + OpenSSL has no viable WASM path; compiled out
- Browser-UI / JS API surface beyond loading the `.wasm` — this milestone verifies the extension in DuckDB-Wasm, not a bespoke web frontend

## Context

- Three-layer architecture: C++ DuckDB adapters (`src/`) → C FFI (`crates/anofox-stats-ffi`) → Rust core (`crates/anofox-stats-core`). See `.planning/codebase/`.
- WASM build path: `extension_config.cmake` declares the extension to DuckDB's
  build; for Emscripten the loadable target is re-linked by a post-build `emcc`
  step (`duckdb/extension/extension_build_tools.cmake`) that only pulls archives
  named in `DUCKDB_EXTENSION_ANOFOX_STATISTICS_LINKED_LIBS`.
- Known WASM failure modes (per query.farm "Testing DuckDB WASM Extensions"):
  (1) static/Rust libs dropped unless in `LINKED_LIBS` — fixed in #103;
  (2) direct filesystem access; (3) raw HTTP/sockets — hit us via telemetry;
  (4) deps without WASM support.
- Rust core is pure computation (faer/statrs/argmin) with no filesystem or
  network I/O, so failure modes (2)/(3) do not apply on the Rust side.

## Constraints

- **Tech stack**: DuckDB extension C++17 + Rust 2021 FFI; built via `duckdb/extension-ci-tools` (v1.5-variegata for DuckDB v1.5.5, v1.4-andium for LTS v1.4.5).
- **Compatibility**: Must load on DuckDB-Wasm builds (wasm_mvp, wasm_eh, wasm_threads) matching the shipped DuckDB versions.
- **Dependencies**: Rust deps must compile for `wasm32-unknown-emscripten` (faer, statrs, statrs, argmin fork, getrandom).
- **No raw HTTP/filesystem** in the WASM code path — route through DuckDB abstractions or compile out.

## Key Decisions

<!-- Decisions that constrain future work. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Link Rust FFI archive via `LINKED_LIBS` in `extension_config.cmake` | WASM post-build emcc link ignores `target_link_libraries`; symbols else unresolved | ✓ Good (#103) |
| Disable telemetry on Emscripten (like MinGW) | `CaptureExtensionLoad` makes a raw HTTPS socket call at load; WASM has no sockets → load throws | ✓ Good — CI-verified, extension loads in DuckDB-Wasm (v0.2.0) |
| Verify WASM via Node harness running `test/sql` | Compile+link (ci-tools) can't catch load/runtime failures; Node has a real FS so it installs/loads like production | ✓ Good — 2095/2095 assertions pass; gating CI job + badge (v0.2.0) |
| Pin `@duckdb/duckdb-wasm` to engine v1.5.5 (dev build `1.33.1-dev64.0`) | Extensions are ABI-locked to the engine version; only this dev build bundles v1.5.5 | ⚠️ Revisit — dev-tag dependency; bump when a stable duckdb-wasm ships 1.5.x |
| Format harness results via DuckDB `::VARCHAR` | duckdb-wasm Arrow-JS renders DECIMAL unscaled (1.0→10); `::VARCHAR` applies scale, matching native | ✓ Good (v0.2.0) |
| FFI result arrays stay `libc::malloc`-backed (`FfiVec`), never `Box`/`Vec` | C++ frees them with C `free()`; Rust's global allocator can differ from libc malloc on musl (WASM/CI) → UB | ✓ Good — Phase 4; `ffi_vec_ptr_is_freeable_by_libc` guards it |
| Convert only the 6 strict inference sites to the macro; leave 6 GLM + 1 ALM hand-written | GLM maps `z_values`→t_values with lenient OOM; ALM uses different field names — forcing the strict macro would change behavior | ✓ Good — Phase 4; documented, suites green |
| Breaking cross-family rename with NO deprecated aliases (drop `anofox_stats_` prefix, `theilsen`→`theil_sen`) | Early-dev; one clean convention beats carrying dual names; `ergo03_naming.test` locks old names as errors | ✓ Good — v0.3.0; all suites + docs green against new names |
| Route FFI errors through `ThrowFromFfiError` dispatching on `AnofoxError.code`; use `InternalException` for numerical failures | `FunctionException` is absent in the embedded DuckDB build; intent (distinct class for computational vs user-data errors) preserved | ✓ Good — v0.3.0 |
| Validate documented SQL by running fenced blocks against the built extension in CI (` ```sql skip ` for illustrative blocks) | Docs drift silently as the API changes; a hard CI gate makes drift fail the build | ✓ Good — v0.3.0; `DocsSqlValidation.yml`, 50 blocks green |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

> **Resolved in v0.3.0 (ERGO-01):** the degenerate sub-`(n_features+1)` frame that returned a
> saturated/NaN result now returns NULL via the scaled `min_obs` guard across all fit_predict
> finalize paths (incl. BLS/NNLS). A separate `ols_fit_agg OVER(...)` window edge remains
> skip-marked in guides and is tracked as tech debt in the v0.3.0 milestone audit.

---
*Last updated: 2026-09-02 after v0.3.0 (Performance & Polish) milestone*
