# Phase 1: WASM Build Green - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning
**Mode:** Autonomous (implement-now/verify-in-CI; local WASM toolchain unavailable)

<domain>
## Phase Boundary

The extension compiles and links cleanly into a `.wasm` artifact for every WASM
arch (wasm_mvp, wasm_eh, wasm_threads) on both shipped DuckDB versions (v1.5.5,
v1.4.5 LTS), with no unresolved FFI symbols and no telemetry/OpenSSL dependency
in the WASM code path.

Requirements: WASM-01, WASM-02, WASM-03, WASM-04.
</domain>

<decisions>
## Implementation Decisions

- **Verification gate = CI, not local.** This sandbox has no `emcc` and no
  `wasm32-unknown-emscripten` Rust target, and a full DuckDB-Wasm build is out
  of scope locally. Per user direction ("implement now, verify in CI"), the
  WASM build/link criteria are validated by the `wasm_*` legs of
  `MainDistributionPipeline.yml` → `_extension_distribution.yml`, not locally.
- **No speculative Rust changes for WASM-01.** Dependency-graph audit (below)
  shows the graph is already WASM-clean; do not add `.cargo` overrides or
  getrandom feature flags unless CI proves a concrete failure.
</decisions>

<code_context>
## Existing Code Insights

- **WASM-02 (already fixed, #103):** `extension_config.cmake` passes
  `LINKED_LIBS "$<TARGET_FILE:anofox_stats_ffi-static>"`, so the emcc post-build
  link (`duckdb/extension/extension_build_tools.cmake:196`) pulls the Rust FFI
  archive into the final `.wasm`.
- **WASM-03 (fixed this session, commit caf9079):** telemetry compiled out on
  Emscripten (`if(NOT MINGW AND NOT EMSCRIPTEN)` in `CMakeLists.txt`; defines
  `POSTHOG_TELEMETRY_DISABLED`), and `vcpkg.json` makes `openssl` a `"!wasm32"`
  dependency. Removes the raw-HTTP-at-load failure and the OpenSSL-for-Emscripten
  build.
- **WASM-01 (confirm-in-CI):** Corrosion is pointed at
  `wasm32-unknown-emscripten` by `CMakeLists.txt:49-52` when that target is
  installed (ci-tools installs it in the wasm legs). Dependency-graph audit of
  `Cargo.lock`: `web-time 1.1` (argmin's wasm `Instant` shim) is present, and
  `getrandom` 0.2/0.3 on `wasm32-unknown-emscripten` uses libc `getentropy` — no
  `js` feature required (that is a `wasm32-unknown-unknown`-only need). No
  filesystem/socket/thread-spawn usage in `crates/`. Graph is WASM-clean.
- **WASM-04 (confirm-in-CI):** the wasm archs are already IN the build matrix —
  `exclude_archs` in `MainDistributionPipeline.yml` only drops Windows.
</code_context>

<specifics>
## Specific Ideas

- Confirmed no additional source changes are needed to make Phase 1's criteria
  achievable; the substantive work was the two committed fixes.
- Real signal comes from pushing the branch and watching the `wasm_mvp`,
  `wasm_eh`, `wasm_threads` build legs go green on v1.5.5 and v1.4.5.
</specifics>

<deferred>
## Deferred Ideas

- If a wasm leg fails on a specific dependency (e.g. a getrandom backend on a
  newer emscripten), address it then with a targeted `Cargo.toml`/`.cargo`
  change — not preemptively.
</deferred>
