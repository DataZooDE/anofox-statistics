# Phase 1 Summary: WASM Build Green

**Completed:** 2026-08-30
**Status:** Implemented — verification deferred to CI (per run decision)
**Requirements:** WASM-01, WASM-02, WASM-03, WASM-04

## What Was Done

Phase 1's substantive changes were the two build/link fixes, both already in the
branch:

- **WASM-02** — Rust FFI static archive linked into the `.wasm` via
  `LINKED_LIBS "$<TARGET_FILE:anofox_stats_ffi-static>"` in `extension_config.cmake`
  (#103, pre-existing).
- **WASM-03** — Telemetry (raw HTTPS via httplib + OpenSSL) compiled out on
  Emscripten: `CMakeLists.txt` guard widened to `if(NOT MINGW AND NOT EMSCRIPTEN)`
  (defines `POSTHOG_TELEMETRY_DISABLED`), and `vcpkg.json` `openssl` made a
  `"!wasm32"` dependency (commit `caf9079`).

For **WASM-01** and **WASM-04** no code change was warranted:

- **WASM-01** — `Cargo.lock`/`crates/` audit found no WASM landmines. The graph
  already carries `web-time` (argmin's wasm `Instant` shim); `getrandom` on
  `wasm32-unknown-emscripten` uses libc `getentropy` (no `js` feature needed);
  no filesystem, socket, or thread-spawn usage in the core/FFI crates. Corrosion
  is routed to `wasm32-unknown-emscripten` by `CMakeLists.txt` when the target is
  present (ci-tools installs it in the wasm legs).
- **WASM-04** — the wasm archs are already in the CI build matrix;
  `MainDistributionPipeline.yml` excludes only Windows.

## Verification

- **Locally verifiable (done):** build wiring coherent; dependency graph
  WASM-clean; telemetry/OpenSSL absent from the Emscripten path by construction.
- **CI gate (open — T5):** the `wasm_mvp`/`wasm_eh`/`wasm_threads` build legs on
  DuckDB v1.5.5 and v1.4.5 LTS must go green. Not reproducible in this
  environment (no `emcc`, no `wasm32-unknown-emscripten` target); validated by
  pushing the branch.

## Follow-ups / Handoff

- Phase 2/3's Node harness will actually load the built `.wasm`, closing the loop
  that CI's compile+link legs cannot (load + runtime correctness).
- If a wasm leg fails on a concrete dependency, fix it there with a targeted
  change rather than preemptive config.
