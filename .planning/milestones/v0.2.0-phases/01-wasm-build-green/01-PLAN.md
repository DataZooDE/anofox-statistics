# Phase 1 Plan: WASM Build Green

**Created:** 2026-08-30
**Requirements:** WASM-01, WASM-02, WASM-03, WASM-04
**Verification mode:** implement-now / verify-in-CI (no local WASM toolchain)

## Approach

The substantive build/link fixes are already applied and committed. This phase's
work is (a) confirm the build wiring is coherent end-to-end, (b) audit the Rust
dependency graph for WASM portability, and (c) hand off to CI as the verification
gate. No new production code is expected unless the audit surfaces a concrete gap.

## Tasks

- [x] **T1 (WASM-02)** — Confirm `LINKED_LIBS "$<TARGET_FILE:anofox_stats_ffi-static>"`
  in `extension_config.cmake` feeds the emcc post-build link. ✓ present (#103).
- [x] **T2 (WASM-03)** — Compile telemetry out on Emscripten and drop OpenSSL on
  wasm. ✓ `CMakeLists.txt` guard `if(NOT MINGW AND NOT EMSCRIPTEN)` + `vcpkg.json`
  `openssl` `"!wasm32"` (commit caf9079).
- [x] **T3 (WASM-01)** — Audit `Cargo.lock`/`crates/` for WASM landmines
  (getrandom backend, threads, filesystem, sockets, timing). ✓ graph is
  WASM-clean; `web-time` present, no `js` feature needed on emscripten.
- [x] **T4 (WASM-04)** — Confirm wasm archs are in the CI build matrix. ✓
  `MainDistributionPipeline.yml` excludes only Windows; wasm_mvp/eh/threads build.
- [ ] **T5 (verify)** — Push branch; confirm the `wasm_*` legs build green on
  DuckDB v1.5.5 and v1.4.5 LTS. **← CI gate; not locally reproducible.**

## Success Criteria → Evidence

1. Rust FFI + deps compile for `wasm32-unknown-emscripten` → dep-graph audit
   clean (T3); confirmed by CI wasm legs (T5).
2. Final `.wasm` has the FFI archive, zero unresolved `anofox_*` imports →
   `LINKED_LIBS` (T1); confirmed by CI + Phase 3 load test.
3. No telemetry/HTTP/OpenSSL symbols in the Emscripten build → guard + vcpkg
   exclusion (T2).
4. CI matrix green for wasm_mvp/eh/threads on v1.5.5 and v1.4.5 → T5 (CI gate).

## Notes

- T1–T4 are complete/confirmed in the working tree and prior commits. T5 is the
  only open item and is intentionally a CI-side gate per the run's decision.
