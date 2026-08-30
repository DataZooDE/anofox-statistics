# Phase 3 Plan: Automated Harness & CI Gate

**Created:** 2026-08-30
**Requirements:** TEST-01, TEST-02, TEST-03
**Verification mode:** implement-now / verify-in-CI

## Tasks

- [x] **T1 (TEST-01)** — `test/wasm/` Node project that loads the built `.wasm` via
  `@duckdb/duckdb-wasm` and runs `test/sql/*.test` through a minimal
  sqllogictest-subset runner (`run.mjs` + `sqllogic.mjs`, `package.json`).
  Curated subset by default; `--all` runs the full 99-file suite; skips are logged.
- [x] **T2 (TEST-02)** — `wasm-runtime-test` job in `MainDistributionPipeline.yml`,
  `needs: duckdb-stable-build`, downloads the `wasm_eh` artifact and runs the
  harness; gating (fails the build on any load/runtime error).
- [x] **T3 (TEST-03)** — `test/wasm/README.md` documents local usage
  (`npm --prefix test/wasm install` + `ANOFOX_WASM_EXT=… npm --prefix test/wasm test`),
  options, internals, and the version-matching caveat.
- [ ] **T4 (verify)** — First CI run green (or reconcile the duckdb-wasm pin). **← CI gate.**

## Success Criteria → Evidence

1. Node harness loads the `.wasm` and executes the `.test` suite/subset →
   `run.mjs`/`sqllogic.mjs` (T1), self-checked via `node --check` + parser unit checks.
2. CI job fails the build on any WASM load/runtime error → `wasm-runtime-test`
   gating job (T2), YAML validated.
3. Local WASM test steps documented and working → `test/wasm/README.md` (T3);
   commands mirror the harness CLI.

## Notes

- The CI job consumes the artifact the existing wasm build leg already produces
  (no rebuild), keeping the gate cheap.
- Subset vs `--all` is an explicit, logged policy — no silent truncation of the
  99-file suite.
