# Phase 3 Summary: Automated Harness & CI Gate

**Completed:** 2026-08-30
**Status:** Implemented — verification deferred to CI (per run decision)
**Requirements:** TEST-01, TEST-02, TEST-03

## What Was Done

- **TEST-01 — Node harness** (`test/wasm/`):
  - `run.mjs` — boots DuckDB-Wasm (Node/eh), serves the built `.wasm` from a
    version-agnostic localhost server, `FORCE INSTALL`/`LOAD`s it, and runs
    `.test` files. Auto-discovers the artifact or takes `--ext`; `--all` runs the
    full 99-file suite, default is a curated WASM subset (logged, not silent).
  - `sqllogic.mjs` — minimal sqllogictest-subset parser/runner (`require`,
    `statement ok|error`, `query <types>` + `----`, `mode skip`) with
    numeric-tolerant, type-robust comparison.
  - `package.json` — pins `@duckdb/duckdb-wasm` + `web-worker@1.2.0`.
- **TEST-02 — CI gate:** `wasm-runtime-test` job in `MainDistributionPipeline.yml`
  (`needs: duckdb-stable-build`) downloads the `anofox_statistics-v1.5.5-extension-wasm_eh`
  artifact and runs the harness. Gating — a WASM load/runtime failure fails the
  build. Runs on push/PR/dispatch like the rest of the pipeline.
- **TEST-03 — Docs:** `test/wasm/README.md` covers local run steps, CLI options,
  maintainer internals (pins/gotchas), and the version-matching caveat.

## Verification

- **Locally verifiable (done):** `node --check` passes for both harness modules;
  the parser/comparator were unit-exercised against real `.test` content
  (numeric tolerance, text mismatch, value-count mismatch); `MainDistributionPipeline.yml`
  parses as valid YAML with the new job wired in.
- **CI gate (open — T4):** first run of `wasm-runtime-test` must go green.

## Known Reconciliation (first CI run)

- **duckdb-wasm ↔ DuckDB version pairing** is the one knob that needs the real CI
  signal: `@duckdb/duckdb-wasm` must be ABI-compatible with DuckDB v1.5.5. If the
  first run fails at `LOAD` with a version/ABI error, bump the pin in
  `test/wasm/package.json` (and, if no stable duckdb-wasm bundles 1.5.5, test the
  v1.4.5 LTS artifact against a matching duckdb-wasm instead). The harness prints
  the engine version to make this obvious.
