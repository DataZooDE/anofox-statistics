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

## Reconciliation — RESOLVED

- **duckdb-wasm ↔ DuckDB version pairing** was the one knob needing the real CI
  signal. First CI run (PR #131) confirmed it: `@duckdb/duckdb-wasm@1.29.0`
  bundles engine **v1.1.1**, so loading the **v1.5.5** artifact ABI-mismatched
  (`bad export type for '…SupportStatementCache…'`). The harness surfaced this
  cleanly (it printed `engine: v1.1.1`).
- **Fix:** pinned `@duckdb/duckdb-wasm@1.33.1-dev64.0` — the only published
  version whose bundled engine is exactly **v1.5.5** (verified via the
  storage-version list embedded in `dist/duckdb-eh.wasm`). It is the `@next` dev
  build; no stable duckdb-wasm ships 1.5.x yet, and none bundles exactly v1.4.5,
  so the gate targets the v1.5.5 artifact.
- **Also observed:** all WASM build + deploy legs (wasm_mvp/eh/threads on v1.5.5
  and v1.4.5) went green — WASM-01/02/03/04 confirmed. The concurrent
  `Smoke test (linux_amd64)` failure was an unrelated network flake (connection
  reset downloading the DuckDB CLI), not caused by this milestone.
