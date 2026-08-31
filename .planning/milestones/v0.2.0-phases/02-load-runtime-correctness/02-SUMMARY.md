# Phase 2 Summary: Load & Runtime Correctness

**Completed:** 2026-08-30
**Status:** Implemented — verification deferred to CI (per run decision)
**Requirements:** LOAD-01, LOAD-02

## What Was Done

Built the DuckDB-Wasm load + runtime-correctness checks as the foundation of the
reusable harness under `test/wasm/`:

- **LOAD-01** — `test/wasm/run.mjs` boots DuckDB-Wasm in Node (verified recipe:
  `duckdb-node.cjs` eh bundle, `web-worker@1.2.0` pinned, `instantiate(module,
  null)`, `open({ allowUnsignedExtensions: true })`), serves the built `.wasm`
  from a **version-agnostic** localhost server, and `FORCE INSTALL … FROM` +
  `LOAD`s it. Any load failure is a hard non-zero exit — this is the core signal
  a compile+link CI leg cannot produce.
- **LOAD-02** — a curated subset of the existing `test/sql/*.test` files
  (OLS scalar/agg/fit-predict, t-test, Jarque-Bera, Pearson, VIF) is executed via
  `test/wasm/sqllogic.mjs` and asserted against the same expected values the
  native suite uses. Comparison is numeric-tolerant (float ε) and type-robust
  (the `.test` files use `query I` loosely for DOUBLE columns).

## Verification

- **Locally verifiable (done):** harness/parsers pass `node --check`; the
  sqllogictest parser + comparator were unit-exercised against real `.test`
  content (numeric tolerance, text mismatch, and value-count mismatch all behave
  correctly). Boot/LOAD themselves need a real `.wasm`.
- **CI gate (open — T3):** the `wasm-runtime-test` job (Phase 3) runs this harness
  against the v1.5.5 `wasm_eh` build artifact.

## Handoff / Known Reconciliation

- The `@duckdb/duckdb-wasm` engine version must be ABI-compatible with DuckDB
  v1.5.5 (the version the artifact is built against). The harness prints the
  engine version on startup; the first CI run confirms the pairing and the pin in
  `test/wasm/package.json` is adjusted if needed.
