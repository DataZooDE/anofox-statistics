# Milestones

## v0.3.0 — Performance & Polish (in progress)

**Started:** 2026-08-31

**Goal:** Make the extension measurably faster and easier to use — a benchmark
suite + FFI/allocation refactor + hotspot optimization, clearer errors and
consistent APIs, and a refreshed README (anofox-forecast form) with every
documented SQL example validated in CI.

**Requirements:** PERF-01..04, ERGO-01..03, DOCS-01..04 (see REQUIREMENTS.md)

---

## v0.2.0 — WASM Support (Shipped: 2026-08-31)

**Phases:** 3 phases, 3 plans · **Requirements:** 9/9 (WASM-01..04, LOAD-01..02, TEST-01..03)
**Verification:** CI-verified end-to-end (PR #131, run 33365401865)

**Delivered:** the native statistics extension now builds, loads, and computes
correctly in DuckDB-Wasm, with an automated Node harness gating CI against
regressions.

**Key accomplishments:**

- Fixed the WASM load failure — PostHog telemetry (raw HTTPS via httplib+OpenSSL,
  fired at extension load) compiled out on Emscripten; complements the
  `LINKED_LIBS` Rust-archive fix (#103).
- All WASM build + deploy legs green (wasm_mvp/eh/threads) on DuckDB v1.5.5 and
  v1.4.5 LTS.
- New `test/wasm/` Node harness boots DuckDB-Wasm, loads the built `.wasm`, and
  runs the full SQL suite via a sqllogictest-subset runner — **2095/2095**
  assertions passing.
- Gating `wasm-runtime-test` CI job + dedicated `WASM` status badge in README.
- Diagnosed a suspected `MIN(x1)` discrepancy as a duckdb-wasm Arrow-JS DECIMAL
  rendering quirk (unscaled integer) in the harness — fixed via DuckDB `::VARCHAR`
  formatting; confirmed the extension itself is clean on WASM.

**Known tech debt:** `@duckdb/duckdb-wasm` pinned to a dev build
(`1.33.1-dev64.0`, only version bundling engine v1.5.5); v1.4.5 LTS is
compile/link-verified but not runtime-verified (no matching duckdb-wasm). See
`milestones/v0.2.0-MILESTONE-AUDIT.md`.

---

## v0.1.0 — Native statistics suite (shipped)

The initial published extension: regression, GLM, GLMM, AFT, 40+ hypothesis
tests, and diagnostics as aggregate/table/scalar/window functions, distributed
for linux/osx (amd64+arm64) via extension-ci-tools. Codebase mapped in
`.planning/codebase/` (2026-08-11).
