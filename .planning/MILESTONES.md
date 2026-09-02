# Milestones

## v0.3.0 Performance & Polish (Shipped: 2026-09-02)

**Phases completed:** 3 phases, 10 plans, 23 tasks

**Key accomplishments:**

- One-command bash+SQL benchmark harness that loads the local release extension by explicit path and times three representative workloads (aggregate dispatch, fit/predict, FFI marshalling) into a diffable results file.
- Profiled the release build with DuckDB EXPLAIN ANALYZE + differential bench workloads; top-3 hotspots dispositioned — DuckDB HASH_GROUP_BY dispatch and the 5-array FFI inference count are inherent, and DataArray::to_vec got a safe bulk-copy fast path (~3–4%), full suite green.
- Dropped anofox_stats_ prefix from all SQL registrations, fixed theilsen→theil_sen, deleted all alias blocks, and shipped docs/API_CONVENTIONS.md with the written convention; make test (103 tests) and cargo test --workspace (295 tests) fully green against the renamed API
- Python harness scripts/validate_docs_sql.py extracts and runs all non-skipped sql blocks from 7 doc files against the locally-built DuckDB extension, establishing the baseline: 6/7 files fail before the DOCS-03 fix sweep
- Systematic fix of all failing SQL blocks across 5 documentation files: strip `anofox_stats_` prefix, convert positional-boolean calls to MAP options, replace external table references with inline data, and skip-mark blocks that reference DuckDB extensions not loaded by the harness or that crash the current build.
- README rewritten to the anofox-forecast form: emoji section headers, ToC, Key Features with Phase-4 benchmark data and Phase-5 ergonomics subsections, validated three-step Quick Start (ols_fit_agg → predict → residuals_diagnostics_agg on a concrete houses dataset), API Reference linking docs/ instead of duplicating the surface, Development section, License last — harness exits 0.
- `.github/workflows/DocsSqlValidation.yml` added — ubuntu-24.04 self-contained build-then-validate gate that hard-fails on any doc-SQL drift; full 7-file harness sweep is green (50 blocks, 0 failures) and SQL regression suite stays clean (506 assertions)

---

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
