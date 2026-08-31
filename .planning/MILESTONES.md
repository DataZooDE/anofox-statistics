# Milestones

## v0.2.0 — WASM Support (in progress)

**Started:** 2026-08-30

**Goal:** The extension builds, loads, and runs correctly on DuckDB-Wasm, with an
automated Node-based test harness gating CI against regressions.

**Requirements:** WASM-01..04, LOAD-01..02, TEST-01..03 (see REQUIREMENTS.md)

---

## v0.1.0 — Native statistics suite (shipped)

The initial published extension: regression, GLM, GLMM, AFT, 40+ hypothesis
tests, and diagnostics as aggregate/table/scalar/window functions, distributed
for linux/osx (amd64+arm64) via extension-ci-tools. Codebase mapped in
`.planning/codebase/` (2026-08-11).
