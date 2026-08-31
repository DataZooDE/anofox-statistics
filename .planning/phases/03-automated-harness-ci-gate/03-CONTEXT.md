# Phase 3: Automated Harness & CI Gate - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning
**Mode:** Autonomous (implement-now/verify-in-CI)

<domain>
## Phase Boundary

WASM verification is a repeatable, automated gate: a Node harness loads the built
`.wasm` via `@duckdb/duckdb-wasm`, runs the SQL test suite (or a WASM-appropriate
subset), is wired into CI to fail the build on any WASM load/runtime error, and
is documented for local use.

Requirements: TEST-01, TEST-02, TEST-03.
</domain>

<decisions>
## Implementation Decisions

- **Location:** `test/wasm/` — a self-contained Node ESM project
  (`package.json`, harness runner, minimal sqllogictest parser).
- **Runner:** a minimal, dependency-light sqllogictest subset parser (handles
  `require`, `statement ok|error`, `query <types>` + `----` expected rows,
  comments, `mode skip`), sufficient for this repo's `.test` files. Full
  sqllogictest semantics (hashing, sort modes, labels) are out of scope.
- **Extension loading:** serve the locally built extension repo dir over a
  localhost HTTP server and `INSTALL ... FROM` it with `allowUnsignedExtensions`,
  matching the `<duckdb_version>/<wasm_platform>/` layout DuckDB-Wasm expects.
- **CI:** a dedicated job in `MainDistributionPipeline.yml` (or a new workflow)
  that (a) obtains the built wasm extension artifact, (b) runs the harness, (c)
  fails on any load/runtime error. Prefer consuming the artifact the existing
  wasm build legs already produce over rebuilding.
- **Subset policy:** if some `.test` files exercise features unavailable/slow on
  WASM, the harness runs a curated allowlist and LOGS what it skips (no silent
  truncation).
</decisions>

<code_context>
## Existing Code Insights

- 99 `.test` files under `test/sql/`. Native CI runs them via ci-tools'
  sqllogictest. The WASM harness reuses the same files through its own parser.
- No root `package.json` today — the Node project is new and isolated under
  `test/wasm/` so it does not affect the Rust/C++ build.
</code_context>

<specifics>
## Specific Ideas

- `test/wasm/run.mjs` — entrypoint: boot DuckDB-Wasm, load extension, iterate
  `.test` files, assert, exit non-zero on failure.
- `test/wasm/sqllogic.mjs` — the parser/runner.
- `test/wasm/package.json` — pins `@duckdb/duckdb-wasm` + a Node HTTP static
  server (built-in `http` to avoid deps).
- `npm --prefix test/wasm test` runs it locally; documented in README.
</specifics>

<deferred>
## Deferred Ideas

- Reporting/JUnit output and per-function coverage dashboards — future.
</deferred>
