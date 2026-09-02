# Phase 6: Docs Refresh & SQL Validation - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Refresh the documentation to the anofox-forecast form and make documentation drift a
build failure. Three strands: (DOCS-01) restructure README.md to the anofox-forecast
form; (DOCS-02) build a doc-SQL validation harness that extracts every SQL example from
README + guides + API reference and runs each against the built extension, reporting
pass/fail; (DOCS-03) fix every example that fails — including the ones broken by the
Phase-5 rename (e.g. the `.residual_standard_error` guide references flagged as IN-03 in
Phase 5's review, and any lingering `anofox_stats_*` / `.r2` / `theilsen` references in
docs); (DOCS-04) run the harness in CI so future drift fails the build.

Out of scope: new documentation content beyond restructuring + fixing; changing the API
(that was Phase 5); a docs website/site generator.

</domain>

<decisions>
## Implementation Decisions

### README Structure (DOCS-01)
- Section order (anofox-forecast form): Title + badges → Table of Contents → Key Features (with a **⚡ Performance** subsection referencing the Phase-4 benchmark work and a **🎨 User-Friendly API** subsection referencing the Phase-5 ergonomics) → Quick Start → Installation → API Reference → Development → Support → Citation → License.
- Emoji section headers: yes (matches anofox-forecast; success criterion 1 requires them).
- Quick Start: one concrete small dataset walked end-to-end (fit → predict → diagnostics) using the NEW renamed function names (no `anofox_stats_` prefix, `.r_squared` not `.r2`, `theil_sen`).
- `docs/API_REFERENCE.md` remains the authoritative API list; README's API Reference section links to it and to `docs/API_CONVENTIONS.md` rather than duplicating the full surface (avoids drift).

### Doc-SQL Validation Harness (DOCS-02/03)
- Extraction scope: fenced ` ```sql ` blocks from `README.md` + `guides/*.md` + `docs/API_REFERENCE.md` (and `docs/API_CONVENTIONS.md`).
- Execution: pipe each block into the built DuckDB CLI with the extension `LOAD`ed; report pass/fail per example with the source file + line.
- Multi-block setup dependencies: concatenate a single file's blocks in document order into one DuckDB session, so a later example can see tables/state set up by an earlier block in the same file.
- Harness lives under `scripts/` (e.g. `scripts/validate_docs_sql.*`), consistent with Phase-4's `scripts/bench.sh`; emits a per-example pass/fail report and a non-zero exit on any failure.
- DOCS-03: every extracted example must pass — fix drift/rename breakage in the docs (this is where the Phase-5 IN-03 `.residual_standard_error` → `residual_std_error` guide fixes land, plus any stale names).

### CI Integration (DOCS-04)
- New workflow `.github/workflows/DocsSqlValidation.yml`, mirroring `WasmTest.yml`'s build-then-run shape: build the extension, then run the harness.
- Gating: hard-fail the job on any doc-SQL example failure (documentation drift fails the build).
- Non-executable/illustrative snippets: a documented skip marker (e.g. ` ```sql skip ` or an HTML-comment sentinel) so intentionally illustrative blocks are excluded; the convention is documented alongside the harness.
- Triggers: on pull_request + push to `main` (same as `WasmTest.yml`).

### Claude's Discretion
- Exact harness language (bash vs Python) and the precise skip-marker syntax are at Claude's discretion, guided by `scripts/bench.sh` conventions and the DuckDB CLI available in CI.
- Exact README wording/copy is at Claude's discretion within the agreed section structure.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/bench.sh` (Phase 4) — established pattern for a repo script that builds/runs against the extension; mirror its structure + a `bench/README.md`-style doc.
- `.github/workflows/WasmTest.yml` — model for a build-extension-then-run-harness CI job (Node harness loading the built extension); mirror its build steps for the DuckDB CLI + LOAD.
- `docs/API_CONVENTIONS.md` (Phase 5) — the naming convention doc; README/API_REFERENCE examples must conform to it.
- Existing docs to restructure/validate: `README.md` (341 lines, sections present but not anofox-forecast form), `docs/API_REFERENCE.md`, `guides/01_quick_start.md`..`04_advanced_use_cases.md`.

### Established Patterns
- The `test/sql/*.test` DuckDB sqllogictest harness (`build/release/test/unittest`) is the existing SQL-execution mechanism; the doc-SQL harness may reuse the built extension + CLI rather than reinventing loading.
- Phase-5 renamed API: no `anofox_stats_` prefix, snake_case option keys, `.r_squared`/`std_errors`/`t_values` return fields, `theil_sen`, GLM/AFT `z_values` exception.

### Integration Points
- CI: `.github/workflows/` (WasmTest.yml, MainDistributionPipeline.yml) — add DocsSqlValidation.yml.
- The harness reads docs at repo paths and needs the built loadable extension (`make release` → `build/release/`).

</code_context>

<specifics>
## Specific Ideas

- Carry the Phase-5 review's deferred **IN-03** here: 10 guide `.sql`/examples reference `.residual_standard_error`, but the actual struct field is `residual_std_error`. The harness must catch these and DOCS-03 fixes them.
- Quick Start should double as a validated example (it runs through the harness).

</specifics>

<deferred>
## Deferred Ideas

- A rendered documentation website / static-site generator — out of scope; this phase is README + guides + API reference in-repo, validated in CI.

</deferred>
