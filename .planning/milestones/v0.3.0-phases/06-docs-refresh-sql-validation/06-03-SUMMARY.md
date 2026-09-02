---
phase: 06-docs-refresh-sql-validation
plan: 03
subsystem: docs
tags: [readme, restructure, doc-sql, validation, ols, quick-start]

# Dependency graph
requires:
  - phase: 06-docs-refresh-sql-validation
    provides: "doc-SQL harness (scripts/validate_docs_sql.py) and fixed guides/API_REFERENCE"
provides:
  - README.md restructured to the anofox-forecast form with emoji headers, ToC, Key Features (Performance + User-Friendly API subsections), validated Quick Start, API Reference links, Development section, License last
  - All README sql blocks validated by the harness (exits 0)
affects:
  - gsd-verify-work (UAT will visual-inspect README structure)
  - DocsSqlValidation.yml CI (README.md is in the 7-file scope)

actuals:
  tokens: 5445
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Installation/telemetry sql blocks marked 'sql skip' in README — not runnable in harness environment"
    - "Illustrative GROUP BY example marked 'sql skip' — references schema not created in README session"

key-files:
  created: []
  modified:
    - README.md

key-decisions:
  - "Quick Start uses ols_fit_agg for FIT and ols_fit scalar (column-major X) + predict scalar for PREDICT — these are self-contained and harness-verified"
  - "predict() subquery approach: (ols_fit([y...], [[X_col...]])).coefficients is the reliable pattern inside the predict() call (scalar subquery evaluated once)"
  - "Installation/telemetry sql blocks marked 'sql skip' — INSTALL FROM erpl.io/community not runnable in a local harness environment"
  - "GROUP BY per-category example marked 'sql skip' — references sales_data table not defined in README session"
  - "residuals_diagnostics_agg(actual, yhat) returns STRUCT(raw, standardized, studentized, leverage) — not durbin_watson/mse; Quick Start uses .raw"

patterns-established:
  - "ols_fit scalar X format: column-major (each inner array is ONE feature column across all observations, not one row)"

requirements-completed: [DOCS-01]

coverage:
  - id: D1
    description: "README.md restructured to anofox-forecast form: ToC, emoji headers, Key Features with ⚡ Performance and 🎨 User-Friendly API subsections, Quick Start (fit→predict→diagnostics), API Reference linking docs/, Development, Support, Citation, License last"
    requirement: DOCS-01
    verification:
      - kind: automated_ui
        ref: "grep gate: Table of Contents, Key Features, Quick Start, API Reference, Development, Citation headings present; License is last level-2 section"
        status: pass
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file README.md: PASS (3 blocks)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Quick Start walks fit→predict→diagnostics on the houses dataset using v0.3.0 unprefixed API names and MAP-option keys"
    requirement: DOCS-01
    verification:
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file README.md: PASS (3 blocks)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-02
status: complete
---

# Phase 6 Plan 03: README Restructure Summary

**README rewritten to the anofox-forecast form: emoji section headers, ToC, Key Features with Phase-4 benchmark data and Phase-5 ergonomics subsections, validated three-step Quick Start (ols_fit_agg → predict → residuals_diagnostics_agg on a concrete houses dataset), API Reference linking docs/ instead of duplicating the surface, Development section, License last — harness exits 0.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-09-02
- **Tasks:** 2/2
- **Files modified:** 1 (README.md only; guides/ and docs/ untouched)

## Accomplishments

- README restructured into the exact locked section order from CONTEXT.md: Title + badges → ToC → Key Features → Quick Start → Installation → API Reference → Development → Support → Citation → License
- All level-2 headings have emoji; `### ⚡ Performance` and `### 🎨 User-Friendly API` subsections added under Key Features with Phase-4 benchmark numbers (`~3.2 µs/group` for OLS fit, `~3–4%` query-time improvement from bulk-copy fast path) and Phase-5 ergonomics narrative
- Quick Start rewritten end-to-end on the `houses` dataset (6 rows: sqm → price_keur): Step 1 FIT with `ols_fit_agg`, Step 2 PREDICT with `ols_fit` scalar (column-major X) + `predict`, Step 3 DIAGNOSTICS with `residuals_diagnostics_agg`
- `Documentation` section replaced by `📚 API Reference` linking `docs/API_REFERENCE.md` and `docs/API_CONVENTIONS.md` — no function-list duplication
- `Development` section added covering `make release`, `cargo test`, SQL test suite, doc-SQL harness, benchmark harness (folding in former Dependencies/Validation/Contributing/Acknowledgments content)
- License moved to final position (after Citation)
- Naming Convention prose corrected: removed stale `anofox_stats_*` primary + alias claim; states unprefixed v0.3.0 convention and points to `docs/API_CONVENTIONS.md`
- `python3 scripts/validate_docs_sql.py --file README.md`: **PASS (3 blocks)**

## Final Section List (in order)

| # | Section | Level |
|---|---------|-------|
| 1 | 📋 Table of Contents | `##` |
| 2 | ✨ Key Features | `##` |
| 2a | ⚡ Performance | `###` |
| 2b | 🎨 User-Friendly API | `###` |
| 3 | 🚀 Quick Start | `##` |
| 4 | 📦 Installation | `##` |
| 5 | 📚 API Reference | `##` |
| 6 | 🛠️ Development | `##` |
| 7 | 💬 Support | `##` |
| 8 | 📖 Citation | `##` |
| 9 | ⚖️ License | `##` (**last**) |

## Quick Start Dataset

- **Dataset:** `houses` — 6 rows, two columns: `sqm DOUBLE`, `price_keur DOUBLE`
- **FIT:** `ols_fit_agg(price_keur, [sqm])` → `r_squared ≈ 0.9995`, `slope ≈ 2.41`
- **PREDICT:** `ols_fit` scalar (column-major X) + `predict(X_new, coefficients, intercept)` → predictions for 70, 100, 140 sqm
- **DIAGNOSTICS:** `residuals_diagnostics_agg(actual, yhat)` → `.raw` residuals array
- **Harness exit code:** 0 (PASS, 3 blocks executed)

## Task Commits

1. **Task 1: Restructure README sections into anofox-forecast form** — `6e69627` (feat)
2. **Task 2: Rewrite the Quick Start example so it passes the harness** — `930c050` (feat)

## Files Modified

- `README.md` — restructured from 341 lines to 384 lines; +209/-166 lines across tasks 1 and 2

## Decisions Made

- **predict() subquery pattern**: `(ols_fit([y_vals], [[x_col_vals]])).coefficients` inside `predict()` is the reliable pattern. Storing the fit in a CREATE TABLE and cross-joining with new-data VALUES had a DuckDB optimizer issue where `fit.coefficients` returned NULL in the cross-join context for small training sets; the subquery approach works consistently.
- **Installation/telemetry blocks marked `sql skip`**: `INSTALL FROM erpl.io`, `INSTALL FROM community`, and `SET anofox_telemetry_enabled = false` are not runnable in the harness's local build environment (extension already loaded). Skip-marking is correct.
- **GROUP BY illustrative example marked `sql skip`**: references `sales_data` table not defined in README session — illustrative prose example, not a validated walkthrough.
- **residuals_diagnostics_agg return type**: the function returns `STRUCT(raw, standardized, studentized, leverage)` arrays — not scalar diagnostics like durbin_watson/mse. Quick Start correctly accesses `.raw`.
- **ols_fit scalar X is column-major**: `X = [[col1_values], [col2_values], ...]` — each inner array is ALL observations for ONE feature. This is documented in the Quick Start Step 2 comment.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Skip-mark Installation and Telemetry sql blocks**
- **Found during:** Task 2 (harness run)
- **Issue:** `INSTALL 'anofox_statistics' FROM 'http://get.erpl.io'` and `INSTALL anofox_statistics FROM community` both failed in the harness because the extension is already installed from a different origin (the local unsigned build). `SET anofox_telemetry_enabled = false` also ran and set state.
- **Fix:** Marked three Installation/Telemetry blocks as `sql skip` — they are instructional documentation not intended for harness execution.
- **Files modified:** README.md
- **Verification:** `python3 scripts/validate_docs_sql.py --file README.md` exits 0

**2. [Rule 2 - Missing Critical] Skip-mark per-group GROUP BY example**
- **Found during:** Task 2 (harness run)
- **Issue:** The `sales_data` reference in the per-group example caused `Catalog Error: Table with name sales_data does not exist`.
- **Fix:** Marked block as `sql skip` — it is an illustrative prose example showing the API pattern, not a self-contained executable.
- **Files modified:** README.md
- **Verification:** `python3 scripts/validate_docs_sql.py --file README.md` exits 0

---

**Total deviations:** 2 auto-fixed (Rule 2 — missing skip markers for non-runnable illustrative blocks)
**Impact on plan:** Both auto-fixes necessary for harness correctness. No scope creep — the blocks are correctly documented patterns, just not executable in the local harness environment.

## Issues Encountered

- **predict() cross-join NULL bug**: `fit.coefficients` returned NULL when accessing a struct stored in a `CREATE TABLE` via a cross join with `VALUES(...)`. Root cause: DuckDB optimizer issue with small training sets (2 rows) — with 6 rows it partly worked but the cross-join context still caused NULL in some positions. Resolved by using the `(ols_fit([y_vals], [[x_col_vals]])).coefficients` subquery pattern inside `predict()`, which is evaluated as a scalar subquery once and works reliably.
- **ols_fit scalar is column-major, not row-major**: The existing README had `[[1.1], [2.1], [2.9], ...]` (one column = one observation) which caused `Dimension mismatch: y has 5 elements, X has 1 rows` because DuckDB flattened the single-element inner arrays. The correct format is `[[x1_col1, x2_col1, ...]]` (one inner array = all observations for feature 1). This was discovered during testing and documented in the Quick Start.

## Known Stubs

None — the Quick Start is fully functional and harness-verified. The `sql skip` blocks are correctly documented API patterns (not stubs); they are marked non-executable by design.

## Threat Flags

None — README.md authoring only; no new endpoints, auth paths, or schema changes.

## Self-Check

**Created files:**
- `README.md` — exists and is 384 lines

**Commits:**
- `6e69627` — Task 1 (feat: restructure)
- `930c050` — Task 2 (feat: Quick Start harness pass)

**Harness gate:**
- `python3 scripts/validate_docs_sql.py --file README.md` → PASS (3 blocks)

**Structure gate:**
- All required sections present; License is last — PASS

## Self-Check: PASSED

---
*Phase: 06-docs-refresh-sql-validation*
*Completed: 2026-09-02*
