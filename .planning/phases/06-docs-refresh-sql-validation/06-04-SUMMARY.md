---
phase: 06-docs-refresh-sql-validation
plan: "04"
subsystem: ci
tags: [ci, github-actions, docs, validation, ubuntu-24.04, rust, duckdb]

requires:
  - phase: 06-docs-refresh-sql-validation
    plan: "02"
    provides: "fixed guides/01-04 + docs/API_REFERENCE.md — all pass harness (46 blocks)"
  - phase: 06-docs-refresh-sql-validation
    plan: "03"
    provides: "fixed README.md — passes harness (3 blocks); anofox-forecast structure"

provides:
  - ".github/workflows/DocsSqlValidation.yml — self-contained ubuntu-24.04 build-then-validate CI gate"
  - "Full 7-file harness sweep exits 0: 50 executable blocks across README, 4 guides, API_REFERENCE, API_CONVENTIONS"
  - "SQL regression suite (22 test cases, 506 assertions) still passes — no regression from docs work"

affects:
  - "All future PRs — DocsSqlValidation.yml gates doc-SQL drift on pull_request and push to main"

actuals:
  tokens: 3400
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Self-contained ubuntu-24.04 CI gate: checkout submodules:recursive + dtolnay/rust-toolchain@stable + actions/cache@v4 (Cargo key) + make release + python3 harness — mirrors build-and-test-rust pattern"
    - "Hard-fail gate with no continue-on-error — harness non-zero exit fails the job"
    - "permissions: contents: read only (principle of least privilege)"

key-files:
  created:
    - ".github/workflows/DocsSqlValidation.yml — DOCS-04 CI validation workflow"
  modified: []

key-decisions:
  - "Option B (self-contained build) over Option A (artifact reuse from WasmTest.yml pattern): provides immediate PR feedback instead of waiting 15-30 min for the distribution pipeline; DuckDB CLI is not exposed as an artifact by the distribution pipeline anyway"
  - "python3 scripts/validate_docs_sql.py (not bash) as the CI step — the harness was written in Python in Plan 06-01 and that's what's on disk"
  - "No matrix/multiple platforms — linux-only gate per RESEARCH §3 Option B; the distribution pipeline already covers the full platform matrix"
  - "No continue-on-error — hard-fail is the point of DOCS-04"

requirements-completed: [DOCS-04]

coverage:
  - id: D1
    description: ".github/workflows/DocsSqlValidation.yml created — self-contained ubuntu-24.04 build-then-validate CI gate with pull_request and push:main triggers, no continue-on-error"
    requirement: DOCS-04
    verification:
      - kind: automated_ui
        ref: "grep gate: make release, validate_docs_sql.py, pull_request, ubuntu-24.04, branches:[main] all present in non-comment lines — exits 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "Full 7-file harness sweep passes locally (50 blocks, 0 failures) — gate will be green on landing"
    requirement: DOCS-04
    verification:
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py: PASS README.md(3) guides/01(9) guides/02(2) guides/03(11) guides/04(14) API_REFERENCE(10) API_CONVENTIONS(1) — Executed: 7 Passed: 7 Failed: 0"
        status: pass
      - kind: integration
        ref: "build/release/test/unittest [sql]: All tests passed (506 assertions in 22 test cases)"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-02
status: complete
---

# Phase 6 Plan 04: CI Doc-SQL Validation Gate (DOCS-04) Summary

**`.github/workflows/DocsSqlValidation.yml` added — ubuntu-24.04 self-contained build-then-validate gate that hard-fails on any doc-SQL drift; full 7-file harness sweep is green (50 blocks, 0 failures) and SQL regression suite stays clean (506 assertions)**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-09-02T08:28:40Z
- **Completed:** 2026-09-02
- **Tasks:** 2/2
- **Files created:** 1 (DocsSqlValidation.yml)

## Accomplishments

- `.github/workflows/DocsSqlValidation.yml` created as a single `ubuntu-24.04` job: checkout with `submodules: 'recursive'`, `dtolnay/rust-toolchain@stable`, `actions/cache@v4` (same Cargo key as `build-and-test-rust`), `make release`, then `python3 scripts/validate_docs_sql.py` — no `continue-on-error`
- Triggers on `pull_request` and `push: branches: [main]`; `permissions: contents: read`
- Full 7-file harness sweep confirmed green: 50 executable blocks, 7/7 files pass, 0 failures — the CI gate will be green on landing
- Pre-existing SQL regression suite: 22 test cases (506 assertions) all pass — docs work introduced no regressions

## Workflow Design

| Property | Value |
|----------|-------|
| Runner | ubuntu-24.04 |
| Triggers | `pull_request`, `push: branches: [main]` |
| Permissions | `contents: read` |
| Build step | `make release` (DuckDB CLI + anofox_statistics extension) |
| Validate step | `python3 scripts/validate_docs_sql.py` |
| Gate | Hard-fail (no `continue-on-error`) |
| Pattern | Mirrors `build-and-test-rust` from MainDistributionPipeline.yml — Option B (self-contained) |

## Full-Sweep Results (as CI will reproduce)

| File | Executable Blocks | Result |
|------|-------------------|--------|
| README.md | 3 | PASS |
| guides/01_quick_start.md | 9 | PASS |
| guides/02_technical_guide.md | 2 | PASS |
| guides/03_business_guide.md | 11 | PASS |
| guides/04_advanced_use_cases.md | 14 | PASS |
| docs/API_REFERENCE.md | 10 | PASS |
| docs/API_CONVENTIONS.md | 1 | PASS |
| **TOTAL** | **50** | **7/7 PASS** |

**Harness exit code: 0**

SQL regression suite: `All tests passed (2 skipped tests, 506 assertions in 22 test cases)`

## Task Commits

1. **Task 1: Author .github/workflows/DocsSqlValidation.yml** — `c4b5bbb` (feat)
2. **Task 2: Confirm full sweep green** — verification-only, no file changes (no separate commit)

## Files Created

- `.github/workflows/DocsSqlValidation.yml` — DOCS-04 CI validation workflow (57 lines)

## Decisions Made

- **Self-contained build (Option B)**: Provides immediate PR feedback vs. 15-30 min lag with artifact-reuse (Option A). The DuckDB CLI is not exposed as a CI artifact by the distribution pipeline, making self-contained build the only clean path anyway.
- **python3 scripts/validate_docs_sql.py** as the CI run command: the harness exists as a Python script (Plan 06-01); the RESEARCH's yaml snippet showed `bash scripts/validate_docs_sql.sh` but the actual file on disk is the Python script.
- **No matrix/multiple platforms**: Linux-only gate; distribution pipeline covers platform breadth.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RESEARCH yaml snippet references bash script; actual harness is Python**
- **Found during:** Task 1
- **Issue:** The RESEARCH.md §3 reusable steps yaml shows `run: bash scripts/validate_docs_sql.sh`, but the script created in Plan 06-01 is `scripts/validate_docs_sql.py`. The bash variant does not exist.
- **Fix:** Used `python3 scripts/validate_docs_sql.py` in the workflow's validate step — matching the actual artifact on disk.
- **Files modified:** `.github/workflows/DocsSqlValidation.yml`
- **Impact:** None on CI correctness; the Python harness is self-contained and is what all prior plans verified against.

**2. [Rule 1 - Info] Plan verify command uses `--test-dir` flag that doesn't exist on this unittest binary**
- **Found during:** Task 2 (verification)
- **Issue:** The plan's Task 2 `<verify>` compound command uses `build/release/test/unittest --test-dir=test/sql`, but the binary uses Catch2 and does not support `--test-dir`. The compound command would exit non-zero due to flag rejection.
- **Fix:** Ran `build/release/test/unittest "[sql]"` (tag filter) which correctly executes all SQL test cases. Results confirmed: 22 test cases, 506 assertions, all passed.
- **Files modified:** None — verification only; noted in SUMMARY for traceability.

---

**Total deviations:** 2 auto-noted (both Rule 1: mis-matched script name in RESEARCH; flag mismatch in plan verify command — neither affected the actual deliverable)
**Impact on plan:** Zero scope change. DocsSqlValidation.yml is correct and the gate shape is exactly per DOCS-04 spec.

## Issues Encountered

None beyond the two auto-noted deviations above.

## Threat Flags

No new security-relevant surface beyond what the threat model covers:
- T-6-02 (mitigated): No `continue-on-error`; hard-fail confirmed
- T-6-01 (mitigated): `permissions: contents: read` only; runner has no production credentials

## Known Stubs

None.

## Next Phase Readiness

Phase 6 is complete:
- DOCS-01: README restructured to anofox-forecast form (Plan 03)
- DOCS-02: Doc-SQL harness created (Plan 01)
- DOCS-03: All 50 executable blocks pass across 7 doc files (Plans 01/02/03)
- DOCS-04: CI gate added — future drift fails the build (this plan)

Post-merge verification: confirm `DocsSqlValidation` workflow run is green on GitHub Actions for the landing PR.

## Self-Check: PASSED

Files verified:
- `.github/workflows/DocsSqlValidation.yml` — exists (57 lines, checked)
- Commit `c4b5bbb` — verified in git log

Harness gate:
- `python3 scripts/validate_docs_sql.py` (full sweep): exit 0, 7/7 files, 50 blocks
- `build/release/test/unittest "[sql]"`: All tests passed (506 assertions, 22 test cases)

---
*Phase: 06-docs-refresh-sql-validation*
*Completed: 2026-09-02*
