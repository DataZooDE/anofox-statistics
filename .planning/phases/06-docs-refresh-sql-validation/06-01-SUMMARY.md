---
phase: 06-docs-refresh-sql-validation
plan: "01"
subsystem: testing
tags: [duckdb, validation, harness, python, docs, sql, tracer]

requires:
  - phase: 05-ergo-api-consistency
    provides: "renamed API (no anofox_stats_ prefix, theil_sen, r_squared) — the harness validates against this"

provides:
  - "scripts/validate_docs_sql.py — doc-SQL validation harness with skip-marker support"
  - "Baseline failure count: 6/7 doc files fail (expected before DOCS-03 fix sweep)"
  - "Skip convention established: ```sql skip info-string excludes illustrative blocks"
  - "docs/API_CONVENTIONS.md migration blocks correctly skip-marked (5 blocks total)"

affects:
  - "06-02 — harness is the measurement tool for each fixed file"
  - "06-03 — harness gates the full sweep exit-0 requirement"
  - "06-04 — harness is the CI command in DocsSqlValidation.yml"

actuals:
  tokens: 5050
  tasks: 2
  commits: 2

tech-stack:
  added: ["python3 subprocess harness", "DuckDB CLI invocation via -unsigned -cmd LOAD -f"]
  patterns:
    - "Per-file session: all blocks from one doc file concatenated in document order into one DuckDB session"
    - ".bail on prepended so mid-file errors abort remaining blocks"
    - "sql skip info-string as the skip marker for illustrative/migration blocks"
    - "Path-traversal guard on --file: reject .. components + out-of-tree resolved paths"

key-files:
  created:
    - "scripts/validate_docs_sql.py — doc-SQL validation harness (the phase tracer)"
  modified:
    - "docs/API_CONVENTIONS.md — 5 illustrative/migration sql blocks skip-marked"

key-decisions:
  - "Python harness (not bash) chosen for cleaner regex extraction + subprocess API; mirrors bench.sh binary paths and precondition pattern"
  - "Per-file concatenation into one DuckDB session (not per-block) preserves cross-block state (CREATE TABLE used by later SELECT)"
  - ".bail on dot-command prepended so DuckDB CLI exits non-zero on first failing statement"
  - "Skip marker: ```sql skip info-string — single-regex decision, renders as plain sql in markdown viewers"
  - "5 blocks skip-marked in API_CONVENTIONS.md: 2 migration Before blocks + 3 syntax-illustration blocks referencing FROM tbl"

patterns-established:
  - "validate_docs_sql.py --file <path>: fast single-file iteration path for Plans 02/03 verify commands"
  - "Full sweep: python3 scripts/validate_docs_sql.py — exits 0 only when all files pass"

requirements-completed:
  - DOCS-02

coverage:
  - id: D1
    description: "scripts/validate_docs_sql.py extracts non-skipped sql blocks, runs per-file DuckDB sessions, prints PASS/FAIL per file, exits non-zero on failure"
    requirement: "DOCS-02"
    verification:
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file .scratch_docs_selftest.md (known-bad: exit 1)"
        status: pass
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file .scratch_docs_selftest.md (trivial-good: exit 0)"
        status: pass
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file docs/API_CONVENTIONS.md (skip-marked: exit 0)"
        status: pass
    human_judgment: false
  - id: D2
    description: "docs/API_CONVENTIONS.md migration and illustrative blocks skip-marked so harness excludes them"
    requirement: "DOCS-02"
    verification:
      - kind: integration
        ref: "python3 scripts/validate_docs_sql.py --file docs/API_CONVENTIONS.md; conv=$?; test $conv -eq 0"
        status: pass
    human_judgment: false

duration: 3min
completed: "2026-09-02"
status: complete
---

# Phase 6 Plan 01: Build doc-SQL validation harness Summary

**Python harness scripts/validate_docs_sql.py extracts and runs all non-skipped sql blocks from 7 doc files against the locally-built DuckDB extension, establishing the baseline: 6/7 files fail before the DOCS-03 fix sweep**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-09-02T07:38:07Z
- **Completed:** 2026-09-02T07:41:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `scripts/validate_docs_sql.py` created and self-tested end-to-end: harness extracts fenced sql blocks, honors skip info-string, runs per-file DuckDB sessions with `.bail on`, prints per-file PASS/FAIL with stderr, exits non-zero on any failure
- Self-test triad verified: known-bad file exits 1, trivial-good file exits 0, API_CONVENTIONS.md exits 0
- Baseline full-sweep failure count captured: **6 out of 7 files fail** (expected at this point — DOCS-03 fix sweep is Plans 02/03)
- 5 blocks skip-marked in `docs/API_CONVENTIONS.md`: 2 migration "Before" blocks (correct documentation of breaking changes) + 3 syntax-illustration blocks referencing non-existent `tbl`

## Tracer Self-Test Proof

| Test | File | Exit code | Result |
|------|------|-----------|--------|
| known-bad | `.scratch_docs_selftest.md` (nonexistent function) | 1 (non-zero) | PASS |
| trivial-good | `.scratch_docs_selftest.md` (SELECT 1) | 0 | PASS |
| API_CONVENTIONS | `docs/API_CONVENTIONS.md` (5 blocks skipped, 1 executable) | 0 | PASS |

Compound assertion: `bad=1 good=0 conv=0` — all three pass.

## Baseline Full-Sweep Results

```
FAIL README.md
     Catalog Error: Table Function with name ols_fit does not exist!
FAIL guides/01_quick_start.md
     IO Error: Extension "build/release/extension/anofox_stats/anofox_stats.duckdb_extension" not found.
FAIL guides/02_technical_guide.md
     Catalog Error: Table with name data does not exist!
FAIL guides/03_business_guide.md
     Catalog Error: Scalar Function with name anofox_stats_ols_fit does not exist!
FAIL guides/04_advanced_use_cases.md
     Catalog Error: Table with name historical_sales does not exist!
FAIL docs/API_REFERENCE.md
     Binder Error: Referenced column "y_array" was not found because the FROM clause is missing
PASS docs/API_CONVENTIONS.md (1 block(s))

Executed: 7 file(s)  Passed: 1  Failed: 6
```

**Baseline count: 6/7 files fail.** Failure categories:
- Old extension path (`anofox_stats` directory vs `anofox_statistics`): guides/01_quick_start.md
- Old function prefix (`anofox_stats_ols_fit`): guides/03_business_guide.md
- Missing table references (external schema assumed): guides/02, guides/04, docs/API_REFERENCE.md
- Wrong function form (scalar `ols_fit` vs aggregate pattern): README.md

## Skip Convention Established

```
```sql skip
-- This block is excluded from harness validation.
```
```

- Info-string `sql skip` → excluded from execution
- Bare `sql` → executed
- Single-regex decision per fence; renders as `sql`-highlighted code in markdown viewers
- Documented in the harness module docstring

## Task Commits

1. **Task 1: Build scripts/validate_docs_sql.py end-to-end** — `3f5e277` (feat)
2. **Task 2: Self-test + skip-mark API_CONVENTIONS.md migration blocks** — `d4a5216` (fix)

## Files Created/Modified

- `scripts/validate_docs_sql.py` — 275-line Python harness; REPO/DUCKDB/EXT path resolution, block extraction, skip filtering, per-file DuckDB session, .bail on, --file flag with path-traversal guard
- `docs/API_CONVENTIONS.md` — 5 sql blocks changed to `sql skip` (2 migration Before examples + 3 syntax-illustration blocks referencing FROM tbl)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Skip-marked 3 additional illustrative blocks in API_CONVENTIONS.md**
- **Found during:** Task 2 (running self-test triad)
- **Issue:** Plan assumed the 4 "other" API_CONVENTIONS.md blocks would pass unchanged. Three of them (at lines 95, 103, 209) use `FROM tbl` without a CREATE TABLE — these reference non-existent tables and fail with Catalog Error. The plan's assumption that `conv=0` after skip-marking only the 2 migration blocks was incorrect.
- **Fix:** Skip-marked all 3 syntax-illustration blocks (they show API syntax, not runnable examples). The one remaining executable block (r_squared at line 252) uses inline scalar data and passes cleanly.
- **Files modified:** `docs/API_CONVENTIONS.md`
- **Verification:** Self-test triad: `bad=1 good=0 conv=0` — compound test exits 0.
- **Committed in:** `d4a5216` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in plan's assumption about pre-existing block validity)
**Impact on plan:** Fix is strictly correct — illustrative convention blocks showing syntax with placeholder `tbl` were always meant to be skipped; the plan mis-categorized them as executable. No scope creep; `conv=0` success criterion met.

## Issues Encountered

None beyond the deviation above.

## Stub Scan

No stubs, TODOs, or FIXMEs introduced in this plan.

## Threat Flags

No new security-relevant surface introduced beyond what the plan's threat model covers (T-6-01 / T-6-01b — path-traversal mitigation implemented and documented).

## Next Phase Readiness

- Harness fully operational; Plans 02/03 can use `python3 scripts/validate_docs_sql.py --file <file>` for per-file fast iteration
- Baseline failure count recorded: 6/7 files; Plans 02/03 fix sweep starts from this baseline
- `docs/API_CONVENTIONS.md` already passes the harness; no further fixes needed there

---
*Phase: 06-docs-refresh-sql-validation*
*Completed: 2026-09-02*
