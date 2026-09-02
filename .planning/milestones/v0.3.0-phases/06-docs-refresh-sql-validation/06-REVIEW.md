---
phase: 06-docs-refresh-sql-validation
reviewed: 2026-09-02T00:00:00Z
depth: deep
files_reviewed: 9
files_reviewed_list:
  - scripts/validate_docs_sql.py
  - .github/workflows/DocsSqlValidation.yml
  - docs/API_CONVENTIONS.md
  - docs/API_REFERENCE.md
  - guides/01_quick_start.md
  - guides/02_technical_guide.md
  - guides/03_business_guide.md
  - guides/04_advanced_use_cases.md
  - README.md
findings:
  critical: 2
  warning: 2
  info: 0
  total: 4
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-09-02
**Depth:** deep
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 6 adds a Python validation harness (`scripts/validate_docs_sql.py`), a CI workflow
(`.github/workflows/DocsSqlValidation.yml`), and extensive doc edits to README, four guides,
and the two `docs/` files.  The harness architecture — extract fenced `sql` blocks, concatenate
per-file in document order with `.bail on`, run via `duckdb -unsigned -f`, report non-zero on
failure — is sound.  The workflow structure, trigger conditions, permissions, and submodule
checkout are all correct.  The path-traversal guard for `--file` is robust.  The `skip`
convention is correctly specified and the existing skip marks in the docs are legitimate
(migration examples, network-dependent installs, engine-buggy window patterns, illustrative
patterns requiring external tables).

Two blockers found, both in `guides/01_quick_start.md`: the Troubleshooting section contains
two executable blocks that are intentionally designed to produce errors, but neither carries
the `skip` info-string.  Because blocks are concatenated into one `.bail on` session, the first
failing statement aborts the entire file, so the harness will permanently report
`guides/01_quick_start.md` as FAIL.  This means the CI drift gate is broken on day one: every
PR fails this check regardless of content, so developers will learn to ignore it — the worst
possible failure mode for a drift gate.

Two warnings: a latent false-negative in the regex closer design (harmless in the current docs
but fragile for future edits), and two missing fields in the API_CONVENTIONS.md standard-field
table.

---

## Critical Issues

### CR-01: "This will error" block is not skip-marked — harness permanently fails

**File:** `guides/01_quick_start.md:322`

**Issue:** The "Insufficient Observations" troubleshooting block is a plain `sql` fence
(not `sql skip`) and contains a statement that deliberately throws `InvalidInputException`:

```sql
-- This will error
SELECT ols_fit([1.0, 2.0], [[1.0, 2.0]]);

-- Need at least 3 points
SELECT ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]);
```

`error_handling.test` confirms `ols_fit` with two observations raises "Insufficient data".
The harness prepends `.bail on` to the concatenated SQL session; the failing `SELECT` statement
aborts the session and DuckDB exits non-zero before the "Need at least 3 points" recovery
statement can run.  The harness therefore reports `guides/01_quick_start.md` as `FAIL` on
every run.  The CI job fails every PR, making the drift gate permanently useless.

**Fix:** Change the fence info-string to `sql skip` and add a comment explaining why:

```markdown
### Insufficient Observations
Minimum 3 observations required for single-feature regression with intercept:

```sql skip
-- Illustrative error: ols_fit([1.0, 2.0], [[1.0, 2.0]]) raises InvalidInputException
-- because n < n_features + 1.  Not executed by the harness.
SELECT ols_fit([1.0, 2.0], [[1.0, 2.0]]);

-- Correct form — this is the runnable version:
SELECT ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]);
```
```

(Alternatively, split into two blocks: one `sql skip` showing the error, one plain `sql`
showing the working form.)

---

### CR-02: "Wrong: integer arrays" block is not skip-marked — may also fail the harness

**File:** `guides/01_quick_start.md:309`

**Issue:** The "Type Errors" troubleshooting block (immediately before CR-01 in the same
concatenated session) is a plain `sql` fence and contains:

```sql
-- Wrong: integer arrays
SELECT ols_fit([1, 2, 3], [[1, 2, 3]]);
```

`ols_fit` declares its argument types as `LIST(DOUBLE)` and `LIST(LIST(DOUBLE))`.  The
comment labels this the "wrong" form, implying the author expects a type error.  Whether
DuckDB auto-casts `INTEGER[]` to `DOUBLE[]` at bind time is version-dependent and not
guaranteed by the extension's type declarations.  If it does fail (or if a future DuckDB
version stops accepting the implicit cast), `.bail on` aborts the session before the
"Correct" statement runs and the harness again reports `FAIL`.

Even if DuckDB currently accepts the cast silently (making the "teaching" intent incorrect),
the block contains a statement the author explicitly labelled as wrong.  Running it under a
drift gate is misleading: a passing result would mean the "wrong" syntax is actually accepted,
not that the correct syntax works.

**Fix:** Apply `sql skip` to this block:

```markdown
### Type Errors
Ensure all numeric values are DOUBLE:

```sql skip
-- This syntax relies on implicit INTEGER->DOUBLE casting, which is not guaranteed.
-- Prefer explicit DOUBLE literals (see below).
SELECT ols_fit([1, 2, 3], [[1, 2, 3]]);
```

```sql
-- Correct: explicit DOUBLE literals
SELECT ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]);
```
```

---

## Warnings

### WR-01: Regex closer `^``` ` matches any fence opener — latent false-negative risk

**File:** `scripts/validate_docs_sql.py:85-88`

**Issue:** The extraction regex is:

```python
BLOCK_RE = re.compile(
    r"^```sql( skip)?\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)
```

The closing anchor `^``` ` matches any line that *starts with* three backticks — including
other fence openers like ```` ```sql ````, ```` ```python ````, ```` ```bash ````.  If a future
author writes two consecutive sql blocks with no blank line between them (e.g. the output of
an AI-assisted doc edit), the opener of the second block closes the first block early and the
second block body is never extracted.  The harness silently skips that SQL rather than
executing it — a false negative in the drift gate.

Verified by direct test:
```python
text = "```sql\nSELECT 1;\n```sql\nSELECT 2;\n```"
matches = list(BLOCK_RE.finditer(text))
# → 1 match, SELECT 2 is silently lost
```

No such case currently exists in the docs, so this is not an active bug.  But the design
is fragile: a single doc edit by any contributor would silently suppress SQL from the gate.

**Fix:** Require the closing fence to be a bare line (only backticks, optional trailing
whitespace, then end-of-line):

```python
BLOCK_RE = re.compile(
    r"^```sql( skip)?\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
```

The `[ \t]*$` ensures only a bare closing `` ``` `` terminates the block; a line like
```` ```python ```` does not match, so the block stays open until its correct closer.

---

### WR-02: `docs/API_CONVENTIONS.md` §3 standard-field table omits `f_statistic` / `f_pvalue`

**File:** `docs/API_CONVENTIONS.md:136-152`

**Issue:** The "Return-Struct Field Names" section lists the standard field set for regression
families, but does not include `f_statistic` or `f_pvalue`.  Both fields are real and active:
they appear in `src/table_functions/ols_fit.cpp:39-40`, `src/aggregate_functions/*_aggregate.cpp`,
`crates/anofox-stats-ffi/src/types.rs:284`, and `test/sql/comprehensive_tests.test:791,1030`.
`guides/01_quick_start.md:71-72` references them in a non-skipped executable block.

The conventions doc is intended as the authoritative field contract.  Omitting two real output
fields means callers relying on the conventions doc will not know these fields exist, and doc
authors reviewing the table may wrongly skip-mark examples that use them.

**Fix:** Add two rows to the standard-field table (they are only present when
`compute_inference: true`):

```markdown
| `f_statistic` | DOUBLE | F-statistic for the overall regression (when `compute_inference: true`) |
| `f_pvalue`    | DOUBLE | p-value for the F-statistic (when `compute_inference: true`) |
```

Add a footnote explaining these fields appear alongside `std_errors`, `t_values`, etc.

---

_Reviewed: 2026-09-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
