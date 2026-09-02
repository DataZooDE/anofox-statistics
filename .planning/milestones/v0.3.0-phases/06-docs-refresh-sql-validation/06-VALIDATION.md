---
phase: "6"
slug: "docs-refresh-sql-validation"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-02"
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | The doc-SQL harness itself (`scripts/validate_docs_sql.py`) is the DOCS-02/03 test runner; existing `ctest`/sqllogictest (`build/release/test/unittest --test-dir=test/sql`) covers ERGO/PERF regression |
| **Config file** | None — harness self-configures (resolves repo root from `__file__`) |
| **Quick run command** | `python3 scripts/validate_docs_sql.py --file <path>` |
| **Full suite command** | `python3 scripts/validate_docs_sql.py` (all 7 doc files) + `build/release/test/unittest --test-dir=test/sql` |
| **Estimated runtime** | ~30–90s for the doc sweep once the extension is built |

---

## Sampling Rate

- **After every task commit:** `python3 scripts/validate_docs_sql.py --file <file-being-edited>`
- **After every plan wave:** full `python3 scripts/validate_docs_sql.py` sweep
- **Before `/gsd-verify-work`:** full harness green AND `build/release/test/unittest --test-dir=test/sql` green (no regression to ERGO/PERF)
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 1 | DOCS-02 | T-6-01 / untrusted doc SQL | Harness runs each block against the built ext, isolates failures, exits non-zero on any failure | harness self-test | `python3 scripts/validate_docs_sql.py --file <known-bad tmp>` exits 1; trivial-good exits 0 | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 2 | DOCS-03 | — | Every extracted sql block (minus `sql skip`) passes against the built extension | full harness | `python3 scripts/validate_docs_sql.py` exits 0 | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 3 | DOCS-01 | — | README in anofox-forecast form; its Quick Start block passes the harness | structure review + harness | `python3 scripts/validate_docs_sql.py --file README.md` exits 0 | ❌ W0 | ⬜ pending |
| 6-04-01 | 04 | 4 | DOCS-04 | T-6-02 / CI bypass | Harness runs in CI on PR + push to main; drift fails the build | CI run | GitHub Actions `DocsSqlValidation.yml` green | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/validate_docs_sql.py` — the harness (TRACER; must exist + self-test green before the DOCS-03 fix sweep). Self-test: known-bad temp file → exit 1; trivial correct block → exit 0.
- [ ] `.github/workflows/DocsSqlValidation.yml` — CI wrapper (final wave)

*Existing `test/sql/*` regression suite already covers ERGO-03 naming (`ergo03_naming.test`) — not a Wave 0 gap.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README reads as anofox-forecast form (emoji headers, ToC, Key Features ⚡/🎨, section order, License at end) | DOCS-01 | Visual/structural judgment | Review rendered README against CONTEXT.md section order; confirm each required section present |

*All executable-example behavior is covered by the harness; only README structure is manual.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (harness + CI workflow)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
