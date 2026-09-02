---
phase: "5"
slug: "api-ergonomics"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-08-12"
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | DuckDB SQL test runner (`ctest` via `make test`) + `cargo test` |
| **Config file** | `CMakeLists.txt` (DuckDB test discovery) + `test/sql/*` |
| **Quick run command** | `cd build && ctest -R "ols" --output-on-failure` |
| **Full suite command** | `make test && cargo test --workspace` |
| **Estimated runtime** | ~120 seconds (native suite) |

---

## Sampling Rate

- **After every task commit:** Run the targeted `ctest -R <family>` for the touched family
- **After every plan wave:** Run `make test && cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite must be green against the renamed API
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | ERGO-01 | T-5-01 / malformed input | Degenerate window frame (n<p+1) returns NULL, no INTERNAL crash | SQL assertion | `cd build && ctest -R window_null --output-on-failure` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 2 | ERGO-01 | T-5-01 | Invalid input → typed DuckDB exception with actionable message, not panic/opaque | SQL expect-error | `cd build && ctest -R ergo01 --output-on-failure` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 2 | ERGO-02 | T-5-02 / option injection | Unknown MAP option key rejected at bind, valid keys listed | SQL expect-error | `cd build && ctest -R unknown_option --output-on-failure` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 3 | ERGO-03 | — | All function names/option keys/return fields follow the documented convention | SQL smoke test | `cd build && ctest -R naming --output-on-failure` | ❌ W0 | ⬜ pending |
| 5-03-02 | 03 | 3 | ERGO-03 | — | Renamed API green: `.r2`→`.r_squared`, `theilsen`→`theil_sen`, alias blocks removed | Existing suites | `make test && cargo test --workspace` | ✅ (fails until renamed) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/sql/ergo01_window_null.sql` — rolling window with fewer than n_features+1 rows at partition start returns NULL (repro + fix for the INTERNAL crash)
- [ ] `test/sql/ergo01_clear_errors.sql` — error-message format for dimension mismatch, insufficient rows, all-non-finite, constant column
- [ ] `test/sql/ergo02_unknown_option.sql` — `{'unknow_key': 1}` throws at bind, listing valid keys
- [ ] `test/sql/ergo03_naming.sql` — smoke test that the renamed function names resolve and old prefixed names do not

*Existing 150 `test/sql/*` files that reference `.r2` / `theilsen` / `anofox_stats_*` are updated in the rename task (Plan 03), not Wave 0.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Error messages read as "clear and actionable" to a human | ERGO-01 | Message quality is a judgment call beyond string-match | Review a sample of triggered errors; confirm each names the function, offending argument, and expected vs actual shape |
| `docs/API_CONVENTIONS.md` accurately describes the shipped convention | ERGO-03 | Doc/API consistency is validated end-to-end in Phase 6 | Spot-check the written convention against 3 functions per family |

*All behavioral checks below the message-quality bar have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
