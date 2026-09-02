---
phase: 4
slug: benchmarking-performance
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-31
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | DuckDB SQL test harness (`test/sql/*.test`) + `cargo test` (Rust crates) |
| **Config file** | none — existing `test/sql` suite + Cargo workspace |
| **Quick run command** | `cargo test -p anofox-stats-ffi` (Rust FFI unit tests, fast) |
| **Full suite command** | `make test` (DuckDB `test/sql` behavior oracle) + `cargo test` (workspace) |
| **Estimated runtime** | ~120 seconds (build excluded) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p anofox-stats-ffi`
- **After every plan wave:** Run `make test` (full `test/sql`) + `cargo test`
- **Before `/gsd-verify-work`:** Full suite must be green (results unchanged from baseline)
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | PERF-01/02 | — | N/A | integration | `bash scripts/bench.sh` exits 0 and writes a results file | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 2 | PERF-04 | — | N/A (behavior unchanged) | unit | `cargo test -p anofox-stats-ffi` | ✅ | ⬜ pending |
| 4-02-02 | 02 | 2 | PERF-04 | — | N/A (behavior unchanged) | integration | `make test` (test/sql green, results unchanged) | ✅ | ⬜ pending |
| 4-03-01 | 03 | 3 | PERF-03 | — | N/A | integration | before/after benchmark numbers recorded in results file | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/bench.sh` + benchmark SQL scripts — the PERF-01 harness (created in Plan 01) is itself the validation vehicle for PERF-03 before/after numbers
- [ ] `perf` installed on the dev box (`sudo pacman -S perf`) — required for PERF-03 profiling; noted as an environment prerequisite

*Existing `test/sql` + `cargo test` infrastructure covers the behavior-preservation oracle for PERF-04.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Hotspot is "optimized OR documented as inherent" | PERF-03 | Judgment call on whether a win is safe/available; before/after numbers are automated but the optimize-vs-document decision is human | Run `scripts/bench.sh` before and after; inspect profiler output; record decision + numbers in the phase perf notes |
| No new memory leaks after FFI refactor | PERF-04 | Leak check (valgrind/ASan) is a noted, environment-dependent check, not wired into the standard suite | Run a valgrind/ASan pass over a representative fit workload; confirm no new leaks vs. baseline |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
