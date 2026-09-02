# Roadmap: Anofox Statistics

## Milestones

- ✅ **v0.1.0 Native statistics suite** — shipped (regression, GLM, GLMM, AFT, 40+ tests, diagnostics; linux/osx amd64+arm64)
- ✅ **v0.2.0 WASM Support** — Phases 1-3 (shipped 2026-08-31)
- ✅ **v0.3.0 Performance & Polish** — Phases 4-6 (shipped 2026-09-02)

Next milestone: TBD — run `/gsd-new-milestone`.

## Phases

<details>
<summary>✅ v0.2.0 WASM Support (Phases 1-3) — SHIPPED 2026-08-31</summary>

- [x] Phase 1: WASM Build Green — completed 2026-08-31
- [x] Phase 2: Load & Runtime Correctness — completed 2026-08-31
- [x] Phase 3: Automated Harness & CI Gate — completed 2026-08-31

Full detail: [milestones/v0.2.0-ROADMAP.md](milestones/v0.2.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.3.0 Performance & Polish (Phases 4-6) — SHIPPED 2026-09-02</summary>

- [x] Phase 4: Benchmarking & Performance (3/3 plans) — completed 2026-08-31 — bench harness + FFI `FfiVec`/macro refactor + hotspot dispositions (PERF-01..04)
- [x] Phase 5: API Ergonomics (3/3 plans) — completed 2026-09-02 — typed errors + early validation + breaking cross-family rename, no aliases (ERGO-01..03)
- [x] Phase 6: Docs Refresh & SQL Validation (4/4 plans) — completed 2026-09-02 — README anofox-forecast form + `validate_docs_sql.py` harness + CI drift gate (DOCS-01..04)

Full detail: [milestones/v0.3.0-ROADMAP.md](milestones/v0.3.0-ROADMAP.md) · Audit: [milestones/v0.3.0-MILESTONE-AUDIT.md](milestones/v0.3.0-MILESTONE-AUDIT.md)

</details>
