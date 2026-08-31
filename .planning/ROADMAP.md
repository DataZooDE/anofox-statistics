# Roadmap: Anofox Statistics

## Milestones

- ✅ **v0.2.0 WASM Support** — Phases 1-3 (shipped 2026-08-31)
- 🚧 **v0.3.0 Performance & Polish** — Phases 4-6 (in progress)

## Overview

v0.3.0 makes the extension measurably faster and easier to use. First we build a
benchmark harness that becomes the measurement foundation, then profile and
optimize hotspots and refactor the FFI's manual malloc/free pattern with
before/after numbers. Next we make the API ergonomic — clear errors, early
validation, and a consistency pass across model families (breaking renames
allowed in early-dev). Finally we refresh the README to the anofox-forecast
form and validate every documented SQL example against the final API, gating
documentation drift in CI.

## Phases

**Phase Numbering:**

- Integer phases (4, 5, 6): Planned milestone work
- Decimal phases (5.1, 5.2): Urgent insertions (marked with INSERTED)

Continuous numbering from v0.2.0 (which ended at Phase 3).

- [ ] **Phase 4: Benchmarking & Performance** - Bench harness first, then profile/optimize hotspots and refactor the FFI alloc pattern with before/after numbers
- [ ] **Phase 5: API Ergonomics** - Clear errors + early validation, then a consistency pass on signatures/option keys/return fields (breaking renames)
- [ ] **Phase 6: Docs Refresh & SQL Validation** - Restructure README to anofox-forecast form, validate every documented SQL example against the final API, gate in CI

## Archived Phases

<details>
<summary>✅ v0.2.0 WASM Support (Phases 1-3) — SHIPPED 2026-08-31</summary>

- [x] Phase 1: WASM Build Green — completed 2026-08-31
- [x] Phase 2: Load & Runtime Correctness — completed 2026-08-31
- [x] Phase 3: Automated Harness & CI Gate — completed 2026-08-31

Full detail: [milestones/v0.2.0-ROADMAP.md](milestones/v0.2.0-ROADMAP.md)

</details>

### 🚧 v0.3.0 Performance & Polish (In Progress)

**Milestone Goal:** Make the extension measurably faster and easier to use — a
benchmark suite + FFI/allocation refactor + hotspot optimization, clearer errors
and consistent APIs, and a refreshed README (anofox-forecast form) with every
documented SQL example validated in CI. Breaking API changes are permitted
(early-dev); docs and tests are updated to match.

## Phase Details

### Phase 4: Benchmarking & Performance

**Goal**: The extension has a repeatable benchmark harness that measures representative workloads, and the surfaced hotspots plus the FFI allocation pattern are optimized with before/after numbers proving the improvement — behavior unchanged.
**Depends on**: Phase 3 (v0.2.0 shipped; native + WASM suites green)
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04
**Success Criteria** (what must be TRUE):

  1. A user can run one documented command to execute the benchmark suite over representative workloads (aggregate dispatch, fit/predict paths, FFI marshalling) and get reported timings
  2. Benchmark runs reproduce locally with documented scope (what they cover, how to run); optional CI perf tracking is noted
  3. Each top hotspot surfaced by profiling is either optimized or explicitly documented as inherent, each with before/after numbers from the benchmark
  4. The FFI layer's manual `libc::malloc`/`free` pattern is refactored (RAII wrapper and/or codegen macros) with per-call allocation overhead reduced, and the existing `test/sql` + `cargo test` suites stay green (results unchanged)

**Plans**: 3/3 plans executed

- [x] 04-01-PLAN.md — Benchmark harness (tracer): scripts/bench.sh + three workload SQL scripts + bench/README.md (PERF-01/02)
- [x] 04-02-PLAN.md — FFI allocation refactor: FfiVec<T> RAII wrapper + alloc_inference_arrays! macro across 13 sites (PERF-04)
- [x] 04-03-PLAN.md — Profiling & hotspot optimization: top-3 hotspots optimized or documented inherent with before/after numbers (PERF-03)

### Phase 5: API Ergonomics

**Goal**: Fit/predict/test functions fail fast with clear, actionable messages for invalid input, and signatures, option-map keys, and return-struct field names follow one documented convention consistent across model families.
**Depends on**: Phase 4
**Requirements**: ERGO-01, ERGO-02, ERGO-03
**Success Criteria** (what must be TRUE):

  1. Invalid input (dimension mismatch, insufficient rows, non-finite or constant columns) to a fit/predict/test function returns a clear, actionable error message instead of a panic or opaque error
  2. Inputs are validated early (at bind time where possible) with a specific message naming the offending argument and its expected shape
  3. Function signatures, option-map keys, and return-struct field names across model families follow one documented naming convention, with the convention written down
  4. Any breaking renames from the consistency pass are reflected in the test suite (`test/sql` + `cargo test` green against the new names)

**Plans**: TBD

### Phase 6: Docs Refresh & SQL Validation

**Goal**: The README matches the anofox-forecast form and every documented SQL example across README, guides, and API reference is validated against the built extension in CI, so documentation drift fails the build.
**Depends on**: Phase 5 (examples validated against the final, renamed API)
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):

  1. The README follows the anofox-forecast form — emoji section headers, Table of Contents, Key Features (incl. ⚡ Performance and 🎨 User-Friendly API), a Quick Start walkthrough on a concrete dataset, structured API Reference, Development, Support, Citation
  2. A doc-SQL validation harness extracts every SQL example from README + `guides/*.md` + `docs/API_REFERENCE.md` and runs each against the built extension, reporting pass/fail per example
  3. Every extracted SQL example passes — examples broken by drift or by the Phase 5 API changes are fixed
  4. The doc-SQL validation runs in CI so any future documentation drift fails the build

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 4 → 5 → 6

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 4. Benchmarking & Performance | v0.3.0 | 3/3 | Complete    | 2026-08-31 |
| 5. API Ergonomics | v0.3.0 | 0/TBD | Not started | - |
| 6. Docs Refresh & SQL Validation | v0.3.0 | 0/TBD | Not started | - |
