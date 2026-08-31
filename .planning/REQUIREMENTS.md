# Requirements: Anofox Statistics — v0.3.0 (Performance & Polish)

**Defined:** 2026-08-31
**Core Value:** Users can run rigorous statistical models in plain SQL wherever DuckDB runs — this milestone makes that faster and easier to use.

## v0.3.0 Requirements

Requirements for the Performance & Polish milestone. Each maps to a roadmap phase.
API changes may be **breaking** (early-dev); docs/tests are updated to match.

### Performance (PERF)

- [ ] **PERF-01**: A repeatable benchmark harness measures representative workloads (aggregate dispatch, fit/predict paths, FFI marshalling) and reports timings
- [ ] **PERF-02**: Benchmark runs are reproducible locally and documented (how to run, what they cover); optional CI perf tracking noted
- [ ] **PERF-03**: The top hotspots surfaced by profiling are each optimized or explicitly documented as inherent, with before/after numbers from the benchmark
- [ ] **PERF-04**: The FFI layer's manual `libc::malloc`/`free` pattern is refactored (RAII wrapper and/or codegen macros) reducing per-call overhead and leak risk, with results unchanged (tests still green)

### Ergonomics (ERGO)

- [ ] **ERGO-01**: Fit/predict/test functions return clear, actionable error messages for invalid input (dimension mismatch, insufficient rows, non-finite/constant columns) instead of panics or opaque errors
- [ ] **ERGO-02**: Inputs are validated early with specific messages naming the offending argument and expected shape
- [ ] **ERGO-03**: Function signatures, option-map keys, and return-struct field names follow one documented convention consistent across model families

### Documentation (DOCS)

- [ ] **DOCS-01**: README is restructured to match the anofox-forecast form — emoji section headers, Table of Contents, Key Features (incl. ⚡ Performance and 🎨 User-Friendly API), a Quick Start walkthrough on a concrete dataset, structured API Reference, Development, Support, Citation
- [ ] **DOCS-02**: A doc-SQL validation harness extracts every SQL example from README + `guides/*.md` + `docs/API_REFERENCE.md` and runs it against the built extension, reporting pass/fail per example
- [ ] **DOCS-03**: All documented SQL examples pass — broken examples are fixed (whether from drift or intentional API changes)
- [ ] **DOCS-04**: The doc-SQL validation runs in CI so documentation drift fails the build

## Future Requirements

Deferred to a later milestone.

### Ergonomics

- **ERGOX-01**: Named parameters (`param := value`) instead of positional-only — larger surface change; deferred

## Out of Scope

| Feature | Reason |
|---------|--------|
| Named parameters | Larger API surface change; deferred to a dedicated milestone |
| New statistical models | This milestone is polish, not new capability |
| Rewriting the Rust core numerics | Perf work targets marshalling/allocation + hotspots, not re-deriving algorithms |

## Traceability

Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | TBD | Pending |
| PERF-02 | TBD | Pending |
| PERF-03 | TBD | Pending |
| PERF-04 | TBD | Pending |
| ERGO-01 | TBD | Pending |
| ERGO-02 | TBD | Pending |
| ERGO-03 | TBD | Pending |
| DOCS-01 | TBD | Pending |
| DOCS-02 | TBD | Pending |
| DOCS-03 | TBD | Pending |
| DOCS-04 | TBD | Pending |

**Coverage:**
- v0.3.0 requirements: 11 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 11 ⚠️

---
*Requirements defined: 2026-08-31*
*Last updated: 2026-08-31 after initial definition*
