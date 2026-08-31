# Phase 4: Benchmarking & Performance - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a repeatable benchmark harness that measures representative
extension workloads (aggregate dispatch, fit/predict paths, FFI marshalling) and
reports timings; profiles the built extension to surface the top hotspots and
either optimizes each or documents it as inherent (with before/after numbers);
and refactors the FFI layer's manual `libc::malloc`/`free` pattern to reduce
per-call overhead and leak risk — all with behavior unchanged (existing
`test/sql` + `cargo test` suites stay green).

Covers PERF-01, PERF-02, PERF-03, PERF-04. Out of scope: new statistical models,
named parameters, rewriting the Rust core numerics/algorithms.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Harness Design
- Harness is driven by SQL benchmark scripts run via a documented shell wrapper
  (`scripts/bench.sh`): it loads the built extension, runs representative queries
  with DuckDB `.timer`, and is invoked with one documented command.
- Workloads cover the three named paths: aggregate dispatch over many GROUP BY
  groups, scalar array fit/predict, and an FFI-marshalling micro-benchmark.
  Reuse the `examples/performance_1m_groups` dataset shape for representative data.
- Results are reported as a timings table to stdout and also written to a results
  file (markdown/CSV) under a `bench/` location so before/after runs are diffable.
- CI perf tracking is documented/noted as optional only this phase — NOT wired
  into a CI gate (PERF-02 requires it "noted", not enforced).

### Profiling & Hotspot Optimization
- Profile using `perf` + DuckDB `EXPLAIN ANALYZE` on a Linux native release build
  (the LTO / codegen-units=1 / O3 release profile is already configured);
  `cargo flamegraph` for the Rust core where useful.
- Address the top 3 surfaced hotspots.
- Bar for action: optimize when a safe, behavior-preserving win exists; otherwise
  document the hotspot as inherent with rationale (the goal permits "optimized OR
  documented as inherent").
- Behavior-preservation gate: the full `test/sql` + `cargo test` suites stay green;
  before/after numbers are produced by the Phase-4 harness.

### FFI Allocation Refactor (PERF-04)
- Approach: a Rust-side allocation helper / RAII abstraction PLUS a codegen macro
  for the repetitive fit-result marshalling (CONCERNS.md flags 185+ `libc::malloc`
  sites in a 7,893-line `lib.rs`).
- Keep the existing C++ `anofox_free_*` free contract byte-identical — this is a
  Rust-side refactor only, no C++ changes — so behavior is unchanged and risk is low.
- Scope: cover the bulk fit-result marshalling pattern (the dominant, repeated
  allocation sites); genuinely one-off or risky sites may be left as-is with a
  documented safety comment rather than force-converted.
- Verification: `cargo test` + `test/sql` stay green, plus a noted leak check
  (valgrind / ASan) to confirm no new leaks were introduced.

### Claude's Discretion
- Exact file layout under `bench/`, results file format details (markdown vs CSV),
  the specific query shapes/row counts per workload, the macro name/signature, and
  which specific hotspots get optimized vs documented — all at Claude's discretion,
  guided by the decisions above and codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `examples/performance_1m_groups/` — existing large-group performance example;
  reuse its dataset shape for the aggregate-dispatch workload.
- Release profile already tuned for profiling representativeness: LTO enabled,
  codegen-units=1, opt-level=3 (STACK.md).
- `test/sql/` (50+ files) and `cargo test` in `crates/anofox-stats-*` are the
  behavior-preservation oracle — results must be unchanged.

### Established Patterns
- Three-layer architecture: C++ DuckDB adapters (`src/`) → C FFI
  (`crates/anofox-stats-ffi/src/lib.rs`) → Rust core (external crates).
- FFI allocation contract: Rust allocates result arrays with `libc::malloc`
  (e.g. `lib.rs:164, 193-197`), C++ frees them via the paired `anofox_free_*`
  functions (e.g. `lib.rs:277-308`). The refactor must preserve this contract.
- Aggregate/fit/window functions follow consistent per-file templates
  (`{method}_aggregate.cpp`, `{method}_fit.cpp`, `{method}_fit_predict.cpp`).

### Integration Points
- New benchmark harness: `scripts/bench.sh` + SQL scripts + `bench/` results dir;
  documented via a run command (likely README/guide + Makefile target).
- FFI refactor lives entirely inside `crates/anofox-stats-ffi/src/` (lib.rs, types.rs).
- Profiling targets the native release build of the extension.

</code_context>

<specifics>
## Specific Ideas

- CONCERNS.md "Scaling and Performance Concerns" already names the target pattern:
  large inference requests allocate many independent arrays via `libc::malloc`
  (one per coefficient, std error, CI bound, etc.) — this is the primary PERF-04
  and hotspot candidate.
- The FFI bridge file is 7,893 lines of repetitive wrapping; a `generate_fit_function!`
  style macro is explicitly suggested in CONCERNS.md as the reduction lever.

</specifics>

<deferred>
## Deferred Ideas

- Wiring perf tracking into a gating CI job (this phase only notes it as optional).
- Broad FFI integration test suite / cross-platform FFI tests (a test-coverage gap
  noted in CONCERNS.md, beyond this phase's behavior-preservation scope).
- Extracting a telemetry abstraction, argmin fork upstreaming, and other CONCERNS.md
  tech-debt items unrelated to performance.

</deferred>
