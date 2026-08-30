# Phase 2: Load & Runtime Correctness - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning
**Mode:** Autonomous (implement-now/verify-in-CI)

<domain>
## Phase Boundary

The `.wasm` produced by Phase 1 loads without error in DuckDB-Wasm under Node,
and representative statistical functions return correct results — proving the
extension is functional at runtime, not merely buildable.

Requirements: LOAD-01, LOAD-02.
</domain>

<decisions>
## Implementation Decisions

- **The Phase 3 Node harness IS Phase 2's verification.** Rather than a throwaway
  manual smoke script, the load check and representative-result assertions are
  implemented as the first checks of the reusable harness (`test/wasm/`). Phase 2
  owns the *load + spot-correctness* contract; Phase 3 owns *generalizing it over
  the suite + CI + docs*.
- **Correctness = parity with native/known values.** Representative functions
  (`anofox_stats_ols_fit` r²/intercept/coefficient; an aggregate; a hypothesis
  test) are asserted against the same expected values already encoded in the
  `.test` files, which are validated on native.
- **No local `.wasm` available** → the assertions are authored against the
  documented DuckDB-Wasm behavior and run in CI once Phase 1's wasm legs produce
  the artifact.
</decisions>

<code_context>
## Existing Code Insights

- SQL tests are DuckDB sqllogictest `.test` files (`require anofox_statistics`,
  `query <types>` / `statement ok|error`, `----` expected rows). 99 files under
  `test/sql/`. A small subset drives the Phase 2 smoke (e.g.
  `test/sql/ols_basic.test`, one aggregate, one hypothesis test).
- Representative anchor (`test/sql/anofox_stats.test`): `ols_fit` on
  y=2x+1 → r²=1.0, intercept=1.0, coefficient[1]=2.0.
</code_context>

<specifics>
## Specific Ideas

- Load path: DuckDB-Wasm with `allowUnsignedExtensions`, install the locally
  built extension from a served repo dir matching `<duckdb_version>/<wasm_platform>/`.
- Smoke asserts: (1) `LOAD` succeeds; (2) `ols_fit` returns the known coefficients
  within tolerance; (3) one aggregate + one test function return correct stats.
</specifics>

<deferred>
## Deferred Ideas

- Full-suite execution and CI wiring belong to Phase 3.
</deferred>
