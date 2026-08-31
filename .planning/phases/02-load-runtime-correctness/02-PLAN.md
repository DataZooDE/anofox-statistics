# Phase 2 Plan: Load & Runtime Correctness

**Created:** 2026-08-30
**Requirements:** LOAD-01, LOAD-02
**Verification mode:** implement-now / verify-in-CI

## Approach

Implement the load + representative-correctness checks as the first checks of the
reusable `test/wasm/` harness (rather than a throwaway smoke script). The harness
boots DuckDB-Wasm in Node, loads the locally-built `.wasm`, and asserts
representative function results against the native-validated expected values.

## Tasks

- [x] **T1 (LOAD-01)** — Harness boots DuckDB-Wasm (eh bundle, `web-worker@1.2.0`,
  `allowUnsignedExtensions`), serves the built `.wasm` locally, and
  `FORCE INSTALL … FROM` + `LOAD`s it. A load error is a hard, non-zero exit.
- [x] **T2 (LOAD-02)** — Curated `.test` subset asserts representative results:
  OLS (`test_fit_agg`, `ols_fit_predict_basic`, `ols_basic`), a hypothesis test
  (`t_test_agg`), normality (`jarque_bera_agg`), correlation (`pearson_agg`),
  diagnostics (`vif_agg`) — compared against the same expected values as native.
- [ ] **T3 (verify)** — Run in CI against the v1.5.5 `wasm_eh` artifact. **← CI gate.**

## Success Criteria → Evidence

1. `.wasm` LOADs under Node with no error → harness `LOAD` step, hard-fail on error (T1).
2. `ols_fit` coefficients match native within tolerance → curated OLS files +
   numeric-tolerant comparison (T2).
3. Representative aggregate/test returns correct stats → t-test/jarque-bera/pearson
   files (T2).

## Notes

- Correctness parity is achieved by reusing the existing `.test` expectations
  (validated on native) rather than re-deriving values.
- Full-suite + CI wiring + docs are Phase 3.
