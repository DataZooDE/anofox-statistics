# Tests

All extension tests are [sqllogictests](https://duckdb.org/dev/sqllogictest/intro.html)
under `test/sql/`, run by DuckDB's `unittest` runner:

```bash
make test                                             # all default tests
build/release/test/unittest "test/sql/reference/*"    # one directory
build/release/test/unittest "test/sql/parallel/*.test_slow"   # slow tests (hidden by default)
```

| Directory | Contents |
|-----------|----------|
| `test/sql/reference/` | Results compared against R / scipy reference values (generators in `validation/generators/`). |
| `test/sql/consistency/` | `*_fit_predict` results equal predictions from the matching `*_fit_agg` coefficients. |
| `test/sql/parallel/` | Multi-threaded (`PRAGMA verify_parallelism`) results equal single-threaded ones; `.test_slow` files use millions of rows. |
| `test/sql/window/` | Window-frame semantics of the fit/fit_predict aggregates. |
| `test/sql/edge_cases/` | ±Inf, empty input, n <= p, collinearity, NULLs inside feature lists, invalid options. |
| other `test/sql/*/` | Functional tests per function family. |

Files ending in `.test_slow` carry the Catch tag `[.]` and are skipped unless selected explicitly.
`test/wasm/` holds the DuckDB-Wasm load/runtime harness used in CI.
