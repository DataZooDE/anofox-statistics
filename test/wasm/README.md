# DuckDB-Wasm test harness

Compiling and linking a DuckDB extension for WebAssembly does **not** prove it
loads or runs — see query.farm's [_Testing DuckDB-WASM Extensions_](https://query.farm/blog/testing-duckdb-wasm-extensions/).
This harness closes that gap: it boots DuckDB-Wasm in Node, loads the
locally-built `anofox_statistics.duckdb_extension.wasm`, and runs the SQL tests
against it.

## What it checks

1. **Load** — the built `.wasm` extension `LOAD`s in DuckDB-Wasm with no error
   (catches missing FFI symbols, raw-HTTP/socket calls at load, ABI mismatch).
2. **Runtime correctness** — representative functions return correct results,
   asserted against the same expected values as the native `test/sql/*.test`
   suite.

## Requirements

- Node.js ≥ 18
- A **WASM build** of the extension (`*.duckdb_extension.wasm`). Produce it with
  the DuckDB extension-ci-tools WASM build, or download the CI artifact.

## Run locally

```bash
# 1. install harness deps (isolated; does not touch the Rust/C++ build)
npm --prefix test/wasm install

# 2. point it at your built extension and run
ANOFOX_WASM_EXT=/path/to/anofox_statistics.duckdb_extension.wasm \
  npm --prefix test/wasm test
```

If `--ext` / `ANOFOX_WASM_EXT` is omitted, the harness searches `build/` for the
artifact.

Options (`node test/wasm/run.mjs ...`):

- `--ext <path>` — path to the built `*.duckdb_extension.wasm`.
- `--all` — run every `test/sql/**/*.test` file (default: a curated WASM subset).
- `--file <t.test>` — run specific file(s).

Exit code is non-zero on any load or assertion failure.

## How it works (notes for maintainers)

- Node bundle: `@duckdb/duckdb-wasm/dist/duckdb-node.cjs` with the **`eh`**
  worker/module, and **`web-worker@1.2.0`** (pinned — 1.5.x throws
  `module is not defined` loading the CJS worker).
- `db.instantiate(mainModule, null)` — the pthread worker **must** be `null` for
  the eh/mvp bundles, else the first extension call fails with
  `TypeError: … is not a function`.
- `db.open({ allowUnsignedExtensions: true })` — required to load an unsigned,
  locally-built extension.
- The built `.wasm` is served over a localhost HTTP server and installed with
  `FORCE INSTALL … FROM '<url>'` (`FORCE` busts DuckDB-Wasm's on-disk Node cache).
  The server is **version-agnostic**: it serves the one built artifact for any
  `<duckdb_version>/<wasm_platform>/` path DuckDB requests, so the harness does
  not need to reconstruct the exact version/platform directory.
- `test/wasm/sqllogic.mjs` is a minimal sqllogictest-subset parser/runner
  (`require`, `statement ok|error`, `query <types>` + `----`, `mode skip`). It is
  intentionally tolerant on numeric comparisons (float tolerance) — not a full
  sqllogictest implementation.

## Version matching (important)

The `@duckdb/duckdb-wasm` engine version must be **ABI-compatible** with the
DuckDB version the extension was built against (the extension ships for DuckDB
v1.5.5 and v1.4.5 LTS). If `LOAD` fails with a version/ABI error, align the
`@duckdb/duckdb-wasm` pin in `package.json` with the matching DuckDB release, and
feed the harness the extension built for that same version. The harness prints
the engine version at startup to make this reconciliation obvious.
