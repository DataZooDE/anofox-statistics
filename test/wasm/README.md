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

## Coverage

The CI gate runs the **full suite** (`--all`, all `test/sql/**/*.test`) against
the built `.wasm` and is green: **2090 assertions passing** across 99 files
(`test/sql/quack.test` is skipped — extension-template boilerplate calling
`quack` / `quack_openssl_version`, functions this extension doesn't define).

Notes on how results are compared (learned the hard way):

- **Results are formatted via DuckDB's own `::VARCHAR`**, not read as JS values.
  duckdb-wasm's Arrow-JS extraction mis-renders `DECIMAL` columns — it returns
  the *unscaled* integer (`1.0` → `10`) — whereas `x::VARCHAR` applies the scale
  (`"1.0"`), matching native sqllogictest output. (This is a duckdb-wasm/Arrow
  quirk, independent of this extension; it was the cause of a spurious
  `MIN(x1)` "failure" during bring-up.)
- Each file runs with an **isolated catalog** (DB re-opened + extension re-loaded
  per file) so `CREATE TABLE` / temp state can't leak across files.
- Multi-column `.test` rows are TAB-separated on one line; booleans render per
  the column type (`query I` → 1/0, `query T` → true/false) — both handled.
- Float comparison uses a loose tolerance (abs 1e-6 / rel 1e-4) to absorb benign
  native-vs-WASM floating-point differences.

## Version matching (important)

The `@duckdb/duckdb-wasm` engine version must **exactly match** the DuckDB version
the extension was built against — DuckDB extensions are ABI-locked to the engine
version. **The npm version is not the engine version**: e.g. `1.29.0` → engine
`v1.1.1`, `1.32.0` → `v1.4.3`, `1.33.1-dev64.0` → `v1.5.5`.

This harness is pinned to **`@duckdb/duckdb-wasm@1.33.1-dev64.0`** (engine
**v1.5.5**), matching the extension's stable target, and the CI job feeds it the
**v1.5.5 `wasm_eh`** artifact. Notes:

- `1.33.1-dev64.0` is a **dev/prerelease** (the `@next` dist-tag today) — no
  *stable* duckdb-wasm ships a 1.5.x engine yet.
- There is no published duckdb-wasm bundling exactly `v1.4.5` (LTS), so the gate
  intentionally targets the v1.5.5 artifact only.

When the extension moves to a new DuckDB version, bump this pin in lockstep. To
find the engine version of any candidate:

```bash
url=$(npm view @duckdb/duckdb-wasm@<version> dist.tarball)
curl -sL "$url" | tar xz -C /tmp/dw
strings -n4 /tmp/dw/package/dist/duckdb-eh.wasm | grep -oE 'v1\.[0-9]+\.[0-9]+' | sort -uV | tail -1
```

or at runtime `await db.getVersion()`. The harness prints the engine version at
startup so a mismatch is obvious.
