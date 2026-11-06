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
- `--file <t.test>` — run specific file(s); a file that does not exist fails.

Exit code is non-zero on any load or assertion failure.

`node test/wasm/sqllogic.mjs --self-test` checks the comparator and parser
without booting DuckDB (CI runs it before the suite): 100000 vs 100001, BIGINT
beyond 2^53, text `001` vs `1` and a one-column result under `query II` must
all be rejected.

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
- `test/wasm/sqllogic.mjs` is a sqllogictest-subset parser/runner with the
  semantics of DuckDB's own runner (see *Parser notes* below), so a pass here
  means what a native pass means.

## Coverage

The CI gate runs the **full suite** (`--all`, all `test/sql/**/*.test`) against
the built `.wasm`: 326 files, 8000 records (2026-11-06, v1.5.6 wasm_eh). 32
records are `SET threads` statements DuckDB-Wasm cannot run (reported as
skipped), and 7 `EXECUTE` queries run unwrapped (compared by type letter).
`test/sql/quack.test` is skipped — extension-template boilerplate calling
`quack` / `quack_openssl_version`, functions this extension doesn't define.
Two records fail under the strict comparator, natively as well as in
DuckDB-Wasm: their expected values are only within the native runner's 1%
(`anofox_stats.test:44`, `api/test_stable_model_struct_glm.test:74`).

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
- See *Parser notes* below for how values are compared.

## Version matching (important)

The `@duckdb/duckdb-wasm` engine version must **exactly match** the DuckDB version
the extension was built against — DuckDB extensions are ABI-locked to the engine
version. **The npm version is not the engine version**: e.g. `1.29.0` → engine
`v1.1.1`, `1.32.0` → `v1.4.3`, `1.33.1-dev64.0` → `v1.5.5`, `1.33.1-dev65.0` → `v1.5.6`.

This harness is pinned to **`@duckdb/duckdb-wasm@1.33.1-dev65.0`** (engine
**v1.5.6**), matching the extension's stable target, and the CI job feeds it the
**v1.5.6 `wasm_eh`** artifact. Notes:

- `1.33.1-dev65.0` is a **dev/prerelease** (the `@next` dist-tag today) — no
  *stable* duckdb-wasm ships a 1.5.x engine yet.

When the extension moves to a new DuckDB version, bump this pin in lockstep. To
find the engine version of any candidate:

```bash
url=$(npm view @duckdb/duckdb-wasm@<version> dist.tarball)
curl -sL "$url" | tar xz -C /tmp/dw
strings -n4 /tmp/dw/package/dist/duckdb-eh.wasm | grep -oE 'v1\.[0-9]+\.[0-9]+' | sort -uV | tail -1
```

or at runtime `await db.getVersion()`. The harness prints the engine version at
startup so a mismatch is obvious.

## Parser notes (`sqllogic.mjs`)

Ported from anofox-optimize's harness (DataZooDE/anofox-optimize#21), keeping
this repo's `loop`/`foreach` support.

- **Records:** `require` (handled by the harness), `statement ok`,
  `statement error` with the expected message after `----` (a substring,
  `<REGEX>:` matched against the whole message, or `<!REGEX>:`),
  `statement maybe` (success, or an error matching the message),
  `query <types> [nosort|rowsort|valuesort] [label]`, `loop`/`foreach` …
  `endloop` (expanded up front, `${var}` substituted, nesting supported; a
  failure inside a loop names the line it is written on), `mode skip`/`unskip`,
  `halt`. Any other keyword (`onlyif`, `skipif`, …) **fails** with its file and
  line rather than being skipped and leaving its body unguarded. As natively, an
  error containing `INTERNAL` never satisfies `statement error`/`maybe`.
- **Shape first:** the result's column count must equal the length of the type
  string, and the row count must match. Expected rows are either one per line
  (tab-separated) or one value per line, as in DuckDB. `N values hashing to
  <md5>` is supported.
- **Values:** DuckDB renders them and reports each column's real type in the
  same statement (`COLUMNS(*)::VARCHAR, typeof(COLUMNS(*))`; `NULL`, `(empty)`
  for the empty string). Identical text is equal. Otherwise integers compare
  exactly as BigInt, BOOLEAN as true/false/1/0, DOUBLE/FLOAT/DECIMAL within
  1e-9 relative (DuckDB's runner allows 1%: `|a - e| <= 0.01*|a| + 1e-8`), and
  everything else — VARCHAR, LIST, STRUCT, a STRUCT of doubles included — only
  as identical text, as natively. The run prints how many queries ran
  unwrapped, and so without types.
- **Raw retry:** a query is run without the `::VARCHAR` wrap only when the wrap
  raises a Binder or Parser error, so a runtime error is reported once and
  never hidden by a second run.
- **Timeout:** every statement and query has 60 s
  (`ANOFOX_WASM_RECORD_TIMEOUT_MS` overrides it). A timeout names the file and
  line and aborts the run, since the hung query still occupies the worker.

The previous comparator compared any two values that parsed as numbers within
1e-4 relative (so 100000 matched 100001, text `001` matched `1`, and BIGINT lost
precision beyond 2^53), and flattened results before checking their shape.
