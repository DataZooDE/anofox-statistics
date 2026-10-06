# Contributing to Anofox Statistics

Thanks for your interest in improving the extension. This guide covers building
and testing it, keeping the documentation honest, and getting a change merged.

By contributing, you agree that your contribution is licensed under the terms in
[LICENSE](LICENSE): Business Source License 1.1, which converts to MPL 2.0.

## Where things live

| Path | Contents |
|------|----------|
| `crates/anofox-stats-core/` | Rust: model wrappers and statistics on top of the `anofox-regression` and `anofox-statistics` crates |
| `crates/anofox-stats-ffi/` | Rust: the C ABI exposed to DuckDB (`src/lib.rs`, `src/types.rs`) |
| `src/include/anofox_stats_ffi.h` | The C header for the FFI. Keep it in sync with `crates/anofox-stats-ffi` |
| `src/anofox_statistics_extension.cpp` | Extension entry point. Every `Register*` call lives here |
| `src/aggregate_functions/` | `*_fit_agg`, `*_fit_predict_agg` and the test/statistics aggregates |
| `src/window_functions/` | The window `*_fit_predict` functions |
| `src/table_functions/` | The scalar `*_fit` functions and `predict` |
| `src/scalar_functions/` | `vif`, `aic`/`bic`, `jarque_bera`, `residuals_diagnostics` |
| `src/macros/fit_predict_macros.cpp` | The `*_fit_predict_by`, `aid_by`, `glmm_fit_by` and `eb_shrink_by` table macros |
| `src/include/map_options_parser.cpp` | Option-MAP parsing: every accepted option key and alias |
| `test/sql/` | SQLLogicTests (`*.test`) and SQL snippets that the guides include |
| `test/wasm/` | DuckDB-Wasm load and runtime harness |
| `docs/` | API reference, conventions, migration, methodology, and NULL semantics |
| `docs/api/` | One reference page per function family |
| `guides/` | User guides. `guides/templates/*.md.in` are the sources; see below |
| `bench/`, `scripts/bench.sh` | Benchmark harness ([bench/README.md](bench/README.md)) |
| `plugins/anofox-statistics/` | Claude Code plugin and its skills |

## Building

Prerequisites: a C++17 compiler, CMake, Ninja (optional), and a stable Rust toolchain.

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-statistics.git
cd anofox-statistics
make release          # or: make debug
```

The build produces:

- `build/release/duckdb`, a DuckDB CLI with the extension statically linked; and
- `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension`,
  the loadable extension.

Supported DuckDB versions are **v1.4.5 (LTS)** and **v1.5.x**. CI builds against
v1.4.5 and v1.5.6. The `duckdb` submodule pins the version used by a local build.

## Testing

```bash
make test                   # SQLLogicTests in test/sql/ against the release build
make test_debug             # the same, against the debug build
cargo test --workspace      # Rust unit and FFI tests
```

Add a SQLLogicTest under `test/sql/` for every new function, and for every bug fix.
Cover NULL handling and the degenerate cases (too few rows, constant columns).
For the WASM build, see [test/wasm/README.md](test/wasm/README.md).

## Documentation and doc-SQL validation

Every fenced ` ```sql ` block in `README.md`, `guides/0*.md`, `docs/*.md` and
`docs/api/**/*.md` is executed against the local release build. CI fails if any
block fails:

```bash
python3 scripts/validate_docs_sql.py                                  # everything
python3 scripts/validate_docs_sql.py --file docs/api/regression/ols.md  # one file
```

All the blocks in one file run in a single DuckDB session, in document order, so
a later block can use a table created by an earlier one. When a file fails, the
script reports the line of the first failing block.

Rules for documentation SQL:

- Make examples self-contained. Create their data with `VALUES` or `range()`, so
  they run as written.
- Fence a block as ` ```sql skip ` only when it cannot run: it reads a user's table,
  installs from the network, or shows an API from before 0.10.0 for migration
  purposes.
- Use the real function names from `src/`. Every regression fit takes
  `(y, x[, options MAP])`.
- Use only option keys that the function reads, and the real result field names.
- Don't link to files that don't exist. Check relative links when you move pages.

**Guides:** `guides/0*.md` are generated from `guides/templates/*.md.in` by
`scripts/build_docs.sh`. The pre-commit hook installed by
`scripts/install_hooks.sh` runs that script. Edit the template and regenerate,
or edit both files identically; otherwise the hook overwrites your change.

## Code style

- C++: DuckDB's clang-format style. Run `scripts/fix_format.sh` to format, and
  `scripts/check_code_quality.sh` to check before you push.
- Rust: run `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings`. CI runs
  clippy with warnings as errors.
- SQL names follow [docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md):
  `{model}_fit_agg`, `{model}_fit_predict[_agg|_by]`, no prefix. Option keys are
  `snake_case`.
- New option keys go in `RegressionMapOptions::ParseFromValue`, including the
  "valid keys" error message. A function must reject the keys it does not support.

## Adding a function

1. Implement the model in `crates/anofox-stats-core`.
2. Expose it through `crates/anofox-stats-ffi` and update `src/include/anofox_stats_ffi.h`.
3. Write the C++ wrapper in the matching `src/*_functions/` directory, and register
   it from `src/anofox_statistics_extension.cpp`. Add the file to `CMakeLists.txt`.
4. Add SQLLogicTests in `test/sql/`.
5. Document it: add a page in `docs/api/`, a row in the README function tables,
   an entry in `docs/API_REFERENCE.md`, and a line in `CHANGELOG.md` under
   `[Unreleased]`.
6. Run `make test`, `cargo test --workspace` and `python3 scripts/validate_docs_sql.py`.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat: ...`, `fix: ...`, `docs: ...`, `ci: ...`, `chore: ...`,
  `refactor: ...`, `test: ...`. Mark breaking changes with `!` (`feat!: ...`)
  or a `BREAKING CHANGE:` footer.
- Keep each pull request focused, and branch from `main`. Describe the change,
  why it is needed, and how you tested it.
- Record user-visible changes in `CHANGELOG.md` under `[Unreleased]`. For a
  breaking change, also update `docs/MIGRATION.md`.
- Issues are tracked on GitHub:
  <https://github.com/DataZooDE/anofox-statistics/issues>.
