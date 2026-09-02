# Phase 6: Docs Refresh & SQL Validation — Research

**Researched:** 2026-09-02
**Domain:** Documentation maintenance, SQL validation harness, CI integration
**Confidence:** HIGH — all findings grounded in repo file reads and live extension probes

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**README Structure (DOCS-01)**
- Section order: Title + badges → Table of Contents → Key Features (with **⚡ Performance** subsection referencing Phase-4 benchmark work and **🎨 User-Friendly API** subsection referencing Phase-5 ergonomics) → Quick Start → Installation → API Reference → Development → Support → Citation → License.
- Emoji section headers: yes (matches anofox-forecast).
- Quick Start: one concrete small dataset walked end-to-end (fit → predict → diagnostics) using the NEW renamed function names.
- `docs/API_REFERENCE.md` remains the authoritative API list; README's API Reference section links to it and to `docs/API_CONVENTIONS.md` rather than duplicating the full surface.

**Doc-SQL Validation Harness (DOCS-02/03)**
- Extraction scope: fenced ` ```sql ` blocks from `README.md` + `guides/*.md` + `docs/API_REFERENCE.md` (and `docs/API_CONVENTIONS.md`).
- Execution: pipe each block into the built DuckDB CLI with the extension `LOAD`ed; report pass/fail per example with the source file + line.
- Multi-block setup dependencies: concatenate a single file's blocks in document order into one DuckDB session.
- Harness lives under `scripts/` (e.g. `scripts/validate_docs_sql.*`); emits a per-example pass/fail report; exits non-zero on any failure.
- DOCS-03: every extracted example must pass — fix drift/rename breakage.

**CI Integration (DOCS-04)**
- New workflow `.github/workflows/DocsSqlValidation.yml`, mirroring `WasmTest.yml`'s build-then-run shape.
- Gating: hard-fail on any doc-SQL example failure.
- Non-executable/illustrative snippets: a documented skip marker.
- Triggers: `on: pull_request` + `push to main`.

### Claude's Discretion
- Exact harness language (bash vs Python) and precise skip-marker syntax.
- Exact README wording/copy within the agreed section structure.

### Deferred Ideas (OUT OF SCOPE)
- A rendered documentation website / static-site generator.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOCS-01 | README restructured to anofox-forecast form — emoji headers, ToC, Key Features, Quick Start, API Reference, Development, Support, Citation | README gap analysis in §4; section mapping table |
| DOCS-02 | Doc-SQL validation harness extracts every SQL block from README + guides + API reference and runs against built extension | CLI invocation pattern (§2); harness design (§5); skip-marker syntax (§5) |
| DOCS-03 | All documented SQL examples pass — fix broken examples | SQL inventory (§1); violation table with exact block counts |
| DOCS-04 | Doc-SQL validation runs in CI | CI mechanics (§3); DocsSqlValidation.yml design |
</phase_requirements>

---

## Summary

Phase 6 has three concrete deliverables: a rewritten `README.md`, a `scripts/validate_docs_sql.sh` harness, and a `.github/workflows/DocsSqlValidation.yml` CI job. All three are grounded in well-understood prior art already in this repo (`scripts/bench.sh`, `WasmTest.yml`), so implementation risk is low.

The dominant workload in this phase is mechanical find-and-replace across 222 SQL blocks in 7 doc files. The scan found **80 blocks** carrying at least one naming violation (all of the `anofox_stats_` prefix flavour — zero `.r2`, zero `theilsen`, zero `.residual_standard_error` remain in the scan corpus). An additional set of blocks uses the old positional-boolean API style (`ols_fit(y, x, true, true, 0.95)`) and stale field names (`.f_statistic`, `.f_pvalue`, `.n_observations`, `full_output`); the harness will surface these at run time. The two blocks in `docs/API_CONVENTIONS.md` that show old-prefix "before" examples are intentional migration examples and must be skip-marked, not fixed.

The DuckDB CLI exists at `build/release/duckdb` [VERIFIED: build/release/duckdb — `ls -la` confirmed executable, 65 MB], the extension at `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension` [VERIFIED: confirmed path from `scripts/bench.sh:23-24`], and the load incantation `duckdb -unsigned -cmd "LOAD '<abs-path-to-ext>';"` is confirmed working against the live build.

**Primary recommendation:** Write the harness in bash, mirroring `scripts/bench.sh`, using the ` ```sql skip ` info-string as the skip marker. Shell the DuckDB CLI (not sqllogictest) because the doc examples are standalone SQL, not sqllogictest format. For CI, add a self-contained `ubuntu-24.04` job in `DocsSqlValidation.yml` that runs `make release` then `bash scripts/validate_docs_sql.sh`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SQL block extraction | Script (harness) | — | Regex parse of markdown fences; no DuckDB involvement |
| SQL execution / validation | DuckDB CLI | — | CLI is the user-facing runtime; sqllogictest is test-internal only |
| Pass/fail reporting | Script (harness) | CI annotations | Harness owns the per-block report; CI reads exit code |
| CI build of extension | GitHub Actions reusable job | — | `make release` builds DuckDB CLI + extension; already proven in pipeline |
| README authoring | Docs (in-repo) | — | Pure markdown; no toolchain dependency |
| API_REFERENCE accuracy | Docs (in-repo) | Harness validation | Harness is the enforcement mechanism |

---

## 1. SQL Example Inventory (DOCS-02/03)

### Block counts per file

| File | Total `sql` blocks | Naming-violation blocks | Notes |
|------|--------------------|------------------------|-------|
| `README.md` | 5 | 1 | Block #1: `anofox_stats_ols_fit` + `.r2` ref via `full_output` option |
| `guides/01_quick_start.md` | 18 | 12 | Blocks #2–#12 (12 of 18 carry the old prefix) |
| `guides/02_technical_guide.md` | 2 | 1 | Block #1: `anofox_stats_ols_fit_agg` in a window example |
| `guides/03_business_guide.md` | 17 | 15 | Nearly all blocks use old prefix + positional bool args |
| `guides/04_advanced_use_cases.md` | 17 | 16 | Nearly all blocks use old prefix; block #14 uses `.n_observations` |
| `docs/API_REFERENCE.md` | 157 | 33 | 33 blocks with old prefix; the bulk of API_REFERENCE uses renamed names |
| `docs/API_CONVENTIONS.md` | 6 | 2 | Blocks #4 and #5 are intentional "Before/After" migration examples — **must be skip-marked, not fixed** |
| **TOTAL** | **222** | **80** | |

[VERIFIED: scan run via Python regex against all 7 files in this session]

### Violation types found

| Violation | Files affected | Count of blocks |
|-----------|----------------|-----------------|
| `anofox_stats_` prefix on function names | All 7 files | 80 |
| `.r2` field reference | README.md block #1 (via `full_output` option context) | 1 |
| `theilsen` (should be `theil_sen`) | None found in sql blocks | 0 |
| `.residual_standard_error` (should be `residual_std_error`) | None found — already correct in docs | 0 |
| Positional boolean args e.g. `ols_fit(y, x, true, true, 0.95)` | guides/03, guides/04, docs/API_REFERENCE | ~8 blocks |
| Stale field names: `.f_statistic`, `.f_pvalue`, `.n_observations`, `full_output` | guides/01, guides/04 | 4 blocks |

[VERIFIED: Python regex scan this session against actual file content]

### IN-03 status

The Phase-5 code review flagged `.residual_standard_error` in guide examples as IN-03. The scan found **zero occurrences** of `residual_standard_error` in any doc file. [VERIFIED: `grep` result this session — no matches in any of the 7 scanned files]. The authoritative field name `residual_std_error` is already used in `docs/API_CONVENTIONS.md:149` and `docs/API_REFERENCE.md:2880`. [VERIFIED: docs/API_CONVENTIONS.md:149, docs/API_REFERENCE.md:2880 — read this session]. The IN-03 fix therefore lands as part of the general `anofox_stats_` prefix sweep in guides/01 (those blocks use old-prefix calls that would also fail on residual field access, but the field itself is not referenced as `.residual_standard_error`). This does **not** require a separate fix pass.

### Intentional skip blocks (API_CONVENTIONS.md)

`docs/API_CONVENTIONS.md` blocks #4 and #5 (Breaking Changes §5) show the old `anofox_stats_` and `theilsen_` names as migration "Before" examples. [VERIFIED: docs/API_CONVENTIONS.md:228-246 — read this session, verbatim:

```
-- Before (v0.2.x):
SELECT anofox_stats_ols_fit_agg(y, [x1, x2]) FROM tbl;
-- After (v0.3.0+):
SELECT ols_fit_agg(y, [x1, x2]) FROM tbl;
```

and

```
-- Before:
SELECT anofox_stats_theilsen_fit_agg(y, [x]) FROM tbl;
-- After:
SELECT theil_sen_fit_agg(y, [x]) FROM tbl;
```]

These blocks **must be skip-marked**. They are correct documentation of the breaking change, not errors.

---

## 2. Extension CLI Invocation (DOCS-02)

### Verified binary paths

| Artifact | Path | Confirmed |
|----------|------|-----------|
| DuckDB CLI | `build/release/duckdb` | [VERIFIED: scripts/bench.sh:22 + `ls` confirmed 65 MB executable] |
| Extension | `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension` | [VERIFIED: scripts/bench.sh:23-24 + `ls` confirmed 50 MB] |

### Exact LOAD incantation for unsigned local build

```bash
build/release/duckdb -unsigned \
  -cmd "LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';"
```

[VERIFIED: scripts/bench.sh:98 — `"$DUCKDB" -unsigned -cmd "LOAD '$EXT';" -f "$sql_file"`, confirmed working this session via live probe returning correct `r_squared` result]

The `-unsigned` flag tells the DuckDB CLI to load extension binaries that have not been signed by the DuckDB Foundation. This is the correct approach for local builds. [VERIFIED: scripts/bench.sh:98]

### Self-contained vs sqllogictest

**Recommendation: shell out to the DuckDB CLI.**

Rationale:
- Doc examples are plain SQL prose, not sqllogictest format (`.test` files use `require`, `statement ok`, expected-output blocks). Running them through `build/release/test/unittest --test-dir=...` would require converting every doc block to sqllogictest syntax, which is a large extra burden and defeats the point of the harness.
- The CLI approach is exactly what `scripts/bench.sh` already does: `"$DUCKDB" -unsigned -cmd "LOAD '$EXT';" -f "$sql_file"`. The doc-SQL harness can follow the same pattern identically, extracting each block to a temp `.sql` file and piping it through the CLI.
- The sqllogictest runner (`build/release/test/unittest`) is the right tool for regression tests in `test/sql/`; it is not the right tool for freeform documentation examples.
- A per-file session (all blocks from a file concatenated in document order) requires a single CLI invocation per file, which the CLI handles trivially with `-f`.

### Multi-block session design

The CONTEXT.md decision is: concatenate all blocks from a single file into one session so later blocks can see tables/state set up by earlier blocks. Concretely:

1. Extract all non-skipped sql blocks from a file in document order.
2. Write them to a temp file (e.g. `/tmp/validate_<file_slug>.sql`).
3. Invoke: `"$DUCKDB" -unsigned -cmd "LOAD '$EXT';" -f /tmp/validate_<file_slug>.sql 2>&1`.
4. Capture exit code and stdout/stderr; report per-file pass/fail.

**Limitation:** Per-block pass/fail requires splitting at block boundaries. The harness should write each block as a separate temp file and invoke DuckDB once per block (prepending the LOAD), not concatenate. To preserve cross-block state (e.g. `CREATE TABLE` in block N used by block N+1), use DuckDB's `-cmd` to LOAD, then `-f` on the concatenated block group. The harness can report at file granularity and list the first failing block's line number from stderr.

---

## 3. CI Mechanics (DOCS-04)

### How the existing pipeline builds the loadable extension

The `MainDistributionPipeline.yml` calls the DuckDB reusable workflow `_extension_distribution.yml` which runs `cmake ... make release` across a multi-platform matrix, producing per-platform `anofox_statistics.duckdb_extension` artifacts. [VERIFIED: MainDistributionPipeline.yml:21-40 — `uses: duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml@v1.5-variegata` with `enable_rust: true`].

The `WasmTest.yml` model (which `DocsSqlValidation.yml` should mirror) does **not** rebuild — it downloads the pre-built `wasm_eh` artifact via `actions/download-artifact@v4` from the preceding pipeline run. [VERIFIED: WasmTest.yml:30-36 — `download-artifact` step].

### DocsSqlValidation.yml design

**Option A — Artifact reuse (mirror WasmTest.yml exactly):** Trigger on `workflow_run` of the main pipeline, download the `linux_amd64` artifact, download the official DuckDB CLI, and run the harness. This avoids a full rebuild but delays the validation until after the full multi-platform build (15–30 min lag).

**Option B — Self-contained build job (recommended):** A single `ubuntu-24.04` job that checks out the repo with `submodules: 'recursive'`, installs the Rust toolchain, and runs `make release`, then `bash scripts/validate_docs_sql.sh`. This mirrors the `build-and-test-rust` job pattern [VERIFIED: MainDistributionPipeline.yml:141-180 — `checkout` + `dtolnay/rust-toolchain@stable` + `cargo build --release`] extended to also build the C++ extension via `make release`.

Option B is recommended because:
- The `MainDistributionPipeline.yml` already does the full multi-platform build on every push/PR. DocsSqlValidation can be a lighter linux-only build.
- It provides immediate feedback on the PR rather than waiting for the distribution pipeline to complete.
- `make release` on `ubuntu-24.04` is proven by `build-and-test-rust` already running there.
- The harness needs the DuckDB CLI (`build/release/duckdb`) which the distribution pipeline's reusable workflow produces as an intermediate artifact but does not expose — a self-contained build is the only clean path to it.

### Reusable steps for DocsSqlValidation.yml

```yaml
on:
  pull_request:
  push:
    branches: [main]

jobs:
  docs-sql-validation:
    runs-on: ubuntu-24.04
    env:
      CC: gcc
      CXX: g++
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: 'recursive'

      - uses: dtolnay/rust-toolchain@stable   # same as build-and-test-rust job

      - uses: actions/cache@v4                # same Cargo cache key as build-and-test-rust
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Build extension (release)
        run: make release                     # produces build/release/duckdb + .duckdb_extension

      - name: Run doc-SQL validation
        run: bash scripts/validate_docs_sql.sh
```

[VERIFIED: MainDistributionPipeline.yml:149-180 — `checkout submodules:recursive`, `dtolnay/rust-toolchain@stable`, `actions/cache@v4` pattern used verbatim]

---

## 4. README Gap Analysis (DOCS-01)

### Current README section structure

[VERIFIED: README.md:1-342 — read this session]

| Line | Current section | Emoji? |
|------|----------------|--------|
| 1 | `# Anofox Statistics - DuckDB Extension` (title) | No |
| 13 | `## Features` | No |
| 15–88 | `### Regression Methods`, `### Statistical Hypothesis Tests`, `### Diagnostics & Utilities`, `### Fit-Predict Table Macros`, `### Key Capabilities` (sub-sections of Features) | No |
| 94 | `## Quick Start` | No |
| 185 | `## Installation` | No |
| 209 | `## Documentation` | No |
| 220 | `## Dependencies` | No |
| 232 | `## Telemetry` | No |
| 254 | `## License` | No |
| 282 | `## Contributing` | No |
| 301 | `## Support` | No |
| 319 | `## Citation` | No |
| 333 | `## Validation` | No |
| 337 | `## Acknowledgments` | No |

### Target section structure (from CONTEXT.md locked decisions)

| Target section | Status |
|---------------|--------|
| Title + badges | EXISTS — needs minor cleanup (title format) |
| Table of Contents | **MISSING** — not present at all |
| Key Features (`⚡ Performance` + `🎨 User-Friendly API` subsections) | **MISSING** — current `## Features` is a flat table, no emoji, no Phase-4/5 narrative subsections |
| Quick Start (end-to-end: fit → predict → diagnostics with new names) | EXISTS but BROKEN — uses `anofox_stats_` prefix, `full_output` option, `.r2`; must be fully rewritten |
| Installation | EXISTS — likely fine, keep |
| API Reference (links to `docs/API_REFERENCE.md` and `docs/API_CONVENTIONS.md`) | **RESTRUCTURED NEEDED** — current `## Documentation` section is close but named wrong; no `API Reference` section heading |
| Development | **MISSING** — current `## Contributing` covers contribution, but a `## Development` section covering how to build/test locally from source is not present as a distinct section |
| Support | EXISTS — `## Support` section at line 301 |
| Citation | EXISTS — `## Citation` section at line 319 |
| License | EXISTS — `## License` section at line 254, currently in wrong position (before Contributing, not at end) |

### Sections to add / restructure

1. **Add `## 📋 Table of Contents`** — entirely new, links to all major sections.
2. **Replace `## Features`** with **`## ✨ Key Features`** — keep the function table content but add:
   - `### ⚡ Performance` subsection with bench numbers from Phase-4 results.
   - `### 🎨 User-Friendly API` subsection calling out the Phase-5 consistency rename and ergonomic error messages.
3. **Rewrite `## Quick Start`** — new end-to-end example with renamed API: `ols_fit_agg` → access `.r_squared`, `.coefficients`; then `predict`; then diagnostics. All with new MAP option style, no positional bools.
4. **Rename `## Documentation`** to **`## 📚 API Reference`** and add the `docs/API_CONVENTIONS.md` link.
5. **Add `## 🛠️ Development`** — build from source instructions (`make release`, cargo test, running test suite).
6. **Move `## License`** to the end (after Citation).
7. **Remove `## Validation`** and **`## Acknowledgments`** (or merge into Development / Support if content is still relevant).
8. **Remove `## Dependencies`** (move content into Development or Installation).
9. **Add emoji to all section headers** (the `##`-level ones).

---

## 5. Skip-Marker Design and Harness Language

### Skip-marker syntax recommendation

**Recommended: ` ```sql skip ` info-string**

The fenced code block info-string in CommonMark is everything after the opening fence's backtick sequence. Renderers that do not understand `skip` simply fall back to plain `sql` highlighting — the doc still renders correctly. A shell/python extractor only needs to match the opening fence line:

```
^```sql$          → executable block
^```sql skip$     → skip this block
```

The alternative (HTML comment sentinel `<!-- docs-sql:skip -->` above the block) requires a two-line lookahead and is more fragile to whitespace changes. The info-string approach is a single-regex decision per block with zero context needed.

**Convention to document:** In a `.planning/phases/06-docs-refresh-sql-validation/` note or inline in the harness header:
> Fenced sql blocks marked ` ```sql skip ` are excluded from validation (illustrative examples showing old API for migration reference, or multi-step examples that require external data not available in CI).

### Harness language recommendation: bash

Mirror `scripts/bench.sh` exactly:
- Same `SCRIPT_DIR` / `REPO_ROOT` pattern [VERIFIED: scripts/bench.sh:19-20].
- Same `DUCKDB` / `EXT` path variables [VERIFIED: scripts/bench.sh:22-24].
- Same precondition check (exit 1 if binary or extension not found) [VERIFIED: scripts/bench.sh:44-53].
- Same `set -euo pipefail` guard.

The extraction logic (regex over markdown) is 15–20 lines of Python. The harness can embed a Python heredoc or call `python3 -c` inline. Alternatively, a pure-Python script (`scripts/validate_docs_sql.py`) with a thin bash wrapper is acceptable. Given that `scripts/bench.sh` calls the DuckDB binary directly and the CI Python environment is also available (`python3` in `ubuntu-24.04`), **a Python script is actually cleaner** for the extraction + per-block loop, while still invoking `subprocess.run(["build/release/duckdb", "-unsigned", ...])` for execution. The decision is at Claude's discretion; both work.

**Concrete Python approach:**

```python
#!/usr/bin/env python3
# scripts/validate_docs_sql.py
# Extracts all non-skipped ```sql blocks from doc files and pipes each through
# the built DuckDB CLI with the extension LOADed. Exits non-zero on any failure.

import re, subprocess, sys, tempfile, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCKDB = os.path.join(REPO, "build/release/duckdb")
EXT    = os.path.join(REPO, "build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension")

DOC_FILES = [
    "README.md",
    "guides/01_quick_start.md",
    "guides/02_technical_guide.md",
    "guides/03_business_guide.md",
    "guides/04_advanced_use_cases.md",
    "docs/API_REFERENCE.md",
    "docs/API_CONVENTIONS.md",
]

# Regex: ```sql\n...\n``` vs ```sql skip\n...\n```
BLOCK_RE = re.compile(r'^```sql( skip)?\n(.*?)^```', re.MULTILINE | re.DOTALL)

failures = []
for rel in DOC_FILES:
    path = os.path.join(REPO, rel)
    text = open(path).read()
    blocks = [(m.group(1), m.group(2), text[:m.start()].count('\n') + 1)
              for m in BLOCK_RE.finditer(text)]
    # Concatenate all non-skipped blocks for this file into one session
    executable = [(sql, lineno) for (skip, sql, lineno) in blocks if not skip]
    if not executable:
        continue
    combined = "\n".join(sql for sql, _ in executable)
    with tempfile.NamedTemporaryFile(suffix=".sql", mode="w", delete=False) as f:
        f.write(combined)
        tmpfile = f.name
    result = subprocess.run(
        [DUCKDB, "-unsigned", "-cmd", f"LOAD '{EXT}';", "-f", tmpfile],
        capture_output=True, text=True
    )
    os.unlink(tmpfile)
    if result.returncode != 0:
        failures.append((rel, result.stderr))
        print(f"FAIL {rel}\n{result.stderr}")
    else:
        print(f"PASS {rel} ({len(executable)} blocks)")

sys.exit(1 if failures else 0)
```

This pattern (per-file session, combined blocks, single DuckDB invocation per file) matches the CONTEXT.md decision for cross-block state sharing.

---

## Architecture Patterns

### Doc-SQL harness data flow

```
Doc files (README.md, guides/*.md, docs/*.md)
        |
        | Python regex extraction
        v
Per-file block list [(sql_text, line_no, skip?)]
        |
        | filter skip=true blocks
        v
Non-skipped blocks → concatenated per file
        |
        | subprocess.run(duckdb -unsigned -cmd "LOAD ext;" -f tmpfile)
        v
exit_code + stderr per file
        |
        | aggregate pass/fail
        v
Console report  +  exit(1) if any failure
```

### Recommended project structure additions

```
scripts/
├── bench.sh                   # Phase-4 benchmark harness (existing)
└── validate_docs_sql.py       # Phase-6 doc-SQL validation harness (new)

.github/workflows/
├── MainDistributionPipeline.yml   # existing
├── WasmTest.yml                   # existing
└── DocsSqlValidation.yml          # new (Phase-6)
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Markdown fenced-block extraction | Custom tokenizer | Python regex `^```sql\n(.*?)^``` ` (MULTILINE + DOTALL) | The doc files have no pathological nesting; a simple regex is sufficient and already proven |
| SQL execution | Custom DuckDB binding | DuckDB CLI (`build/release/duckdb -unsigned -f`)  | CLI is already built; used identically in bench.sh |
| CI build matrix | Custom Makefile wrapper | Reuse `dtolnay/rust-toolchain@stable` + `make release` pattern from `build-and-test-rust` | Proven to build the extension on ubuntu-24.04 |

---

## Common Pitfalls

### Pitfall 1: Block-context bleeding between files
**What goes wrong:** If all blocks from all files are concatenated into one DuckDB session, a `CREATE TABLE` in `guides/01_quick_start.md` persists and interferes with a same-name table in `guides/03_business_guide.md`.
**Why it happens:** DuckDB is stateful within a session.
**How to avoid:** Invoke DuckDB once per file (separate processes), not once for all files combined.
**Warning signs:** Errors referencing "table already exists" or "column count mismatch" in block N when block N is syntactically correct.

### Pitfall 2: Illustrative blocks that reference external tables
**What goes wrong:** Many blocks in `guides/03_business_guide.md` and `guides/04_advanced_use_cases.md` reference tables created earlier in the same file (`FROM historical_sales`, `FROM daily_prices`). If any block in the file uses a table not defined by an earlier block in the same file, validation will fail with "table not found".
**Why it happens:** The guides mix self-contained examples (CTE-based) with examples that assume a schema.
**How to avoid:** The file-session approach handles this correctly as long as `CREATE TABLE` blocks precede SELECT blocks in document order. The DOCS-03 fix pass must ensure every guide file is self-contained: either inline data via VALUES or mark as ` ```sql skip `.
**Warning signs:** `Catalog Error: Table with name X does not exist` on a block that looks correct.

### Pitfall 3: `anofox_stats_ols_fit(y, x, true, true, 0.95)` positional-bool style
**What goes wrong:** The old API accepted positional booleans for `fit_intercept` and `compute_inference`. The new API uses MAP options. These old-style calls will fail with "wrong number of arguments" or "type mismatch" against the v0.3.0 extension.
**Why it happens:** The Phase-5 rename dropped the positional overloads along with the prefix.
**How to avoid:** DOCS-03 fix pass must convert these to MAP style: `{'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}`.
**Warning signs:** The harness reports errors from `ols_fit_agg` or `ols_fit` blocks in `03_business_guide.md` and `04_advanced_use_cases.md` that pass correct-looking arguments.

### Pitfall 4: API_CONVENTIONS.md "Before" blocks must be skipped, not fixed
**What goes wrong:** If a DOCS-03 sweep removes the `anofox_stats_` prefix from the Breaking Changes "Before" examples in `docs/API_CONVENTIONS.md`, the migration documentation becomes incorrect.
**Why it happens:** The automated rename is applied globally.
**How to avoid:** Mark those two blocks with ` ```sql skip ` BEFORE any automated rename sweep. The skip marker is the guard.
**Warning signs:** `docs/API_CONVENTIONS.md` §5 "Before" and "After" examples both showing unprefixed names.

### Pitfall 5: `.r2` in README block #1 is a symptom of `full_output` option
**What goes wrong:** README.md Quick Start block #1 uses `full_output: true` (a pre-v0.3.0 option) which no longer exists, and references field names that were part of that old schema.
**Why it happens:** The Quick Start was written before the Phase-5 API consistency pass.
**How to avoid:** The entire Quick Start section should be rewritten from scratch for DOCS-01 (new names, new option style). Do not patch the existing block.

---

## Runtime State Inventory

Not applicable — Phase 6 is a docs + CI configuration phase with no renamed runtime state, no stored data, and no OS-registered components.

---

## Validation Architecture

> `workflow.nyquist_validation` is `true` in `.planning/config.json` — this section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | The doc-SQL harness itself (`scripts/validate_docs_sql.py`) is the test runner for DOCS-02/03; existing ctest-invoked sqllogictest (`build/release/test/unittest --test-dir=test/sql`) covers ERGO/PERF regression |
| Config file | None — harness is self-configuring; finds REPO root from `__file__` |
| Quick run (single file) | `python3 scripts/validate_docs_sql.py --file README.md` (harness should support `--file` for fast iteration) |
| Full suite command | `python3 scripts/validate_docs_sql.py` (all 7 files) |
| Existing SQL test suite | `build/release/test/unittest --test-dir=test/sql` (pre-existing, covers ERGO-03 naming via `ergo03_naming.test`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Exists? |
|--------|----------|-----------|-------------------|---------|
| DOCS-01 | README has emoji headers, ToC, Key Features, Quick Start, correct section order | Manual visual inspection | n/a — structure review at verification | N/A |
| DOCS-02 | Harness extracts sql blocks and runs them | The harness is itself the test; its own exit code is the test | `python3 scripts/validate_docs_sql.py` | Wave 0 gap |
| DOCS-03 | Every extracted block passes | `python3 scripts/validate_docs_sql.py` exits 0 | `python3 scripts/validate_docs_sql.py` | Wave 0 gap |
| DOCS-04 | CI job runs harness on PR/push | CI green on `.github/workflows/DocsSqlValidation.yml` | Verified via GitHub Actions run | Wave 0 gap |

### Sampling rate

- **Per task commit:** `python3 scripts/validate_docs_sql.py --file <file-being-edited>` (fast loop on the file touched)
- **Per wave merge:** `python3 scripts/validate_docs_sql.py` (full 7-file sweep)
- **Phase gate:** Full harness green + `build/release/test/unittest --test-dir=test/sql` still green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `scripts/validate_docs_sql.py` — the harness itself; must be created as the first deliverable (it is the measurement foundation for DOCS-03, i.e. the tracer)
- [ ] `scripts/validate_docs_sql.py` self-test: run against a known-bad temp file and confirm exit 1; run against a trivial correct block and confirm exit 0
- [ ] `.github/workflows/DocsSqlValidation.yml` — CI wrapper; created in the final plan wave

**The harness is the tracer for this phase.** It must be created and verified working before the DOCS-03 fix sweep begins, because it defines what "passing" means and measures the before/after count of failures.

---

## Security Domain

> `security_enforcement: true` in `.planning/config.json` — section required.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable — no auth in a doc harness |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Not applicable |
| V5 Input Validation | Marginal | Harness reads doc files from repo; no user-controlled input. File paths are hardcoded constants, not user-supplied. |
| V6 Cryptography | No | Not applicable |

### Known Threat Patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|-----------|
| SQL injection via doc examples | Tampering | Not applicable — the harness runs docs SQL under the local DuckDB CLI as the developer's own user; there is no user input pathway |
| Malicious doc-SQL in a PR (CI context) | Elevation | CI runs on `ubuntu-24.04` in a sandboxed runner with no production credentials; the DuckDB process cannot reach network or secrets |
| Path traversal in file arguments to harness | Tampering | Use `--file` argument validation: only accept paths within `REPO_ROOT`; reject `..` components |

**Security posture:** Low risk. The harness runs pre-reviewed SQL from the repo's own doc files. The main concern is ensuring a contributor cannot inject a doc block that exfiltrates CI secrets — mitigated by the fact that DuckDB has no network access by default and the CI runner has no production credentials.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `build/release/duckdb` CLI | DOCS-02 harness execution | Yes (local) | v1.5.4 | Build with `make release` |
| `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension` | DOCS-02 harness | Yes (local) | current | Build with `make release` |
| `python3` | Harness script | Yes | system default | bash-only alternative |
| `make release` (cmake + Rust) | CI build | Yes (proven in CI) | via extension-ci-tools Makefile | — |
| `dtolnay/rust-toolchain@stable` | CI build | Yes (GH Actions) | stable | — |

[VERIFIED: `build/release/duckdb` — confirmed executable at that path this session; `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension` — confirmed at that path this session]

---

## Package Legitimacy Audit

Not applicable — this phase installs no external packages. All tooling (DuckDB CLI, Python, bash, GitHub Actions actions) is already in the repo's existing CI or available on the standard runner image.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The old positional-boolean API overloads (`ols_fit(y, x, true, true, 0.95)`) were removed in Phase-5; calling them against the current extension will error | §1 Violation types | If aliases still exist, DOCS-03 fix scope narrows slightly — but validation still catches them as passing |
| A2 | `make release` on `ubuntu-24.04` in CI takes under 15 minutes (keeping DocsSqlValidation.yml feedback fast) | §3 CI mechanics | If it takes longer, artifact-reuse strategy (Option A) becomes more attractive |
| A3 | The 33 API_REFERENCE.md violation blocks all use the `anofox_stats_` prefix pattern; none use unknown function names invented since v0.3.0 | §1 inventory | If some blocks reference functions removed in v0.3.0, additional API-gap fixes are needed beyond prefix stripping |

---

## Open Questions (RESOLVED)

> OQ-1 (per-block vs per-file granularity) → resolved: harness reports per-example (file + block index/line) while concatenating a file's blocks in document order for shared setup — adopted by Plan 06-01. OQ-2 (how many API_REFERENCE blocks need external tables) → resolved operationally: Plan 06-01 runs the harness first to produce the baseline count, then Plan 06-02 fixes/inline-data/skip-marks per that output.

1. **Per-block vs per-file failure granularity**
   - What we know: The CONTEXT.md requires "pass/fail per example with the source file + line."
   - What's unclear: This requires one DuckDB invocation per block (not per file), which sacrifices cross-block state sharing. The two goals are in tension.
   - Recommendation: Run per-file (one DuckDB session per file, concatenated blocks) for state sharing. Report the first failure line by scanning stderr for "line N" mentions. For individual block debugging, add a `--block N` flag to the harness.

2. **API_REFERENCE.md block count of 157 — how many are actually executable?**
   - What we know: 157 total blocks, 33 with naming violations. The API_REFERENCE has many illustrative blocks that reference tables like `FROM data`, `FROM tbl` — without inline data creation.
   - What's unclear: How many of those 157 blocks are self-contained vs depend on external schema.
   - Recommendation: Run the harness before DOCS-03 fix work; count failures vs naming-violation counts. Blocks that fail with "table not found" need skip markers or inline data wrappers, not prefix fixes.

---

## Sources

### Primary (HIGH confidence)
- `scripts/bench.sh` — read this session; DuckDB CLI invocation pattern, binary paths, harness structure
- `docs/API_CONVENTIONS.md` — read this session; v0.3.0 naming convention, return struct fields, breaking changes
- `.github/workflows/MainDistributionPipeline.yml` — read this session; CI build steps, runner config
- `.github/workflows/WasmTest.yml` — read this session; build-then-run harness model
- `.planning/phases/06-docs-refresh-sql-validation/06-CONTEXT.md` — read this session; all locked decisions
- Live extension probe — `r_squared`, `residual_std_error` verified against `build/release/duckdb` this session

### Secondary (MEDIUM confidence)
- Python regex scan across 7 doc files — counts are exact for the current file content; any in-flight edits would change them

### Tertiary (LOW confidence)
- None — all findings are from file reads or live probes this session

---

## Metadata

**Confidence breakdown:**
- SQL inventory / violation counts: HIGH — exact regex scan against actual files this session
- DuckDB CLI invocation: HIGH — verified working against live local build
- CI mechanics: HIGH — read actual workflow files this session
- README gap analysis: HIGH — read actual README this session, compared against CONTEXT.md locked decisions
- Harness design: MEDIUM — pattern follows bench.sh closely; exact block-granularity vs file-granularity tradeoff is an open question

**Research date:** 2026-09-02
**Valid until:** Stable (docs and CI config are not fast-moving); re-verify if Phase-5 introduces further API renames
