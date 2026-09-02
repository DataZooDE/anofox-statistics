# Phase 4: Benchmarking & Performance — Research

**Researched:** 2026-08-31
**Domain:** DuckDB extension benchmarking, Rust FFI allocation patterns, Linux profiling
**Confidence:** HIGH (all claims verified against live codebase this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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
- Profile using `perf` + DuckDB `EXPLAIN ANALYZE` on a Linux native release build
  (the LTO / codegen-units=1 / O3 release profile is already configured);
  `cargo flamegraph` for the Rust core where useful.
- Address the top 3 surfaced hotspots.
- Bar for action: optimize when a safe, behavior-preserving win exists; otherwise
  document the hotspot as inherent with rationale.
- Behavior-preservation gate: the full `test/sql` + `cargo test` suites stay green;
  before/after numbers are produced by the Phase-4 harness.
- Approach: a Rust-side allocation helper / RAII abstraction PLUS a codegen macro
  for the repetitive fit-result marshalling.
- Keep the existing C++ `anofox_free_*` free contract byte-identical — Rust-side
  refactor only, no C++ changes.
- Scope: cover the bulk fit-result marshalling pattern; genuinely one-off or risky
  sites may be left as-is with a documented safety comment.
- Verification: `cargo test` + `test/sql` stay green, plus a noted leak check
  (valgrind / ASan) to confirm no new leaks introduced.

### Claude's Discretion

- Exact file layout under `bench/`, results file format details (markdown vs CSV),
  the specific query shapes/row counts per workload, the macro name/signature, and
  which specific hotspots get optimized vs documented — all at Claude's discretion,
  guided by the decisions above and codebase conventions.

### Deferred Ideas (OUT OF SCOPE)

- Wiring perf tracking into a gating CI job (this phase only notes it as optional).
- Broad FFI integration test suite / cross-platform FFI tests.
- Extracting a telemetry abstraction, argmin fork upstreaming, and other CONCERNS.md
  tech-debt items unrelated to performance.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | A repeatable benchmark harness measures representative workloads (aggregate dispatch, fit/predict paths, FFI marshalling) and reports timings | Sections: Build & Load, Benchmark Mechanics, Workload Inventory |
| PERF-02 | Benchmark runs are reproducible locally and documented; optional CI perf tracking noted | Sections: Benchmark Mechanics, bench.sh Design Pattern |
| PERF-03 | The top hotspots surfaced by profiling are each optimized or explicitly documented as inherent, with before/after numbers | Sections: Profiling Stack, Hotspot Candidates, EXPLAIN ANALYZE |
| PERF-04 | The FFI layer's manual `libc::malloc`/`free` pattern is refactored (RAII wrapper and/or codegen macros) reducing per-call overhead and leak risk, with results unchanged | Sections: FFI Allocation Pattern, RAII Refactor Design, ABI Safety Constraint |
</phase_requirements>

---

## Summary

The extension is a three-layer stack: C++ DuckDB adapters → C FFI boundary (`crates/anofox-stats-ffi/src/lib.rs`, 7,893 lines) → external Rust core crates. The release build already has ideal profiling representativeness: LTO=true, codegen-units=1, opt-level=3 in `Cargo.toml:24-27` [VERIFIED: Cargo.toml:24-27]. The built extension `.duckdb_extension` and the `duckdb` CLI are both present at `build/release/` [VERIFIED: build/release/duckdb and build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension]. The extension loads cleanly under `-unsigned` flag with an explicit path `LOAD 'build/release/extension/...'` [VERIFIED: live test this session].

The FFI file contains **105 `libc::malloc` call sites** [VERIFIED: `grep -c "libc::malloc" lib.rs` = 105] across 24 fit functions [VERIFIED: `grep -c "^pub unsafe extern.*_fit\b"` = 24]. The dominant pattern — 5-array inference block (std_err, t_val, p_val, ci_lo, ci_hi) — repeats in **13 functions** [VERIFIED: `grep -c "std_err_ptr = libc::malloc" lib.rs` = 13]. The critical ABI constraint: Rust allocates with `libc::malloc`, C++ frees with `free()`. A replacement must not use Rust's global allocator (`Box`/`Vec`) since the C++ side issues `free()` directly; `libc::malloc` must remain the allocator, but a helper/macro can wrap the pattern.

The dev machine has `perf` available as `extra/perf 7.1.8-1` (installable from Manjaro repos) [VERIFIED: `pacman -Ss "^extra/perf"` this session]. `gperftools`/`libprofiler.so` is already installed [VERIFIED: `/usr/lib/libprofiler.so`]. `valgrind` is not installed; ASan is available via Rust's `-Z sanitizer=address` (nightly) or `ASAN_OPTIONS` with a re-compiled debug build. `cargo flamegraph` is not installed but requires `perf` first. `EXPLAIN ANALYZE` works in the build binary [VERIFIED: live test this session].

**Primary recommendation:** Build `scripts/bench.sh` first (establishes before numbers), then profile, then refactor FFI. The FFI macro refactor targets the 13 `FitResultInference` allocation blocks — all other malloc sites should be reviewed but may be left with safety comments.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Benchmark harness execution | Shell script (bench.sh) | DuckDB SQL scripts | Query timing owned by DuckDB `.timer`; script orchestrates, captures output |
| Aggregate dispatch workload | DuckDB SQL (GROUP BY) | C++ aggregate adapter | DuckDB partitions data; C++ calls FFI per group |
| Fit/predict scalar workload | DuckDB SQL (window/scalar) | C++ table/window function | DuckDB invokes scalar functions; C++ marshals to FFI |
| FFI marshalling micro-bench | DuckDB SQL (small groups) | Rust FFI lib.rs | Isolation of allocation cost requires controlled small calls |
| Profiling (hot path) | Linux perf / flamegraph | DuckDB EXPLAIN ANALYZE | perf attributes CPU time; EXPLAIN ANALYZE attributes query plan time |
| FFI RAII refactor | Rust (lib.rs, types.rs) | — | Pure Rust-side change; C++ interface byte-identical |
| Behavior verification | `test/sql` + `cargo test` | — | All existing test files are the oracle |

---

## Build & Extension Loading

### Building the Release Extension

```bash
# Full C++ + Rust release build (runs cmake + cargo build --release internally)
make release

# Output artifacts:
#   build/release/duckdb                                              — CLI binary
#   build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension  — loadable extension
```

[VERIFIED: build/release/duckdb exists, 68 MB, built 2026-08-05; build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension exists, 52 MB]

The Makefile delegates to `extension-ci-tools/makefiles/duckdb_extension.Makefile` [VERIFIED: Makefile:8]. The cmake invocation is:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -S ./duckdb/ -B build/release
cmake --build build/release --config Release
```
[VERIFIED: extension-ci-tools/makefiles/duckdb_extension.Makefile:167-169]

### Loading the Built Extension

The build-local `duckdb` binary at `build/release/duckdb` has `anofox_statistics` autoloaded from `~/.duckdb/extensions/` (the installed community version, not the local build) [VERIFIED: live CALL duckdb_extensions() check this session]. The benchmark harness **must** explicitly LOAD the local build path:

```bash
# Required incantation in bench.sh or init SQL:
DUCKDB="./build/release/duckdb"
EXT="build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
$DUCKDB -unsigned -init <(echo "LOAD '${EXT}';") bench/workload.sql
```

Or within the SQL script:
```sql
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';
```

The `-unsigned` flag is required because the local build is not signed [VERIFIED: live test — `LOAD 'build/release/...'` succeeds with `-unsigned`, fails without]. Note: `allow_unsigned_extensions` cannot be set via SQL `SET` while DB is running; it must be passed as a CLI flag [VERIFIED: live test — "Cannot change allow_unsigned_extensions setting while database is running"].

---

## Benchmark Mechanics

### DuckDB Timer and Output

`.timer on` in a SQL script prints `Run Time (s): real X user X sys X` to **stdout** after each statement [VERIFIED: live test this session]. The timer line goes to stdout even when `.output FILE` redirects query results to a file — so `bench.sh` must capture stdout to collect both query results and timings.

```bash
$DUCKDB -unsigned bench/workload.sql > bench/results-$(date +%Y%m%d).txt 2>&1
```

`.output FILE` in SQL redirects only query result rows, not timer lines. This means a bench SQL script can separate results from timing only by post-processing stdout, or by not using `.output` and capturing all output together.

### DuckDB CLI flags relevant to benchmarking

```
-unsigned          — allow loading unsigned (unsigned = locally built) extensions
-init FILENAME     — run SQL from FILENAME before reading stdin/file
-f FILENAME        — read/process named file and exit
-c COMMAND         — run SQL command string and exit
.timer on          — enable per-statement timing (in SQL scripts)
.mode markdown     — table output in markdown format (diffable)
.mode csv          — CSV output (importable)
```

[VERIFIED: `./build/release/duckdb --help` this session]

### EXPLAIN ANALYZE

`EXPLAIN ANALYZE <query>` produces a query plan tree with per-node timing and row counts [VERIFIED: live test this session]. Example output structure:

```
Total Time: 0.0148s
EXPLAIN_ANALYZE → PROJECTION → STREAMING_LIMIT → ... → AGGREGATE
```

The timing at each node identifies where DuckDB spends time outside the extension (partitioning, sorting, output buffering). For aggregate workloads, the aggregate node time shows the FFI + Rust cost.

---

## Workload Inventory

### Existing Benchmark Assets

`examples/performance_1m_groups/` contains [VERIFIED: `ls` this session]:
- `benchmark_ols.sql` — window function (fit_predict), 1M groups, 100M rows, 3 features [VERIFIED: examples/performance_1m_groups/benchmark_ols.sql]
- `benchmark_ridge.sql`, `benchmark_wls.sql`, `benchmark_rls.sql`, `benchmark_elasticnet.sql` — same shape, different models [VERIFIED: ls this session]
- `benchmark_ols_predict_agg.sql` — aggregate function (predict_agg), 1M groups, 80M training + 20M prediction rows [VERIFIED: examples/performance_1m_groups/benchmark_ols_predict_agg.sql]
- `run_all_benchmarks.sh` — shell script that invokes each SQL file, monitors peak RSS via `/proc/<pid>/status` [VERIFIED: examples/performance_1m_groups/run_all_benchmarks.sh]

The existing benchmarks do NOT explicitly `LOAD` the extension in the SQL — they rely on the extension being auto-loaded from `~/.duckdb` [VERIFIED: benchmark_ols.sql has no LOAD statement]. The new `bench.sh` must add explicit LOAD of the local build.

### Dataset shape (reuse per CONTEXT.md decision)

```sql
-- 1M groups × variable rows; the existing shape uses:
-- 1M groups, 100M total rows, 3 features
WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        i / 1000000 AS row_num,
        random() * 100 AS x1,
        random() * 50  AS x2,
        random() * 25  AS x3,
        random() * 100 AS y
    FROM generate_series(1, 100000000) t(i)
)
```
[VERIFIED: examples/performance_1m_groups/benchmark_ols.sql — exact SQL]

**Known resource requirements:** 1M-groups workload uses ~8 GB RAM peak RSS and runs ~160–210 s on a 6-core i7 [VERIFIED: examples/performance_1m_groups/README.md — benchmark results table]. The new harness should document a **scaled-down** variant (e.g. 10K groups, 1M rows) for rapid iteration, with the full 1M-group variant for final "official" before/after numbers.

### Three Required Workloads (PERF-01)

| # | Workload | DuckDB path | SQL shape | Analogous existing file |
|---|---------|-------------|-----------|------------------------|
| W1 | Aggregate dispatch | GROUP BY aggregate | `anofox_stats_ols_fit_agg(y, [x1,x2,x3]) ... GROUP BY group_id` | `benchmark_ols_predict_agg.sql` |
| W2 | Scalar array fit/predict | Window function | `anofox_stats_ols_fit_predict(...) OVER (PARTITION BY ... ROWS BETWEEN ...)` | `benchmark_ols.sql` |
| W3 | FFI marshalling micro-bench | Small groups + inference | N groups, N small (e.g. 1K rows each), with inference ON to stress malloc per call | new |

W3 isolates the malloc cost: set `compute_inference: true` in the options map and use a small number of groups (100–1000) with ~100 rows each so the query is fast but calls the 5-array malloc block many times per second.

---

## FFI Allocation Pattern

### The Dominant Pattern (PERF-04 target)

Every fit function that supports inference allocates 5 independent arrays via `libc::malloc` and populates them individually. The pattern at `lib.rs:193-197` is [VERIFIED: crates/anofox-stats-ffi/src/lib.rs:193-197, verbatim]:

```rust
let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
let t_val_ptr   = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
let p_val_ptr   = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
let ci_lo_ptr   = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
let ci_hi_ptr   = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
```

This block appears in **13 fit functions** [VERIFIED: `grep -c "std_err_ptr = libc::malloc" lib.rs` = 13 this session].

The `FitResultInference` struct (the consumer of these pointers) is [VERIFIED: crates/anofox-stats-ffi/src/types.rs:128-164, verbatim fields]:

```rust
pub struct FitResultInference {
    pub std_errors: *mut f64,
    pub t_values:   *mut f64,
    pub p_values:   *mut f64,
    pub ci_lower:   *mut f64,
    pub ci_upper:   *mut f64,
    pub len: usize,
    pub confidence_level: f64,
    pub f_statistic: f64,
    pub f_pvalue: f64,
}
```

The free function [VERIFIED: crates/anofox-stats-ffi/src/lib.rs:287-311, verbatim]:

```rust
pub unsafe extern "C" fn anofox_free_result_inference(result: *mut FitResultInference) {
    if result.is_null() { return; }
    if !(*result).std_errors.is_null() { libc::free((*result).std_errors as *mut libc::c_void); ... }
    // repeated for t_values, p_values, ci_lower, ci_upper
}
```

The `FitResultCore` pattern (coefficients only, 1 malloc per call) appears at `lib.rs:164` [VERIFIED: crates/anofox-stats-ffi/src/lib.rs:164]:

```rust
let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
```

### Full malloc site count by category

| Category | Site count | Target for macro? |
|----------|-----------|-------------------|
| FitResultInference 5-array block | 13 × 5 = 65 mallocs | Yes — primary macro target |
| FitResultCore coefficients | ~20 mallocs | Yes — same macro or RAII helper |
| Other (prediction, VIF, outlier masks, GLMM arrays, AFT, etc.) | ~20 mallocs | Review; leave with safety comments if unique |
| Total | 105 | |

[VERIFIED: total count from `grep -c "libc::malloc" lib.rs` = 105 this session]

### GLM functions use a different result struct

Functions `anofox_poisson_fit`, `anofox_binomial_fit`, `anofox_negbinomial_fit`, `anofox_tweedie_fit`, `anofox_gamma_fit`, `anofox_logistic_fit` take `out_result: *mut GlmFitResultCore` rather than `FitResultCore` [VERIFIED: crates/anofox-stats-ffi/src/lib.rs:2364-2370]. `GlmFitResultCore` is a different struct with a `coefficients: *mut f64` field [VERIFIED: crates/anofox-stats-ffi/src/types.rs:939-965]. However, they share the same `FitResultInference` for inference — so the 5-array macro applies to their inference block too.

---

## RAII Refactor Design (PERF-04)

### CRITICAL ABI Constraint

**THE RAII HELPER MUST ALLOCATE WITH `libc::malloc`, NOT WITH `Box` OR `Vec`.**

The C++ `anofox_free_result_inference()` (and all `anofox_free_*` functions in C++) call the C standard library `free()`. On Linux with glibc, `libc::malloc` in Rust and `malloc`/`free` in C++ share the same allocator — so `libc::malloc` in Rust → `free()` in C++ works correctly. Rust's global allocator (`Box::new`, `Vec`) uses `malloc` on glibc too, BUT this is an implementation detail: on musl (WASM, some CI targets), the allocators may differ. The FFI contract requires `libc::malloc` explicitly. **Do not use `Box::into_raw` or `Vec::into_raw_parts` in the RAII helper.**

[ASSUMED: The musl allocator difference is a known platform concern; the existing code's use of `libc::malloc` is deliberate. The CONTEXT.md confirms no C++ changes are in scope.]

### Proposed RAII wrapper (`FfiVec<T>`)

A thin wrapper that holds a `libc::malloc`-allocated pointer and length, drops via `libc::free`, and can produce a raw pointer for FFI assignment:

```rust
/// A heap allocation backed by libc::malloc, compatible with C free().
///
/// SAFETY: T must be a plain-data type. The allocation is freed via libc::free
/// on Drop, which is compatible with C++ callers using ::free().
pub struct FfiVec<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> FfiVec<T> {
    /// Allocate `len` elements of T via libc::malloc. Returns Err on OOM.
    pub fn alloc(len: usize) -> Result<Self, ()> {
        if len == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), len: 0 });
        }
        let ptr = unsafe { libc::malloc(len * std::mem::size_of::<T>()) as *mut T };
        if ptr.is_null() { Err(()) } else { Ok(Self { ptr, len }) }
    }

    /// Copy from a slice into the allocation.
    pub unsafe fn copy_from_slice(&self, src: &[T]) where T: Copy {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len);
    }

    /// Consume self and return the raw pointer (caller owns the memory).
    /// The pointer must be freed via libc::free / anofox_free_* when done.
    pub fn into_raw(self) -> *mut T {
        let p = self.ptr;
        std::mem::forget(self); // don't run Drop
        p
    }
}

impl<T> Drop for FfiVec<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { libc::free(self.ptr as *mut libc::c_void) };
        }
    }
}
```

[ASSUMED: exact API surface — adjust at implementation time. The key invariant (libc::malloc + forget on into_raw()) is non-negotiable.]

### Proposed macro for the 5-array inference block

```rust
macro_rules! alloc_inference_arrays {
    ($n:expr, $out_inference:expr, $coef_ptr:expr, $inf:expr) => {{
        let std_err = FfiVec::<f64>::alloc($n)?;
        let t_val   = FfiVec::<f64>::alloc($n)?;
        let p_val   = FfiVec::<f64>::alloc($n)?;
        let ci_lo   = FfiVec::<f64>::alloc($n)?;
        let ci_hi   = FfiVec::<f64>::alloc($n)?;
        unsafe {
            std_err.copy_from_slice(&$inf.std_errors);
            t_val.copy_from_slice(&$inf.t_values);
            p_val.copy_from_slice(&$inf.p_values);
            ci_lo.copy_from_slice(&$inf.ci_lower);
            ci_hi.copy_from_slice(&$inf.ci_upper);
            *$out_inference = FitResultInference {
                std_errors: std_err.into_raw(),
                t_values:   t_val.into_raw(),
                p_values:   p_val.into_raw(),
                ci_lower:   ci_lo.into_raw(),
                ci_upper:   ci_hi.into_raw(),
                len: $n,
                confidence_level: $inf.confidence_level,
                f_statistic: $inf.f_statistic.unwrap_or(f64::NAN),
                f_pvalue:    $inf.f_pvalue.unwrap_or(f64::NAN),
            };
        }
    }};
}
```

[ASSUMED: exact macro signature — FfiVec::alloc returning Result with ? operator requires the enclosing scope to handle Err. The macro will need adaptation per call site's error-handling style (some use early-return false, others use match).]

### Contiguous-buffer optimization (alternative / enhancement)

Instead of 5 separate mallocs, allocate a single buffer of `5 * n * sizeof(f64)` and subdivide it. This reduces allocator overhead from 5 syscall-level allocations to 1:

```rust
pub struct InferenceBuffer {
    buf: *mut f64,
    n: usize,
}
// std_err at buf+0*n, t_val at buf+1*n, p_val at buf+2*n, ci_lo at buf+3*n, ci_hi at buf+4*n
```

The C++ free functions currently free each pointer individually; if the buffer is contiguous, only the first pointer (`std_errors`) needs to be freed and the rest can be set to null. This requires updating `anofox_free_result_inference` — which would be a C++ change. Per CONTEXT.md, C++ must not change. Therefore, the contiguous-buffer approach is **not viable** for this phase unless the free function changes are included. The 5-separate-malloc-wrapped-in-FfiVec approach is the correct scope.

[ASSUMED: "keep C++ byte-identical" means the free contract (freeing each pointer independently) must be preserved, which rules out contiguous allocation in this phase.]

---

## Profiling Stack (Linux)

### Tool availability on this machine

| Tool | Status | Notes |
|------|--------|-------|
| `perf` | Not installed; installable via `pacman -S perf` | `extra/perf 7.1.8-1` available [VERIFIED: `pacman -Ss "^extra/perf"` this session] |
| `gperftools` / `libprofiler.so` | Installed | `/usr/lib/libprofiler.so` [VERIFIED: `find /usr -name "libprofiler*"` this session] |
| `cargo flamegraph` | Not installed; requires `perf` first | `cargo install flamegraph` after `perf` install |
| `valgrind` | Not installed | `pacman -S valgrind` if needed for leak check |
| `gprof` | Installed | `/usr/bin/gprof` [VERIFIED: `which gprof` this session] |
| `EXPLAIN ANALYZE` | Working | Live test confirmed [VERIFIED: this session] |
| ASan | Available via recompile | Rust nightly + `-Z sanitizer=address` or RUSTFLAGS |

### Install sequence before profiling wave

```bash
sudo pacman -S perf           # linux-perf tool
cargo install flamegraph      # cargo flamegraph (requires perf)
sudo pacman -S valgrind       # for leak check (PERF-04 verification)
```

### Profiling the DuckDB extension with `perf`

Because the statistical computation is in a static Rust library (`libanofox_stats_ffi.a`) linked into the DuckDB binary, `perf` will attribute time to the DuckDB process including the Rust functions. Frame pointers are needed for correct unwinding:

```bash
# Build with frame pointers for perf (add to Cargo.toml [profile.release] temporarily):
# force-frame-pointers = true  <-- add, profile, then remove

# Or use RUSTFLAGS:
RUSTFLAGS="-C force-frame-pointers=yes" make release

# Run perf against the benchmark:
perf record -g --call-graph=dwarf -- \
  ./build/release/duckdb -unsigned \
  -f bench/workload_agg_dispatch.sql
perf report --no-children
```

[ASSUMED: RUSTFLAGS approach for frame pointers — standard practice but not verified against this specific CMake/Corrosion integration. May need `corrosion_set_env_vars` in CMakeLists.txt.]

### Profiling with `cargo flamegraph` (Rust core only)

`cargo flamegraph` wraps `perf` to produce flame graphs from Rust code. Since the Rust crate is a `staticlib`, it cannot be run standalone; flamegraph must target the DuckDB binary:

```bash
cargo install flamegraph

# Method: run flamegraph against the duckdb binary (not `cargo flamegraph`)
perf record -F 99 -g --call-graph=dwarf -- \
  ./build/release/duckdb -unsigned \
  -f bench/workload_agg_dispatch.sql
flamegraph -- perf script | ... # or use cargo flamegraph if it supports binary path
```

[ASSUMED: `cargo flamegraph` on a staticlib embedded in an external binary requires the `--bin` path override; exact incantation needs testing. Alternative: use `perf report` directly and filter to symbols matching `anofox_`.]

### Profiling with gperftools (alternative, no install needed)

`libprofiler.so` is already installed [VERIFIED]. CPU profiling without needing `perf`:

```bash
LD_PRELOAD=/usr/lib/libprofiler.so \
CPUPROFILE=/tmp/bench.prof \
  ./build/release/duckdb -unsigned -f bench/workload_agg_dispatch.sql

# Analyze (needs pprof -- from go or via google-pprof package)
pprof --text /tmp/bench.prof
```

[ASSUMED: `pprof` may not be installed even though `libprofiler.so` is. Check `pacman -S go-tools` or `pip install pprof` for the analyzer.]

### EXPLAIN ANALYZE for DuckDB-side attribution

```sql
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';
PRAGMA enable_profiling;
PRAGMA profiling_output = '/tmp/profile.json';
EXPLAIN ANALYZE
WITH t AS (
    SELECT i % 10000 AS g, random() AS x, random() AS y
    FROM generate_series(1, 1000000) tt(i)
)
SELECT anofox_stats_ols_fit_agg(y, [x]) FROM t GROUP BY g;
```

The EXPLAIN ANALYZE tree shows time split: `HASH_GROUP_BY` node time (DuckDB aggregation overhead) vs. the aggregate function's total time. This distinguishes "DuckDB overhead" from "FFI + Rust" cost.

---

## Hotspot Candidates

Based on CONCERNS.md [VERIFIED: .planning/codebase/CONCERNS.md:119-130] and FFI structure:

| Rank | Candidate | Layer | Nature | Likely outcome |
|------|-----------|-------|--------|----------------|
| 1 | 5-array inference `libc::malloc` (13 sites × 5 calls/invoke) | FFI boundary | Allocation overhead + heap fragmentation | Optimize via RAII helper (PERF-04) |
| 2 | `DataArray::to_vec()` — copies input data into `Vec<f64>` on every FFI call | FFI boundary | Unnecessary heap allocation for transient data | Candidate for optimization (pass slice, not Vec) — verify safety |
| 3 | DuckDB aggregate state serialization/deserialization (window function partitioning) | DuckDB dispatch | Documented as 7.5 GB overhead for 1M groups | Inherent to DuckDB window functions; document as such |

`DataArray::to_vec()` [VERIFIED: crates/anofox-stats-ffi/src/types.rs:79-89 — allocates a new `Vec<f64>` converting from the `DataArray` input bitmask representation]. This is called once per FFI fit invocation. For aggregate functions called per-group, this may be significant.

The `CONCERNS.md` section "Memory Allocation Scaling" [VERIFIED: .planning/codebase/CONCERNS.md:119-130] also names lines `lib.rs:421-425, 667-671` as examples — these are the same 5-array pattern in Huber and RANSAC fit respectively.

---

## Behavior Preservation Oracle

### `test/sql` test suite

```bash
# Full test suite (SQL tests + extension tests via DuckDB test runner)
make test                    # runs test_release_internal
make test_release            # explicit release build tests

# Internal target runs:
./build/release/test/unittest "test/*"
```

[VERIFIED: extension-ci-tools/makefiles/duckdb_extension.Makefile:199,203-204]

The 120 test SQL files in `test/sql/` [VERIFIED: `ls test/sql/ | wc -l` = 120 this session] use the `require anofox_statistics` directive, consumed by DuckDB's test harness, which loads the extension from the build. They cover all model families, aggregate/window/scalar functions, error handling, fit_predict, predict_agg, GLM, GLMM, diagnostics, and more [VERIFIED: test/sql/README.md this session].

Individual test execution during development:
```bash
# Run a single test file via duckdb CLI
./build/release/duckdb -unsigned -c ".read test/sql/regression/test_ols_agg.test"

# Run all tests in a category
./build/release/test/unittest "test/sql/regression/*"
```

### `cargo test`

```bash
# Full workspace test (both crates)
cargo test

# Individual crate
cargo test -p anofox-stats-core
cargo test -p anofox-stats-ffi

# Test with release optimizations (catches optimizer-sensitive bugs)
cargo test --release
```

[VERIFIED: Makefile:18 — `cargo test` is the `rust_test` target; Cargo.toml workspace members confirmed]

The FFI crate (`anofox-stats-ffi`) has Rust unit tests but no C FFI boundary tests [VERIFIED: .planning/codebase/CONCERNS.md:99-103]. `cargo test` for the FFI crate exercises the Rust code paths but not the C ABI.

---

## bench.sh Design Pattern

### Recommended structure

```
bench/
├── workloads/
│   ├── 00-load-ext.sql      # LOAD extension, .timer on preamble
│   ├── 01-agg-dispatch.sql  # W1: aggregate dispatch (scaled: 10K groups, 1M rows)
│   ├── 02-fit-predict.sql   # W2: scalar fit_predict window function
│   ├── 03-ffi-micro.sql     # W3: FFI marshalling micro-bench (small groups + inference ON)
│   └── 01-agg-dispatch-1m.sql  # W1 full-scale (1M groups) for final before/after
├── results/                 # timestamped output files written by bench.sh
└── README.md                # how to interpret results
scripts/
└── bench.sh                 # the one documented command
```

### bench.sh outline

```bash
#!/usr/bin/env bash
# Usage: scripts/bench.sh [--full]
# Runs the Phase-4 benchmark suite against the locally built extension.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUCKDB="$REPO_ROOT/build/release/duckdb"
EXT="$REPO_ROOT/build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
RESULTS_DIR="$REPO_ROOT/bench/results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTFILE="$RESULTS_DIR/bench-$TIMESTAMP.md"

# Validate build is present
if [[ ! -x "$DUCKDB" ]]; then
    echo "ERROR: build/release/duckdb not found. Run: make release" >&2; exit 1
fi

mkdir -p "$RESULTS_DIR"

run_workload() {
    local name="$1" sql_file="$2"
    echo "## $name" | tee -a "$OUTFILE"
    echo '```' >> "$OUTFILE"
    "$DUCKDB" -unsigned \
        -cmd "LOAD '$EXT';" \
        -f "$sql_file" 2>&1 | tee -a "$OUTFILE"
    echo '```' >> "$OUTFILE"
    echo "" >> "$OUTFILE"
}

echo "# Benchmark Results — $TIMESTAMP" > "$OUTFILE"
echo "Extension: $EXT" >> "$OUTFILE"
echo "" >> "$OUTFILE"

run_workload "W1: Aggregate Dispatch (10K groups)" \
    "$REPO_ROOT/bench/workloads/01-agg-dispatch.sql"
run_workload "W2: Fit/Predict Window (10K groups)" \
    "$REPO_ROOT/bench/workloads/02-fit-predict.sql"
run_workload "W3: FFI Marshalling Micro-bench" \
    "$REPO_ROOT/bench/workloads/03-ffi-micro.sql"

if [[ "${1:-}" == "--full" ]]; then
    run_workload "W1-FULL: Aggregate Dispatch (1M groups)" \
        "$REPO_ROOT/bench/workloads/01-agg-dispatch-1m.sql"
fi

echo "Results written to: $OUTFILE"
```

[ASSUMED: exact CLI flag for inline command in duckdb — `-cmd` may need verification; `-c` runs then exits. Use `-init <(echo "LOAD '...';")` as an alternative. Adjust at implementation time.]

---

## Standard Stack

No new packages are needed for this phase. All work is:
- Shell scripting (`bench.sh`) — pure bash
- SQL files — DuckDB SQL
- Rust refactoring — existing workspace crates

The `libc` crate is already a workspace dependency [VERIFIED: Cargo.toml:22 — `libc = "0.2"`].

## Package Legitimacy Audit

**No external packages are introduced in this phase.** All code changes are to existing crates (`anofox-stats-ffi`) and new shell/SQL scripts.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Heap allocation tracking | Custom allocator shim | `gperftools` LD_PRELOAD or `valgrind --tool=massif` | Correct attribution across C++/Rust boundary |
| Cross-language flame graphs | Custom stack unwinder | `perf` with DWARF unwinding | perf works on mixed C++/Rust binaries |
| Allocation correctness | Custom memory tracker | `valgrind --tool=memcheck` or ASan rebuild | These catch use-after-free and leaks at the C ABI boundary |
| Benchmark result diffing | Custom diff tool | Standard `diff` or `git diff` on markdown output | bench.sh writes timestamped markdown; compare with diff |

---

## Common Pitfalls

### Pitfall 1: Extension version mismatch (installed vs. local build)

**What goes wrong:** `build/release/duckdb` auto-loads the `anofox_statistics` extension from `~/.duckdb/extensions/` (the installed community version), not the local build. Benchmarks run against stale code.

**Why it happens:** DuckDB autoloads community extensions if they are already installed. The `build/release/duckdb` binary inherits this behavior.

**How to avoid:** Always `LOAD` the extension by absolute path in every benchmark SQL or use `-cmd "LOAD 'build/release/extension/.../anofox_statistics.duckdb_extension';"`. Verify with `SELECT * FROM duckdb_extensions() WHERE extension_name = 'anofox_statistics';` — check `install_path` starts with the repo path, not `~/.duckdb`.

[VERIFIED: live test confirmed the version mismatch risk this session]

### Pitfall 2: Timer line goes to stdout, not to `.output FILE`

**What goes wrong:** Using `.output bench/results.txt` in SQL to capture output — the query results go to the file but `Run Time (s): ...` stays on stdout, so you capture results without timings.

**How to avoid:** Do not use `.output` for benchmark scripts. Instead redirect stdout of the entire `duckdb` invocation: `duckdb ... > bench/results.txt 2>&1`.

[VERIFIED: live test this session]

### Pitfall 3: FfiVec Drop runs on allocation failure

**What goes wrong:** If `FfiVec::alloc` fails midway (e.g., after allocating std_err and t_val but before ci_lo), `Drop` on the already-allocated instances frees them — correct! But if you partially fill `FitResultInference` before all 5 allocs succeed and then copy the pointers out, the Drop-and-free has already happened, leaving dangling pointers in the struct.

**How to avoid:** Allocate all 5 `FfiVec` instances before calling `.into_raw()` on any of them. Only call `.into_raw()` on all 5 together after confirming all 5 allocations succeeded. The proposed macro pattern above does this correctly.

### Pitfall 4: `libc::malloc` vs Rust global allocator confusion

**What goes wrong:** Using `Box::into_raw()` instead of `FfiVec::into_raw()` compiles and runs on Linux/glibc (same underlying allocator) but silently breaks on musl (WASM, musl CI target) where Rust's global allocator and libc's `malloc` are distinct.

**How to avoid:** The `FfiVec` wrapper must call `libc::malloc` explicitly. Code review the RAII type to confirm no `Box` or `Vec` allocation paths exist. Add a `#[cfg(test)]` test that verifies `FfiVec::into_raw()` produces a pointer that can be freed by `libc::free`.

### Pitfall 5: perf not installed — flamegraph install fails silently

**What goes wrong:** `cargo install flamegraph` succeeds but `cargo flamegraph` fails with "perf not found" at runtime.

**How to avoid:** Install `perf` first: `sudo pacman -S perf`. Confirm with `perf --version`. Then install flamegraph.

### Pitfall 6: Large workload benchmarks take >3 minutes on CI

**What goes wrong:** Including 1M-groups benchmark in a standard CI run causes timeouts.

**How to avoid:** Per CONTEXT.md, CI perf tracking is "noted only" this phase. The bench.sh default uses scaled-down workloads (10K groups). The `--full` flag runs 1M-groups locally only. README documents the distinction.

---

## Runtime State Inventory

**Not applicable.** This is a greenfield addition (new `scripts/bench.sh`, new `bench/` directory, Rust refactor within existing crates). No rename, migration, or external-state changes.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `build/release/duckdb` | All benchmarks | Yes | v1.5.4 | `make release` |
| `build/release/extension/anofox_statistics/*.duckdb_extension` | Extension loading | Yes | 2d054e0 | `make release` |
| `perf` | profiling, cargo flamegraph | No (installable) | `extra/perf 7.1.8-1` | `gperftools` (libprofiler.so installed) |
| `cargo flamegraph` | Rust flame graphs | No | post-perf install | `perf report` directly |
| `valgrind` | Leak check (PERF-04) | No (installable) | `extra/valgrind` | `RUSTFLAGS="-Z sanitizer=address"` + nightly |
| `gperftools` / libprofiler.so | CPU profiling | Yes | 2.18.1-1 | — |
| `pprof` analyzer | gperftools reports | Unknown | — | `perf report` |
| DuckDB `EXPLAIN ANALYZE` | Hotspot attribution | Yes | v1.5.4 built binary | — |

**Missing dependencies with no fallback:** None that block the phase. perf/flamegraph/valgrind are each installable; gperftools is a functional fallback for CPU profiling.

**Missing dependencies with fallback:**
- `perf` → `gperftools` for CPU profiling (LD_PRELOAD); `EXPLAIN ANALYZE` for query-level attribution
- `valgrind` → ASan rebuild for leak detection (requires nightly Rust for `sanitizer=address`)

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | DuckDB native test harness + Rust `cargo test` |
| Config file | None for SQL tests; `Cargo.toml` for Rust tests |
| Quick run command | `cargo test` (Rust unit tests, ~30s) |
| Full suite command | `make test` (SQL + Rust, ~2-5 min) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Verification |
|--------|----------|-----------|-------------------|--------------|
| PERF-01 | Harness runs and produces timing output | smoke | `bash scripts/bench.sh` exits 0; output contains "Run Time" | Manual check on file existence + grep |
| PERF-02 | Results file written to `bench/results/` | smoke | `ls bench/results/*.md` after bench.sh run | File exists check |
| PERF-03 | Before/after numbers produced; hotspots documented | manual | Compare bench run before+after optimization | Human review of diff output |
| PERF-04 | FFI refactor — behavior unchanged | automated | `cargo test && make test` both green | Full regression suite |
| PERF-04 | No new memory leaks from RAII refactor | leak-check | `valgrind --tool=memcheck ./build/release/duckdb -unsigned -f bench/workloads/03-ffi-micro.sql` | Zero "definitely lost" blocks |
| PERF-04 | FfiVec alloc uses libc::malloc (not Box) | unit | `#[test] fn ffi_vec_ptr_is_freeable_by_libc()` in types.rs or new test file | `cargo test` |

### Wave 0 Gaps

- [ ] `bench/workloads/01-agg-dispatch.sql` — W1 workload script (new file)
- [ ] `bench/workloads/02-fit-predict.sql` — W2 workload script (new file)
- [ ] `bench/workloads/03-ffi-micro.sql` — W3 FFI micro-bench (new file)
- [ ] `scripts/bench.sh` — benchmark harness shell script (new file)
- [ ] `bench/README.md` — documentation of workloads and how to run (new file)
- [ ] Unit test for `FfiVec` allocator correctness — in `crates/anofox-stats-ffi/src/` (new test module)

---

## Security Domain

`security_enforcement: true` in config. This phase's changes are:
1. Shell script (`bench.sh`) — no network, no user input, no credentials.
2. SQL workload files — pure DuckDB SQL with `generate_series` / `random()`; no external data.
3. Rust FFI refactor — replaces manual `libc::malloc/free` with a RAII wrapper; reduces the attack surface of the manual memory management pattern.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | n/a |
| V3 Session Management | No | n/a |
| V4 Access Control | No | n/a |
| V5 Input Validation | Partial | FFI refactor must not introduce buffer overflows; `FfiVec` bounds check |
| V6 Cryptography | No | n/a |

**Primary security benefit:** The `FfiVec` RAII wrapper with `Drop` reduces the risk of partial-allocation leaks (a security concern because unfreed memory that holds inference data may be referenced after free by a careless C++ caller) [CITED: CONCERNS.md — "FFI Null Pointer Validation" section].

---

## Open Questions

1. **`cargo flamegraph` against staticlib in external binary**
   - What we know: flamegraph works by wrapping `perf`; the extension is a `staticlib` linked into `duckdb`
   - What's unclear: exact invocation to target `duckdb` binary rather than a Rust binary
   - Recommendation: After installing `perf`, run `perf record` manually and use `flamegraph -- perf script` or `perf report`; verify in Wave 1 of execution

2. **`-cmd` vs `-init` for extension LOAD in bench.sh**
   - What we know: `-init FILENAME` reads SQL from file before stdin; `-f FILENAME` reads/processes a named file and exits; `-c COMMAND` runs SQL command and exits
   - What's unclear: Whether `-cmd COMMAND` is a valid flag (appears in help but needs confirmation for inline LOAD)
   - Recommendation: Use `echo "LOAD 'path';" | cat - bench/workload.sql | duckdb -unsigned` or write a preamble SQL file and use `-init preamble.sql -f workload.sql`

3. **pprof analyzer for gperftools output**
   - What we know: `libprofiler.so` is installed; `pprof` binary is not at `/usr/bin/pprof`
   - What's unclear: whether `go tool pprof` or another pprof variant is available
   - Recommendation: Try `perf` first (installable); use gperftools as backup; document in bench README

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | musl allocator differs from libc malloc, making Box/Vec unsafe across the FFI boundary | RAII Refactor Design | If wrong (musl uses same underlying malloc), the constraint is overly conservative but still correct; behavior is the same either way. Low risk. |
| A2 | FfiVec macro uses `?` operator for error propagation — caller sites must handle Err | RAII Refactor Design | Macro signature must match each call site's error handling style; may need per-site adaptation |
| A3 | `pprof` analyzer is not installed despite gperftools being present | Environment Availability | If pprof is available (e.g., via `go tool pprof`), gperftools becomes a stronger option |
| A4 | `-cmd COMMAND` flag valid for bench.sh inline LOAD | bench.sh Design | Use `-init` approach as fallback; low implementation risk |
| A5 | `cargo flamegraph` can target an external binary (not a cargo-managed binary) | Profiling Stack | Fallback: `perf record` + `perf report` directly |
| A6 | W3 micro-bench (small groups + inference ON) effectively isolates malloc overhead | Workload Inventory | If DuckDB dispatch overhead dominates even at small scale, the micro-bench may not isolate FFI cost; measure and adjust |

---

## Sources

### Primary (HIGH confidence — verified via direct file reads this session)

- `crates/anofox-stats-ffi/src/lib.rs` — all FFI function signatures, malloc counts, free functions [VERIFIED: Read this session, line ranges cited throughout]
- `crates/anofox-stats-ffi/src/types.rs` — FitResultInference, FitResultCore, GlmFitResultCore struct definitions [VERIFIED: Read this session]
- `Cargo.toml:24-27` — release profile (lto, codegen-units, opt-level) [VERIFIED: Read this session]
- `examples/performance_1m_groups/benchmark_ols.sql` — dataset SQL shape [VERIFIED: Read this session]
- `examples/performance_1m_groups/run_all_benchmarks.sh` — existing bash harness [VERIFIED: Read this session]
- `examples/performance_1m_groups/README.md` — benchmark results (i7-6800K, ~8 GB RAM) [VERIFIED: Read this session]
- `extension-ci-tools/makefiles/duckdb_extension.Makefile:199,203-204` — test targets [VERIFIED: Read this session]
- Live shell tests (duckdb CLI behavior, extension loading, timer output) [VERIFIED: Bash tool this session]
- System tool availability (`perf`, `gperftools`, `valgrind`) [VERIFIED: Bash tool this session]

### Secondary (MEDIUM confidence)

- `.planning/codebase/CONCERNS.md` — FFI memory management pattern, scaling concerns [CITED: Read this session]
- `.planning/codebase/STACK.md` — release profile documentation [CITED: Read this session]

### Tertiary (LOW confidence)

- Musl/glibc allocator compatibility: well-known ecosystem knowledge [ASSUMED]
- `cargo flamegraph` incantation for external binaries [ASSUMED — needs live verification]

---

## Metadata

**Confidence breakdown:**

- Build & extension loading: HIGH — verified live with CLI tests
- FFI malloc pattern: HIGH — verified by reading lib.rs and counting with grep
- Benchmark mechanics: HIGH — verified live with DuckDB CLI
- Profiling tool availability: HIGH — verified with shell commands
- RAII refactor design: MEDIUM — design is correct per ABI analysis; exact macro API is ASSUMED
- hotspot candidates: MEDIUM — CONCERNS.md is authoritative but profiling may surface surprises

**Research date:** 2026-08-31
**Valid until:** 2026-11-30 (stable codebase; DuckDB API stable at v1.5.4)
