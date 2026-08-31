# Profiling & hotspots (PERF-03)

Profiling of the native **release** build (LTO, `codegen-units=1`, `opt-level=3`)
to surface the top hotspots and, for each, either apply a safe behavior-preserving
optimization or document it as inherent — every decision backed by before/after
numbers from the Plan-01 harness (`scripts/bench.sh`).

## Methodology & tooling

`perf`, `cargo flamegraph`, and `valgrind` all require `sudo` to install on this
box and were **not available**; `gperftools` `libprofiler.so` is present but its
`pprof` analyzer is not, so it could not produce symbol-level attribution. Per
04-CONTEXT / 04-RESEARCH the sanctioned fallback is **DuckDB's own profiler**,
which needs no install:

- **`EXPLAIN ANALYZE` / `PRAGMA enable_profiling='json'`** — per-operator timing
  and cardinality. This splits DuckDB-node time (`HASH_GROUP_BY`, `TABLE_SCAN`,
  `PROJECTION`) from the aggregate-function time (the FFI + Rust fit), which is
  the attribution that matters here.
- **Differential workloads** — W1 (aggregate dispatch, no inference) vs W3
  (tiny fits with `compute_inference: true`, so per-call FFI marshalling/allocation
  dominates). Comparing per-group cost isolates the FFI-boundary contribution.
- **`scripts/bench.sh`** timings for whole-workload before/after, and a controlled
  A/B (rebuild with/without the change) for the one optimization landed.
- **Code reading** of the FFI marshalling path (`DataArray::to_vec`, the inference
  block) for candidates the operator view cannot name.

> `perf`/flamegraph would add function-level symbol attribution. If wanted later:
> `sudo pacman -S perf && cargo install flamegraph`, then re-run this profiling —
> the conclusions below (DuckDB dispatch dominates; FFI marshalling is a minority
> cost) are unlikely to change, but perf would confirm them at symbol granularity.

## Operator attribution (BEFORE — post Plan-02 refactor)

`PRAGMA enable_profiling='json'` on the default-scale workloads:

**W1 — aggregate dispatch, 10K groups / 1M rows, OLS fit per group (no inference):**

| Operator | Time | Share |
|----------|------|-------|
| `HASH_GROUP_BY` (incl. per-group OLS fit via FFI) | ~0.032 s | ~66% |
| data-gen `PROJECTION` (`random()`) | ~0.005 s | ~10% |
| `GENERATE_SERIES` | ~0.0003 s | <1% |
| **Total query** | ~0.049 s | 100% |

**W3 — FFI micro-bench, 500 groups / 50K rows, `compute_inference: true`:**

| Operator | Time | Share |
|----------|------|-------|
| `HASH_GROUP_BY` (fit + 5-array inference marshalling) | ~0.0024 s | ~34% |
| data-gen `PROJECTION` | ~0.0003 s | ~4% |
| **Total query** | ~0.0070 s | 100% |

The aggregate work lives inside `HASH_GROUP_BY` (DuckDB runs the extension's
aggregate `Update`/`Finalize` there). Per-group: W1 ≈ 3.2 µs (3-feature fit,
no inference), W3 ≈ 4.8 µs (1-feature fit **plus** the 5-array inference path) —
the inference marshalling is the measurable per-call add-on, as predicted.

## Top 3 hotspots & dispositions

### 1. DuckDB `HASH_GROUP_BY` aggregate dispatch + per-group state — **INHERENT**

The single dominant cost (~66% of W1; the bulk of the ~0.08 s at 5M rows/50K
groups). This is DuckDB's group-by hashing/partitioning plus the per-group
aggregate `Finalize` dispatch. The extension **registers** an aggregate function
but cannot change how DuckDB schedules and dispatches group-by aggregation, and
rewriting the core numerics is explicitly out of scope (REQUIREMENTS). Documented
as inherent: the cost is DuckDB dispatch machinery, not extension code we control.
*Before:* W1 `HASH_GROUP_BY` ~0.032 s / 10K groups. No safe extension-side change.

### 2. FFI inference 5-array marshalling (`compute_inference`) — **OPTIMIZED (leak-safety) / count INHERENT**

The per-call allocation of `std_errors / t_values / p_values / ci_lower /
ci_upper` — the W3-heavy path. **Plan 02** replaced the 6 strict hand-written
5-array `libc::malloc` blocks with the `FfiVec<T>` RAII wrapper + the
`alloc_inference_arrays!` macro, removing the manual free/OOM boilerplate and its
leak risk (RAII `Drop` + allocate-all-before-`into_raw`). The **number** of
allocations (5 arrays) is inherent to the FFI contract — five separate arrays
must cross to C++ and be freed by `anofox_free_result_inference` — so it is not
reduced further without an ABI change (excluded by CONTEXT).
*Before:* W3 `HASH_GROUP_BY` ~0.0024 s / 500 groups. *After (Plan 02):* behavior
unchanged (make test 2421 assertions green); leak risk removed; per-call cycle
count ~unchanged (same allocation count, now RAII-managed).

### 3. `DataArray::to_vec` per-call input marshalling — **OPTIMIZED**

Each FFI fit call converts every input column `DataArray` into an owned
`Vec<f64>` (NULL→NaN). The original did a per-element `is_valid()` branch + push
for every value. Added a **bulk-copy fast path** for the common no-validity-mask
case (dense/non-nullable columns — every benchmark column, and most real data):

```rust
if self.validity.is_null() {
    return std::slice::from_raw_parts(self.data, self.len).to_vec();
}
```

Safe and behavior-preserving: it still returns an owned `Vec<f64>` (no borrow
crosses the FFI boundary → no aliasing/lifetime risk, cf. threat T-04-07); only
the unreachable-in-this-case NULL→NaN branch is skipped. The nullable path is
unchanged.

*Before/after* (controlled A/B: rebuild without vs with the change; 50K groups /
5M rows, 6 warm runs each, `.timer` query time):

| | mean | runs (s) |
|---|------|----------|
| BEFORE (per-element loop) | ~0.0845 s | 0.091 0.081 0.083 0.087 0.083 0.082 |
| AFTER (bulk-copy fast path) | ~0.0813 s | 0.083 0.082 0.081 0.080 0.081 0.081 |

A small but consistent **~3–4%** query-time improvement (tighter, lower AFTER
runs). Modest because `HASH_GROUP_BY` (hotspot #1) dominates; `to_vec` is a
minority of total time. Behavior unchanged: `make test` 2421 assertions and
`cargo test` (289 core + 6 ffi) stay green.

## Summary

| # | Hotspot | Disposition | Evidence |
|---|---------|-------------|----------|
| 1 | DuckDB `HASH_GROUP_BY` dispatch/state | **inherent** | ~66% of W1; DuckDB machinery, not extension-controllable |
| 2 | FFI 5-array inference marshalling | **optimized (Plan 02, leak-safety)**; count inherent to ABI | W3 path; FfiVec+macro; make test green |
| 3 | `DataArray::to_vec` per-call Vec alloc | **optimized** | bulk-copy fast path; ~3–4% at 5M/50K, A/B-controlled; tests green |

All optimizations are behavior-preserving (the full `test/sql` + `cargo test`
suites stay green with results unchanged). The honest finding: at these scales
the extension's per-call marshalling is a **minority** of total time — DuckDB's
group-by dispatch dominates — so the safe wins are modest and the largest cost is
inherent to DuckDB, documented rather than forced.
