# Phase 4: Benchmarking & Performance - Pattern Map

**Mapped:** 2026-08-31
**Files analyzed:** 7 new/modified files
**Analogs found:** 6 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/bench.sh` | utility / harness | batch | `examples/performance_1m_groups/run_all_benchmarks.sh` | role-match |
| `bench/workloads/00-load-ext.sql` | config | request-response | `examples/performance_1m_groups/benchmark_ols.sql` | partial (structure) |
| `bench/workloads/01-agg-dispatch.sql` | utility | batch | `examples/performance_1m_groups/benchmark_ols_predict_agg.sql` | exact (agg shape) |
| `bench/workloads/02-fit-predict.sql` | utility | batch | `examples/performance_1m_groups/benchmark_ols.sql` | exact (window shape) |
| `bench/workloads/03-ffi-micro.sql` | utility | batch | `examples/performance_1m_groups/benchmark_ols.sql` | partial (no analog at small scale with inference) |
| `crates/anofox-stats-ffi/src/types.rs` | model / utility | — | `crates/anofox-stats-ffi/src/types.rs` (itself, extended) | self |
| `crates/anofox-stats-ffi/src/lib.rs` | service | request-response | `crates/anofox-stats-ffi/src/lib.rs` (itself, refactored) | self |

---

## Pattern Assignments

### `scripts/bench.sh` (utility, batch)

**Analog:** `examples/performance_1m_groups/run_all_benchmarks.sh`

**Shell script header pattern** (run_all_benchmarks.sh:1-8):
```bash
#!/bin/bash
# Run all fit_predict benchmarks with memory monitoring
# Usage: ./run_all_benchmarks.sh [path_to_duckdb]

set -e

DUCKDB=${1:-duckdb}
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
```

**run_benchmark function shape** (run_all_benchmarks.sh:18-41):
```bash
run_benchmark() {
    local name=$1
    local sql_file=$2

    echo "=== $name ==="

    # Run benchmark in background for memory monitoring
    $DUCKDB < "$sql_file" &
    local pid=$!

    # Monitor peak RSS
    local peak_rss=0
    while [ -d /proc/$pid ] 2>/dev/null; do
        local rss=$(awk '/^VmRSS:/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
        if [ "$rss" -gt "$peak_rss" ] 2>/dev/null; then
            peak_rss=$rss
        fi
        sleep 0.5
    done

    wait $pid
    echo "Peak RSS: $((peak_rss / 1024)) MB"
    echo ""
}
```

**Dispatch loop pattern** (run_all_benchmarks.sh:43-47):
```bash
run_benchmark "OLS" "$SCRIPT_DIR/benchmark_ols.sql"
run_benchmark "Ridge" "$SCRIPT_DIR/benchmark_ridge.sql"
run_benchmark "WLS" "$SCRIPT_DIR/benchmark_wls.sql"
run_benchmark "RLS" "$SCRIPT_DIR/benchmark_rls.sql"
run_benchmark "Elastic Net" "$SCRIPT_DIR/benchmark_elasticnet.sql"
```

**Key divergences from analog for `scripts/bench.sh`:**
- Use `set -euo pipefail` (stricter than analog's `set -e`) — matches `scripts/check_code_quality.sh:2`
- Use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` for reliable path resolution — matches `scripts/check_code_quality.sh:12`
- Must pass `-unsigned` flag to duckdb (analog does not; analog relies on autoload)
- Must explicitly LOAD the extension by path before running workloads (not via autoload)
- Capture stdout to a timestamped markdown file under `bench/results/` instead of just printing
- Support a `--full` flag to gate the 1M-group variant
- Remove RSS monitoring loop (not needed for this harness; simplify)

**LOAD incantation (verified, from RESEARCH.md):**
```bash
DUCKDB="$REPO_ROOT/build/release/duckdb"
EXT="$REPO_ROOT/build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
# -init approach (safe; -cmd may not be a valid flag):
"$DUCKDB" -unsigned -init <(printf "LOAD '%s';\n" "$EXT") -f "$sql_file"
```

---

### `bench/workloads/01-agg-dispatch.sql` (utility, batch — W1)

**Analog:** `examples/performance_1m_groups/benchmark_ols_predict_agg.sql`

**Full analog file** (benchmark_ols_predict_agg.sql:1-33):
```sql
-- OLS Predict Aggregate Benchmark: 1M Groups, 100M Rows, 3 Features
-- This benchmark tests the non-rolling predict aggregate function that:
-- - Fits model once per group on training rows (y IS NOT NULL)
-- - Returns predictions for ALL rows (including out-of-sample)
-- Usage: duckdb < benchmark_ols_predict_agg.sql

.timer on

-- Load the extension
LOAD 'anofox_statistics';

-- Generate test data with ~80% training rows (y not null) and 20% prediction rows (y null)
WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        i / 1000000 AS row_num,
        random() * 100 AS x1,
        random() * 50 AS x2,
        random() * 25 AS x3,
        CASE WHEN (i % 100) < 80 THEN random() * 100 ELSE NULL END AS y
    FROM generate_series(1, 100000000) t(i)
)
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN (pred).is_training THEN 1 ELSE 0 END) AS training_rows,
    SUM(CASE WHEN NOT (pred).is_training THEN 1 ELSE 0 END) AS prediction_rows
FROM (
    SELECT
        group_id,
        UNNEST(anofox_stats_ols_predict_agg(y, [x1, x2, x3])) AS pred
    FROM test_data
    GROUP BY group_id
) t;
```

**Key divergences for `01-agg-dispatch.sql`:**
- Omit `LOAD 'anofox_statistics';` — the harness loads the local build via `-init` before running this file
- Scale down to 10K groups / 1M rows for the default variant: `i % 10000` and `generate_series(1, 1000000)`
- Use `anofox_stats_ols_fit_agg(y, [x1,x2,x3])` aggregate (fit only, not predict_agg) to isolate dispatch overhead
- Output `.mode markdown` + simple `SELECT COUNT(*)` so timings are the only meaningful output
- Add `.timer on` at top

---

### `bench/workloads/02-fit-predict.sql` (utility, batch — W2)

**Analog:** `examples/performance_1m_groups/benchmark_ols.sql`

**Full analog file** (benchmark_ols.sql:1-23):
```sql
-- OLS Fit Predict Benchmark: 1M Groups, 100M Rows, 3 Features
-- Usage: duckdb < benchmark_ols.sql

.timer on

WITH test_data AS (
    SELECT
        i % 1000000 AS group_id,
        i / 1000000 AS row_num,
        random() * 100 AS x1,
        random() * 50 AS x2,
        random() * 25 AS x3,
        random() * 100 AS y
    FROM generate_series(1, 100000000) t(i)
)
SELECT COUNT(*) AS total_predictions FROM (
    SELECT anofox_stats_ols_fit_predict(y, [x1, x2, x3], {'fit_intercept': true}) OVER (
        PARTITION BY group_id ORDER BY row_num
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pred
    FROM test_data
) t WHERE pred IS NOT NULL;
```

**Key divergences for `02-fit-predict.sql`:**
- Scale down to 10K groups / 1M rows for default variant
- No `LOAD` statement (handled by bench.sh via `-init`)
- Keep `.timer on` at top, keep the window function shape (PARTITION BY + ROWS BETWEEN) — this is the exact W2 path

---

### `bench/workloads/03-ffi-micro.sql` (utility, batch — W3, no close analog)

**No close analog** — this workload is new. It targets FFI marshalling cost specifically by using a small number of groups with inference enabled. Closest structural reference is benchmark_ols.sql for SQL shape.

**Design from RESEARCH.md (W3 isolation goal):**
```sql
-- W3: FFI Marshalling Micro-bench
-- Small groups (100–1000), ~100 rows each, inference ON
-- Isolates per-call malloc overhead (5-array FitResultInference block)
.timer on

WITH test_data AS (
    SELECT
        i % 500 AS group_id,      -- 500 groups
        random() AS x1,
        random() AS y
    FROM generate_series(1, 50000) t(i)  -- 50K rows total (~100 rows/group)
)
SELECT COUNT(*) FROM (
    SELECT anofox_stats_ols_fit_agg(y, [x1], {'compute_inference': true})
    FROM test_data
    GROUP BY group_id
) t;
```

The key is `compute_inference: true` in the options map — this forces the 5-array malloc block to execute on every group invocation.

---

### `crates/anofox-stats-ffi/src/types.rs` — add `FfiVec<T>` (model/utility)

**Analog:** existing struct/impl patterns in `crates/anofox-stats-ffi/src/types.rs`

**Existing struct + impl pattern to follow** (types.rs:92-125, FitResultCore):
```rust
/// Core fit result (always returned)
#[repr(C)]
pub struct FitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    // ... scalar fields ...
}

impl Default for FitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            // ...
        }
    }
}
```

**Existing DataArray::to_vec() unsafe impl pattern** (types.rs:75-89) — shows the project's convention for unsafe blocks with Safety doc comments:
```rust
    /// Convert to Vec<f64>, replacing NULL with NaN
    ///
    /// # Safety
    /// Caller must ensure pointers are valid and len is correct
    pub unsafe fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            if self.is_valid(i) {
                result.push(*self.data.add(i));
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }
```

**FitResultInference struct** (types.rs:128-165) — the consumer struct whose fields `FfiVec<f64>::into_raw()` pointers will populate:
```rust
#[repr(C)]
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

**Where to add FfiVec:** Insert after the `DataArray` impl block (before line 92) or at the bottom of types.rs. No existing macros in the file to follow — `FfiVec` will be the first generic type added.

---

### `crates/anofox-stats-ffi/src/lib.rs` — add `macro_rules!` + refactor 13 sites (service)

**The exact current-state code to be replaced** — the OLS 5-array malloc block (lib.rs:188-254, canonical instance):

```rust
// lib.rs:188-254 — anofox_ols_fit, inference allocation block (CANONICAL INSTANCE)
            if !out_inference.is_null() {
                if let Some(inf) = result.inference {
                    let n = inf.std_errors.len();

                    // Allocate arrays
                    let std_err_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let t_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let p_val_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_lo_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;
                    let ci_hi_ptr = libc::malloc(n * std::mem::size_of::<f64>()) as *mut f64;

                    if n > 0
                        && (std_err_ptr.is_null()
                            || t_val_ptr.is_null()
                            || p_val_ptr.is_null()
                            || ci_lo_ptr.is_null()
                            || ci_hi_ptr.is_null())
                    {
                        // Free any allocated memory
                        if !std_err_ptr.is_null() {
                            libc::free(std_err_ptr as *mut libc::c_void);
                        }
                        if !t_val_ptr.is_null() {
                            libc::free(t_val_ptr as *mut libc::c_void);
                        }
                        if !p_val_ptr.is_null() {
                            libc::free(p_val_ptr as *mut libc::c_void);
                        }
                        if !ci_lo_ptr.is_null() {
                            libc::free(ci_lo_ptr as *mut libc::c_void);
                        }
                        if !ci_hi_ptr.is_null() {
                            libc::free(ci_hi_ptr as *mut libc::c_void);
                        }
                        libc::free(coef_ptr as *mut libc::c_void);

                        if !out_error.is_null() {
                            (*out_error).set(
                                ErrorCode::AllocationFailure,
                                "Failed to allocate inference arrays",
                            );
                        }
                        return false;
                    }

                    // Copy data
                    std::ptr::copy_nonoverlapping(inf.std_errors.as_ptr(), std_err_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.t_values.as_ptr(), t_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.p_values.as_ptr(), p_val_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_lower.as_ptr(), ci_lo_ptr, n);
                    std::ptr::copy_nonoverlapping(inf.ci_upper.as_ptr(), ci_hi_ptr, n);

                    (*out_inference) = FitResultInference {
                        std_errors: std_err_ptr,
                        t_values: t_val_ptr,
                        p_values: p_val_ptr,
                        ci_lower: ci_lo_ptr,
                        ci_upper: ci_hi_ptr,
                        len: n,
                        confidence_level: inf.confidence_level,
                        f_statistic: inf.f_statistic.unwrap_or(f64::NAN),
                        f_pvalue: inf.f_pvalue.unwrap_or(f64::NAN),
                    };
                } else {
                    (*out_inference) = FitResultInference::default();
                }
            }
```

This identical block (identical except variable names) also appears at:
- lib.rs:417-476 (Huber fit)
- lib.rs:663-722 (RANSAC fit)
- lib.rs:896-955 (line 900, 4th occurrence)
- lib.rs:1079-1138 (5th occurrence)
- ... 8 more at regular intervals through the file

**The paired free functions that must NOT change** (lib.rs:272-311):
```rust
/// Free memory allocated by anofox_ols_fit for core results
///
/// # Safety
/// `result` must be a pointer to a FitResultCore previously filled by anofox_ols_fit
#[no_mangle]
pub unsafe extern "C" fn anofox_free_result_core(result: *mut FitResultCore) {
    if result.is_null() {
        return;
    }
    if !(*result).coefficients.is_null() {
        libc::free((*result).coefficients as *mut libc::c_void);
        (*result).coefficients = std::ptr::null_mut();
    }
}

/// Free memory allocated by anofox_ols_fit for inference results
///
/// # Safety
/// `result` must be a pointer to a FitResultInference previously filled by anofox_ols_fit
#[no_mangle]
pub unsafe extern "C" fn anofox_free_result_inference(result: *mut FitResultInference) {
    if result.is_null() {
        return;
    }
    if !(*result).std_errors.is_null() {
        libc::free((*result).std_errors as *mut libc::c_void);
        (*result).std_errors = std::ptr::null_mut();
    }
    if !(*result).t_values.is_null() {
        libc::free((*result).t_values as *mut libc::c_void);
        (*result).t_values = std::ptr::null_mut();
    }
    if !(*result).p_values.is_null() {
        libc::free((*result).p_values as *mut libc::c_void);
        (*result).p_values = std::ptr::null_mut();
    }
    if !(*result).ci_lower.is_null() {
        libc::free((*result).ci_lower as *mut libc::c_void);
        (*result).ci_lower = std::ptr::null_mut();
    }
    if !(*result).ci_upper.is_null() {
        libc::free((*result).ci_upper as *mut libc::c_void);
        (*result).ci_upper = std::ptr::null_mut();
    }
}
```

**lib.rs header and import pattern** (lib.rs:1-27) — no macros exist today; the new `macro_rules!` will be the first:
```rust
//! C FFI boundary for anofox-statistics
//!
//! This crate provides C-compatible functions for calling from the C++ DuckDB extension layer.

mod types;

pub use types::*;

use anofox_stats_core::{
    // ... core imports ...
};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::slice;
```

**Error-return pattern in lib.rs** (lib.rs:164-173) — the convention used by all fit functions for OOM:
```rust
let coef_ptr = libc::malloc(n_coef * std::mem::size_of::<f64>()) as *mut f64;
if coef_ptr.is_null() && n_coef > 0 {
    if !out_error.is_null() {
        (*out_error).set(
            ErrorCode::AllocationFailure,
            "Failed to allocate coefficients",
        );
    }
    return false;
}
```

The macro must replicate this pattern: on OOM, set `(*out_error).set(ErrorCode::AllocationFailure, ...)` and `return false`. The `?` operator is NOT used in lib.rs — the file uses explicit `return false` for error paths throughout.

---

## Shared Patterns

### Shell script conventions
**Source:** `scripts/check_code_quality.sh:1-15`
**Apply to:** `scripts/bench.sh`
```bash
#!/bin/bash
set -e   # check_code_quality uses set -e; bench.sh should use set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
```

### DuckDB timer SQL preamble
**Source:** `examples/performance_1m_groups/benchmark_ols.sql:4`
**Apply to:** all `bench/workloads/*.sql` files
```sql
.timer on
```
Every workload file must have `.timer on` as the first line. The timer output goes to stdout, not to `.output FILE`, so the harness must redirect stdout to capture it (not use `.output` inside the SQL).

### Dataset generation shape
**Source:** `examples/performance_1m_groups/benchmark_ols.sql:6-14`
**Apply to:** `bench/workloads/01-agg-dispatch.sql`, `bench/workloads/02-fit-predict.sql`
```sql
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
Scale the modulo and series limit: `i % 10000` + `generate_series(1, 1000000)` for 10K-group default; keep as-is for the `*-1m.sql` full-scale variants.

### FFI unsafe doc comment convention
**Source:** `crates/anofox-stats-ffi/src/lib.rs:268-271`, `crates/anofox-stats-ffi/src/types.rs:76-78`
**Apply to:** `FfiVec<T>` in types.rs, any new unsafe functions
```rust
/// # Safety
/// Caller must ensure pointers are valid and len is correct
pub unsafe fn ...
```

### ABI-preserving free contract
**Source:** `crates/anofox-stats-ffi/src/lib.rs:287-311` (verbatim above)
**Apply to:** NOTHING — these functions must be copied/preserved byte-identical. The RAII refactor in lib.rs changes only the allocation side; the free functions are out of scope.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `bench/workloads/03-ffi-micro.sql` | utility | batch | No existing small-scale inference benchmark; all existing benchmarks use 1M groups and do not set `compute_inference: true` |

---

## Implementation Notes for Planner

### FfiVec placement
Add `FfiVec<T>` to `crates/anofox-stats-ffi/src/types.rs` after the `DataArray` impl block (before `FitResultCore` at line 92). It does not need `#[repr(C)]` — it is a pure Rust helper that never crosses the FFI boundary itself; only the raw pointer from `into_raw()` does.

### Macro placement
Add `macro_rules! alloc_inference_arrays { ... }` in `crates/anofox-stats-ffi/src/lib.rs` after the `use` statements and before the first `fn` (i.e., after line 27). Macros defined at module scope are available throughout the file.

### Error-handling style in macro
The macro must use `return false` + `(*out_error).set(ErrorCode::AllocationFailure, ...)` — NOT the `?` operator. The `FfiVec::alloc` method should return `Option<Self>` (or check for null inline) and the macro expands to the explicit `return false` path on OOM. This matches the existing error-handling style at lib.rs:165-173.

### 13 call sites
All 13 `std_err_ptr = libc::malloc` blocks are structurally identical to the canonical instance at lib.rs:188-254. The second instance (Huber, lib.rs:421-476) is also verbatim identical. The macro replacement is mechanical at all 13 sites.

### bench/workloads/ load-ext preamble
The `00-load-ext.sql` preamble file is only needed if the harness uses `-init 00-load-ext.sql`; if using `-init <(printf ...)` inline, the preamble file is not needed. The planner should choose one approach and be consistent.

---

## Metadata

**Analog search scope:** `scripts/`, `examples/performance_1m_groups/`, `crates/anofox-stats-ffi/src/`
**Files scanned:** 6 source files read directly; 2 grep passes on lib.rs
**Pattern extraction date:** 2026-08-31
