---
phase: 04-benchmarking-performance
reviewed: 2026-08-31
diff_base: b52c720
status: resolved
---

# Phase 4 Code Review — Findings & Resolutions

Adversarial deep review of the Phase-4 source changes (FFI RAII refactor,
`DataArray::to_vec` fast path, benchmark harness). Behavior oracle (`make test`
2421 assertions + `cargo test` 289+6) was green before and after the fixes below.

## Findings

### CR-01 (BLOCKER → FIXED) — dangling `*out_core.coefficients` on inference-OOM path
On an inference-array OOM, the 6 converted sites returned `false` with
`*out_core.coefficients` still holding the just-freed `coef_ptr`. **Pre-existing**
(the hand-written blocks had it too) and safe in practice today — every C++ caller
returns immediately on `false` without touching `*out_core` (verified in
`ols_fit.cpp`, `ols_aggregate.cpp`). Fixed as defense-in-depth: the macro's OOM
cleanup now also sets `*out_core = FitResultCore::default()` (nulls the freed
pointer) at all 6 sites. Only affects the (unreachable-in-tests) OOM error path;
success-path behavior unchanged. — `crates/anofox-stats-ffi/src/lib.rs`.

### WR-01 (WARNING → FIXED) — `copy_from_slice` length check elided in release
`debug_assert_eq!` was compiled out in release, so a future core-crate bug
producing unequal inference-array lengths could read past a slice/allocation (UB).
Promoted to `assert_eq!` so a mismatch panics in release too — a hard crash is
strictly safer than silent memory corruption at the FFI boundary. —
`crates/anofox-stats-ffi/src/types.rs`.

### WR-02 (WARNING → FIXED) — `bench.sh --help` dumped all `#` lines
`grep '^#'` printed the shebang and internal comments. Replaced with a `sed` range
that prints only the `Usage:` block. — `scripts/bench.sh`.

### IN-01 / IN-02 (INFO → acknowledged, no change)
`FfiVec` is implicitly `!Send + !Sync` (contains `*mut T`) — correct for
single-threaded FFI use; no change needed. Zero-length `FfiVec` (null ptr, len 0)
flows through the macro success path writing a `FitResultInference` with null
pointers and `len: 0` — safe because `anofox_free_result_inference` null-checks and
C++ callers loop on `len`; already covered by the `alloc` docstring + unit test.

## Verified (no defects)
ABI/allocator soundness (libc::malloc/free, `into_raw` suppresses Drop, zero-len →
null); OOM partial-allocation safety (all 5 alloc before any `into_raw`; `Some`
FfiVecs dropped/freed on the `return false`); byte-identical field mappings at the
6 converted sites; the 6 GLM (`z_values`, lenient OOM) + 1 ALM
(`standard_errors`/`conf_int_*`) sites genuinely unchanged and correctly
documented; `to_vec` fast-path correctness (len==0 guard, owned Vec, unchanged
nullable path); `bench.sh` `set -euo pipefail` + PIPESTATUS capture + quoting.

_Resolution commit: see `fix(04):` review-fixes commit on this branch._
