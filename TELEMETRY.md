# Anofox Statistics Telemetry

The `anofox_statistics` DuckDB extension collects **anonymous, privacy-preserving
usage telemetry** so we can see which statistical functions are used, on which
platforms, and where they fail — and prioritise accordingly. It is **on by
default** and **trivial to turn off**.

Telemetry is emitted through the shared
[`DataZooDE/posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry)
library and follows the cross-product **`telemetry_schema: 2`** envelope
(`posthog-telemetry/TELEMETRY-SCHEMA.md`). Ingestion is the EU PostHog cloud.
Telemetry is compiled in only when the extension is built with
`ANOFOX_TELEMETRY_ENABLED`; otherwise every call site is a no-op stub.

## How to turn it off

Any one of these fully short-circuits telemetry — when disabled, **nothing
leaves the machine** (the opt-out is enforced at the transport, not just at the
call sites):

```sql
SET anofox_telemetry_enabled = false;   -- DuckDB setting (per session)
```

```bash
export DATAZOO_DISABLE_TELEMETRY=1       # environment (1|true|yes)
```

## The guarantee: bounded, enumerated, non-PII

Every property we send is **either** a constant drawn from a small,
code-controlled enumeration **or** a pure number (durations, counts). The
library additionally clamps every outgoing string to 512 bytes as a backstop.

We **never** send: table names, column names, `FILTER`/`WHERE` clauses, SQL
text, row/result data, model coefficients, input values, or error messages. The
only free-form-looking strings sent are **function names**, and those are drawn
from the fixed set of functions this extension registers — not from user data.

## What is collected

### Envelope (attached to every event)

`product` (`anofox_statistics`), `product_version`, `product_edition` (`oss`),
`telemetry_schema` (`2`), `duckdb_version`, `os`, `arch`, `platform`, `is_ci`,
`is_container`, a per-process `$session_id`, and — once associated — the
`deployment` group. `distinct_id` is the SHA-256 of a machine id: a **stable,
pseudonymous** identifier, not tied to any personal data.

### Events

| Event | When | Properties (beyond the envelope) |
|---|---|---|
| `extension_loaded` | the `anofox_statistics` extension loads | — |
| `function_executed` | a DuckDB function runs — **aggregated** per function per session (not per row) | `function_name`, `call_count`, `duration_ms_p50` |

That is the complete set of events emitted by this repository today. The shared
schema also defines `feature_used` (`CaptureFeature`) and `$exception`
(`CaptureError`), but `anofox_statistics` does **not** emit them yet — that is a
later, per-repo instrumentation pass.

## Function-call aggregation

DuckDB function calls are recorded via `RecordFunctionCall(function_name)`, which
aggregates in-process into a single `function_executed` event per function per
session (carrying `call_count` and `duration_ms_p50`), flushed at session end.

The instrumented functions are the extension's statistical routines — regression
fits (`ols_fit`, `ridge_fit`, `elasticnet_fit`, `wls_fit`, `rls_fit`, …), their
aggregate/window forms (`ols_fit_agg`, `ols_fit_predict`, …), hypothesis tests
(`t_test`, `mann_whitney`, `kruskal_wallis`, `shapiro_wilk`, …), and diagnostics
(`aic`, `bic`, `vif`, `jarque_bera`, …). Each `RecordFunctionCall` sits at the
function's bind/registration step or at the top of the per-chunk execute path —
**never inside a per-row loop** — so a million-row scan produces O(1) telemetry
rows, not a firehose.

## Enterprise / account analytics

`anofox_statistics` is OSS and associates only the `deployment` group. It has no
license key, so no `account` group is associated.
