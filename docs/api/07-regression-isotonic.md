# Isotonic Functions

Isotonic (monotonic) regression for non-decreasing/non-increasing constraints.

## anofox_stats_isotonic_fit

**Signature:**
```sql
anofox_stats_isotonic_fit(
    y LIST(DOUBLE),
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| increasing | BOOLEAN | true | Enforce increasing constraint |

## anofox_stats_isotonic_fit_agg

Streaming isotonic regression aggregate function.

## Short Aliases

- `isotonic_fit` -> `anofox_stats_isotonic_fit`
- `isotonic_fit_agg` -> `anofox_stats_isotonic_fit_agg`
