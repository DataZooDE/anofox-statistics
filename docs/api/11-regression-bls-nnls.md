# BLS/NNLS Functions

Bounded and Non-negative Least Squares for constrained optimization.

## anofox_stats_bls_fit_agg

Bounded Least Squares with coefficient constraints.

**Signature:**
```sql
anofox_stats_bls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [lower_bounds LIST(DOUBLE)],
    [upper_bounds LIST(DOUBLE)],
    [options MAP]
) -> STRUCT
```

**Example:**
```sql
-- Coefficients bounded between 0 and 1
SELECT anofox_stats_bls_fit_agg(y, [x1, x2], [0, 0], [1, 1])
FROM data;
```

## anofox_stats_nnls_fit_agg

Non-negative Least Squares (all coefficients >= 0).

**Signature:**
```sql
anofox_stats_nnls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Example:**
```sql
-- Non-negative coefficients (e.g., mixture proportions)
SELECT anofox_stats_nnls_fit_agg(y, [x1, x2, x3])
FROM mixture_data;
```

## Short Aliases

- `bls_fit_agg` -> `anofox_stats_bls_fit_agg`
- `nnls_fit_agg` -> `anofox_stats_nnls_fit_agg`
