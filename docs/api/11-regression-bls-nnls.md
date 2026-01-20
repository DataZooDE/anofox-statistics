# BLS/NNLS Functions

Bounded and Non-negative Least Squares for constrained optimization.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `bls_fit_agg` | Aggregate | Bounded LS with GROUP BY support |
| `bls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `bls_fit_predict_by` | Table Macro | Fit per group, return predictions table |
| `nnls_fit_agg` | Aggregate | Non-negative LS with GROUP BY support |

## NULL Handling

The `null_policy` option controls how NULL values are handled:

| Value | Behavior |
|-------|----------|
| `'drop'` (default) | Drop rows with NULL y from training, include in output with predictions |
| `'drop_y_zero_x'` | Drop rows with NULL y OR zero x values from training |

See [Common Options](19-common-options.md#null-handling-options) for details.

## bls_fit_agg

Bounded Least Squares with coefficient constraints.

**Signature:**
```sql
bls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [lower_bounds LIST(DOUBLE)],
    [upper_bounds LIST(DOUBLE)],
    [options MAP]
) -> STRUCT
```

**Example:**
```sql
SELECT bls_fit_agg(y, [x1, x2], [0, 0], [1, 1])
FROM data;
```

## bls_fit_predict_agg

Aggregate function returning predictions array.

## bls_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM bls_fit_predict_by('mixture', batch, y, [x1, x2], {'lower_bounds': [0, 0], 'upper_bounds': [1, 1]});
```

**Options:** `lower_bounds`, `upper_bounds`, `fit_intercept`, `null_policy`

## nnls_fit_agg

Non-negative Least Squares (all coefficients >= 0).

**Signature:**
```sql
nnls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Example:**
```sql
SELECT nnls_fit_agg(y, [x1, x2, x3])
FROM mixture_data;
```

## Short Aliases

- `bls_fit_agg` -> `anofox_stats_bls_fit_agg`
- `bls_fit_predict_agg` -> `anofox_stats_bls_fit_predict_agg`
- `nnls_fit_agg` -> `anofox_stats_nnls_fit_agg`
