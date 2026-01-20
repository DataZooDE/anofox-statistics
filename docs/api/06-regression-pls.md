# PLS Functions

Partial Least Squares regression for high-dimensional data and multicollinearity.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `pls_fit` | Scalar | Fit on array data |
| `pls_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `pls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `pls_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## pls_fit

PLS regression using the SIMPLS algorithm.

**Signature:**
```sql
pls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_components | INTEGER | min(n_features, 10) | Number of PLS components |
| fit_intercept | BOOLEAN | true | Include intercept term |
| scale | BOOLEAN | true | Standardize features |

## pls_fit_agg

Streaming PLS aggregate function.

## pls_fit_predict_agg

Aggregate function returning predictions array.

## pls_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM pls_fit_predict_by('spectral', sample_id, concentration, [w1, w2, w3], {'n_components': 3});
```

**Options:** `n_components`, `fit_intercept`, `scale`, `null_policy`

## Short Aliases

- `pls_fit` -> `anofox_stats_pls_fit`
- `pls_fit_agg` -> `anofox_stats_pls_fit_agg`
- `pls_fit_predict_agg` -> `anofox_stats_pls_fit_predict_agg`
