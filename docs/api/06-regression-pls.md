# PLS Functions

Partial Least Squares regression for high-dimensional data and multicollinearity.

## anofox_stats_pls_fit

PLS regression using the SIMPLS algorithm to find latent components that maximize covariance between X scores and y.

**Signature:**
```sql
anofox_stats_pls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_components | INTEGER | min(n_features, 10) | Number of PLS components |
| fit_intercept | BOOLEAN | true | Include intercept term |
| scale | BOOLEAN | true | Standardize features |

## anofox_stats_pls_fit_agg

Streaming PLS aggregate function.

## Short Aliases

- `pls_fit` -> `anofox_stats_pls_fit`
- `pls_fit_agg` -> `anofox_stats_pls_fit_agg`
