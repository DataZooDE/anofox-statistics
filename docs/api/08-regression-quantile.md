# Quantile Functions

Quantile regression for modeling conditional quantiles (including median regression).

## anofox_stats_quantile_fit

**Signature:**
```sql
anofox_stats_quantile_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| quantile | DOUBLE | 0.5 | Target quantile (0-1, 0.5 = median) |
| fit_intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 1000 | Max optimization iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

## anofox_stats_quantile_fit_agg

Streaming quantile regression aggregate function.

## Short Aliases

- `quantile_fit` -> `anofox_stats_quantile_fit`
- `quantile_fit_agg` -> `anofox_stats_quantile_fit_agg`
