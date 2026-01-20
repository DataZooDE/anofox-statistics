# Quantile Functions

Quantile regression for modeling conditional quantiles (including median regression).

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `quantile_fit` | Scalar | Fit on array data |
| `quantile_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `quantile_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `quantile_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## NULL Handling

The `null_policy` option controls how NULL values are handled:

| Value | Behavior |
|-------|----------|
| `'drop'` (default) | Drop rows with NULL y from training, include in output with predictions |
| `'drop_y_zero_x'` | Drop rows with NULL y OR zero x values from training |

See [Common Options](19-common-options.md#null-handling-options) for details.

## quantile_fit

**Signature:**
```sql
quantile_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| quantile | DOUBLE | 0.5 | Target quantile (0-1, 0.5 = median) |
| fit_intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 1000 | Max optimization iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

## quantile_fit_agg

Streaming quantile regression aggregate function.

## quantile_fit_predict_agg

Aggregate function returning predictions array.

## quantile_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM quantile_fit_predict_by('sales', region, revenue, [price, promo], {'quantile': 0.5});
```

**Options:** `quantile`, `fit_intercept`, `max_iterations`, `tolerance`, `null_policy`

## Short Aliases

- `quantile_fit` -> `anofox_stats_quantile_fit`
- `quantile_fit_agg` -> `anofox_stats_quantile_fit_agg`
- `quantile_fit_predict_agg` -> `anofox_stats_quantile_fit_predict_agg`
