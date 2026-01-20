# GLM Functions

Generalized Linear Models for non-Gaussian distributions.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `poisson_fit_agg` | Aggregate | Poisson regression with GROUP BY support |
| `poisson_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `poisson_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## poisson_fit_agg

Poisson regression for count data using IRLS.

**Signature:**
```sql
poisson_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 25 | Max IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |

**Example:**
```sql
SELECT poisson_fit_agg(count, [exposure, factor])
FROM count_data;
```

## poisson_fit_predict_agg

Aggregate function returning predictions array.

## poisson_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM poisson_fit_predict_by('counts', region, events, [exposure, risk]);
```

**Options:** `fit_intercept`, `max_iterations`, `tolerance`, `null_policy`

## Short Aliases

- `poisson_fit_agg` -> `anofox_stats_poisson_fit_agg`
- `poisson_fit_predict_agg` -> `anofox_stats_poisson_fit_predict_agg`
