# GLM Functions

Generalized Linear Models for non-Gaussian distributions.

## anofox_stats_poisson_fit_agg

Poisson regression for count data using IRLS (Iteratively Reweighted Least Squares).

**Signature:**
```sql
anofox_stats_poisson_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 25 | Max IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |

**Example:**
```sql
-- Count data regression
SELECT anofox_stats_poisson_fit_agg(count, [exposure, factor])
FROM count_data;
```

## Short Aliases

- `poisson_fit_agg` -> `anofox_stats_poisson_fit_agg`
