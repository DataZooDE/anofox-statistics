# Elastic Net Regression

Elastic Net regression with combined L1/L2 regularization. Combines Lasso's feature selection with Ridge's stability.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `elasticnet_fit` | Scalar | Process complete arrays in a single call |
| `elasticnet_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `elasticnet_fit_predict` | Window | Fit and predict in a single pass |
| `elasticnet_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `elasticnet_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_elasticnet_fit

**Signature:**
```sql
anofox_stats_elasticnet_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    alpha DOUBLE,
    l1_ratio DOUBLE,
    [fit_intercept BOOLEAN DEFAULT true],
    [max_iterations INTEGER DEFAULT 1000],
    [tolerance DOUBLE DEFAULT 1e-6]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| y | LIST(DOUBLE) | Response variable values |
| x | LIST(LIST(DOUBLE)) | Feature arrays |
| alpha | DOUBLE | Regularization strength (>= 0) |
| l1_ratio | DOUBLE | L1 ratio: 0=Ridge, 1=Lasso (range: 0-1) |
| max_iterations | INTEGER | Max coordinate descent iterations |
| tolerance | DOUBLE | Convergence tolerance |

**Returns:** [FitResult](../reference/return_types.md#fitresult-structure) STRUCT

**Example:**
```sql
SELECT anofox_stats_elasticnet_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1,  -- alpha
    0.5   -- l1_ratio (50% L1, 50% L2)
);
```

## anofox_stats_elasticnet_fit_agg

Streaming Elastic Net aggregate function.

```sql
SELECT elasticnet_fit_agg(y, [x1, x2], 0.1, 0.5)
FROM data;
```

## Understanding l1_ratio

- **l1_ratio = 0**: Pure Ridge (L2 only)
- **l1_ratio = 0.5**: Equal mix of L1 and L2
- **l1_ratio = 1**: Pure Lasso (L1 only)

## Use Cases

- Feature selection with grouped correlated features
- When Lasso is unstable due to collinearity
- High-dimensional data with correlated predictors
- Sparse models with stability

## See Also

- [OLS](ols.md) - Unregularized baseline
- [Ridge](ridge.md) - L2 regularization only
- [Table Macros](../macros/table_macros.md#elasticnet_fit_predict_by) - Per-group predictions
