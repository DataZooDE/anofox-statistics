# Ridge Regression

Ridge regression with L2 regularization. Shrinks coefficients toward zero to handle multicollinearity.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ridge_fit` | Scalar | Process complete arrays in a single call |
| `ridge_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `ridge_fit_predict` | Window | Fit and predict in a single pass |
| `ridge_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `ridge_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_ridge_fit

**Signature:**
```sql
anofox_stats_ridge_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    alpha DOUBLE,
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| y | LIST(DOUBLE) | Response variable values |
| x | LIST(LIST(DOUBLE)) | Feature arrays |
| alpha | DOUBLE | L2 regularization strength (>= 0) |
| fit_intercept | BOOLEAN | Include intercept term (default: true) |
| compute_inference | BOOLEAN | Compute t-tests, p-values, CIs (default: false) |
| confidence_level | DOUBLE | CI confidence level (default: 0.95) |

**Returns:** [FitResult](../reference/return_types.md#fitresult-structure) STRUCT

**Example:**
```sql
SELECT anofox_stats_ridge_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1  -- alpha
);
```

## anofox_stats_ridge_fit_agg

Streaming Ridge regression aggregate function.

```sql
SELECT
    (anofox_stats_ridge_fit_agg(y, [x1, x2], 0.5)).coefficients
FROM data;
```

## Choosing Alpha

- **alpha = 0**: Equivalent to OLS
- **alpha = 0.01-0.1**: Light regularization
- **alpha = 1.0**: Moderate regularization
- **alpha = 10+**: Strong regularization

## Use Cases

- Multicollinearity in predictors
- When you have more features than observations
- Preventing overfitting
- Stable coefficient estimates

## See Also

- [OLS](ols.md) - Unregularized baseline
- [Elastic Net](elasticnet.md) - Combined L1+L2 for feature selection
- [Table Macros](../macros/table_macros.md#ridge_fit_predict_by) - Per-group predictions
