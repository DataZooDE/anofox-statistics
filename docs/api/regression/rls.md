# RLS (Recursive Least Squares)

Recursive Least Squares for online/adaptive regression with exponential forgetting.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `rls_fit` | Scalar | Process complete arrays in a single call |
| `rls_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `rls_fit_predict` | Window | Fit and predict in a single pass |
| `rls_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `rls_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_rls_fit

**Signature:**
```sql
anofox_stats_rls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [forgetting_factor DOUBLE DEFAULT 1.0],
    [fit_intercept BOOLEAN DEFAULT true],
    [initial_p_diagonal DOUBLE DEFAULT 100.0]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| y | LIST(DOUBLE) | Response variable values |
| x | LIST(LIST(DOUBLE)) | Feature arrays |
| forgetting_factor | DOUBLE | Exponential forgetting (0.95-1.0 typical) |
| fit_intercept | BOOLEAN | Include intercept term (default: true) |
| initial_p_diagonal | DOUBLE | Initial covariance matrix diagonal |

**Returns:** [FitResult](../reference/return_types.md#fitresult-structure) STRUCT

**Example:**
```sql
SELECT anofox_stats_rls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.99,  -- forgetting_factor
    true,  -- fit_intercept
    100.0  -- initial_p_diagonal
);
```

## anofox_stats_rls_fit_agg

Streaming RLS aggregate function. Ideal for adaptive/online learning.

```sql
-- Adaptive regression with exponential forgetting
SELECT anofox_stats_rls_fit_agg(y, [x], 0.95) FROM streaming_data;
```

## Understanding Forgetting Factor

- **λ = 1.0**: No forgetting, all observations weighted equally (converges to OLS)
- **λ = 0.99**: Slight forgetting, recent data weighted slightly more
- **λ = 0.95**: Moderate forgetting, adapts to recent trends
- **λ = 0.90**: Strong forgetting, rapid adaptation

**Effective window size**: Approximately `1 / (1 - λ)` observations

## Use Cases

- Time-varying parameters
- Adaptive control systems
- Online learning / streaming data
- Concept drift in machine learning
- Real-time parameter tracking

## See Also

- [OLS](ols.md) - Static regression
- [WLS](wls.md) - Weighted regression
- [Table Macros](../macros/table_macros.md#rls_fit_predict_by) - Per-group predictions
