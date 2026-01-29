# WLS (Weighted Least Squares)

Weighted Least Squares regression for heteroscedastic data.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `wls_fit` | Scalar | Process complete arrays in a single call |
| `wls_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `wls_fit_predict` | Window | Fit and predict in a single pass |
| `wls_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `wls_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_wls_fit

**Signature:**
```sql
anofox_stats_wls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    weights LIST(DOUBLE),
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
| weights | LIST(DOUBLE) | Observation weights (same length as y) |
| fit_intercept | BOOLEAN | Include intercept term (default: true) |
| compute_inference | BOOLEAN | Compute t-tests, p-values, CIs (default: false) |
| confidence_level | DOUBLE | CI confidence level (default: 0.95) |

**Returns:** [FitResult](../reference/return_types.md#fitresult-structure) STRUCT

**Example:**
```sql
SELECT anofox_stats_wls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    [1.0, 2.0, 3.0, 2.0, 1.0]  -- higher weight for middle observations
);
```

## anofox_stats_wls_fit_agg

Streaming WLS aggregate function.

```sql
SELECT anofox_stats_wls_fit_agg(y, [x], weight) FROM data;
```

## Choosing Weights

- **Inverse variance**: `weight = 1 / variance` when variance is known
- **Sample size**: `weight = n` when observations are group means
- **Reliability**: Higher weights for more reliable observations

## Use Cases

- Heteroscedastic data (non-constant variance)
- Aggregated data (weighted by sample size)
- When observation reliability varies
- Survey data with sampling weights

## See Also

- [OLS](ols.md) - Equal-weighted regression
- [RLS](rls.md) - Adaptive/online regression
- [Table Macros](../macros/table_macros.md#wls_fit_predict_by) - Per-group predictions
