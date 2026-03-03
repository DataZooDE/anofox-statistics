# OLS (Ordinary Least Squares)

Ordinary Least Squares regression. Supports SVD, QR, and Cholesky decomposition with optional heteroscedasticity-consistent standard errors.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit` | Scalar | Process complete arrays in a single call |
| `ols_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `ols_fit_predict` | Window | Fit and predict in a single pass |
| `ols_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `ols_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_ols_fit

**Signature:**
```sql
anofox_stats_ols_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| y | LIST(DOUBLE) | Response variable values |
| x | LIST(LIST(DOUBLE)) | Feature arrays (each inner list is one feature) |
| fit_intercept | BOOLEAN | Include intercept term (default: true) |
| compute_inference | BOOLEAN | Compute t-tests, p-values, CIs (default: false) |
| confidence_level | DOUBLE | CI confidence level (default: 0.95) |

**Returns:** [FitResult](../reference/return_types.md#fitresult-structure) STRUCT

**Example:**
```sql
-- Simple regression: y = 2x + 1
SELECT anofox_stats_ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]]
);

-- With inference
SELECT anofox_stats_ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    true, true, 0.95
);
```

## anofox_stats_ols_fit_agg

Streaming OLS regression aggregate function. Supports `GROUP BY` and window functions via `OVER`.

**Signature:**
```sql
anofox_stats_ols_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Example:**
```sql
-- Per-group regression
SELECT
    category,
    (anofox_stats_ols_fit_agg(sales, [price, ads])).r_squared
FROM data
GROUP BY category;

-- Rolling regression (window function)
SELECT
    date,
    (anofox_stats_ols_fit_agg(y, [x]) OVER (
        ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    )).coefficients[1] as rolling_beta
FROM time_series;
```

## MAP Options

All OLS functions accept an optional MAP parameter for advanced configuration:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include intercept term |
| `compute_inference` | BOOLEAN | `false` | Compute t-tests, p-values, CIs |
| `confidence_level` | DOUBLE | `0.95` | CI confidence level |
| `solver` | VARCHAR | `'svd'` | Decomposition method: `'qr'`, `'svd'`, `'cholesky'` |
| `hc_type` | VARCHAR | `'none'` | Heteroscedasticity-consistent SEs: `'none'`, `'hc0'`, `'hc1'`, `'hc2'`, `'hc3'` |

**Example with MAP options:**
```sql
-- OLS with QR decomposition and HC3 robust standard errors
SELECT ols_fit_agg(
    y, [x1, x2],
    {'solver': 'qr', 'hc_type': 'hc3', 'compute_inference': true}
) FROM data;
```

## Use Cases

- Standard linear regression
- Baseline model before trying regularization
- Small to medium datasets with well-conditioned features
- When inference (p-values, confidence intervals) is needed

## See Also

- [Ridge](ridge.md) - L2 regularization for multicollinearity
- [Elastic Net](elasticnet.md) - Combined L1+L2 regularization
- [Table Macros](../macros/table_macros.md#ols_fit_predict_by) - Per-group predictions
