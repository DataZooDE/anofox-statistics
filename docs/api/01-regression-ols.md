# OLS Functions

Ordinary Least Squares regression using QR decomposition.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit` | Scalar | Fit on array data |
| `ols_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `ols_fit_predict` | Window | Fit and predict in window context |
| `ols_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `ols_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## ols_fit

Scalar function for array-based fitting.

**Signature:**
```sql
ols_fit(
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

**Returns:** [FitResult](20-return-types.md#fitresult-structure) STRUCT

**Example:**
```sql
SELECT ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    true, true, 0.95
);
```

## ols_fit_agg

Streaming aggregate function with GROUP BY support.

**Signature:**
```sql
ols_fit_agg(
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
    (ols_fit_agg(sales, [price, ads])).r_squared
FROM data
GROUP BY category;
```

## ols_fit_predict

Window function for fit and predict in rolling/expanding windows.

**Signature:**
```sql
ols_fit_predict(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) OVER (...) -> DOUBLE
```

**Example:**
```sql
SELECT
    date,
    ols_fit_predict(y, [x]) OVER (
        ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as predicted
FROM time_series;
```

## ols_fit_predict_agg

Aggregate function returning predictions for all input rows.

**Signature:**
```sql
ols_fit_predict_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> LIST(STRUCT)
```

**Returns:** List of structs with `y`, `x`, `yhat`, `yhat_lower`, `yhat_upper`, `is_training`

**Example:**
```sql
SELECT ols_fit_predict_agg(sales, [ads, price])
FROM monthly_data
GROUP BY region;
```

## ols_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict returning a table.

**Signature:**
```sql
FROM ols_fit_predict_by(
    source VARCHAR,
    group_col COLUMN,
    y_col COLUMN,
    x_cols LIST(COLUMN),
    [options MAP]
)
```

**Returns:** Table with columns: `group_id`, `y`, `x`, `yhat`, `yhat_lower`, `yhat_upper`, `is_training`

**Example:**
```sql
FROM ols_fit_predict_by('sales_data', store_id, revenue, [ads, traffic]);
```

**Options:** `fit_intercept`, `confidence_level`, `null_policy`

## Short Aliases

All functions have short aliases (without `anofox_stats_` prefix):
- `ols_fit` -> `anofox_stats_ols_fit`
- `ols_fit_agg` -> `anofox_stats_ols_fit_agg`
- `ols_fit_predict` -> `anofox_stats_ols_fit_predict`
- `ols_fit_predict_agg` -> `anofox_stats_ols_fit_predict_agg`
