# Fit-Predict Window Functions

Window functions that fit a model and return predictions in a single pass.

## Overview

Fit-predict window functions use the `OVER` clause to define partitions and ordering:

```sql
SELECT
    ols_fit_predict(y, [x1, x2]) OVER (
        PARTITION BY group_id
        ORDER BY date
    ) as prediction
FROM data;
```

## Available Functions

| Function | Description |
|----------|-------------|
| `ols_fit_predict` | OLS regression |
| `ridge_fit_predict` | Ridge regression |
| `elasticnet_fit_predict` | Elastic Net regression |
| `wls_fit_predict` | Weighted Least Squares |
| `rls_fit_predict` | Recursive Least Squares |

## Example

```sql
-- Rolling regression with predictions
SELECT
    date,
    actual,
    ols_fit_predict(actual, [feature1, feature2]) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as predicted
FROM time_series;
```
