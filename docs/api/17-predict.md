# Predict Function

Apply a fitted model to new data.

## anofox_stats_predict

**Signature:**
```sql
anofox_stats_predict(
    model STRUCT,
    x LIST(LIST(DOUBLE))
) -> LIST(DOUBLE)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| model | STRUCT | Fitted model from any `*_fit` function |
| x | LIST(LIST(DOUBLE)) | New feature values |

**Example:**
```sql
-- Fit model and predict on new data
WITH fitted AS (
    SELECT ols_fit(y, [x1, x2]) as model
    FROM training_data
)
SELECT anofox_stats_predict(
    model,
    [[1.5, 2.5], [3.0, 4.0]]  -- new x values
) as predictions
FROM fitted;
```

## Short Alias

- `predict` -> `anofox_stats_predict`
