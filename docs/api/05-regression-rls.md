# RLS Functions

Recursive Least Squares for online/adaptive regression.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `rls_fit` | Scalar | Fit on array data |
| `rls_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `rls_fit_predict` | Window | Fit and predict in window context |
| `rls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `rls_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## rls_fit

**Signature:**
```sql
rls_fit(
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
| forgetting_factor | DOUBLE | Exponential forgetting (0.95-1.0 typical) |
| initial_p_diagonal | DOUBLE | Initial covariance matrix diagonal |

**Example:**
```sql
SELECT rls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.99, true, 100.0
);
```

## rls_fit_agg

Streaming RLS aggregate function. Ideal for adaptive/online learning.

```sql
SELECT rls_fit_agg(y, [x], 0.95) FROM streaming_data;
```

## rls_fit_predict

Window function for adaptive rolling regression.

## rls_fit_predict_agg

Aggregate function returning predictions array.

## rls_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM rls_fit_predict_by('streaming', sensor_id, value, [temp, pressure], {'forgetting_factor': 0.95});
```

**Options:** `forgetting_factor`, `initial_p_diagonal`, `fit_intercept`, `confidence_level`, `null_policy`

## Short Aliases

- `rls_fit` -> `anofox_stats_rls_fit`
- `rls_fit_agg` -> `anofox_stats_rls_fit_agg`
- `rls_fit_predict` -> `anofox_stats_rls_fit_predict`
- `rls_fit_predict_agg` -> `anofox_stats_rls_fit_predict_agg`
