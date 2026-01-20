# Ridge Functions

Ridge regression with L2 regularization.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `ridge_fit` | Scalar | Fit on array data |
| `ridge_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `ridge_fit_predict` | Window | Fit and predict in window context |
| `ridge_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `ridge_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## ridge_fit

**Signature:**
```sql
ridge_fit(
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
| alpha | DOUBLE | L2 regularization strength (>= 0) |

**Example:**
```sql
SELECT ridge_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1
);
```

## ridge_fit_agg

Streaming Ridge regression aggregate function.

```sql
SELECT (ridge_fit_agg(y, [x1, x2], 0.5)).coefficients
FROM data;
```

## ridge_fit_predict

Window function for rolling ridge regression.

```sql
SELECT ridge_fit_predict(y, [x], {'alpha': 0.1}) OVER (
    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
) FROM time_series;
```

## ridge_fit_predict_agg

Aggregate function returning predictions array.

```sql
SELECT ridge_fit_predict_agg(y, [x], {'alpha': 0.1})
FROM data GROUP BY category;
```

## ridge_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM ridge_fit_predict_by('data', category, y, [x1, x2], {'alpha': 0.1});
```

**Options:** `alpha`, `fit_intercept`, `confidence_level`, `null_policy`

## Short Aliases

- `ridge_fit` -> `anofox_stats_ridge_fit`
- `ridge_fit_agg` -> `anofox_stats_ridge_fit_agg`
- `ridge_fit_predict` -> `anofox_stats_ridge_fit_predict`
- `ridge_fit_predict_agg` -> `anofox_stats_ridge_fit_predict_agg`
