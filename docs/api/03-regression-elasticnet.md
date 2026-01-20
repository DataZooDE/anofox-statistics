# Elastic Net Functions

Elastic Net regression with combined L1/L2 regularization.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `elasticnet_fit` | Scalar | Fit on array data |
| `elasticnet_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `elasticnet_fit_predict` | Window | Fit and predict in window context |
| `elasticnet_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `elasticnet_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## elasticnet_fit

**Signature:**
```sql
elasticnet_fit(
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
| alpha | DOUBLE | Regularization strength (>= 0) |
| l1_ratio | DOUBLE | L1 ratio: 0=Ridge, 1=Lasso (range: 0-1) |
| max_iterations | INTEGER | Max coordinate descent iterations |
| tolerance | DOUBLE | Convergence tolerance |

**Example:**
```sql
SELECT elasticnet_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1, 0.5
);
```

## elasticnet_fit_agg

Streaming Elastic Net aggregate function.

## elasticnet_fit_predict

Window function for rolling elastic net.

## elasticnet_fit_predict_agg

Aggregate function returning predictions array.

## elasticnet_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM elasticnet_fit_predict_by('data', grp, y, [x1, x2], {'alpha': 0.1, 'l1_ratio': 0.5});
```

**Options:** `alpha`, `l1_ratio`, `max_iterations`, `tolerance`, `fit_intercept`, `confidence_level`, `null_policy`

## Short Aliases

- `elasticnet_fit` -> `anofox_stats_elasticnet_fit`
- `elasticnet_fit_agg` -> `anofox_stats_elasticnet_fit_agg`
- `elasticnet_fit_predict` -> `anofox_stats_elasticnet_fit_predict`
- `elasticnet_fit_predict_agg` -> `anofox_stats_elasticnet_fit_predict_agg`
