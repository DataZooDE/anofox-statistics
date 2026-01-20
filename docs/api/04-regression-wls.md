# WLS Functions

Weighted Least Squares regression.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `wls_fit` | Scalar | Fit on array data |
| `wls_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `wls_fit_predict` | Window | Fit and predict in window context |
| `wls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `wls_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## NULL Handling

The `null_policy` option controls how NULL values are handled:

| Value | Behavior |
|-------|----------|
| `'drop'` (default) | Drop rows with NULL y from training, include in output with predictions |
| `'drop_y_zero_x'` | Drop rows with NULL y OR zero x values from training |

See [Common Options](19-common-options.md#null-handling-options) for details.

## wls_fit

**Signature:**
```sql
wls_fit(
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
| weights | LIST(DOUBLE) | Observation weights (same length as y) |

**Example:**
```sql
SELECT wls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    [1.0, 2.0, 3.0, 2.0, 1.0]
);
```

## wls_fit_agg

Streaming WLS aggregate function.

```sql
SELECT wls_fit_agg(y, [x], weight) FROM data;
```

## wls_fit_predict

Window function for rolling WLS.

## wls_fit_predict_agg

Aggregate function returning predictions array.

## wls_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict. Note: requires weight column.

```sql
FROM wls_fit_predict_by('data', grp, y, [x1, x2], weight_col);
```

**Options:** `fit_intercept`, `confidence_level`, `null_policy`

## Short Aliases

- `wls_fit` -> `anofox_stats_wls_fit`
- `wls_fit_agg` -> `anofox_stats_wls_fit_agg`
- `wls_fit_predict` -> `anofox_stats_wls_fit_predict`
- `wls_fit_predict_agg` -> `anofox_stats_wls_fit_predict_agg`
