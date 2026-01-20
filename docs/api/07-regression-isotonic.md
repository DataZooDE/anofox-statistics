# Isotonic Functions

Isotonic (monotonic) regression for non-decreasing/non-increasing constraints.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `isotonic_fit` | Scalar | Fit on array data |
| `isotonic_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `isotonic_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `isotonic_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## isotonic_fit

**Signature:**
```sql
isotonic_fit(
    y LIST(DOUBLE),
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

Note: Isotonic takes a single x column (not a list of features).

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| increasing | BOOLEAN | true | Enforce non-decreasing constraint |

## isotonic_fit_agg

Streaming isotonic regression aggregate function.

## isotonic_fit_predict_agg

Aggregate function returning predictions array.

## isotonic_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM isotonic_fit_predict_by('calibration', model_id, actual, predicted, {'increasing': true});
```

**Options:** `increasing`, `null_policy`

## Short Aliases

- `isotonic_fit` -> `anofox_stats_isotonic_fit`
- `isotonic_fit_agg` -> `anofox_stats_isotonic_fit_agg`
- `isotonic_fit_predict_agg` -> `anofox_stats_isotonic_fit_predict_agg`
