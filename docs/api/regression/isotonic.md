# Isotonic Regression

Isotonic regression fits a monotone (non-decreasing or non-increasing)
step function of a single predictor, using the Pool Adjacent Violators
Algorithm (PAVA).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `isotonic_fit_agg` | Aggregate | Fit an isotonic regression per group <!-- TODO(lead): verify after *_fit_agg lands --> |
| `isotonic_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `isotonic_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#isotonic_fit_predict_by) |

Unlike the other regressions, `x` is a single `DOUBLE`, not a list.

## isotonic_fit_agg

<!-- TODO(lead): verify after *_fit_agg lands -->

**Signature:**

```text
isotonic_fit_agg(y DOUBLE, x DOUBLE [, options MAP]) -> STRUCT
```

Options: `increasing` (see [Options](#options)).

```sql skip
-- TODO(lead): un-skip once isotonic_fit_agg is registered
SELECT isotonic_fit_agg(response, dose, {'increasing': true}) AS fit
FROM isotonic_demo;
```

## isotonic_fit_predict_agg

**Signature:**

```text
isotonic_fit_predict_agg(y DOUBLE, x DOUBLE [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
isotonic_fit_predict_agg(y DOUBLE, x DOUBLE, split_col VARCHAR [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
```

Rows with a NULL `y` (or, with the split form, rows whose `split_col` is not
`'train'`/`'training'`) are not used for fitting but still receive a
prediction. There are no prediction intervals.

**Example:**

```sql
CREATE OR REPLACE TABLE isotonic_demo AS
SELECT * FROM (VALUES
    (1.0, 1.5), (2.0, 2.0), (3.0, 1.8), (4.0, 3.5),
    (5.0, 4.0), (6.0, 3.9), (7.0, 5.2), (8.0, NULL)
) t(dose, response);

-- Noisy but generally increasing dose-response curve
SELECT p.y, p.yhat, p.is_training
FROM (
    SELECT unnest(isotonic_fit_predict_agg(response, dose, {'increasing': true} ORDER BY dose)) AS p
    FROM isotonic_demo
);
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `increasing` | BOOLEAN | `true` | Fit a non-decreasing (`true`) or non-increasing (`false`) function |

Option keys the function does not support raise an error.

## How it works

PAVA starts from the raw `y` values ordered by `x`, scans for adjacent pairs
that violate monotonicity, replaces each violating block by its mean, and
repeats until the sequence is monotone.

## NULL handling

Rows where `x` is NULL are skipped. Rows with NULL `y` are prediction rows.

## Use Cases

- Dose-response curves
- Probability calibration (scores to probabilities)
- Monotone trend estimation when domain knowledge implies a direction

## See Also

- [Quantile](quantile.md) - Conditional quantiles
- [Fit-predict aggregates](fit_predict_agg.md)
