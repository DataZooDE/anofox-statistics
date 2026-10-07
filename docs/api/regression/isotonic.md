# Isotonic Regression

Isotonic regression fits a monotone (non-decreasing or non-increasing)
step function of a single predictor, using the Pool Adjacent Violators
Algorithm (PAVA).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `isotonic_fit_agg` | Aggregate | Fit an isotonic regression per group |
| `isotonic_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `isotonic_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#isotonic_fit_predict_by) |

Unlike the other regressions, `x` is a single `DOUBLE`, not a list.

## isotonic_fit_agg

**Signature:**

```text
isotonic_fit_agg(y DOUBLE, x DOUBLE [, options MAP]) -> STRUCT
```

Options: `increasing` (see [Options](#options)).

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `x` | `DOUBLE[]` | Sorted distinct `x` values (the knots of the fitted function) |
| `fitted` | `DOUBLE[]` | Fitted value at each knot |
| `increasing` | `BOOLEAN` | Direction of the fit |
| `r_squared` | `DOUBLE` | In-sample R² |
| `n_observations` | `BIGINT` | Rows used in the fit |

The fitted model is a monotone function given by its knots. Evaluate it at new
points with [`predict(model, [x])`](model_tools.md#predict), which interpolates
linearly between neighbouring knots and clamps to the first/last fitted value
outside the range of the training `x`. Rows with a NULL or non-finite `y` or
`x` are skipped.

**Example:**

```sql
CREATE OR REPLACE TABLE isotonic_demo AS
SELECT * FROM (VALUES
    (1.0, 1.5), (2.0, 2.0), (3.0, 1.8), (4.0, 3.5),
    (5.0, 4.0), (6.0, 3.9), (7.0, 5.2), (8.0, NULL)
) t(dose, response);

SELECT
    m.x AS knots,
    m.fitted,
    predict(m, [3.5]) AS at_3_5,     -- interpolated between knots 3 and 4
    predict(m, [10.0]) AS at_10      -- clamped to the last fitted value
FROM (SELECT isotonic_fit_agg(response, dose, {'increasing': true}) AS m FROM isotonic_demo);
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
