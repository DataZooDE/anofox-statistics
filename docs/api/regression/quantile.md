# Quantile Regression

Quantile regression estimates a conditional quantile of `y` (for example the
median) instead of the conditional mean. Median regression is robust to
outliers in `y`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `quantile_fit_agg` | Aggregate | Fit a quantile regression per group |
| `quantile_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `quantile_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#quantile_fit_predict_by) |

## quantile_fit_agg

**Signature:**

```text
quantile_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

Options: `tau` (alias `quantile`), `fit_intercept`, `max_iterations`,
`tolerance` (see [Options](#options)).

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Slope coefficients, one per feature |
| `intercept` | `DOUBLE` | Intercept (0 when `fit_intercept` is false) |
| `tau` | `DOUBLE` | The quantile that was estimated |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

Rows with a NULL `y`, a NULL `x` or a NULL list element are skipped. A group
with too few usable rows returns NULL. Apply the model to new rows with
[`predict(model, x)`](model_tools.md#predict).

**Example:**

```sql
CREATE OR REPLACE TABLE quantile_demo AS
SELECT
    i AS id,
    i::DOUBLE AS x,
    -- spread grows with x, plus two gross outliers
    2.0 + 0.5 * i + (i % 5 - 2) * 0.2 * i + CASE WHEN i IN (7, 19) THEN 50.0 ELSE 0.0 END AS y
FROM range(1, 41) t(i);

-- Slopes of the 10th, 50th and 90th percentile lines
SELECT
    (quantile_fit_agg(y, [x], {'tau': 0.1})).coefficients AS p10,
    (quantile_fit_agg(y, [x], {'tau': 0.5})).coefficients AS p50,
    (quantile_fit_agg(y, [x], {'tau': 0.9})).coefficients AS p90
FROM quantile_demo;

-- Median prediction for a new x
SELECT round(predict(quantile_fit_agg(y, [x], {'tau': 0.5}), [45.0]), 3) AS median_at_45
FROM quantile_demo;
```

## quantile_fit_predict_agg

**Signature:**

```text
quantile_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
quantile_fit_predict_agg(y DOUBLE, x DOUBLE[], split_col VARCHAR [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
```

Rows with a NULL `y` (or, with the split form, rows whose `split_col` is not
`'train'`/`'training'`) are not used for fitting but still receive a
prediction. Quantile regression returns point predictions only; there are no
`yhat_lower` / `yhat_upper` fields.

**Example:**

```sql
-- Upper (90th percentile) band per row (quantile_demo is created above)
SELECT p.y, round(p.yhat, 2) AS p90
FROM (
    SELECT unnest(quantile_fit_predict_agg(y, [x], {'tau': 0.9})) AS p
    FROM quantile_demo
)
LIMIT 5;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tau` (alias `quantile`) | DOUBLE | `0.5` | Quantile to estimate, 0 < tau < 1 |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `max_iterations` | INTEGER | `1000` | Maximum solver iterations |
| `tolerance` | DOUBLE | `1e-6` | Convergence tolerance |

Option keys the function does not support raise an error.

## Common tau values

| tau | Meaning |
|-----|---------|
| 0.10 | 10th percentile (lower tail) |
| 0.25 | First quartile |
| 0.50 | Median (robust central tendency) |
| 0.75 | Third quartile |
| 0.90 | 90th percentile (upper tail) |

## NULL handling

Rows where `x` is NULL are skipped. Rows with NULL `y` are prediction rows.

## Use Cases

- Robust regression (median)
- Prediction bands and service-level planning (upper quantiles)
- Heteroscedastic data where effects differ across the distribution

## See Also

- [OLS](ols.md) - Mean regression
- [ALM](../glm/alm.md) - Asymmetric Laplace distribution for quantile models
- [Isotonic](isotonic.md) - Monotone regression
