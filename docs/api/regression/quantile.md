# Quantile Regression

Quantile regression estimates a conditional quantile of `y` (for example the
median) instead of the conditional mean. Median regression is robust to
outliers in `y`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `quantile_fit_agg` | Aggregate | Fit a quantile regression per group <!-- TODO(lead): verify after *_fit_agg lands --> |
| `quantile_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `quantile_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#quantile_fit_predict_by) |

## quantile_fit_agg

<!-- TODO(lead): verify after *_fit_agg lands -->

**Signature:**

```text
quantile_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

Options: `tau`, `fit_intercept` (see [Options](#options)).

```sql skip
-- TODO(lead): un-skip once quantile_fit_agg is registered
SELECT
    (quantile_fit_agg(y, [x], {'tau': 0.1})).coefficients AS p10,
    (quantile_fit_agg(y, [x], {'tau': 0.5})).coefficients AS p50,
    (quantile_fit_agg(y, [x], {'tau': 0.9})).coefficients AS p90
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
CREATE OR REPLACE TABLE quantile_demo AS
SELECT
    i AS id,
    i::DOUBLE AS x,
    -- spread grows with x, plus two gross outliers
    2.0 + 0.5 * i + (i % 5 - 2) * 0.2 * i + CASE WHEN i IN (7, 19) THEN 50.0 ELSE 0.0 END AS y
FROM range(1, 41) t(i);

-- Upper (90th percentile) band per row
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
| `tau` | DOUBLE | `0.5` | Quantile to estimate, 0 < tau < 1 |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |

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
