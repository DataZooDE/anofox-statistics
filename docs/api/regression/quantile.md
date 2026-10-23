# Quantile Regression

Quantile regression estimates a conditional quantile of `y` (for example the
median) instead of the conditional mean. Median regression is robust to
outliers in `y`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `quantile_fit_agg` | Aggregate | Fit a quantile regression per group, optionally with standard errors |
| `quantile_process_agg` | Aggregate | The regression-quantile process: one fit per tau with inference |
| `quantile_process_by`, `quantile_tidy_by` | Table macro | The process / the coefficient table per group (`terms` schema) |
| `quantile_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `quantile_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#quantile_fit_predict_by) |

## quantile_fit_agg

**Signature:**

```text
quantile_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

Options: `tau` (alias `quantile`), `fit_intercept`, `max_iterations`,
`tolerance`, and for inference `se`, `compute_inference`, `confidence_level`,
`n_boot`, `seed` (see [Options](#options)).

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Slope coefficients, one per feature |
| `intercept` | `DOUBLE` | Intercept (0 when `fit_intercept` is false) |
| `tau` | `DOUBLE` | The quantile that was estimated |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |
| `std_errors`, `t_values`, `p_values` | `DOUBLE[]` | Slope inference (NULL without inference) |
| `conf_low`, `conf_high`, `conf_level` | `DOUBLE[]`, `DOUBLE` | Slope confidence limits and their level |
| `intercept_std_error`, `intercept_statistic`, `intercept_p_value`, `intercept_conf_low`, `intercept_conf_high` | `DOUBLE` | Intercept inference |

Plus the shared model fields (`model_type`, `log_likelihood`, ...; see
[Model tools](model_tools.md)).

### Inference

With `{'se': 'iid' | 'nid' | 'boot'}` (or `{'compute_inference': true}`, which
uses `'nid'`) the fit carries standard errors as
`quantreg::summary.rq(se = ...)`, computed by anofox-regression's
`solvers::quantile_inference`:

- `'iid'`: `τ(1 − τ) s² (X'X)⁻¹` with the sparsity `s` estimated from the
  ordered residuals (Hall-Sheather bandwidth).
- `'nid'`: the Hendricks-Koenker sandwich, locally iid errors.
- `'boot'`: `n_boot` (default 200) `(x, y)`-pair bootstrap resamples drawn with
  the seed `seed` (default 42). The generator differs from R's, so the values
  differ from R's bootstrap for the same seed; rows are put in a canonical order
  first, so the result does not depend on the thread count.

The statistic is `estimate / std_error` with a two-sided Student-t p-value on
`n − k` degrees of freedom (`k` counts the intercept), and the limits are
`estimate ∓ t · std_error` at `confidence_level` (default 0.95). quantreg's
rank-inversion intervals (`se = "rank"`, its default for small n) are not
available. `quantile_tidy_by` computes `'nid'` inference by default.

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

```sql
-- Median regression with Hendricks-Koenker standard errors
SELECT t.term, round(t.estimate, 3) AS estimate, round(t.std_error, 3) AS std_error, round(t.p_value, 4) AS p_value
FROM (SELECT unnest(tidy(quantile_fit_agg(y, [x], {'tau': 0.5, 'se': 'nid'}))) AS t FROM quantile_demo);
```

## quantile_process_agg

**Signature:**

```text
quantile_process_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT(...)[]
```

The regression-quantile process, the table behind
`plot(summary(rq(y ~ x, tau = taus)))`: one quantile regression per tau with
its coefficient table, from `solvers::quantile_inference::quantile_process`.
Each element is a `terms` row (`term`, `estimate`, `std_error`, `statistic`,
`p_value`, `conf_low`, `conf_high`, `conf_level`) with `index_name = 'tau'` and
`index_value = tau`. Options: `taus` (default `[0.1, 0.2, ..., 0.9]`), `se`
(`'nid'` default, `'iid'`, `'boot'`), `compute_inference` (false: estimates
only), `confidence_level`, `n_boot`, `seed`, `fit_intercept`, `feature_names`.

`quantile_process_by(source, group_col, y_col, x_cols [, options := ..., names := [...]])`
returns it per group with `id` (the group) and `model_id` = `'quantile'`, in the columns of `<model>_tidy_by`
(an integration-contract `terms` producer).

```sql
-- How the slope changes across the conditional distribution
SELECT index_value AS tau, round(estimate, 3) AS slope, round(conf_low, 3) AS conf_low, round(conf_high, 3) AS conf_high
FROM quantile_process_by('quantile_demo', 1, y, [x], options := {'taus': [0.1, 0.25, 0.5, 0.75, 0.9]})
WHERE term = 'x'
ORDER BY tau;
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
| `se` (alias `se_type`) | VARCHAR | - | `'iid'`, `'nid'` or `'boot'`; turns inference on |
| `compute_inference` | BOOLEAN | `false` | Inference with `'nid'` standard errors |
| `confidence_level` | DOUBLE | `0.95` | Level of `conf_low` / `conf_high` |
| `n_boot` | INTEGER | `200` | Bootstrap resamples (`se = 'boot'`) |
| `seed` (alias `random_state`) | BIGINT | `42` | Bootstrap seed |

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

Rows where `x` is NULL, or has a NULL element, are not used for fitting. The fit-predict
aggregate keeps them in its output with a NULL `yhat`. Rows with NULL `y` are
prediction rows.

## Use Cases

- Robust regression (median)
- Prediction bands and service-level planning (upper quantiles)
- Heteroscedastic data where effects differ across the distribution

## See Also

- [OLS](ols.md) - Mean regression
- [Regularisation paths](paths.md) - other `terms` producers with an index
- [ALM](../glm/alm.md) - Asymmetric Laplace distribution for quantile models
- [Isotonic](isotonic.md) - Monotone regression
