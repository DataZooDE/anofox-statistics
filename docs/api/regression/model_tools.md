# Model Tools: predict, tidy, glance

Every `*_fit_agg` aggregate and `*_fit` scalar returns the fitted model as a
`STRUCT`. These scalar functions work on that struct directly, so a model can
be fitted once per group and then applied, summarised or reported in plain SQL.

| Function | Returns | Description |
|----------|---------|-------------|
| [`predict(model, x [, options])`](#predict) | `DOUBLE` | Prediction for one row from a fitted model |
| [`predict(x, coefficients, intercept)`](#linear_predict) | `DOUBLE[]` | Column-layout linear prediction (also `linear_predict`) |
| [`linear_predict(x, coefficients, intercept)`](#linear_predict) | `DOUBLE[]` | Same as the column-layout `predict` |
| [`tidy(model [, names])`](#tidy) | `STRUCT(...)[]` | One entry per term: estimate and inference |
| [`glance(model)`](#glance) | `STRUCT` | Model-level summary, the same fields for every model |
| [`<model>_tidy_by`, `<model>_glance_by`](#tidy_by-and-glance_by) | table | Per-group coefficient table and fit statistics |

The examples on this page use this table:

```sql
CREATE OR REPLACE TABLE mt_demo AS
SELECT
    i AS id,
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS grp,
    i::DOUBLE AS x1,
    ((i * 3) % 7)::DOUBLE AS x2,
    1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) AS y,
    round(exp(0.2 + 0.05 * i))::DOUBLE AS y_count
FROM range(1, 41) t(i);
```

## predict

**Signature:**

```text
predict(model STRUCT, x DOUBLE[] [, options MAP]) -> DOUBLE
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `STRUCT` | The result of any `*_fit_agg` or `*_fit` function |
| `x` | `DOUBLE[]` | The features of one row, in the order used for fitting |
| `options` | `MAP` / `STRUCT` | `{'type': 'response'}` (default) or `{'type': 'link'}` |

How the prediction is computed depends on the model:

- **Linear models** (OLS, Ridge, WLS, Elastic Net, RLS, Huber, RANSAC,
  Theil-Sen, BLS/NNLS, LARS, PLS, Quantile, ALM, ...):
  `intercept + Σ coefficients[j] · x[j]`. A `NaN` coefficient (an aliased or
  constant column) contributes nothing.
- **GLMs** (`poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`):
  the linear predictor `η` is mapped through the inverse of the model's `link`
  field, so the default prediction is on the response scale (a rate, a
  probability, a mean). With `{'type': 'link'}` the linear predictor `η` itself
  is returned.
- **Isotonic models** (`isotonic_fit_agg`): `x` holds a single value. The
  prediction interpolates linearly between the neighbouring knots and is
  clamped to the first/last fitted value outside the training range.

NULL handling: a NULL model (for example a group whose fit was degenerate), a
NULL `x`, or a NULL element in `x` gives NULL. An `x` whose length does not
match the number of coefficients raises an error.

**Examples:**

```sql
-- Fit per group, then score every row with its group's model
WITH models AS (
    SELECT grp, ols_fit_agg(y, [x1, x2]) AS m
    FROM mt_demo
    GROUP BY grp
)
SELECT d.id, d.grp, round(d.y, 3) AS y, round(predict(m.m, [d.x1, d.x2]), 3) AS yhat
FROM mt_demo d
JOIN models m USING (grp)
ORDER BY d.id
LIMIT 4;
```

```sql
-- GLM: response scale (expected count) by default, linear predictor on request
WITH fit AS (SELECT poisson_fit_agg(y_count, [x1]) AS m FROM mt_demo)
SELECT
    m.link,
    round(predict(m, [45.0]), 3) AS expected_count,
    round(predict(m, [45.0], {'type': 'link'}), 3) AS log_rate
FROM fit;
```

```sql
-- One-step-ahead forecasts: fit on strictly earlier rows, predict the current row
SELECT
    id, round(y, 3) AS y,
    round(predict((ols_fit_agg(y, [x1, x2]) OVER (
        ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )), [x1, x2]), 3) AS yhat_next
FROM mt_demo
ORDER BY id
LIMIT 6;
```

The first rows of the one-step-ahead query are NULL: the frame does not yet
hold enough rows to fit, so the model is NULL.

## linear_predict

**Signature:**

```text
predict(x DOUBLE[][], coefficients DOUBLE[], intercept DOUBLE) -> DOUBLE[]
linear_predict(x DOUBLE[][], coefficients DOUBLE[], intercept DOUBLE) -> DOUBLE[]
```

The column-layout form takes the features as a list of columns (each inner
list is one feature, like the `*_fit` scalars) and returns one prediction per
position. `linear_predict` is the same function under a name that cannot be
confused with the model-aware `predict`. NULL inputs give NULL.

```sql
WITH model AS (
    SELECT ols_fit([3.0, 5.0, 7.0, 9.0, 11.0], [[1.0, 2.0, 3.0, 4.0, 5.0]]) AS fit
)
SELECT
    linear_predict([[6.0, 7.0, 8.0]], fit.coefficients, fit.intercept) AS predictions,
    predict(fit, [6.0]) AS first_prediction
FROM model;
```

## tidy

**Signature:**

```text
tidy(model STRUCT [, names VARCHAR[]])
    -> STRUCT(term VARCHAR, estimate DOUBLE, std_error DOUBLE, statistic DOUBLE,
              p_value DOUBLE, conf_low DOUBLE, conf_high DOUBLE, conf_level DOUBLE,
              index_name VARCHAR, index_value DOUBLE)[]
```

One entry per term, in the style of R's `broom::tidy`:

- The intercept row comes first, with term `'(Intercept)'`.
- Slope terms are named `x1 .. xk` unless `names` (one name per feature) is
  given.
- `std_error`, `statistic` (t or z), `p_value`, `conf_low` and `conf_high`
  come from the model's inference fields (`std_errors`, `t_values` /
  `z_values`, `p_values`, `conf_low`, `conf_high`; for the intercept its
  `intercept_*` fields). They are NULL when the model has no inference, for
  example a fit without `compute_inference`, or PLS, Quantile and LARS, which
  do not compute it.
- `conf_level` is the confidence level of the interval; `index_name` and
  `index_value` identify a point on a coefficient path or process (lambda,
  tau, ...) and are NULL for a single fit.

Use `unnest(..., recursive := true)` to turn the list into rows and columns:

```sql
SELECT unnest(tidy(ols_fit_agg(y, [x1, x2], {'compute_inference': true}),
                   ['trend', 'cycle']), recursive := true)
FROM mt_demo;
```

Per group, `GROUP BY` the fit and unnest the result:

```sql
SELECT grp, unnest(tidy(ols_fit_agg(y, [x1, x2], {'compute_inference': true}),
                        ['trend', 'cycle']), recursive := true)
FROM mt_demo
GROUP BY grp
ORDER BY grp;
```

`tidy` needs a model with a `coefficients` field; isotonic models are not
supported.

## glance

**Signature:**

```text
glance(model STRUCT) -> STRUCT(model_type, family, link, n_observations, n_features,
    r_squared, adj_r_squared, residual_std_error, f_statistic, f_pvalue,
    log_likelihood, aic, bic, deviance, null_deviance, pseudo_r_squared,
    dispersion, iterations, converged)
```

Returns the same fields for every model, in the style of R's `broom::glance`;
fields a model does not have are NULL. Model-specific values (Huber's `scale`,
RANSAC's `n_inliers`, ...) stay on the model struct. Expand it into columns
with `unnest`, or pick a field with dot notation:

```sql
SELECT unnest(glance(ols_fit_agg(y, [x1, x2]))) FROM mt_demo;
```

```sql
-- One row of fit statistics per group
SELECT grp,
       round(glance(ols_fit_agg(y, [x1, x2])).r_squared, 4) AS r_squared,
       glance(ols_fit_agg(y, [x1, x2])).n_observations AS n
FROM mt_demo
GROUP BY grp
ORDER BY grp;
```

## tidy_by and glance_by

`<model>_tidy_by(source, group_col, y_col, x_cols [, weight_col] [, options := ..., names := [...]])`
and `<model>_glance_by(source, group_col, y_col, x_cols [, weight_col] [, options := ...])`
fit one model per group and return long tables (for `ols`, `wls`, `ridge`,
`elasticnet`, `huber`, `ransac`, `theil_sen`, `rls`, `lars`; `wls` takes
`weight_col`):

- `_tidy_by`: the group column, `model_id`, and the `tidy` columns, one row per
  term. Inference is computed by default for models that have it.
- `_glance_by`: the group column, `model_id`, `model_type`, `metric`, `value`,
  one row per available numeric metric.

```sql
SELECT * FROM ols_tidy_by('mt_demo', grp, y, [x1, x2], names := ['trend', 'cycle']);
```

```sql
-- Compare two models per group
SELECT * FROM ols_glance_by('mt_demo', grp, y, [x1, x2])
UNION ALL
SELECT * FROM huber_glance_by('mt_demo', grp, y, [x1, x2])
ORDER BY grp, metric, model_type;
```
## See Also

- [Window fit-predict](fit_predict_window.md) - predictions per window frame
- [Fit-predict aggregates](fit_predict_agg.md) - one prediction per row of a group
- [Table macros](../macros/table_macros.md) - per-group fit and predict in long format
