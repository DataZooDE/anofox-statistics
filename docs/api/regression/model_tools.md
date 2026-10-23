# Model Tools: predict, tidy, glance

Every `*_fit_agg` aggregate and `*_fit` scalar returns the fitted model as a
`STRUCT`. These scalar functions work on that struct directly, so a model can
be fitted once per group and then applied, summarised or reported in plain SQL.

| Function | Returns | Description |
|----------|---------|-------------|
| [`predict(model, x [, options])`](#predict) | `DOUBLE` or `STRUCT` | Prediction for one row from a fitted model, optionally with standard error and interval |
| [`predict(x, coefficients, intercept)`](#linear_predict) | `DOUBLE[]` | Column-layout linear prediction (also `linear_predict`) |
| [`linear_predict(x, coefficients, intercept)`](#linear_predict) | `DOUBLE[]` | Same as the column-layout `predict` |
| [`tidy(model [, names])`](#tidy) | `STRUCT(...)[]` | One entry per term: estimate and inference |
| [`glance(model)`](#glance) | `STRUCT` | Model-level summary, the same fields for every model |
| [`<model>_tidy_by`, `<model>_glance_by`](#tidy_by-and-glance_by) | table | Per-group coefficient table and fit statistics |
| [`<model>_prediction_by`, `<model>_effects_by`](#prediction_by-and-effects_by) | table | Fitted values with bands, and marginal-effect grids, in the contract `prediction` shape |
| [`<model>_rolling_tidy_by`](#rolling_tidy_by) | table | Coefficients of a trailing-window fit, per window |
| [`ols_augment_by`, `wls_augment_by`, `<glm>_augment_by`](#augment_by) | table | Per-row fitted values, residuals, leverage, Cook's distance, DFFITS |
| [`<model>_partial_residuals_by`](#partial_residuals_by) | table | Partial-residual (crPlots) and added-variable (avPlots) coordinates |

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
predict(model STRUCT, x DOUBLE[], {'se_fit': true | 'interval': ... [, 'level': ...]})
    -> STRUCT(yhat DOUBLE, se_fit DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `STRUCT` | The result of any `*_fit_agg` or `*_fit` function |
| `x` | `DOUBLE[]` | The features of one row, in the order used for fitting |
| `options` | `MAP` / `STRUCT` | `type`, `se_fit`, `interval`, `level` (below) |

| Option | Default | Description |
|--------|---------|-------------|
| `type` | `'response'` | `'link'` returns a GLM prediction on the link scale |
| `se_fit` | `false` | Return the STRUCT with the standard error of the fitted mean |
| `interval` | `'none'` | `'confidence'` (mean response) or `'prediction'` (new observation, linear models only); returns the STRUCT |
| `level` | `0.95` | Level of the interval |

How the prediction is computed depends on the model:

- **Linear models** (OLS, Ridge, WLS, Elastic Net, RLS, Huber, RANSAC,
  Theil-Sen, BLS/NNLS, LARS, Bayesian Ridge, ARD, PLS, Quantile, ALM, ...):
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

### Standard errors and intervals

With `se_fit` or `interval` the result is
`STRUCT(yhat, se_fit, yhat_lower, yhat_upper)` (without `interval` the bounds
are NULL). Without these options `predict` returns a `DOUBLE` exactly as before.
They need the model's coefficient covariance, the `vcov` field that OLS, WLS,
Ridge and the GLM `*_fit_agg` aggregates carry when fitted with
`{'compute_inference': true}` (row-major `k × k`, intercept first, with
`df_residual = n − k`); a model without it raises an error. Bayesian Ridge and
ARD models predict point values with `predict(model, x)` but carry no `vcov`,
so `se_fit` / `interval` raise an error; their posterior predictive band comes
from `bayesian_ridge_` / `ard_fit_predict_agg`, `_fit_predict_by` and
`_prediction_by`. The values are
computed by anofox-regression (`intervals_from_variance_factor`,
`GlmResult::predict_se`):

- **OLS, WLS, Ridge**, as R `predict.lm(se.fit = TRUE, interval = ...)`:
  `se_fit = sqrt(x0' V x0)`, a confidence interval `yhat ∓ t(df) · se_fit` and a
  prediction interval `yhat ∓ t(df) · sqrt(se_fit² + σ²)` with
  `σ = residual_std_error` (for WLS a new observation of weight 1, R's
  default). For Ridge `V` is the sandwich covariance of the shrunken
  coefficients, as the fit-predict intervals use. `vcov` is the classical
  covariance even when `hc_type` asks for robust standard errors.
- **GLMs** (`poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`),
  as R `predict.glm(se.fit = TRUE)`: on the link scale `se_fit = sqrt(x0' V x0)`
  (`V` includes the estimated dispersion of Gamma and Tweedie); on the response
  scale (the default `type`) `se_fit` is the delta-method value. The interval
  is `η ∓ z · se` on the link scale, mapped through the inverse link for the
  response scale, so it stays inside the range of the mean. A prediction
  interval is an error for a GLM.

```sql
-- 90% prediction interval and the standard error of the mean at a new x
SELECT predict(m, [41.0, 3.0], {'se_fit': true, 'interval': 'prediction', 'level': 0.9}) AS p
FROM (SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS m FROM mt_demo);

-- Expected count with a 95% confidence interval (link-scale interval, back-transformed)
SELECT (p).yhat, (p).yhat_lower, (p).yhat_upper
FROM (SELECT predict(poisson_fit_agg(y_count, [x1], {'compute_inference': true}), [45.0], {'interval': 'confidence'}) AS p
      FROM mt_demo);
```

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
  example a fit without `compute_inference`, or PLS and LARS, which do not
  compute it (Quantile computes it with its `se` option).
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
    dispersion, iterations, converged, press, press_rmse, loo_r_squared)
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

**Model identifiers.** In every `<model>_*_by` macro that fits one of the
extension's models (`tidy_by`, `glance_by`, `augment_by`, `rolling_tidy_by`,
`partial_residuals_by`, the path, CV and quantile-process macros,
`prediction_by`, `effects_by`):

- `model_id` and `model_name` are the method, for example `'ols'`;
- `id` is the group value as text, and the group column itself is kept. (`augment_by`, which returns the source columns, has no `id`.)

So the outputs of several methods stack with `UNION ALL` and stay
distinguishable, as anofox-forecast's forecasts do. The test and curve macros
whose groups are themselves the models (`roc_curve_by`, `kaplan_meier_by`, the
post-hoc `_by` tests, ...) keep `model_id` = the group.

`<model>_tidy_by(source, group_col, y_col, x_cols [, weight_col] [, options := ..., names := [...]])`
and `<model>_glance_by(source, group_col, y_col, x_cols [, weight_col] [, options := ...])`
fit one model per group and return long tables (for `ols`, `wls`, `ridge`,
`elasticnet`, `huber`, `ransac`, `theil_sen`, `rls`, `lars`, `bayesian_ridge`,
`ard`, `passive_aggressive` and `lm_dynamic` (these two take an `order_col`
after `group_col`; `lm_dynamic_tidy_by` returns the coefficients of every row
with `index_name = 'window_end'`), `bls`, `nnls`,
`pls`, `quantile`, the GLMs `poisson`, `binomial`, `logistic`, `negbinom`,
`gamma`, `tweedie`, and `alm`, `aft`; `wls` takes `weight_col`, `aft`
`event_col`).

RLS is sequential too: `rls_tidy_by` and `rls_glance_by` take an optional
`order_by := <column>` and fit each group's rows in that order. Without it the
row order, and so the fit, is undefined (it can change between runs under
parallel scans).

The columns:

- `_tidy_by`: the group column, `id` (the group as text), `model_id` and `model_name` (the method, e.g. `'ols'`), and the `tidy` columns, one row per
  term. Inference is computed by default for models that have it, also when
  other `options` are passed.
- `_glance_by`: the group column, `id` (the group as text), `model_id` and `model_name` (the method, e.g. `'ols'`), `model_type`, `metric`, `value`,
  one row per available numeric metric. `ols_glance_by` and `wls_glance_by`
  include the leave-one-out metrics `press`, `press_rmse` and `loo_r_squared`
  (see [Leave-one-out diagnostics](#leave-one-out-diagnostics)).

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

```sql
-- RLS: feed each group's rows in id order
SELECT * FROM rls_tidy_by('mt_demo', grp, y, [x1, x2], options := {'forgetting_factor': 0.95}, order_by := id);
```
## prediction_by and effects_by

Both return the integration contract's `prediction` schema, the shape of
anofox-forecast's forecasts. One fan/band plot (`anofox_plot`) therefore draws
fitted values, effect grids and forecasts alike. The columns are:

- the group column and `id` (the group as text, for faceting);
- `model_id` and `model_name` (the method, for example `'ols'`, so several models can be overlaid);
- `x`, `y`, `yhat`, `yhat_lower` and `yhat_upper`;
- `level`, the interval level, NULL where the model has no band;
- `split`.

`<model>_prediction_by(source, group_col, x_col, y_col, x_cols [, weight_col] [, options := ..., level := 0.95, split := NULL])`
fits the model per group with `<model>_fit_predict_by` and returns one row per
input row:
- `x_col` is the plot position, for example a time or the main feature, and the row order for order-sensitive models.
- `split` is `'train'` for fitted rows, `'future'` for rows with a NULL `y`, and `'test'` for rows held out with `split :=`, as in `_fit_predict_by`.
- `level` sets the interval and overrides any `confidence_level` in `options`.

It exists for `ols`, `wls`, `ridge`, `elasticnet`, `huber`, `ransac`,
`theil_sen`, `rls`, `bls`, `bayesian_ridge`, `ard`, `passive_aggressive`,
`lm_dynamic`, `pls`, `quantile`, `alm`, the GLMs and `isotonic`. PLS, quantile, isotonic, Passive-Aggressive and lmDynamic have no band; for Bayesian
Ridge and ARD the band is the posterior predictive interval. For isotonic there is no `x_cols`:
`x_col` is the one feature.

`<model>_effects_by(source, group_col, y_col, x_cols [, weight_col] [, focal := 1, n_points := 50, at_values := NULL, options := ..., level := 0.95, names := [...]])`
is a marginal-effect (effect-plot) grid:
- the fitted mean over `n_points` values spanning the range of feature `focal`, its 1-based position in `x_cols`;
- the other features held at their means over the fitted rows, or at `at_values`, one value per feature;
- a confidence band from `predict(model, x, {'interval': 'confidence', 'level': level})`, on the response scale for GLMs;
- `y` and `split` are NULL, `focal` and `se_fit` are added;
- `focal_name` (VARCHAR, last column) names the focal feature: `names[focal]` when
  `names := [...]` is given, else `'x' || focal` (`x1`, `x2`, …), the same labels as
  the terms of `<model>_tidy_by`. A table macro cannot see the expressions in
  `x_cols`, so pass `names` to get the column names.

It exists for the models whose `predict` gives standard errors: `ols`, `wls`,
`ridge` and the GLMs.

```sql
-- Fitted line with a 90% band per group, ready for a band plot
SELECT * FROM ols_prediction_by('mt_demo', grp, id, y, [x1, x2], level := 0.9) LIMIT 5;
```

```sql
-- How the expected count changes with x1, other features at their means
SELECT focal_name, x, yhat, yhat_lower, yhat_upper
FROM poisson_effects_by('mt_demo', grp, y_count, [x1, x2], focal := 1, n_points := 5,
                        names := ['x1', 'x2'])
WHERE grp = 'A';
```

## rolling_tidy_by

`<model>_rolling_tidy_by(source, group_col, order_col, y_col, x_cols [, weight_col] [, window_size := 30, min_obs := NULL, options := ..., names := [...]])`
fits the model over a trailing window of `window_size` rows (ordered by
`order_col`) within each group and returns the coefficients of every window, for
`ols`, `wls`, `ridge`, `elasticnet`, `huber` and `poisson`. It is
`<model>_fit_agg(...) OVER (PARTITION BY group_col ORDER BY order_col ROWS BETWEEN window_size - 1 PRECEDING AND CURRENT ROW)`
in the integration contract's `terms` shape, for rolling-coefficient plots.

Columns: the group column, `id` (the group as text), `model_id` and `model_name` (the method, e.g. `'ols'`), `window_end` (the `order_col` value of
the window's last row, in its own type), the `tidy` columns, `index_name`
(`'window_end'`), `index_value` (the window's 1-based end position in its group)
and `n_window` (rows in the window). Windows with fewer than `min_obs` rows
(default `window_size`, so only full windows) are skipped. Inference is computed
by default where the model has it.

```sql
SELECT window_end, term, estimate, conf_low, conf_high
FROM ols_rolling_tidy_by('mt_demo', grp, id, y, [x1, x2], window_size := 12, names := ['trend', 'cycle'])
WHERE grp = 'A' AND term = 'trend'
ORDER BY window_end;
```

## augment_by

`ols_augment_by(source, group_col, y_col, x_cols [, options := ..., split := ..., order_by := ...])`
and `wls_augment_by(source, group_col, y_col, x_cols, weight_col [, ...])` are
broom's `augment`: they fit one model per group and return every source row
with the integration contract's `obs` columns appended: `model_id` and
`model_name` (the method; the group column is among the source columns), `row_id`
(source row number), `fitted`, `residual` (response residual), `resid_type`
(`'response'`), `std_residual` (R `rstandard`), `stud_residual` (R `rstudent`),
`leverage` (R `hatvalues`), `cooks_d` (R `cooks.distance`), `dffits`, `weight`
(the prior weight; 1 for OLS), `is_training`, `n_params` (coefficients
including the intercept), and the leave-one-out columns `loo_residual` and
`loo_fitted` (see [Leave-one-out diagnostics](#leave-one-out-diagnostics)).

Training rows follow the `_fit_predict_by` macros: `y_col` NOT NULL and `split`
NULL or `'train'` (WLS also needs a positive weight). Every other row gets the
model's prediction as `fitted` and NULL diagnostics. Options: `fit_intercept`,
`null_policy`, `solver`. The values match R `lm()` / `lm(weights = )`.

```sql
-- The five most influential observations per group
SELECT grp, id, fitted, residual, leverage, cooks_d
FROM ols_augment_by('mt_demo', grp, y, [x1, x2])
QUALIFY row_number() OVER (PARTITION BY grp ORDER BY cooks_d DESC) <= 5
ORDER BY grp, cooks_d DESC;
```

```sql
-- Hold out the last rows: they get a prediction but no diagnostics
SELECT id, fitted, residual, is_training
FROM wls_augment_by('mt_demo', grp, y, [x1, x2], x1,
                    split := CASE WHEN id > 36 THEN 'test' ELSE 'train' END)
WHERE grp = 'A' AND id > 30
ORDER BY id;
```

The window aggregates behind them, `ols_augment_agg(y, x [, options])` and
`wls_augment_agg(y, x, weight [, options])`, return one STRUCT with these fields
per row of a whole-partition frame, in frame order.

### Leave-one-out diagnostics

For OLS and WLS the exact leave-one-out quantities follow from the leverage
`h` without refitting (#220):

| Column / metric | Where | Definition | R |
|---|---|---|---|
| `loo_residual` | `ols_augment_by`, `wls_augment_by` | `residual / (1 - leverage)`: `y` minus the prediction of the fit without the row (the PRESS residual) | `residuals(fit) / (1 - hatvalues(fit))`; OLS: `rstandard(fit, type = "predictive")` |
| `loo_fitted` | `ols_augment_by`, `wls_augment_by` | `y - loo_residual`, the prediction for the row from the fit without it | |
| `press` | `ols_glance_by`, `wls_glance_by`; `ols_fit_agg`, `wls_fit_agg`, `ols_fit`, `wls_fit` with `compute_loo` (and `glance()` of them) | `sum(weight * loo_residual^2)` | `sum(rstandard(fit, type = "predictive")^2)` |
| `press_rmse` | as `press` | `sqrt(press / n)` | |
| `loo_r_squared` | as `press` | predicted R² (Q²) `1 - press / SST`, with the SST of `r_squared` (centred with an intercept, uncentred without) | |

For WLS `loo_residual` is on the response scale like `residual`, the leverage
is that of the weighted fit, and `press` weights the squares by the prior
weights (R's predictive residual is `sqrt(w)` times `loo_residual`).
`loo_r_squared` is never above `r_squared`; a large gap points to rows that
the fit leans on.

On the fit functions the three statistics cost one more QR of the design and
are opt-in: pass `{'compute_loo': true}` (independent of `compute_inference`);
otherwise the fields are NULL. `ols_glance_by` / `wls_glance_by` turn it on
(an explicit `compute_loo := false` in `options` wins). The `_augment_by`
columns are always computed, from the leverage they already have.

A row with leverage 1 (for example the only observation of a dummy level, or
any row of a fit with as many coefficients as rows) determines its own fitted
value and has no leave-one-out prediction: its `loo_residual` and `loo_fitted`
are NULL, and `press`, `press_rmse` and `loo_r_squared` of the group are NULL
(absent from `_glance_by`). Rows outside training (`split`, NULL `y`) have NULL
leave-one-out columns. The GLM `_augment_by` macros do not have these columns,
and the other models' `glance()` has them as NULL. Ridge is not covered yet.

```sql
-- How far each fitted value moves when its row is left out, and PRESS per group
SELECT grp, id, round(fitted, 3) AS fitted, round(loo_fitted, 3) AS loo_fitted,
       round(loo_residual, 3) AS loo_residual
FROM ols_augment_by('mt_demo', grp, y, [x1, x2])
QUALIFY row_number() OVER (PARTITION BY grp ORDER BY abs(loo_residual) DESC) <= 3
ORDER BY grp, abs(loo_residual) DESC;
```

```sql
SELECT grp, metric, round(value, 4) AS value
FROM ols_glance_by('mt_demo', grp, y, [x1, x2])
WHERE metric IN ('r_squared', 'press', 'press_rmse', 'loo_r_squared')
ORDER BY grp, metric;
```

The GLMs have the same macros: `poisson_augment_by`, `binomial_augment_by`,
`logistic_augment_by`, `negbinom_augment_by`, `gamma_augment_by` and
`tweedie_augment_by` (arguments as `ols_augment_by`). For them `fitted` is the
mean on the response scale, `residual` is the deviance residual with
`resid_type = 'deviance'` (as R and broom), `std_residual` is R
`rstandard(fit, type = "deviance")` and `weight` is 1. The dispersion is 1 for
Poisson, binomial, logistic and Negative Binomial (as `MASS::glm.nb`) and the
Pearson estimate for Gamma and Tweedie. Options: `fit_intercept`,
`max_iterations`, `tolerance`, `offset`, and `link` (Poisson, binomial),
`theta` (Negative Binomial) or `power` (Tweedie). A penalized fit
(`glm_lambda > 0` or a `prior`) is an error. The values match R `glm()`,
`MASS::glm.nb()` and `statmod::tweedie`.

```sql
-- Poisson counts: deviance residuals and influence, held-out rows predicted
SELECT id, y_count, fitted, residual, resid_type, cooks_d, is_training
FROM poisson_augment_by('mt_demo', grp, y_count, [x1],
                        split := CASE WHEN id > 36 THEN 'test' ELSE 'train' END)
WHERE grp = 'A' AND id > 30
ORDER BY id;
```

## partial_residuals_by

`<model>_partial_residuals_by(source, group_col, y_col, x_cols [, weight_col] [, options := ..., names := [...], order_by := ...])`
for `ols`, `wls` (with `weight_col`), `poisson`, `binomial`, `logistic`,
`negbinom`, `gamma` and `tweedie` fits one model per group and returns one row
per source row and fitted term with the data behind component-plus-residual
plots (`car::crPlots`, statsmodels `plot_ccpr`) and added-variable plots
(`car::avPlots`, statsmodels `plot_partregress`):

| Column | Description |
|--------|-------------|
| group column, `id`, `model_id`, `model_name`, `row_id` | the group, the group as text, the method, and the source row number |
| `term_index`, `term` | 1-based position in `x_cols`, and its label (`names` or `x1`, `x2`, ...) |
| `x` | the feature value |
| `component` | `b_j (x_j − mean(x_j))` (uncentred without an intercept) |
| `partial_residual` | residual + component, R `residuals(fit, type = "partial")`; for the GLMs the working residual |
| `x_residual`, `y_residual` | `x_j` and `y` residualised on the other columns (weighted for WLS); NULL for the GLMs |

They are computed by anofox-regression
(`diagnostics::{partial_residuals, partial_residuals_glm, added_variable}`).
The slope of `y_residual` on `x_residual` through the origin is the coefficient
of `x_j`. Rows with a NULL `y` (or a NULL feature, or for WLS a non-positive
weight) are not used for fitting and get NULL residuals; a column the fit drops
(constant) and a GLM offset column (option `offset`) are not terms. Options are
those of the matching `<model>_augment_by`. The window aggregates behind them,
`<model>_partial_residuals_agg(y, x [, weight] [, options])`, return per row of a
whole-partition frame a LIST with one STRUCT per term. Added-variable
coordinates for GLMs are not available upstream.

```sql
-- Component-plus-residual data for x1, per group
SELECT grp, row_id, x, partial_residual, component
FROM ols_partial_residuals_by('mt_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
WHERE term = 'trend'
ORDER BY grp, row_id
LIMIT 5;
```

## Fit per series, score new rows per series

A common task is to fit one model per series (`unique_id`) on history and then
score new rows, such as a forecast horizon, with each series' own model. The
two sides rarely hold the same series. A series can have history but no new
rows, or new rows but no history. There are two ways to do this.

The examples use three series with history (`A`, `B`, `C`) and new rows for
`A`, `B` and `D`:

```sql
CREATE OR REPLACE TABLE series_history AS
SELECT s.unique_id, i AS t, i::DOUBLE AS trend, (i % 3)::DOUBLE AS promo,
       s.base + 0.8 * i + 2.0 * (i % 3) + sin(i) AS y
FROM (VALUES ('A', 10.0), ('B', 20.0), ('C', 30.0)) s(unique_id, base), range(1, 21) r(i);

CREATE OR REPLACE TABLE series_new AS
SELECT s.unique_id, i AS t, i::DOUBLE AS trend, (i % 3)::DOUBLE AS promo
FROM (VALUES ('A'), ('B'), ('D')) s(unique_id), range(21, 25) r(i);
```

**1. One table, `y` NULL for the new rows, `<model>_fit_predict_by`.** Stack
history and new rows, with a NULL `y` for the new rows. The macro fits each
series on its rows with a `y` and predicts every row of that series. The new
rows come back with `is_training = false`. The same input works with
`<model>_prediction_by`, which labels these rows `split = 'future'` for plotting.

```sql
CREATE OR REPLACE VIEW series_all AS
SELECT unique_id, t, trend, promo, y FROM series_history
UNION ALL
SELECT unique_id, t, trend, promo, NULL::DOUBLE FROM series_new;

SELECT unique_id, t, round(yhat, 2) AS yhat,
       round(yhat_lower, 2) AS yhat_lower, round(yhat_upper, 2) AS yhat_upper
FROM ols_fit_predict_by('series_all', unique_id, y, [trend, promo], order_by := t)
WHERE is_training IS NOT TRUE
ORDER BY unique_id, t;
```

Each series is fitted on its own rows only, so:

- `C` (history, no new rows) has no rows with `is_training = false`;
- `D` (new rows, no history) has no training rows, so its fit fails. Its rows are
  returned with `yhat`, `yhat_lower`, `yhat_upper` and `is_training` all NULL, which
  is why the filter is `is_training IS NOT TRUE` and not `NOT is_training`.

**2. A fit table, joined to the new rows, `predict(model, x)`.** Fit once per
series with `GROUP BY` and keep the models in a table. Then join the new rows
to it on the series column and call the model-aware
[`predict`](#predict) on each row. This also scores rows that arrive later
without refitting.

```sql
CREATE OR REPLACE TABLE series_models AS
SELECT unique_id, ols_fit_agg(y, [trend, promo], {'compute_inference': true}) AS model
FROM series_history
GROUP BY unique_id;

-- INNER JOIN: only series with both a model and new rows (A, B)
SELECT n.unique_id, n.t, round(predict(m.model, [n.trend, n.promo]), 2) AS yhat
FROM series_new n
JOIN series_models m USING (unique_id)
ORDER BY n.unique_id, n.t;
```

The join decides what happens to series that are missing on one side:

| Join | Series with new rows but no model (`D`) | Series with a model but no new rows (`C`) |
|------|------------------------------------------|-------------------------------------------|
| `series_new JOIN series_models` | dropped | dropped |
| `series_new LEFT JOIN series_models` | kept, `yhat` is NULL | dropped |
| `series_new FULL JOIN series_models` | kept, `yhat` is NULL | one row with NULL features and a NULL `yhat` |

A `LEFT JOIN` from the new rows keeps every row that has to be scored and makes
a missing model visible as a NULL `yhat`, because `predict` of a NULL model is
NULL. For an interval (the model needs `compute_inference`, as in the fit table
above), call `predict` once in a subquery and read the fields from the returned
struct:

```sql
SELECT unique_id, t, round(p.yhat, 2) AS yhat,
       round(p.yhat_lower, 2) AS yhat_lower, round(p.yhat_upper, 2) AS yhat_upper
FROM (
    SELECT n.unique_id, n.t,
           predict(m.model, [n.trend, n.promo], {'interval': 'prediction'}) AS p
    FROM series_new n
    LEFT JOIN series_models m USING (unique_id)
)
ORDER BY unique_id, t;
```

Both ways give the same predictions for `A` and `B`. Use way 1 when history and
new rows are scored in one pass. Use way 2 to keep the models, to inspect them
with [`tidy`](#tidy) / [`glance`](#glance), or to score new rows later.

## See Also

- [Window fit-predict](fit_predict_window.md) - predictions per window frame
- [Fit-predict aggregates](fit_predict_agg.md) - one prediction per row of a group
- [Table macros](../macros/table_macros.md) - per-group fit and predict in long format
- [Regularisation paths](paths.md) - coefficient paths and CV curves (`terms` / `curve`)
