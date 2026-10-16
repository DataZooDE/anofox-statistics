# Passive-Aggressive, lmDynamic and LOWESS

Three estimators that do not fit the batch least-squares pattern:

- **Passive-Aggressive** (scikit-learn `PassiveAggressiveRegressor`) is an
  online regressor. It updates the coefficients row by row and leaves them
  alone while a row's error stays inside `epsilon` (passive). Otherwise it
  makes the smallest change that removes the error, capped by `c`
  (aggressive).
- **lmDynamic** (greybox `lmDynamic`) gives time-varying coefficients. It
  fits every subset of the regressors, weights the subsets per observation by
  their pointwise information criterion (optionally smoothed over time with
  LOWESS), and uses the weighted coefficients of each row.
- **LOWESS** (R `stats::lowess`) is Cleveland's robust locally weighted
  scatterplot smoother.

Passive-Aggressive and lmDynamic are **sequential**: the result depends on the
row order. Give it with `ORDER BY` inside the aggregate, `order_by :=` in
`_fit_predict_by`, and the `order_col` argument of `_tidy_by` / `_glance_by`.
The results agree with scikit-learn 1.9.1, greybox 2.0.9 and R's `lowess` to
1e-8 or better (`test/sql/reference/regression_passive_aggressive.test`,
`lm_dynamic_reference.test`, `lowess_reference.test`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `passive_aggressive_fit_agg(y, x[, options] ORDER BY t)` | Aggregate | Online fit over the ordered rows |
| `passive_aggressive_fit_predict(y, x[, options]) OVER (...)` | Window | Prediction for the last row of the frame |
| `passive_aggressive_fit_predict_agg(y, x[, split][, options] ORDER BY t)` | Aggregate | Per-row predictions from the final coefficients |
| `passive_aggressive_fit_predict_by`, `_prediction_by` | Table macro | Per group, ordered by `order_by` / `x_col` |
| `passive_aggressive_tidy_by`, `_glance_by` | Table macro | `(source, group_col, order_col, y_col, x_cols, ...)` |
| `lm_dynamic_fit_agg(y, x[, options] ORDER BY t)` | Aggregate | Time-averaged and per-row coefficients |
| `lm_dynamic_fit_predict_agg(y, x[, split][, options] ORDER BY t)` | Aggregate | Per-row fits with the row's coefficients |
| `lm_dynamic_fit_predict_by`, `_prediction_by` | Table macro | Per group, ordered by `order_by` / `x_col` |
| `lm_dynamic_tidy_by`, `_glance_by` | Table macro | Per-row coefficients as `terms` (`index_name = 'window_end'`), fit statistics |
| `lowess_agg(y, x[, frac[, iterations]])` | Aggregate | The smoothed curve as `LIST(STRUCT(x, yhat))` |
| `lowess_by(source, group_col, x_col, y_col[, frac := 2/3, iterations := 3])` | Table macro | The curve per group (`curve`, `curve_type = 'lowess'`) |

## Passive-Aggressive

```sql
CREATE OR REPLACE TABLE stream AS
SELECT
    i AS t,
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    sin(i * 0.37) AS x1,
    cos(i * 0.11) AS x2,
    1.0 + 2.0 * sin(i * 0.37) - cos(i * 0.11) + 0.2 * sin(i * 1.7) AS y
FROM range(1, 201) t(i);

-- One online pass over the rows in time order (scikit-learn partial_fit)
SELECT list_transform(m.coefficients, lambda c: round(c, 3)) AS coefficients, round(m.intercept, 3) AS intercept
FROM (SELECT passive_aggressive_fit_agg(y, [x1, x2] ORDER BY t) AS m FROM stream);

-- The prediction at every time from the rows up to it
SELECT t, round((passive_aggressive_fit_predict(y, [x1, x2], {'c': 0.5})
                 OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 3) AS yhat
FROM stream
ORDER BY t
LIMIT 3;
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `c` (also `C`) | DOUBLE | `1.0` | Aggressiveness: the largest step (PA-I) or the regularisation of PA-II; > 0 |
| `epsilon` | DOUBLE | `0.1` | Width of the epsilon-insensitive zone; >= 0 |
| `variant` | VARCHAR | `'pa1'` | `'pa1'` (epsilon-insensitive loss) or `'pa2'` (squared, smoother updates) |
| `fit_intercept` | BOOLEAN | `true` | Learn an intercept |
| `max_iterations` (alias `max_iter`) | INTEGER | `1` | Passes over the rows; 1 is the online update, more passes use scikit-learn's early stopping |
| `tolerance` (alias `tol`) | DOUBLE | `0.001` | Early-stopping tolerance of multi-pass fits |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict_agg` / `_by` only: `'drop'`, `'drop_y_zero_x'`, `'zero'` or `'ignore'`, see [null_policy](fit_predict_agg.md#null_policy) |

The model STRUCT has `coefficients`, `intercept`, `r_squared`,
`n_observations`, `n_features` and `iterations` (passes run), then the
inference fields and the shared tail, which are all NULL because the model
has neither inference nor a likelihood; `model_type` is `'passive_aggressive'`.
The model has no prediction interval, so `yhat_lower` and `yhat_upper` are
NULL in the fit-predict forms. The window form, like `rls_fit_predict`, runs
the update over the frame's rows (including the last row when its `y` is not
NULL) and predicts the frame's last row.

## lmDynamic

```sql
-- x1 matters only in the second half: the per-row x1 coefficient rises
SELECT
    round(list_avg(list_transform(m.dynamic_coefficients[1:80], lambda c: c[1])), 3) AS x1_early,
    round(list_avg(list_transform(m.dynamic_coefficients[121:200], lambda c: c[1])), 3) AS x1_late
FROM (
    SELECT lm_dynamic_fit_agg(1 + CASE WHEN t > 100 THEN 2 ELSE 0 END * x1 + 0.5 * x2 + 0.3 * sin(t * 2.9),
                              [x1, x2], {'span': 0.3} ORDER BY t) AS m
    FROM stream
);

-- Per-row coefficients as `terms` rows, ready for a rolling-coefficient plot
SELECT grp, window_end, term, round(estimate, 3) AS estimate
FROM lm_dynamic_tidy_by('stream', grp, t, y, [x1, x2], names := ['x1', 'x2'])
ORDER BY grp, window_end, term
LIMIT 6;
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ic` (alias `information_criterion`) | VARCHAR | `'aicc'` | Pointwise criterion that weights the subsets: `'aic'`, `'aicc'`, `'bic'` |
| `span` (aliases `lowess_span`, `f`) | DOUBLE | `0.3` | LOWESS span smoothing the weights over time (greybox's `f`); `0` = no smoothing |
| `fit_intercept` | BOOLEAN | `true` | Include an intercept in every candidate |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict_agg` / `_by` only: `'drop'`, `'drop_y_zero_x'`, `'zero'` or `'ignore'`, see [null_policy](fit_predict_agg.md#null_policy) |

All `2^p` subsets of the regressors are fitted (greybox's `bruteforce`), so at
most 14 regressors are accepted. The normal distribution is used (greybox's
default). greybox picks `f` by optimisation when it is not given; here the
span is always explicit.

`lm_dynamic_fit_agg` returns the time-averaged `coefficients` and `intercept`
(greybox's `coef()`), `r_squared`, `adj_r_squared`, `residual_std_error`,
`n_observations`, `n_features`, `n_models` and, for every row aggregated in
order, `dynamic_intercept`, `dynamic_coefficients` (one list per row) and
`fitted_values` (`x_t' beta_t`). `log_likelihood` is greybox's combined
log-likelihood. Inference fields, `aic` and `bic` are NULL.
`lm_dynamic_fit_predict_agg` returns per row `y`, `yhat`, `is_training`,
`intercept` and `coefficients`: training rows get their own time-varying fit,
and the other rows (NULL `y`, held out by `split`) are scored with the
time-averaged coefficients, as greybox's `predict()` scores new data with an
`lmDynamic` model's `coefficients` (checked with greybox 2.0.9). `yhat_lower` and `yhat_upper` are NULL.

## LOWESS

```sql
CREATE OR REPLACE TABLE scatter AS
SELECT ((i * 37) % 200) / 20.0 AS x, sin(((i * 37) % 200) / 20.0) + 0.4 * sin(i * 12.9898) AS y
FROM range(1, 201) t(i);

SELECT round(p.x, 2) AS x, round(p.yhat, 3) AS yhat
FROM (SELECT unnest(lowess_agg(y, x, 0.2)) AS p FROM scatter)
LIMIT 3;

SELECT model_id, curve_type, round(x, 2) AS x, round(y, 3) AS y
FROM lowess_by('scatter', 1, x, y, frac := 0.2)
LIMIT 3;
```

`frac` (R's `f`, default 2/3) is the share of points that influence each
fitted value; `iterations` (R's `iter`, default 3) is the number of
robustifying passes, which downweight outliers. `delta` stays at R's default,
1% of the range of `x`. The curve has one point per complete row, with `x`
increasing (ties in input order), as `lowess()` returns it. Rows with a NULL
or non-finite `x` or `y` are skipped.

`lowess_by` returns the integration contract's `curve` schema: the group
column, `id` (the group), `model_id` and `model_name` (`'lowess'`),
`curve_type` (`'lowess'`), `series`, `x`, `y` (the smoothed value), `y_low`,
`y_high`, `threshold`, `weight` and `n_censor`. The last five are NULL.

## See Also

- [RLS](rls.md): recursive least squares, the other online estimator
- [Model tools](model_tools.md): `rolling_tidy_by` for coefficients over trailing windows
