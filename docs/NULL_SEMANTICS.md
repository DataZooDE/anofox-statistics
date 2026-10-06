# NULL and NaN Semantics

This page explains what the extension does with missing values: SQL `NULL`,
`NULL` elements inside a feature list, and IEEE `NaN`/`Inf` values. It also covers
the cases where the extension itself returns NULL or NaN. Every example is a
self-contained query that runs as written.

## Summary

| Situation | Fit aggregates (`*_fit_agg`) | Window `*_fit_predict` | `*_fit_predict_agg` / `*_fit_predict_by` |
|-----------|------------------------------|------------------------|-------------------------------------------|
| `y` is NULL | Row skipped | Not used for training; the row still gets a prediction | Not used for training; the row gets a prediction and `is_training = false` |
| `x` list is NULL | Row skipped | Not used for training, and no prediction for that row | Not used for training, and the row is **left out** of the returned list |
| `x` list has a NULL element | Row skipped | Not used for training | Not used for training; `yhat` is NULL for that row |
| `y` or `x` is NaN/±Inf | Row dropped before fitting | Row dropped before fitting | Row dropped before fitting |
| Too few usable rows | Result is NULL | Result is NULL | Result is NULL |
| All rows unusable (NULL/NaN) | NULL, or an error when every row is NaN | NULL | NULL |

The row filters are applied **listwise**. A row is used for fitting only when `y`
and every element of `x` are present and finite.

## Fit aggregates

Rows with a NULL `y`, a NULL `x` list or a NULL list element are skipped. The model
is fitted on the rows that remain, and `n_observations` reports how many there were.

```sql
SELECT (ols_fit_agg(y, [x, z])).n_observations AS n_used   -- 4 of 6 rows
FROM (VALUES
    (1.0, 1.0, 1.0),
    (NULL, 2.0, 2.0),      -- NULL y: skipped
    (3.0, 3.0, NULL),      -- NULL element in x: skipped
    (5.0, 5.0, 7.0),
    (4.0, 4.0, 1.0),
    (7.0, 6.0, 3.0)
) t(y, x, z);
```

Non-finite values (`NaN`, `Inf`, `-Inf`) pass through DuckDB as ordinary doubles.
The numerical core drops them before fitting:

```sql
SELECT (ols_fit_agg(y, [x])).n_observations AS n_used      -- 4: the NaN row is dropped
FROM (VALUES (1.0, 1.0), ('NaN'::DOUBLE, 2.0), (3.0, 3.0), (5.1, 5.0), (6.9, 7.0)) t(y, x);
```

### When the result itself is NULL

A fit aggregate returns NULL for a group when:

- the group has no usable rows; or
- the group has too few rows for the model. For OLS this means
  `n <= n_features + 1` with an intercept, or `n <= n_features` without one.
  Other models apply their own minimum.

```sql
SELECT g, ols_fit_agg(y, [x]) IS NULL AS is_null
FROM (VALUES (1, 1.0, 1.0), (1, 2.1, 2.0), (1, 2.9, 3.0), (1, 4.2, 4.0),
             (2, 1.0, 1.0), (2, 2.0, 2.0)) t(g, y, x)     -- group 2 has only 2 rows
GROUP BY g ORDER BY g;
```

If **every** row is NaN, the numerical core raises an error instead
(`ols_fit_agg: All rows filtered due to NULL/NaN values`). Numerical failures, such
as a singular design or a solver that does not converge, either raise an error
that starts with the function name or return NULL. Which one you get depends on
the function.

### Hypothesis tests and correlations

The test aggregates also skip rows listwise. For two-sample tests, a row whose
group is NULL is skipped too. NaN values are dropped. The reported sample sizes
(`n`, `n1`, `n2`) count only the rows that were used.

```sql
SELECT (t_test_agg(v, g)).n1 AS n1, (t_test_agg(v, g)).n2 AS n2  -- 3 and 3
FROM (VALUES (1.0, 0), (2.0, 0), (NULL, 0), (3.0, 0),
             (4.0, 1), (5.0, 1), (6.0, NULL), (7.0, 1), ('NaN'::DOUBLE, 1)) t(v, g);
```

## NaN coefficients: constant and aliased columns

A coefficient is reported as `NaN`, not NULL, when its column cannot be
estimated:

- **Zero-variance (constant) columns** always get a `NaN` coefficient, whichever
  solver is used. The intercept absorbs the constant.
- **Perfectly collinear (aliased) columns** get a `NaN` coefficient with the `qr`
  and `cholesky` solvers. The default `svd` solver returns the minimum-norm
  solution instead, which spreads the effect across the collinear columns.

```sql
SELECT
    (ols_fit_agg(y, [x, 1.0])).coefficients                       AS constant_column,  -- [b, nan]
    (ols_fit_agg(y, [x, 2 * x], {'solver': 'qr'})).coefficients   AS aliased_qr        -- [nan, b]
FROM (VALUES (1.0, 1.0), (2.1, 2.0), (2.9, 3.0), (4.2, 4.0), (5.0, 5.0)) t(y, x);
```

Predictions treat a `NaN` coefficient as contributing 0. The inference arrays
(`std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper`) hold `NaN` in the
same positions.

When `fit_intercept` is `false`, the `intercept` field is `NaN`, not NULL.

Filter on NaN with `isnan(...)`. A NaN is not NULL, so `IS NULL` does not match it.

## Window `*_fit_predict` functions

The window functions (`ols_fit_predict`, `ridge_fit_predict`, `wls_fit_predict`,
`rls_fit_predict`, `elasticnet_fit_predict`, `huber_fit_predict`,
`ransac_fit_predict`, `theil_sen_fit_predict`) fit on the rows in the window frame
that have a non-NULL `y` and a complete `x`. They are window aggregates: they
cannot tell which row is the current one, so they return a prediction for the
**last row of the frame**.

Uses that give the intended result:

- Rolling or expanding in-sample fits whose frame ends at `CURRENT ROW`, over a
  unique ordering. For example,
  `OVER (ORDER BY t ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)` or
  `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Rows with a NULL `y` are not
  used for training but still get a prediction.
- The result is NULL while the frame has too few training rows. A row whose
  `x` is NULL or incomplete gets NULL.

Uses that give misleading results:

- **Frames that end before the current row** (`... AND 1 PRECEDING`). The
  prediction is for the frame's last row, not for the current row.
- **`OVER ()` or `OVER (PARTITION BY g)` with no `ORDER BY`.** The frame is the whole
  partition, so every row gets the same value: the prediction for the
  partition's last row.
- **`RANGE` frames, or an `ORDER BY` key with ties.** Which row is last in the
  frame is then not well defined.

For one prediction per row over a whole partition, use `*_fit_predict_agg` or
`*_fit_predict_by`.

```sql
SELECT t, y,
       round((ols_fit_predict(y, [x]) OVER (
           ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat, 3) AS yhat
FROM (VALUES (1, 3.1, 1.0), (2, 4.9, 2.0), (3, 7.2, 3.0), (4, 8.8, 4.0),
             (5, NULL, 5.0), (6, 13.1, 6.0)) t(t, y, x)
ORDER BY t;
```

For a one-step-ahead (out-of-sample) forecast, fit on the rows *before* the
current row with `ols_fit_agg`, then apply the coefficients to the current row's
features with `predict`. This example is not run during validation: `predict`
currently raises an error for the first rows, where the frame is still too small
and the coefficients are NULL.

<!-- TODO(lead): model-aware predict -->
```sql skip
SELECT t, y,
       predict([[x]],
               (ols_fit_agg(y, [x]) OVER w).coefficients,
               (ols_fit_agg(y, [x]) OVER w).intercept)[1] AS yhat_next
FROM observations
WINDOW w AS (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY t;
```

## `*_fit_predict_agg` and `*_fit_predict_by`

These functions fit once per group and return a prediction for every row:

- Rows with a non-NULL `y` and a complete `x` are training rows. All other rows are
  prediction rows (`is_training = false`), and their `yhat` is still computed.
- A row whose `x` element is NULL gets `yhat`, `yhat_lower` and `yhat_upper` set to NULL.
- A row whose **whole `x` list** is NULL is left out of the list that
  `*_fit_predict_agg` returns, so the list can be shorter than the group. Build
  `x` as a list literal (`[x1, x2]`), so that a missing value becomes a NULL
  element rather than a NULL list.
- With a split column (`*_fit_predict_agg(y, x, split, options)`, or the `split`
  argument of the `*_by` macros), only rows whose split value is `'train'`
  (case-insensitive; `'training'` also works for the aggregate) are used for
  training. A NULL split value marks a prediction row.

```sql
SELECT unnest(ols_fit_predict_agg(y, [x]), recursive := true)
FROM (VALUES (3.1, 1.0), (4.9, 2.0), (NULL, 3.0), (8.8, 4.0), (11.2, 5.0)) t(y, x);
```

### The `null_policy` option

The fit-predict functions accept `null_policy`, which controls which rows are
training rows:

| Value | Meaning |
|-------|---------|
| `'drop'` (default) | Rows with a NULL `y` or an incomplete `x` are not used for training |
| `'drop_y_zero_x'` | Like `'drop'`, but a row with any feature exactly equal to `0.0` is also left out of training. Such rows still get a prediction |

`null_policy` is read by the window `*_fit_predict` functions and by the
`*_fit_predict_agg` functions (and `*_fit_predict_by` macros) for OLS, Ridge, WLS,
RLS, Elastic Net, Huber, RANSAC, Theil-Sen, BLS, ALM and Poisson. PLS, isotonic
and quantile fit-predict do not read it.

```sql
SELECT unnest(ols_fit_predict_agg(y, [x], {'null_policy': 'drop_y_zero_x'}), recursive := true)
FROM (VALUES (3.1, 1.0), (0.4, 0.0), (7.2, 3.0), (8.8, 4.0), (11.2, 5.0)) t(y, x);
```

### Predictions that cannot be computed

When a group's fit fails or has too few training rows, the whole
`*_fit_predict_agg` result for that group is NULL. Within a successful fit, a
row's `yhat` is NULL whenever the prediction is not finite, for example because
the row has a NULL feature.

## Options

An option set to NULL (for example `{'alpha': NULL}`) is treated as unset, and the
default applies. A NULL `options` argument means "all defaults".
