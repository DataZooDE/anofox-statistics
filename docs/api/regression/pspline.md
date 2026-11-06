# P-Spline Smoothing

A P-spline (penalized B-spline, Eilers & Marx 1996) is a one-dimensional
smoother: `y` is fitted on cubic B-splines of a single predictor `x` with a
difference penalty on neighbouring coefficients, and the smoothing parameter is
chosen by generalized cross-validation (GCV) unless you fix it. It is mgcv's
`bs = "ps"` smooth:

```text
mgcv::gam(y ~ s(x, bs = "ps", k = n_basis, m = c(2, penalty_order)), method = "GCV.Cp")
```

The fit comes from anofox-regression's `PSplineRegressor`; the reference test
compares it with mgcv.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pspline_fit_agg` | Aggregate | Fit a P-spline per group; returns the smooth at the data points and the fit statistics |
| `pspline_fit_predict_agg` | Aggregate | Fit and predict every row of a group, with a prediction or confidence interval |
| `pspline_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#pspline_fit_predict_by) |
| `pspline_prediction_by` | Table macro | The same in the integration contract's `prediction` schema |

As in isotonic regression, `x` is a single `DOUBLE`, not a list.

The fitted basis (knots, coefficients, posterior covariance) is not exposed by
the upstream crate, so the model cannot be stored and evaluated later with
`predict`. To predict new points, add them to the group with a NULL `y` (or a
non-`'train'` split) and use `pspline_fit_predict_agg` or
`pspline_fit_predict_by`.

## pspline_fit_agg

**Signature:**

```text
pspline_fit_agg(y DOUBLE, x DOUBLE [, options MAP]) -> STRUCT
```

Options: `n_basis`, `penalty_order`, `lambda` (see [Options](#options)).

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `x` | `DOUBLE[]` | Sorted distinct `x` values of the fitted rows |
| `fitted` | `DOUBLE[]` | The smooth at each `x` |
| `se_fit` | `DOUBLE[]` | Standard error of the smooth at each `x` (Bayesian posterior, mgcv's `predict.gam(se.fit = TRUE)`) |
| `coefficients` | `DOUBLE[]` | B-spline coefficients (`n_basis` values) |
| `lambda` | `DOUBLE` | Smoothing parameter of the difference penalty (mgcv: `sp / S.scale`) |
| `edf` | `DOUBLE` | Effective degrees of freedom, the intercept included (`sum(gam$edf)`) |
| `sigma2` | `DOUBLE` | Residual variance `RSS / (n - edf)` (`gam$sig2`) |
| `r_squared` | `DOUBLE` | In-sample R² (`1 - RSS / TSS`) |
| `n_basis` | `BIGINT` | Number of basis functions used |
| `penalty_order` | `BIGINT` | Order of the difference penalty |
| `n_observations` | `BIGINT` | Rows used in the fit |

The model-summary fields that every fit returns (`model_type` `'pspline'`,
`family` `'gaussian'`, `link`) follow. The coefficient-inference fields and
`log_likelihood`, `aic`, `bic` are NULL: the upstream fit does not report a
likelihood.

Rows with a NULL or non-finite `y` or `x` are skipped. The result is NULL when
fewer than 6 rows remain or `x` is constant.

**Example:**

```sql
CREATE OR REPLACE TABLE pspline_demo AS
SELECT i, (i % 2)::VARCHAR AS grp, i / 10.0 AS x,
       sin(i / 10.0) + 0.4 * ((i * 7919) % 97 / 97.0 - 0.5) AS y
FROM range(1, 101) r(i);

SELECT m.n_basis, round(m.edf, 2) AS edf, round(m.lambda, 4) AS lambda,
       round(m.sigma2, 4) AS sigma2, m.fitted[1:3] AS first_fitted
FROM (SELECT pspline_fit_agg(y, x, {'n_basis': 20}) AS m FROM pspline_demo);
```

A fixed `lambda` skips the GCV search; a larger value gives a smoother curve
(fewer effective degrees of freedom):

```sql
SELECT pspline_fit_agg(y, x, {'n_basis': 20, 'lambda': 0.1}).edf AS edf_rough,
       pspline_fit_agg(y, x, {'n_basis': 20, 'lambda': 100}).edf AS edf_smooth
FROM pspline_demo;
```

## pspline_fit_predict_agg

**Signature:**

```text
pspline_fit_predict_agg(y DOUBLE, x DOUBLE [, options MAP])
    -> STRUCT(y DOUBLE, yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, is_training BOOLEAN)[]
pspline_fit_predict_agg(y DOUBLE, x DOUBLE, split_col VARCHAR [, options MAP])
    -> STRUCT(y DOUBLE, yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, is_training BOOLEAN)[]
```

Fits on the training rows (a non-NULL, finite `y` and `x` and, with the split
form, `split_col` `'train'` / `'training'` in any case) and returns one STRUCT
per input row in input order (use `ORDER BY` inside the aggregate to fix it).
Every row with a finite `x` gets a prediction; a NULL `x` gives a NULL `yhat`.

The interval is `yhat ∓ t(n - edf) · s` at `confidence_level`, with `s` the
standard error of the smooth (`'interval': 'confidence'`) or
`sqrt(se² + sigma2)` for a new observation (`'prediction'`, the default).
`'interval': 'none'` leaves the bounds NULL. Outside the range of the training
`x` the smooth continues linearly from the boundary, as mgcv's.

**Example:**

```sql
-- Predict two new points (y NULL), one beyond the data
SELECT r.y, round(r.yhat, 3) AS yhat, round(r.yhat_lower, 3) AS lower,
       round(r.yhat_upper, 3) AS upper, r.is_training
FROM (
    SELECT unnest(pspline_fit_predict_agg(y, x, {'n_basis': 20, 'interval': 'confidence'} ORDER BY x)) AS r
    FROM (SELECT x, y FROM pspline_demo UNION ALL VALUES (5.05, NULL), (11.0, NULL))
)
WHERE r.y IS NULL;
```

## pspline_fit_predict_by

```text
pspline_fit_predict_by(source, group_col, y_col, x_col, options := NULL, split := NULL, order_by := NULL)
```

Fits one smooth per group and returns every source row with `yhat`,
`yhat_lower`, `yhat_upper` and `is_training` appended:

```sql
SELECT grp, x, y, round(yhat, 3) AS yhat, round(yhat_lower, 3) AS lower, round(yhat_upper, 3) AS upper
FROM pspline_fit_predict_by('pspline_demo', grp, y, x, options := {'n_basis': 15})
ORDER BY grp, x
LIMIT 5;
```

`pspline_prediction_by(source, group_col, x_col, y_col, options := NULL,
level := 0.95, split := NULL)` returns the same fit in the contract's
`prediction` schema (`model_id` `'pspline'`), ready for a band plot:

```sql
SELECT id, x, y, round(yhat, 3) AS yhat, round(yhat_lower, 3) AS lower, round(yhat_upper, 3) AS upper, split
FROM pspline_prediction_by('pspline_demo', grp, x, y, level := 0.9)
LIMIT 5;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_basis` (alias `k`) | INTEGER | `0` | Number of cubic B-spline basis functions, mgcv's `k`. `0` chooses `n / 4 + 4`, clamped to [6, 40]. Values below `penalty_order + 4` are raised to it |
| `penalty_order` | INTEGER | `2` | Order of the difference penalty, mgcv's `m[2]` (1 to 10). With order 2 a straight line is not penalized |
| `lambda` | DOUBLE | GCV | Fixed smoothing parameter (>= 0). Omitted: minimise `GCV(λ) = n · RSS / (n − edf)²` |
| `confidence_level` | DOUBLE | `0.95` | Interval level (`pspline_fit_predict_agg` only) |
| `interval` | VARCHAR | `'prediction'` | `'prediction'`, `'confidence'` or `'none'` (`pspline_fit_predict_agg` only) |

Option keys the function does not support raise an error.

Note that mgcv's default basis size is `k = 10`; pass `{'n_basis': 10}` to
match `s(x, bs = "ps")` without `k`.

## NULL handling

Rows where `y` or `x` is NULL or non-finite are not used for fitting. The
fit-predict aggregate keeps every row in its output: rows with a NULL `y` are
prediction rows, rows with a NULL `x` get a NULL `yhat`. A group with no
training rows, fewer than 6, or a constant `x` returns NULL.

## See Also

- [Online / dynamic](online_dynamic.md#lowess) - LOWESS, a local-regression smoother
- [Isotonic](isotonic.md) - Monotone one-dimensional fit
- [Fit-predict aggregates](fit_predict_agg.md)
