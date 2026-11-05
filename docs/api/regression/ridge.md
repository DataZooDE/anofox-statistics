# Ridge Regression

Ridge regression (L2 penalty) with a choice of SVD, QR or Cholesky solver and
two lambda-scaling conventions.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ridge_fit` | Scalar | Fit on complete arrays in a single call |
| `ridge_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `ridge_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `ridge_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `ridge_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#ridge_fit_predict_by) |
| `ridge_augment_by` | Table macro | Per-row leverage and exact leave-one-out residuals per group, see [Leave-one-out diagnostics](model_tools.md#ridge) |

## ridge_fit

**Signature:**

```text
ridge_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

The penalty is passed as the `alpha` option; there is no positional alpha
argument.

**Example:**

```sql
SELECT ridge_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'alpha': 0.1}
) AS fit;
```

## ridge_fit_agg

**Signature:**

```text
ridge_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE ridge_demo AS
SELECT
    i::DOUBLE AS x1,
    i::DOUBLE + (i % 3) * 0.01 AS x2,          -- nearly collinear with x1
    5.0 + 1.5 * i + cos(i) AS y
FROM range(1, 31) t(i);

SELECT (ridge_fit_agg(y, [x1, x2], {'alpha': 0.5})).coefficients AS coefficients
FROM ridge_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `alpha` (alias `lambda`) | DOUBLE | `1.0` | L2 penalty strength, must be >= 0 |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term (the intercept is not penalized) |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Level of the coefficient confidence intervals |
| `solver` | VARCHAR | `'svd'` | `'svd'`, `'qr'` or `'cholesky'` |
| `lambda_scaling` | VARCHAR | `'raw'` | `'raw'` uses `alpha` as given; `'glmnet'` scales it like R's glmnet |
| `compute_loo` | BOOLEAN | `false` | Fill the exact leave-one-out fields `press`, `press_rmse`, `loo_r_squared` (one SVD of the design); independent of `compute_inference` |
| `conditioning` (alias `conditioning_aware`) | BOOLEAN | `false` | Conditioning-aware λ, see below |
| `lambda_small` | DOUBLE | `0.1` | λ on a well-conditioned design (implies `conditioning`) |
| `cond_threshold` (alias `condition_threshold`) | DOUBLE | `1000` | κ(XᵀX) at or below which the design counts as well-conditioned (implies `conditioning`) |

Option keys the function does not support raise an error.

### Conditioning-aware λ

A fixed λ over-shrinks the coefficients of a well-conditioned design, where the
penalty buys no stability. With `{'conditioning': true}` (or either of
`lambda_small` / `cond_threshold`) the fit measures the condition number
`κ(XᵀX) = (σ_max / σ_min)²` of the design it penalises (columns centred when an
intercept is fitted) and uses `lambda_small` when `κ ≤ cond_threshold`, `alpha`
otherwise (anofox-regression's `RidgeRegressorBuilder::conditioning_aware`; the
defaults 0.1 and 1000 are upstream's). `lambda_small` is on the same scale as
`alpha` (`lambda_scaling` applies to both). The inference, the coefficient
covariance and the fit-predict intervals use the λ actually chosen, so a
`lambda_small` of 0 reports OLS inference. `ridge_fit`, `ridge_fit_agg`,
`ridge_fit_predict`, `ridge_fit_predict_agg`, `ridge_augment_agg` and the
`ridge_*_by` macros take these options; `ridge_fit_from_state` does not (a
sufficient-statistics state keeps no design to measure), nor do the
`ridge_cv` / `ridge_path` grids.

```sql
-- x1 and x2 are nearly collinear: kappa is large, so alpha (5) is kept;
-- x1 alone is well conditioned, so lambda_small (0.1) is used
SELECT
    (ridge_fit_agg(y, [x1, x2], {'alpha': 5, 'conditioning': true})).coefficients AS collinear,
    (ridge_fit_agg(y, [x1], {'alpha': 5, 'conditioning': true})).coefficients AS single,
    (ridge_fit_agg(y, [x1], {'alpha': 0.1})).coefficients AS single_at_0_1
FROM ridge_demo;
```

```sql
-- glmnet-style lambda scaling with the Cholesky solver
SELECT (ridge_fit_agg(
    y, [x1, x2],
    {'alpha': 0.1, 'lambda_scaling': 'glmnet', 'solver': 'cholesky'}
)).coefficients AS coefficients
FROM ridge_demo;
```

## Returns

Same structure as [OLS](ols.md#returns):

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | One coefficient per feature |
| `intercept` | `DOUBLE` | Intercept |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `adj_r_squared` | `DOUBLE` | Adjusted R² |
| `residual_std_error` | `DOUBLE` | Residual standard error |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

With `compute_inference = true`: `std_errors`, `t_values`, `p_values`,
`ci_lower`, `ci_upper` (all `DOUBLE[]`), `f_statistic` and `f_pvalue` (`DOUBLE`).
With `alpha > 0` the standard errors are the ridge sandwich variance, and
`t_values`, `p_values`, `ci_lower`, `ci_upper`, `f_statistic` and `f_pvalue` are
`NULL`: classical tests on shrunken coefficients are not valid. With
`alpha = 0` the fit and its inference are those of OLS.

With `compute_loo = true`: `press`, `press_rmse` and `loo_r_squared` (`DOUBLE`),
the exact leave-one-out statistics of the ridge fit (#235): `press` is
`Σ (eᵢ / (1 − hᵢᵢ))²` with `hᵢᵢ` the diagonal of the ridge hat matrix
`X(XᵀX + λI)⁻¹Xᵀ` (intercept unpenalised), the squared error of predicting each
row from the fit without it at the same λ; `press_rmse = sqrt(press / n)`;
`loo_r_squared = 1 − press / SST`. `ridge_glance_by` turns it on, and
`ridge_augment_by` has the per-row `leverage`, `loo_residual` and `loo_fitted`.
With `'lambda_scaling': 'glmnet'` the effective penalty `alpha · n` of the full
fit is held fixed. See
[Leave-one-out diagnostics](model_tools.md#ridge).

```sql
-- Exact LOO cross-validation over a small alpha grid (options are constants)
SELECT alpha, round((m).press, 3) AS press, round((m).loo_r_squared, 4) AS q2
FROM (SELECT 0.1 AS alpha, ridge_fit_agg(y, [x1, x2], {'alpha': 0.1, 'compute_loo': true}) AS m FROM ridge_demo
      UNION ALL
      SELECT 1.0, ridge_fit_agg(y, [x1, x2], {'alpha': 1.0, 'compute_loo': true}) FROM ridge_demo
      UNION ALL
      SELECT 10.0, ridge_fit_agg(y, [x1, x2], {'alpha': 10.0, 'compute_loo': true}) FROM ridge_demo)
ORDER BY alpha;
```

## Choosing alpha

- **alpha = 0**: no penalty; the fit is identical to OLS
- **alpha = 0.01-0.1**: light regularization
- **alpha = 1.0** (default): moderate regularization
- **alpha >= 10**: strong shrinkage toward zero

Negative values raise an error.

## NULL handling

Rows where `y` or `x` is NULL (or where `x` contains a NULL element) are
skipped by the aggregate; NaN/infinite positions are dropped by `ridge_fit`.
Constant features get a NaN coefficient.

## Use Cases

- Multicollinear predictors
- Stabilising coefficient estimates
- Preventing overfitting with many features

## Coefficient covariance

With `compute_inference` the struct also carries `vcov`, the coefficient covariance
(the sandwich `σ² A X'X A`, `A = (X'X + λI)⁻¹` with an unpenalised intercept) (row-major `k × k`, intercept first), and `df_residual = n − k`;
[`predict`](model_tools.md#standard-errors-and-intervals) uses them for `se_fit` and
confidence / prediction intervals.

## See Also

- [Regularisation paths](paths.md) - `ridge_path_agg`, `ridge_cv_agg`: the coefficient path and the CV curve over lambda

- [OLS](ols.md) - Unregularized baseline
- [Elastic Net](elasticnet.md) - Combined L1+L2 penalty
- [LARS](lars.md) - Least angle regression
- [Table macros](../macros/table_macros.md#ridge_fit_predict_by) - Per-group predictions
