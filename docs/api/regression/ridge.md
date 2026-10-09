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

Option keys the function does not support raise an error.

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

## See Also

- [OLS](ols.md) - Unregularized baseline
- [Elastic Net](elasticnet.md) - Combined L1+L2 penalty
- [LARS](lars.md) - Least angle regression
- [Table macros](../macros/table_macros.md#ridge_fit_predict_by) - Per-group predictions
