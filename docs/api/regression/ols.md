# OLS (Ordinary Least Squares)

Ordinary least squares regression with a choice of SVD, QR or Cholesky solver
and optional heteroscedasticity-consistent (HC0-HC3) standard errors.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit` | Scalar | Fit on complete arrays in a single call |
| `ols_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `ols_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `ols_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `ols_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#ols_fit_predict_by) |

## ols_fit

**Signature:**

```text
ols_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature, the same length as `y` |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
-- Simple regression: y = 2x + 1 (plus noise)
SELECT ols_fit(
    [3.1, 4.9, 7.2, 8.8, 11.1],
    [[1.0, 2.0, 3.0, 4.0, 5.0]]
) AS fit;

-- With inference
SELECT (ols_fit(
    [3.1, 4.9, 7.2, 8.8, 11.1],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'compute_inference': true, 'confidence_level': 0.99}
)).p_values AS p_values;
```

## ols_fit_agg

Aggregate form. Each input row contributes one observation.

**Signature:**

```text
ols_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE ols_demo AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS region,
    i::DOUBLE AS price,
    (i % 5)::DOUBLE AS ads,
    10.0 + 2.0 * i + 0.5 * (i % 5) + sin(i) AS sales
FROM range(1, 41) t(i);

-- One model per group
SELECT
    region,
    (ols_fit_agg(sales, [price, ads])).coefficients AS coefficients,
    (ols_fit_agg(sales, [price, ads])).r_squared AS r_squared
FROM ols_demo
GROUP BY region
ORDER BY region;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Level of the coefficient confidence intervals |
| `solver` | VARCHAR | `'svd'` | `'svd'`, `'qr'` or `'cholesky'` |
| `hc_type` | VARCHAR | `'none'` | Robust standard errors: `'none'`, `'hc0'`, `'hc1'`, `'hc2'`, `'hc3'` |

Option keys the function does not support raise an error.

```sql
-- QR solver with HC3 robust standard errors
SELECT (ols_fit_agg(
    sales, [price, ads],
    {'solver': 'qr', 'hc_type': 'hc3', 'compute_inference': true}
)).std_errors AS robust_se
FROM ols_demo;
```

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | One coefficient per feature (intercept excluded) |
| `intercept` | `DOUBLE` | Intercept (NaN when `fit_intercept` is false) |
| `r_squared` | `DOUBLE` | Coefficient of determination (uncentered, `1 - RSS/Σy²`, when `intercept` is false, as R's `summary.lm`) |
| `adj_r_squared` | `DOUBLE` | Adjusted R² |
| `residual_std_error` | `DOUBLE` | Residual standard error |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

Coefficient inference; the fields are always present and NULL unless
`compute_inference = true`:

| Field | Type | Description |
|-------|------|-------------|
| `std_errors` | `DOUBLE[]` | Coefficient standard errors (HC-adjusted if `hc_type` is set) |
| `t_values` | `DOUBLE[]` | t-statistics |
| `p_values` | `DOUBLE[]` | Two-sided p-values |
| `ci_lower` | `DOUBLE[]` | Deprecated alias of `conf_low` |
| `ci_upper` | `DOUBLE[]` | Deprecated alias of `conf_high` |
| `f_statistic` | `DOUBLE` | Overall F-statistic |
| `f_pvalue` | `DOUBLE` | p-value of the F-test |
| `conf_low` | `DOUBLE[]` | Lower confidence bounds |
| `conf_high` | `DOUBLE[]` | Upper confidence bounds |
| `conf_level` | `DOUBLE` | Confidence level of the intervals |
| `intercept_std_error` | `DOUBLE` | Intercept standard error (HC-adjusted if `hc_type` is set) |
| `intercept_statistic` | `DOUBLE` | Intercept t-statistic |
| `intercept_p_value` | `DOUBLE` | Intercept p-value |
| `intercept_conf_low` | `DOUBLE` | Intercept lower confidence bound |
| `intercept_conf_high` | `DOUBLE` | Intercept upper confidence bound |

The inference lists cover the feature coefficients, in the same order as
`coefficients`; the intercept has its own `intercept_*` fields.

Always present:

| Field | Type | Description |
|-------|------|-------------|
| `log_likelihood` | `DOUBLE` | Gaussian log-likelihood, as R's `logLik(lm(...))` |
| `aic` | `DOUBLE` | AIC, as R's `AIC` (the residual variance counts as a parameter) |
| `bic` | `DOUBLE` | BIC, as R's `BIC` |
| `model_type` | `VARCHAR` | `'ols'` |
| `family` | `VARCHAR` | `'gaussian'` |
| `link` | `VARCHAR` | `'identity'` |

## NULL handling

- `ols_fit_agg`: rows where `y` or `x` is NULL, or where `x` contains a NULL
  element, are skipped.
- `ols_fit`: rows (array positions) with NaN or infinite values are dropped
  before fitting.
- A feature that is constant over the fitted rows cannot be estimated; its
  coefficient is returned as NaN (aliased) rather than failing the whole fit.

## Use Cases

- Standard linear regression and baseline models
- When inference (p-values, confidence intervals) is needed
- Robust inference under heteroscedasticity via `hc_type`

## See Also

- [Ridge](ridge.md) - L2 regularization for multicollinearity
- [WLS](wls.md) - Weighted least squares
- [Huber](huber.md), [RANSAC](ransac.md), [Theil-Sen](theil_sen.md) - Robust alternatives
- [Table macros](../macros/table_macros.md#ols_fit_predict_by) - Per-group predictions
