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
| `ols_anova_agg` | Aggregate | Nested-model F test (reduced vs full model), see [below](#nested-model-f-test-ols_anova_agg--ols_anova_by) |
| `ols_anova_by` | Table macro | Nested-model F test per group, see [below](#nested-model-f-test-ols_anova_agg--ols_anova_by) |

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
| `alias_tolerance` | DOUBLE | `1e-7` | Tolerance of the collinearity (aliasing) test, as R `lm`'s `tol`: a column whose residual norm, relative to its own norm, is below it after projecting out the earlier columns is aliased (NaN coefficient). Must be > 0 |

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
| `vcov` | `DOUBLE[]` | Coefficient covariance `σ²(X'X)⁻¹`, row-major `k × k`, intercept first; only with `compute_inference` (used by [`predict`](model_tools.md#standard-errors-and-intervals) for `se_fit` / intervals) |
| `df_residual` | `DOUBLE` | `n − k`; only with `compute_inference` |

## NULL handling

- `ols_fit_agg`: rows where `y` or `x` is NULL, or where `x` contains a NULL
  element, are skipped.
- `ols_fit`: rows (array positions) with NaN or infinite values are dropped
  before fitting.
- A feature that is constant over the fitted rows cannot be estimated; its
  coefficient is returned as NaN (aliased) rather than failing the whole fit.

## Nested-model F test: ols_anova_agg / ols_anova_by

Do the extra regressors improve the model? `ols_anova_agg` fits two nested OLS
models on the same rows and returns the F test of the reduced model against
the full one, as R's `anova(lm(y ~ x_reduced), lm(y ~ x_full))`, together with
the adjusted R² of both models ("before" and "after").

**Signatures:**

```text
ols_anova_agg(y DOUBLE, x_reduced DOUBLE[], x_full DOUBLE[] [, options MAP]) -> STRUCT
ols_anova_by(source VARCHAR, group_col, y_col, x_reduced := [], x_full := [...], options := NULL) -> TABLE
```

- **Nesting.** Every `x_reduced` column must also be a column of `x_full`; more
  precisely, it must lie in the span of the full design, so a rescaled or
  shifted copy (`2 * price + 1`) also counts. Otherwise the call raises
  `x_reduced is not nested in x_full`.
- **Intercept-only reduced model.** `x_reduced := []` (the macro's default)
  compares the full model with the intercept-only model. That reproduces the
  overall F test of `ols_fit_agg` (its `f_statistic` with `compute_inference`).
- **Same rows.** A row is used by both fits only when `y`, both lists and every
  list element are non-NULL and finite.
- **Options.** `fit_intercept` (default `true`) applies to both models.
- **Rank deficiency.** As in R, parameters are counted by the numerical rank,
  so an aliased (collinear) column adds no degree of freedom. When the full
  model adds nothing (`df1 = 0`) or leaves no residual degrees of freedom
  (`df2 = 0`), `statistic` and `p_value` are NULL (R's `NA`; never NaN, so
  `avg()` and other aggregates over the results skip them). A group with fewer rows than
  the full model has parameters returns NULL.

The aggregate's STRUCT follows the integration contract's `test` schema:

| Field | Type | Description |
|-------|------|-------------|
| `method` | `VARCHAR` | `'nested F'` |
| `statistic` | `DOUBLE` | `F = ((RSS_r − RSS_f) / df1) / (RSS_f / df2)` |
| `df1` | `DOUBLE` | Parameters the full model adds (`p_full − p_reduced`) |
| `df2` | `DOUBLE` | Residual degrees of freedom of the full model (`n − p_full`) |
| `p_value` | `DOUBLE` | Upper tail of `F(df1, df2)` |
| `rss_reduced`, `rss_full` | `DOUBLE` | Residual sums of squares |
| `adj_r_squared_reduced`, `adj_r_squared_full` | `DOUBLE` | Adjusted R² of each model, as `summary(lm(...))$adj.r.squared` |
| `delta_adj_r_squared` | `DOUBLE` | `adj_r_squared_full − adj_r_squared_reduced` |
| `n_observations` | `BIGINT` | Rows used by both fits |
| `alternative`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name` | | Unified `test` fields; NULL here |
| `n` | `BIGINT` | Same as `n_observations` |

`ols_anova_by` returns one row per group: the group column, `id` (the group as
text), `model_id` / `model_name` (`'ols'`), then `method`, `statistic`,
`p_value`, `df1`, `df2`, `rss_reduced`, `rss_full`, `adj_r_squared_reduced`,
`adj_r_squared_full`, `delta_adj_r_squared` and `n_observations`.

**Example** (continues with `ols_demo` from above):

```sql
-- Does ads improve on price alone, per region?
SELECT region, round((r).statistic, 3) AS f, (r).df1, (r).df2, (r).p_value < 0.05 AS significant,
       round((r).delta_adj_r_squared, 4) AS delta_adj_r2
FROM (SELECT region, ols_anova_agg(sales, [price], [price, ads]) AS r FROM ols_demo GROUP BY region)
ORDER BY region;

-- The same as a table macro; x_reduced := [] would test the whole model.
SELECT region, round(statistic, 3) AS f, df1, df2, round(adj_r_squared_reduced, 4) AS before,
       round(adj_r_squared_full, 4) AS after
FROM ols_anova_by('ols_demo', region, sales, x_reduced := [price], x_full := [price, ads]);
```

Sequential (type I) and partial (type II) ANOVA tables per term are not
available yet.

## Use Cases

- Standard linear regression and baseline models
- When inference (p-values, confidence intervals) is needed
- Testing whether added regressors improve a model (`ols_anova_agg` / `ols_anova_by`)
- Robust inference under heteroscedasticity via `hc_type`

## See Also

- [Ridge](ridge.md) - L2 regularization for multicollinearity
- [WLS](wls.md) - Weighted least squares
- [Huber](huber.md), [RANSAC](ransac.md), [Theil-Sen](theil_sen.md) - Robust alternatives
- [Table macros](../macros/table_macros.md#ols_fit_predict_by) - Per-group predictions
