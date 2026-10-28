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
| `ols_anova_table_agg` | Aggregate | Per-term ANOVA table (type I / type II), see [below](#per-term-anova-tables-ols_anova_table_agg--ols_anova_table_by) |
| `ols_anova_table_by` | Table macro | Per-term ANOVA table per group, see [below](#per-term-anova-tables-ols_anova_table_agg--ols_anova_table_by) |

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
| `compute_loo` | BOOLEAN | `false` | Fill the leave-one-out fields `press`, `press_rmse`, `loo_r_squared` (one more QR of the design); independent of `compute_inference` |
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
| `press` | `DOUBLE` | Leave-one-out prediction sum of squares `Σ (eᵢ / (1 − hᵢᵢ))²` (R: `sum(rstandard(fit, type = "predictive")^2)`); NULL unless `compute_loo`, and if a row has leverage 1 |
| `press_rmse` | `DOUBLE` | `sqrt(press / n_observations)` |
| `loo_r_squared` | `DOUBLE` | Predicted R² (Q²) `1 − press / SST`, with the SST of `r_squared`; never above `r_squared` |

The leave-one-out fields come from the hat-matrix diagonal, without refitting
(#220). They cost one more QR of the design, so they are opt-in
(`compute_loo`); `ols_glance_by` turns it on and reports them as metrics and `ols_augment_by` has the
per-row `loo_residual` / `loo_fitted`. See
[Leave-one-out diagnostics](model_tools.md#leave-one-out-diagnostics).

```sql
-- Predicted R² next to R² per group
SELECT region, round((m).r_squared, 4) AS r2, round((m).loo_r_squared, 4) AS q2,
       round((m).press_rmse, 4) AS press_rmse
FROM (SELECT region, ols_fit_agg(sales, [price, ads], {'compute_loo': true}) AS m
      FROM ols_demo GROUP BY region)
ORDER BY region;
```

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

## Per-term ANOVA tables: ols_anova_table_agg / ols_anova_table_by

Which regressors matter? `ols_anova_table_agg` fits one OLS model and returns
one row per term with its sum of squares and F test, as R's `anova(lm(...))`
(type I) or `car::Anova(fit, type = 2)` (type II).

**Signatures:**

```text
ols_anova_table_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT[]
ols_anova_table_by(source VARCHAR, group_col, y_col, x_cols, options := NULL) -> TABLE
```

- **Type I (sequential, the default).** Each term is adjusted for the terms
  before it, in the order of `x`: `SS_k = RSS(terms 1..k−1) − RSS(terms 1..k)`.
  The sums of squares add up to `TSS − RSS`, and the table depends on the
  column order.
- **Type II (partial, `{'type': 'II'}`).** Each term is adjusted for all other
  terms: `SS_k = RSS(all but k) − RSS(full)`, the drop-one nested F test
  (`ols_anova_agg`) of the term. The table does not depend on the order. Models
  with interactions are not supported (there are no interaction terms here).
- Every `F = (SS_k / df_k) / (RSS / df_res)` uses the full model's residual mean
  square; types I and II agree for the last term.
- **Terms.** By default every column of `x` is a term, labelled `x1`, `x2`, ....
  The `terms` option gives one label per column; columns with the same label
  form one term, for example the dummies of a factor
  (`{'terms': ['wt', 'cyl', 'cyl', 'hp']}`), ordered by first appearance.
- **Options.** `type` (`'I'` default, `'II'`; also `1` / `2`, `'sequential'` /
  `'partial'`; alias key `anova_type`), `terms` and `fit_intercept` (default
  `true`; without an intercept the sums of squares are uncentred, as R's
  `y ~ 0 + ...`).
- **Rank deficiency.** Degrees of freedom count the numerical rank. A term that
  adds no rank (aliased with the terms before it, or for type II with all
  other terms) gets `df1 = 0`, `sum_sq = 0` and NULL `mean_sq`, `statistic` and
  `p_value`; R's `anova()` drops such a term. With no residual degrees of
  freedom every `statistic` and `p_value` is NULL.
- **Rows.** A row is used when `y`, the list and every list element are
  non-NULL and finite. A group with too few rows returns NULL.

Each list element follows the integration contract's `test` schema:

| Field | Type | Description |
|-------|------|-------------|
| `term` | `VARCHAR` | Term label |
| `method` | `VARCHAR` | `'type I ANOVA'` or `'type II ANOVA'` |
| `statistic` | `DOUBLE` | F statistic of the term |
| `df1` | `DOUBLE` | Degrees of freedom of the term (the rank it adds) |
| `df2` | `DOUBLE` | Residual degrees of freedom of the full model |
| `p_value` | `DOUBLE` | Upper tail of `F(df1, df2)` |
| `sum_sq`, `mean_sq` | `DOUBLE` | Sum of squares of the term and `sum_sq / df1` |
| `residual_sum_sq`, `residual_mean_sq` | `DOUBLE` | The residual row of R's table |
| `n_observations` | `BIGINT` | Rows used by the fits |
| `alternative`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name` | | Unified `test` fields; NULL here |
| `n` | `BIGINT` | Same as `n_observations` |

`ols_anova_table_by` returns one row per group and term: the group column,
`id`, `model_id` / `model_name` (`'ols'`), `term`, `method`, `statistic`,
`p_value`, `df1`, `df2`, `sum_sq`, `mean_sq`, `residual_sum_sq`,
`residual_mean_sq` and `n_observations`.

**Example** (continues with `ols_demo` from above):

```sql
-- Sequential (type I) table per region: price first, then ads.
SELECT region, (r).term, (r).df1, round((r).sum_sq, 2) AS sum_sq, round((r).statistic, 2) AS f
FROM (SELECT region, unnest(ols_anova_table_agg(sales, [price, ads])) AS r FROM ols_demo GROUP BY region)
ORDER BY region, (r).term;

-- Type II with named terms, as a table macro.
SELECT region, term, df1, df2, round(statistic, 2) AS f, p_value < 0.05 AS significant
FROM ols_anova_table_by('ols_demo', region, sales, [price, ads],
                        options := {'type': 'II', 'terms': ['price', 'ads']});
```

## Use Cases

- Standard linear regression and baseline models
- When inference (p-values, confidence intervals) is needed
- Testing whether added regressors improve a model (`ols_anova_agg` / `ols_anova_by`)
- Per-term ANOVA tables, sequential or partial (`ols_anova_table_agg` / `ols_anova_table_by`)
- Robust inference under heteroscedasticity via `hc_type`

## See Also

- [Ridge](ridge.md) - L2 regularization for multicollinearity
- [WLS](wls.md) - Weighted least squares
- [Huber](huber.md), [RANSAC](ransac.md), [Theil-Sen](theil_sen.md) - Robust alternatives
- [Table macros](../macros/table_macros.md#ols_fit_predict_by) - Per-group predictions
