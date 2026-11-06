---
name: anofox-statistics-batch
description: >
  AI-ready batch and per-group model fitting for the anofox_statistics DuckDB
  extension — training thousands of regression models in a single SQL query.
  Covers the *_fit_predict_by table macros (ols_fit_predict_by,
  huber_fit_predict_by, ransac_fit_predict_by, theil_sen_fit_predict_by,
  ridge_fit_predict_by, elasticnet_fit_predict_by, wls_fit_predict_by,
  rls_fit_predict_by, bls_fit_predict_by, alm_fit_predict_by,
  poisson_fit_predict_by, binomial_fit_predict_by, logistic_fit_predict_by,
  negbinom_fit_predict_by, gamma_fit_predict_by, tweedie_fit_predict_by,
  pls_fit_predict_by, isotonic_fit_predict_by, quantile_fit_predict_by,
  bayesian_ridge_fit_predict_by, ard_fit_predict_by,
  passive_aggressive_fit_predict_by, lm_dynamic_fit_predict_by) with split :=
  and order_by :=, the per-group long-table macros <model>_tidy_by and
  <model>_glance_by (25 models), ols/wls/ridge/GLM _augment_by,
  <model>_partial_residuals_by, <model>_prediction_by (23 models),
  <model>_effects_by (marginal effects), <model>_rolling_tidy_by (rolling
  coefficients), ridge/elasticnet/lars_path_by, ridge/elasticnet_cv_by,
  ridge/elasticnet_nested_cv_by, quantile_process_by, acf_by / pacf_by,
  kaplan_meier_by, roc/pr/calibration_curve_by and pairwise_t_test_by /
  tukey_hsd_by / dunn_test_by, per-group aggregate fitting with GROUP BY plus
  predict / tidy / glance on the per-group model structs, the
  *_fit_predict_agg aggregates, rolling *_fit_predict window functions, the
  glmm_fit_by / eb_shrink_by / aid_by macros, and scaling patterns for fitting
  a separate model per segment / SKU / cohort over large panels. Also
  glmm_tidy_by / glance_by / augment_by, ols_anova_by / ols_anova_table_by,
  mediation_by,
  p_adjust_by, ridge / elasticnet_nested_cv_by, the split := rule and order_by
  := for rls_tidy_by. Since 0.16.0 also cox_fit_by / cox_tidy_by /
  cox_glance_by / cox_augment_by, gls_fit_by / gls_tidy_by / gls_glance_by /
  gls_augment_by, aft_augment_by, ordinal_fit_predict_by,
  continuation_ratio_fit_predict_by, multinomial_fit_predict_by,
  pspline_fit_predict_by / pspline_prediction_by / pspline_fit_predict_agg,
  predict_grid_by, validate_by / calibrate_by / calibrate_cph_by / bootcov_by,
  pentrace_by / penalized_ols_fit_by, glm_score_test_by, areg_impute_by,
  pca_by and dataset_drift_by. Use when the task is "fit one model per group" at scale,
  or "give me a coefficient / fit / residual / prediction table per group",
  rather than a single global fit.
version: 0.16.0
user-invocable: false
---

# Anofox Statistics — Batch / Per-Group Fitting Cheat Sheet

**Extension:** `anofox_statistics` v0.16.0 | **DuckDB:** v1.5.6 / v2.0

> **New in 0.15.0:** contract `*_by` outputs end with `row_order BIGINT` (display order per group). For incremental per-partition fits, store `ls_state_agg` states and refit with `*_fit_from_state(ls_state_combine_agg(state))`.
>
> **New in 0.16.0:** per-group Cox (`cox_*_by`), GLS (`gls_*_by`), AFT residuals (`aft_augment_by`), ordinal / continuation-ratio / multinomial and P-spline `*_fit_predict_by`, `predict_grid_by`, rms validation (`validate_by`, `calibrate_by`, `calibrate_cph_by`, `bootcov_by`), `pentrace_by` / `penalized_ols_fit_by`, `glm_score_test_by`, `areg_impute_by`, `pca_by`, `dataset_drift_by` (Pattern 1, 1b, 1c). Behaviour changes that reach the `_by` outputs: with `hc_type`, the OLS / WLS `vcov` (and so the `predict(model, x, {'se_fit': true})` intervals) is now the HC covariance; `wls_*` with `hc_type` now uses the weights.

The extension's superpower for AI/analytics pipelines: **fit one model per group in a single SQL statement**, using DuckDB's `HASH_GROUP_BY` for parallelism — no Python loop, no per-group round-trips. Three patterns, from highest-level to lowest.

## Pattern 1 — `*_fit_predict_by` table macros (recommended)

Fit + predict per group with one call. The macro renames the target and adds prediction columns.

```text
{model}_fit_predict_by(
  source     VARCHAR,   -- table name (quoted string, NOT a CTE/subquery)
  group_col  COLUMN,    -- segment identifier (unquoted)
  y_col      COLUMN,    -- target (unquoted)
  x_cols     LIST,      -- feature columns, e.g. [x1, x2]
  options    MAP,       -- optional model options (defaults to NULL)
  split      VARCHAR,   -- optional: column name of a train/test split; only rows where it is 'train' / 'training' (any case) are trained on; a NULL split value is predicted
  order_by   COLUMN     -- optional (by name): orders each group's rows; deterministic row alignment, and the feed order for RLS
) -> TABLE(<all source columns>, yhat, yhat_lower, yhat_upper, is_training)
```

Pass `split` and `order_by` by name: `ols_fit_predict_by('sales', category, revenue, [units], split := sp, order_by := t)`.

Options must be constant per query, not literal: a whole option struct can live in a variable (`SET VARIABLE opts = {'alpha': 1.0}`, then `options := getvariable('opts')`, or `$opts` on DuckDB 2.0) or be a prepared-statement parameter (`PREPARE p AS ... options := ?`). That makes it easy to parameterize scripts.

Rows with a NULL `y_col` are not used for fitting but still receive a prediction, so appending future rows with `y = NULL` gives you a forecast.

Available macros: `ols_fit_predict_by`, `huber_fit_predict_by`, `ransac_fit_predict_by`,
`theil_sen_fit_predict_by`, `ridge_fit_predict_by`, `elasticnet_fit_predict_by`,
`wls_fit_predict_by`, `rls_fit_predict_by`, `bls_fit_predict_by`, `alm_fit_predict_by`,
`poisson_fit_predict_by`, `pls_fit_predict_by`, `isotonic_fit_predict_by`,
`quantile_fit_predict_by`, `bayesian_ridge_fit_predict_by`, `ard_fit_predict_by` (posterior predictive band), `passive_aggressive_fit_predict_by`, `lm_dynamic_fit_predict_by` (sequential: pass `order_by := t`), and the GLM macros `binomial_fit_predict_by`,
`logistic_fit_predict_by`, `negbinom_fit_predict_by`, `gamma_fit_predict_by`,
`tweedie_fit_predict_by`. Related grouped macros: `glmm_fit_by(source, group_col, y_col, x_cols[, options])`,
`eb_shrink_by(source, estimate_col, se_col[, options])`, `aid_by`, `aid_anomaly_by`.

New in 0.16.0 (same `(source, group_col, y_col, x_cols, options := NULL, split := NULL)` shape, no `order_by`, no interval columns):

| Macro | Purpose / added columns |
|---|---|
| `ordinal_fit_predict_by` | Cumulative-link model per group (`rms::orm` / `MASS::polr`, option `link`): `yhat` (most probable level), `yhat_mean`, `probabilities` (`DOUBLE[]` in level order), `is_training` |
| `continuation_ratio_fit_predict_by` | Continuation-ratio ordinal model per group (`cr.setup` + `lrm`); same columns as `ordinal_fit_predict_by` |
| `multinomial_fit_predict_by` | Softmax logistic per group (options `C` / `lambda`, `l1_ratio`): `yhat` (most probable class), `probabilities` (class order), `is_training`; a single-class group gets NULLs |
| `pspline_fit_predict_by(source, group_col, y_col, x_col, …)` | P-spline smoother (mgcv `s(x, bs = 'ps')`) of `y` on a **single** `x_col` per group (options `n_basis`, `penalty_order`, `lambda`, `interval`); `yhat`, `yhat_lower`, `yhat_upper` (prediction interval by default), `is_training`; also takes `order_by` |

```sql
CREATE TABLE ratings AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS site,
       (i % 7)::DOUBLE AS dose, (20 + i % 13)::DOUBLE AS age,
       CASE WHEN (i % 7) + (i % 3) < 3 THEN 1 WHEN (i % 7) + (i % 3) < 6 THEN 2 ELSE 3 END::DOUBLE AS grade,
       ['low', 'mid', 'high'][grade::INT] AS rating
FROM range(1, 121) t(i);

-- One proportional-odds model per site (y is numeric: the levels are its ordered values)
SELECT site, id, grade, yhat, round(yhat_mean, 2) AS yhat_mean,
       list_transform(probabilities, lambda p: round(p, 3)) AS probs
FROM ordinal_fit_predict_by('ratings', site, grade, [dose, age])
ORDER BY site, id LIMIT 4;

-- One multinomial model per site, trained on 4/5 of the rows; train vs test accuracy
SELECT site, is_training, round(avg((yhat = rating)::INT), 3) AS accuracy
FROM multinomial_fit_predict_by('ratings', site, rating, [dose, age],
                                split := CASE WHEN id % 5 = 0 THEN 'test' ELSE 'train' END)
GROUP BY ALL ORDER BY ALL;
```

The five GLM macros fit `{family}_fit_agg` per group and apply `predict(model, x)` to every row:
`yhat` is on the **response scale** (probability / rate / mean), `yhat_lower`/`yhat_upper` are
NULL, and they take `(source, group_col, y_col, x_cols, options := NULL, split := NULL)` — no
`order_by` (rows are joined to their group's model). PLS, quantile and isotonic macros return
only `yhat` and `is_training`. Prediction intervals elsewhere are leverage-aware and NULL when
a group has no residual degrees of freedom.

**Signature variants:** `wls_fit_predict_by` takes an extra `weight_col` before `options`
(`…, x_cols, weight_col, options`); `isotonic_fit_predict_by` takes a single `x_col`
(not a list). All others follow the 4-arg `(source, group_col, y_col, x_cols[, options])` shape.

```sql
CREATE TABLE sales AS
SELECT CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
       i::DOUBLE AS units, (i % 5)::DOUBLE AS price,
       (3.0 * i - 2.0 * (i % 5) + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

-- One OLS model per product category, fit + in-sample predictions, in one query
SELECT category, units, revenue, yhat, is_training
FROM ols_fit_predict_by('sales', category, revenue, [units, price], order_by := units)
LIMIT 5;

-- One Gamma GLM per category; yhat is the expected revenue (response scale)
SELECT category, units, revenue, round(yhat, 2) AS yhat
FROM gamma_fit_predict_by('sales', category, revenue, [units])
ORDER BY category, units
LIMIT 5;
```

**Scoring new rows per series** (a forecast horizon, a test set): either stack the
new rows into the source with a NULL `y` and read the rows with `is_training IS NOT TRUE`
from `*_fit_predict_by` (a series without history has NULL `is_training` and `yhat`), or keep a fit table
(`SELECT id, ols_fit_agg(y, [x1, x2]) AS model FROM history GROUP BY id`) and score with
`new_rows LEFT JOIN models USING (id)` plus `predict(model, [x1, x2])`. A series with new
rows but no model gets a NULL `yhat` (with an INNER JOIN it is dropped); a series with a
model but no new rows produces no rows.

```sql
CREATE OR REPLACE TABLE sales_models AS
SELECT category, ols_fit_agg(revenue, [units, price]) AS model
FROM sales
GROUP BY category;

SELECT n.category, n.units, round(predict(m.model, [n.units, n.price]), 2) AS yhat
FROM (VALUES ('A', 41.0, 1.0), ('B', 42.0, 2.0), ('Z', 43.0, 3.0)) n(category, units, price)
LEFT JOIN sales_models m USING (category)   -- 'Z' has no model: yhat is NULL
ORDER BY n.category;
```

## Pattern 1b — per-group long tables (`_tidy_by`, `_glance_by`, `_augment_by`, `_prediction_by`, …)

When you want a **report table** per group instead of predictions, use the output macros. They share the `*_fit_predict_by` call shape (`source` is a table-name string, columns unquoted) and return long tables with fixed columns (the integration contract schemas — see **anofox-statistics-plotting**). Every table starts with the group column, `id` (the group as VARCHAR), `model_id` and `model_name` (the method: `'ols'`, `'poisson'`, …), so tables of several models stack with `UNION ALL BY NAME`. `_augment_by` is the exception: it keeps all source columns and adds `model_id` / `model_name` but no `id`. (The curve and test macros below, whose groups are what is compared, set `model_id` to the group value instead.)

| Macro | Signature | Main columns (besides group, `id`, `model_id`, `model_name`) |
|---|---|---|
| `<model>_tidy_by` | `(source, group_col, y_col, x_cols [, weight_col \| event_col] [, options := …, names := […]])` | `term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, index_value` (one row per term, intercept first) |
| `<model>_glance_by` | `(source, group_col, y_col, x_cols [, weight_col \| event_col] [, options := …])` | `model_type, metric, value` (one row per available fit statistic; OLS / WLS / Ridge add `press`, `press_rmse`, `loo_r_squared`) |
| `ols_augment_by`, `ridge_augment_by`, `<glm>_augment_by` | `(source, group_col, y_col, x_cols [, options := …, split := …, order_by := …])` | all source columns + `model_id, model_name, row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params` (+ `loo_residual, loo_fitted` for OLS / WLS / Ridge; + `approx_loo_fitted, approx_loo_residual` for GLMs; Ridge: NULL `std_residual` / `stud_residual` / `cooks_d` / `dffits`) |
| `glmm_tidy_by` | `(source, group_col, y_col, x_cols, re_group [, options := …, names := […], re_names := […]])` | the tidy columns + `effect` (`'fixed'` / `'ran_pars'` / `'ran_vals'`), `re_group`, `re_level`, `df` (Satterthwaite for gaussian) |
| `glmm_glance_by` | `(source, group_col, y_col, x_cols, re_group [, options := …])` | the glance columns (`n_groups`, `sigma`, `df_residual`, `icc`, `var_group`, …) |
| `glmm_augment_by` | `(source, group_col, y_col, x_cols, re_group [, options := …, order_by := …])` | the augment columns + `fitted_population`, `pearson_residual` |
| `wls_augment_by` | `(source, group_col, y_col, x_cols, weight_col [, …])` | as above |
| `<model>_partial_residuals_by` | `(source, group_col, y_col, x_cols [, weight_col] [, options := …, names := […], order_by := …])` | `row_id, term_index, term, x, partial_residual, component, x_residual, y_residual` |
| `<model>_prediction_by` | `(source, group_col, x_col, y_col, x_cols [, weight_col] [, options := …, level := 0.95, split := …])` | `id, model_name, x, y, yhat, yhat_lower, yhat_upper, level, split` (`'train'` / `'test'` / `'future'`) |
| `<model>_effects_by` | `(source, group_col, y_col, x_cols [, weight_col] [, focal := 1, n_points := 50, at_values := NULL, options := …, level := 0.95, names := […]])` | `id, model_name, focal, x, y (NULL), yhat, yhat_lower, yhat_upper, level, split (NULL), se_fit, focal_name` (`names[focal]`, else `'x' \|\| focal` like `_tidy_by` terms) |
| `<model>_rolling_tidy_by` | `(source, group_col, order_col, y_col, x_cols [, weight_col] [, window_size := 30, min_obs := NULL, options := …, names := […]])` | `window_end` + the tidy columns (`index_name = 'window_end'`, `index_value` = window position) + `n_window` |
| `ridge_path_by`, `elasticnet_path_by`, `lars_path_by`, `quantile_process_by` | `(source, group_col, y_col, x_cols [, options := …, names := […]])` | the tidy columns, `index_name` `'lambda'` (`'alpha'` for LARS, `'tau'` for the quantile process) |
| `ridge_cv_by`, `elasticnet_cv_by` | `(source, group_col, y_col, x_cols [, options := …, cv_key := col])` | `curve_type, series, x, y, y_low, y_high, threshold, weight, n_censor, lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds` |
| `ridge_nested_cv_by`, `elasticnet_nested_cv_by` | `(source, group_col, y_col, x_cols [, options := …, cv_key := col])` | `curve_type ('nested_cv'), series ('fold' / 'mean'), x, y, y_low, y_high, threshold, weight, n_censor, fold, lambda, test_error, test_se, inner_cv_mean, inner_cv_se, n_train, n_test, n_inner_folds, n_folds, metric` |
| `cox_fit_by` (new) | `(source, group_col, time_col, x_cols, event_col [, options := …])` | one row per group with the `cox_fit_agg` fields (`coefficients`, `hazard_ratios`, `hr_conf_low/high`, LR / Wald / score tests, `concordance`, `aic`, …) |
| `cox_tidy_by`, `cox_glance_by`, `cox_augment_by` (new) | as `cox_fit_by` (`+ names := […]` / `order_by := …`) | `terms` (log hazard ratios, no intercept row), `summary`, `obs` (`fitted` = linear predictor, `residual` = martingale, + deviance / score / dfbeta / Schoenfeld residuals). No weight column: weights only via `cox_fit_agg` |
| `gls_fit_by` (new) | `(source, group_col, y_col, x_cols, subject_col, time_col [, options := …, variance_col := …])` | one row per group with the `gls_fit_agg` fields (`nlme::gls`; options `correlation`, `variance`, `method`) |
| `gls_tidy_by`, `gls_glance_by`, `gls_augment_by` (new) | as `gls_fit_by` (`+ names := […]`) | `terms` (`effect` `'coefficient'` / `'correlation'` / `'variance'` / `'sigma'`, always with inference), `summary`, `obs` (+ `se_fit`, `normalized_residual`) |
| `aft_augment_by` (new) | `(source, group_col, time_col, x_cols, event_col [, options := …, order_by := …])` | `obs`: `fitted` = linear predictor, `residual` = Cox-Snell, `std_residual` = standardized, `score_contributions` |
| `pspline_prediction_by` (new) | `(source, group_col, x_col, y_col [, options := …, level := 0.95, split := …])` | the `prediction` schema of a per-group P-spline (`model_id` `'pspline'`) |
| `predict_grid_by` (new) | `(source, group_col, x_cols, model, terms, term [, grid_values := …, simultaneous := false, level := 0.95])` | `rms::Predict` grid per group in the `prediction` schema (`model_name` `'predict_grid'`); `model` / `terms` are columns of `source` (a view joining each group's rows with its `effects_design` model) |

Model coverage:
- `_tidy_by` / `_glance_by` (26): `ols`, `wls` (+`weight_col`), `ridge`, `elasticnet`, `lars`, `bayesian_ridge`, `ard`, `passive_aggressive` and `lm_dynamic` (both `(source, group_col, order_col, y_col, x_cols, …)`; `lm_dynamic_tidy_by` = per-row coefficients with `window_end`), `rls` (sequential: pass `order_by := t` by name, else the row order is undefined), `huber`, `ransac`, `theil_sen`, `bls`, `nnls`, `pls`, `quantile`, `poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`, `alm`, `aft` (+`event_col`); `glmm_tidy_by` / `glmm_glance_by` take the random-effects factor `re_group` after `x_cols` (one mixed model per `group_col`, the outer group). Not `isotonic`
- `_augment_by`: `ols`, `wls`, `ridge`, `poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`, `glmm` (+`re_group`), and new `cox`, `gls`, `aft`. `_partial_residuals_by`: the original list without `glmm`.
- `_prediction_by` (24): every `*_fit_predict_by` model incl. the new `pspline` (not `ordinal`, `continuation_ratio`, `multinomial`); `pls`, `quantile`, `isotonic` have no band; `isotonic_prediction_by(source, group_col, x_col, y_col)` has no `x_cols`.
- `_effects_by` (9): `ols`, `wls`, `ridge` and the six GLMs (models whose `predict` has standard errors); GLM bands on the response scale.
- Survival / correlated errors: `cox_{fit,tidy,glance,augment}_by`, `gls_{fit,tidy,glance,augment}_by`, `aft_tidy_by` / `aft_glance_by` / `aft_augment_by` (all take `event_col` or `subject_col, time_col` after `x_cols`).
- `_rolling_tidy_by` (6): `ols`, `wls`, `ridge`, `elasticnet`, `huber`, `poisson`. Windows shorter than `min_obs` (default `window_size`: full windows only) are skipped; `window_end` keeps the type of `order_col`.

Other per-group long tables: `acf_by` / `pacf_by(source, group_col, value_col, order_col[, max_lag := …])`, `kaplan_meier_by(source, strata_col, time_col, event_col[, conf_level := …, conf_type := …])`, `roc_curve_by` / `pr_curve_by` / `calibration_curve_by(source, group_col, y_true_col, score_col …)` (`curve` schema), and `pairwise_t_test_by` / `tukey_hsd_by` / `dunn_test_by(source, group_col, value_col, factor_col, options := NULL)` (`test` schema).

`_tidy_by` and `_rolling_tidy_by` compute inference by default (quantile: `'nid'` standard errors), also when you pass other `options`; `{'compute_inference': false}` turns it off.

```sql
-- Coefficient table and fit statistics per category
SELECT category, term, round(estimate, 3) AS estimate, round(std_error, 4) AS se, p_value
FROM ols_tidy_by('sales', category, revenue, [units, price], names := ['units', 'price'])
ORDER BY category, term;
SELECT * FROM ols_glance_by('sales', category, revenue, [units, price])
WHERE metric IN ('r_squared', 'aic') ORDER BY category, metric;

-- Compare two models per group: stack their glance tables
SELECT * FROM ols_glance_by('sales', category, revenue, [units, price])
UNION ALL BY NAME
SELECT * FROM huber_glance_by('sales', category, revenue, [units, price])
ORDER BY category, metric, model_type;

-- HC3 robust standard errors
SELECT category, term, std_error
FROM ols_tidy_by('sales', category, revenue, [units, price], options := {'hc_type': 'hc3'});

-- Fitted values with a 90% band, positioned on units (prediction schema)
SELECT id, x, y, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS lo, round(yhat_upper, 2) AS hi, split
FROM ols_prediction_by('sales', category, units, revenue, [units, price], level := 0.9)
ORDER BY id, x LIMIT 4;

-- Marginal effect of price, units held at its mean, per category
SELECT id, focal_name, round(x, 2) AS price, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS lo, round(yhat_upper, 2) AS hi
FROM ols_effects_by('sales', category, revenue, [units, price], focal := 2, n_points := 3, names := ['units', 'price'])
ORDER BY id, price;
```

```sql
-- Rolling 12-row coefficients per category, ordered by units
SELECT category, window_end, term, estimate, conf_low, conf_high
FROM ols_rolling_tidy_by('sales', category, units, revenue, [units, price], window_size := 12, names := ['units', 'price'])
WHERE term = 'price' ORDER BY category, window_end;
```

```sql
-- Survival data: one Cox model per site (hazard ratios, terms schema)
CREATE TABLE patients AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS site,
       (40 + (i * 7) % 30)::DOUBLE AS age, (i % 3 = 0)::INT::DOUBLE AS treated,
       (30 + (i * 37) % 300 + 4 * (i % 3 = 0)::INT * (i % 50))::DOUBLE AS days,
       ((i * 7 + age::INT) % 5 < 3)::INT::DOUBLE AS died
FROM range(1, 161) t(i);

SELECT site, term, round(exp(estimate), 3) AS hr, round(p_value, 4) AS p
FROM cox_tidy_by('patients', site, days, [age, treated], died, names := ['age', 'treated'])
ORDER BY site, term;

-- Repeated measures: one GLS model per clinic with AR(1) errors within patient
CREATE TABLE visits AS
SELECT CASE WHEN p % 2 = 0 THEN 'c1' ELSE 'c2' END AS clinic, p AS patient, w AS week,
       (p % 3)::DOUBLE AS dose,
       (10 + 0.5 * w + 1.5 * (p % 3) + ((p * 13 + w * 7) % 5) * 0.4)::DOUBLE AS score
FROM range(1, 21) a(p), range(1, 6) b(w);

SELECT clinic, correlation_params, round(sigma, 3) AS sigma, round(aic, 2) AS aic
FROM gls_fit_by('visits', clinic, score, [dose, week::DOUBLE], patient, week, options := {'correlation': 'ar1'})
ORDER BY clinic;
```

## Pattern 1c — per-group validation, penalization, imputation, drift (new in 0.16.0)

Same `source`-string convention; one row per group and result element.

| Macro | Signature | Purpose / output |
|---|---|---|
| `validate_by` | `(source, group_col, y_col, x_cols, options := …)` | `rms::validate` per group (`'model'`: `'ols'` / `'logistic'`, `'B'`, `'method'` bootstrap / CV / .632): `index_name, index_orig, training, test, optimism, index_corrected, …` |
| `calibrate_by` | `(source, group_col, y_col, x_cols, options := …)` | `rms::calibrate` overfitting-corrected calibration curve (`curve` schema: `x` = predicted, `y` = corrected) |
| `calibrate_cph_by` | `(source, group_col, time_col, x_cols, event_col, options := …)` | `rms::calibrate.cph` (KM vs predicted survival at `'u'`, groups of `'m'`) |
| `bootcov_by` | `(source, group_col, y_col, x_cols, options := …, cluster := col, names := […])` | `rms::bootcov` bootstrap SEs / percentile CIs (`terms` schema); `cluster :=` resamples whole clusters |
| `pentrace_by` | `(source, group_col, y_col, x_cols, options := {'penalties': […], …})` | `rms::pentrace`: one row per penalty with `df`, `aic`, `bic`, `aic_c`, `is_best` |
| `penalized_ols_fit_by` | `(source, group_col, y_col, x_cols, options := {'penalty': …})` | `rms::ols(penalty =)` per group: coefficients, `effective_df`, `sigma`, … |
| `glm_score_test_by` | `(source, group_col, y_col, x_reduced := […], x_add := […], options := {'family': …})` | Rao score test for adding `x_add` to a GLM (`test` schema); only the reduced model is fitted |
| `areg_impute_by` | `(source, group_col, id_col, x_cols, options := …)` | `Hmisc::aregImpute` per group: one row per imputation and id with the completed `x`; constant `group_col` (e.g. `1`) imputes the whole table. Pool fits with `pool_rubin_agg` |
| `pca_by` | `(source, group_col, x_cols, options := {'n_components': k, 'scale': …})` | every source row with its group's PCA `scores DOUBLE[]` |
| `dataset_drift_by` | `(source, group_col, columns, options := …, drift_share := 0.5)` | evidently dataset drift: one row per column name in `columns` (a list of strings). Here `group_col` is the **reference vs current** split, not a segment: loop segments with a `WHERE` / view |

All resampling macros are seeded (`'seed'`) and independent of thread count and row order; keep `'B'` small in exploratory runs.

```sql
-- Bootstrap-corrected discrimination / calibration slope of a logistic model per site
SELECT site, index_name, round(index_orig, 3) AS apparent, round(index_corrected, 3) AS corrected
FROM validate_by('patients', site, died, [age, treated], options := {'model': 'logistic', 'B': 40})
WHERE index_name IN ('Dxy', 'Slope')
ORDER BY site, index_name;

-- Best rms penalty per site (by corrected AIC)
SELECT site, simple AS best_penalty, round(df, 2) AS eff_df
FROM pentrace_by('patients', site, days, [age, treated], options := {'penalties': [0, 1, 4, 16]})
WHERE is_best ORDER BY site;
```

## Pattern 2 — `*_fit_agg` with `GROUP BY` (extract per-group coefficients)

When you want the fitted **model struct per group** (coefficients, R², inference) rather than predictions:

```sql
-- A separate model per category; keep coefficients + fit quality
SELECT category, fit.r_squared AS r2, fit.coefficients AS betas, fit.p_values AS pvals
FROM (
  SELECT category, ols_fit_agg(revenue, [units, price], {'compute_inference': true}) AS fit
  FROM sales
  GROUP BY category
);
```

The same works for every `*_fit_agg`, including the new ones: `cox_fit_agg(time, x, event)`,
`gls_fit_agg(y, x, subject, time, opts)`, `ordinal_fit_agg`, `continuation_ratio_fit_agg`,
`multinomial_fit_agg`, `pspline_fit_agg(y, x)`, `penalized_ols_fit_agg`, `areg_fit_agg`; score
with `predict` / `ordinal_predict` / `multinomial_predict` / `cox_survival`. Cluster-robust SEs per
group: `ols_fit_agg(y, x, cluster, {...})`, `wls_fit_agg(y, x, w, cluster, {...})`, the GLMs with
`{'vcov': 'cluster'}` (`poisson_fit_agg(y, x, cluster, {'vcov': 'cluster'})`) and
`aft_fit_agg(time, x, event, cluster, opts)`.

> Tip: compute the struct once in a subquery/CTE and unpack fields downstream to avoid repeating the `_agg` call — DuckDB evaluates each `(…_agg(...)).field` independently.

The per-group model struct plugs straight into the model tools — score rows with `predict(model, x)`,
report with `tidy` / `glance`:

```sql
-- Fit per category, then score every row with its own category's model
WITH models AS (
  SELECT category, ols_fit_agg(revenue, [units, price]) AS m FROM sales GROUP BY category
)
SELECT s.category, s.units, round(predict(m.m, [s.units, s.price]), 2) AS yhat
FROM sales s JOIN models m USING (category)
ORDER BY s.category, s.units
LIMIT 4;

-- Coefficient table per group (intercept row first) and one summary row per group
SELECT category, unnest(tidy(ols_fit_agg(revenue, [units, price], {'compute_inference': true}),
                             ['units', 'price']), recursive := true)
FROM sales GROUP BY category ORDER BY category;
SELECT category, unnest(glance(ols_fit_agg(revenue, [units, price])))
FROM sales GROUP BY category ORDER BY category;
```

## Pattern 3 — `*_fit_predict_agg` (fit once per group, predictions as a list)

`{model}_fit_predict_agg(y, [x…][, opts])` fits on the group's rows with non-NULL `y` and returns a
`LIST<STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)>` covering every row of the group.
Available for `ols`, `ridge`, `elasticnet`, `wls` (extra weight arg), `rls`, `huber`, `ransac`,
`theil_sen`, `bls`, `alm`, `poisson`, `pls`, `isotonic`, `quantile`, `bayesian_ridge`, `ard`,
`passive_aggressive`, `lm_dynamic` and (new) `pspline` (`pspline_fit_predict_agg(y, x [, split] [, options])`, single `x`). A variant
`{model}_fit_predict_agg(y, x, split_col[, opts])` trains only on rows where `lower(split_col) IN ('train', 'training')`; a NULL split is a prediction row, as in the `_by` macros.

```sql
SELECT category, unnest(ols_fit_predict_agg(revenue, [units, price])) AS p
FROM sales
GROUP BY category
LIMIT 3;
```

## Pattern 4 — Rolling / expanding `*_fit_predict` window functions

`{model}_fit_predict(y, [x…][, opts]) OVER (…)` fits on the window frame and returns
`STRUCT(yhat, yhat_lower, yhat_upper)` for the **last row of the frame**. Available for `ols`, `ridge`,
`elasticnet`, `wls` (extra weight arg), `rls`, `huber`, `ransac`, `theil_sen`, `passive_aggressive`.
The model-specific window aggregates `cox_augment_agg`, `gls_augment_agg` and `aft_augment_agg` (new)
fit the frame and return one STRUCT per frame row; use them over a whole partition, or simply
call `cox_augment_by` / `gls_augment_by` / `aft_augment_by`.

Use frames that end at `CURRENT ROW` over a unique `ORDER BY`. Do **not** use `*_fit_predict … AND 1 PRECEDING`
(that is not a one-step-ahead forecast), `OVER (PARTITION BY g)` without `ORDER BY` (every row
gets the same prediction), or `RANGE` frames with ties. For a one-step-ahead forecast fit the
aggregate on the earlier rows and score the current row:
`predict((ols_fit_agg(y, [x]) OVER (PARTITION BY g ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), [x])`.

```sql
SELECT
  category, units,
  (ols_fit_predict(revenue, [units]) OVER (
    PARTITION BY category ORDER BY units ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  )).yhat AS rolling_pred
FROM sales
ORDER BY category, units
LIMIT 5;

-- One-step-ahead: model from strictly earlier rows, NULL until the frame can be fitted
SELECT
  category, units, revenue,
  predict((ols_fit_agg(revenue, [units]) OVER (
    PARTITION BY category ORDER BY units ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  )), [units]) AS yhat_next
FROM sales
ORDER BY category, units
LIMIT 5;
```

## Scaling & correctness notes

1. **`source` is a table name string, not a CTE.** `*_fit_predict_by` macros expand to a subselect internally — pass `'my_table'`, not `(SELECT …)`. Materialize a CTE to a temp table first if needed.
2. **Degenerate groups return NULL, not errors.** A window frame with too few rows to fit yields `NULL` predictions rather than raising — filter with `WHERE yhat IS NOT NULL` if you need only fitted rows.
3. **Per-group option keys apply uniformly.** The `opts` MAP is applied to every group's fit; there's no per-group option override in one call.
4. **Parallelism is free.** Grouped fits scale across cores via DuckDB's `HASH_GROUP_BY` — the per-fit cost is a few µs (see the extension's benchmark harness). Prefer one grouped query over N single-group queries.
5. **Choose the model per data regime.** For intermittent/lumpy segments classify with `aid_by` first (see **anofox-statistics-diagnostics**); use robust `*_fit_predict_by` (Huber/RANSAC/Theil-Sen) for outlier-heavy segments.

## Why "AI-ready"

An LLM/agent building a forecasting or scoring pipeline can express "train a model for every
segment and score it" as a single declarative SQL statement — no imperative loop, no data
egress from DuckDB, deterministic and reproducible. This is the batch surface to reach for
whenever the unit of work is *a group*, not *a row*.
