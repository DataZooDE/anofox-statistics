---
name: anofox-statistics-plotting
description: >
  Plot-ready outputs of the anofox_statistics DuckDB extension: the anofox
  integration contract's long-table schemas (terms, summary, obs,
  prediction, curve, test), which table macro produces each one, their
  columns, how to discover producers with duckdb_functions() tags
  (anofox.contract, anofox.output) and anofox_contract_version(), and how
  to hand the tables to the anofox-visualization extension
  (anofox_plot_terms, anofox_plot_prediction, anofox_plot_curve,
  anofox_plot_summary, anofox_plot_obs, anofox_plot_diagnostics,
  anofox_plot). Covers coefficient forests (tidy_by), model comparison
  (glance_by), residual / influence plots (augment_by), fitted lines with
  bands and marginal effects (prediction_by, effects_by), rolling
  coefficients (rolling_tidy_by), regularisation paths and CV curves,
  quantile process, ROC / PR / calibration curves, Kaplan-Meier curves,
  ACF / PACF and post-hoc test tables. Use when the user wants to make a
  model or test result plottable, asks "which table do I need for this
  chart", wants long / tidy tables instead of STRUCTs, or wants to combine
  several models or groups into one plot.
version: 0.12.1
user-invocable: false
---

# Anofox Statistics — Plot-Ready Outputs Cheat Sheet

**Extension:** `anofox_statistics` v0.12.1 | **DuckDB:** v1.4.5 LTS / v1.5.x | **Contract version:** `anofox_contract_version()` = `'1'`

Fit functions return STRUCTs and LISTs. For charts, the `_by` table macros return **long tables with fixed columns** (one row per mark), the schemas of the anofox integration contract shared by anofox-statistics, anofox-forecast and the plotting extension anofox-visualization. Optional columns are always present (NULL when not computed) and the group column is kept.

**Identifiers.** Macros that fit one of the extension's models (`_tidy_by`, `_glance_by`, `_partial_residuals_by`, `_rolling_tidy_by`, `_path_by`, `_cv_by`, `quantile_process_by`, `_prediction_by`, `_effects_by`) return the group column, `id` (the group as VARCHAR), `model_id` and `model_name` (the method: `'ols'`, `'poisson'`, …), in that order, before the schema columns. `_augment_by` keeps all source columns and adds `model_id` / `model_name` but **no `id`** (a source column named `id` would collide). Macros whose groups are the things compared (`roc_curve_by`, `pr_curve_by`, `calibration_curve_by`, `acf_by`, `pacf_by`, `kaplan_meier_by`, `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by`) set `model_id` = the group value.

## Which table do I need?

| Question / chart | Macro (per group of `source`) | Schema | anofox-visualization |
|---|---|---|---|
| Coefficients with intervals (forest plot), several models | `<model>_tidy_by` | `terms` | `anofox_plot_terms` |
| How coefficients change over time | `<model>_rolling_tidy_by` (`ols`, `wls`, `ridge`, `elasticnet`, `huber`, `poisson`) | `terms` (`index_name = 'window_end'`) | `anofox_plot_terms` (one panel per term over the index) |
| Regularisation path | `ridge_path_by`, `elasticnet_path_by`, `lars_path_by` | `terms` (`'lambda'`; `'alpha'` for LARS) | `anofox_plot_terms` |
| Effect across quantiles | `quantile_process_by` | `terms` (`'tau'`) | `anofox_plot_terms` |
| Compare fit statistics (R², AIC, …) across models | `<model>_glance_by` | `summary` | `anofox_plot_summary` |
| Residuals vs fitted, QQ, leverage, Cook's distance | `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` | `obs` | `anofox_plot_obs`, `anofox_plot_diagnostics` (plot.lm 2×2) |
| Fitted line / band, train vs test vs future | `<model>_prediction_by` (23 models) | `prediction` | `anofox_plot_prediction` |
| Marginal effect of one feature (effect plot) | `<model>_effects_by` (`ols`, `wls`, `ridge`, six GLMs) | `prediction` | `anofox_plot_prediction` |
| ROC / PR / calibration curve | `roc_curve_by`, `pr_curve_by`, `calibration_curve_by` | `curve` | `anofox_plot_curve` |
| CV error over lambda (`lambda.min`, `lambda.1se`) | `ridge_cv_by`, `elasticnet_cv_by` | `curve` (`'lambda_cv'`) | `anofox_plot_curve` |
| Survival curves per stratum | `kaplan_meier_by` | `curve` (`'km'`) | `anofox_plot_curve` |
| Residual autocorrelation | `acf_by`, `pacf_by` | `curve` (`'acf'` / `'pacf'`) | `anofox_plot_curve` |
| Scatterplot smoother | `lowess_by(source, group_col, x_col, y_col[, frac, iterations])` | `curve` (`'lowess'`; not yet in the contract's curve_type vocabulary) | `anofox_plot_curve` |
| Time-varying coefficients | `lm_dynamic_tidy_by(source, group_col, order_col, y_col, x_cols)` | `terms` (`'window_end'`, as rolling coefficients) | `anofox_plot_terms` |
| Pairwise comparisons (significance brackets) | `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by` | `test` | no dedicated plot macro |
| Partial-residual / added-variable plot | `<model>_partial_residuals_by` (`ols`, `wls`, six GLMs) | not a contract schema | plot `x` vs `partial_residual` (or `x_residual` vs `y_residual`) with a generic scatter |

Signatures of these macros are in the **anofox-statistics-batch** skill; the source is always a table-name string (`'my_table'`), columns are unquoted.

## Schemas (columns after the group column)

The model macros also carry `id` (not `_augment_by`) and `model_name` next to `model_id` (see **Identifiers** above).

| Schema | Columns |
|---|---|
| `terms` | `model_id, term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, index_value` (rolling adds `window_end` before `term` and `n_window` at the end) |
| `summary` | `model_id, model_type, metric, value` |
| `obs` | all source columns + `model_id, model_name, row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params` |
| `prediction` | `id, model_id, model_name, x, y, yhat, yhat_lower, yhat_upper, level, split` (`effects_by` adds `focal`, `se_fit` and `focal_name`, the axis label: `names[focal]` with `names := [...]`, else `'x' || focal`) |
| `curve` | `model_id, series, curve_type, x, y, y_low, y_high, threshold, weight, n_censor` (+ `n_risk, n_event, std_err` for KM; `lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds` for CV) |
| `test` | `model_id, test_id, method, statistic, p_value, group1, group2, p_adj, df1, df2, estimate, conf_low, conf_high, conf_level, effect_size, effect_name, alternative, p_adjust_method` |

What the `prediction` columns mean: `x` is the `x_col` you pass (time or the main feature), `id` is the group as text (one panel per id), `model_name` is the method (`'ols'`, …) so models overlay, `split` is `'train'`, `'test'` (held out with `split :=`) or `'future'` (NULL `y`), `level` is NULL for models without a band (PLS, quantile, isotonic).

## Discovery

```sql
SELECT anofox_contract_version();   -- '1'

-- Every producer and the schema it emits
SELECT DISTINCT function_name, tags['anofox.output'] AS output
FROM duckdb_functions()
WHERE tags['anofox.contract'] = '1'
ORDER BY output, function_name;
```

All functions of this extension also carry `tags['anofox.family'] = 'statistics'`. The test aggregates (`t_test_agg`, …) are tagged `anofox.output = 'test'`: they return the `test` fields as a STRUCT, not a table.

## Gotchas

1. **One `curve_type` per curve plot.** ROC and calibration rows share columns (so `UNION ALL` works) but are plotted separately: `GROUP BY curve_type` or filter.
2. **Combine models with `UNION ALL BY NAME`.** Same schema, same columns; `model_id` tells the methods apart and `id` the groups. Filter or facet on `id` (or the group column) when one plot should show one group.
3. **LARS paths are on a different scale.** `lars_path_by` has `index_name = 'alpha'` (upstream's LARS alpha), the ridge / elastic-net paths `'lambda'` (glmnet's). Do not overlay them without rescaling.
4. **`*_fit_predict_by` is not the `prediction` schema.** It keeps the source columns and adds `yhat`, `yhat_lower`, `yhat_upper`, `is_training`; use `<model>_prediction_by` for the contract columns (`x`, `id`, `split`, `level`).
5. **STRUCT outputs to long rows:** `unnest(tidy(model), recursive := true)` gives `terms` columns for one model; `unnest(acf_agg(resid, 12 ORDER BY t), recursive := true)` gives the ACF points. The `_by` macros do this per group for you.
6. **`lambda` is a SQL keyword:** quote `"lambda"` in the CV tables, or use `x`.

## Worked examples

```sql
CREATE OR REPLACE TABLE plot_demo AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS grp, i::DOUBLE AS x1,
       ((i * 3) % 7)::DOUBLE AS x2, 1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) AS y,
       round(exp(0.2 + 0.05 * i))::DOUBLE AS y_count, (sin(i) + 0.05 * i > 0.5) AS label,
       1.0 / (1.0 + exp(-(sin(i) + 0.05 * i - 0.5))) AS prob
FROM range(1, 41) t(i);

-- terms: two models' coefficients in one table (model_id = the method)
SELECT * FROM ols_tidy_by('plot_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
UNION ALL BY NAME
SELECT * FROM huber_tidy_by('plot_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
ORDER BY grp, term, model_id;

-- summary: R² and AIC per group
SELECT grp, metric, round(value, 4) AS value
FROM ols_glance_by('plot_demo', grp, y, [x1, x2]) WHERE metric IN ('r_squared', 'aic');

-- obs: the most influential rows
SELECT grp, row_id, round(fitted, 3) AS fitted, round(residual, 3) AS residual, round(cooks_d, 3) AS cooks_d
FROM ols_augment_by('plot_demo', grp, y, [x1, x2])
ORDER BY cooks_d DESC LIMIT 3;

-- prediction: fitted line and 90% band over id, held-out rows marked 'test'
SELECT id, x, y, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS lo, round(yhat_upper, 2) AS hi, split
FROM ols_prediction_by('plot_demo', grp, id, y, [x1, x2], level := 0.9,
                       split := CASE WHEN id > 36 THEN 'test' ELSE 'train' END)
WHERE x > 34 ORDER BY id, x;

-- prediction: expected count as x1 varies, x2 at its mean (GLM band on the response scale)
SELECT id, focal_name, round(x, 1) AS x1, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS lo, round(yhat_upper, 2) AS hi
FROM poisson_effects_by('plot_demo', grp, y_count, [x1, x2], focal := 1, n_points := 3, names := ['x1', 'x2'])
ORDER BY id, x1;

-- curve: ROC and calibration curves of a classifier per group (plot each curve_type separately)
SELECT * FROM roc_curve_by('plot_demo', grp, label, prob)
UNION ALL
SELECT * FROM calibration_curve_by('plot_demo', grp, label, prob, n_bins := 4);

-- test: Tukey HSD of y across the x2 levels, per group
SELECT grp, test_id, round(estimate, 3) AS estimate, round(p_adj, 4) AS p_adj
FROM tukey_hsd_by('plot_demo', grp, y, x2::BIGINT) ORDER BY grp, test_id LIMIT 4;
```

## Hand-off to anofox-visualization

anofox-visualization is a separate DuckDB extension that renders SQL results to SVG. Its contract macros are aggregates over **whole rows**: alias the table and pass the alias. They are not part of this extension; load it first.

```sql skip
LOAD anofox_visualization;

-- Coefficient forest for group A: OLS and Huber dodged and coloured by model_id
SELECT anofox_plot_terms(t)
FROM (SELECT * FROM ols_tidy_by('plot_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
      UNION ALL BY NAME
      SELECT * FROM huber_tidy_by('plot_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])) t
WHERE grp = 'A';

-- Fit + band, one panel per group (id), coloured by split
SELECT anofox_plot_prediction(p)
FROM ols_prediction_by('plot_demo', grp, id, y, [x1, x2], level := 0.9) p;

-- One curve plot per curve_type
SELECT curve_type, anofox_plot_curve(c)
FROM (SELECT * FROM roc_curve_by('plot_demo', grp, label, prob)
      UNION ALL SELECT * FROM calibration_curve_by('plot_demo', grp, label, prob, n_bins := 4)) c
GROUP BY curve_type;

-- plot.lm 2x2 for group A
SELECT anofox_plot_diagnostics(o) FROM ols_augment_by('plot_demo', grp, y, [x1, x2]) o WHERE grp = 'A';

-- Fit statistics of two models side by side
SELECT anofox_plot_summary(s)
FROM (SELECT * FROM ols_glance_by('plot_demo', grp, y, [x1, x2])
      UNION ALL BY NAME SELECT * FROM huber_glance_by('plot_demo', grp, y, [x1, x2])) s
WHERE grp = 'A';
```

Its plot macros: `anofox_plot_terms`, `anofox_plot_prediction(tbl, x := NULL, y := NULL, facet := NULL)`, `anofox_plot_curve`, `anofox_plot_summary`, `anofox_plot_obs(tbl, kind := 'resid_fitted' | 'qq' | 'scale_location' | 'leverage' | 'cooks', label_top := 3)`, `anofox_plot_diagnostics` and `anofox_plot` (picks the chart from the columns); all take `width :=` / `height :=`. It has no macro for the `test` schema. For a `*_fit_predict_by` result name the position column: `anofox_plot_prediction(p, x := 'x1', facet := 'grp')`.
