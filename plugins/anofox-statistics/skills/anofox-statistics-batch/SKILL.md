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
  passive_aggressive_fit_predict_by, lm_dynamic_fit_predict_by)
  with split := and order_by :=, the per-group long-table macros
  <model>_tidy_by and <model>_glance_by (25 models), ols/wls/GLM
  _augment_by, <model>_partial_residuals_by, <model>_prediction_by (23
  models), <model>_effects_by (marginal effects), <model>_rolling_tidy_by
  (rolling coefficients), ridge/elasticnet/lars_path_by,
  ridge/elasticnet_cv_by, quantile_process_by, acf_by / pacf_by,
  kaplan_meier_by, roc/pr/calibration_curve_by and
  pairwise_t_test_by / tukey_hsd_by / dunn_test_by, per-group aggregate
  fitting with GROUP BY plus predict / tidy / glance on the per-group model
  structs, the *_fit_predict_agg aggregates, rolling *_fit_predict window
  functions, the glmm_fit_by / eb_shrink_by / aid_by macros, and scaling
  patterns for fitting a separate model per segment / SKU / cohort over
  large panels. Use when the task is "fit one model per group" at scale, or
  "give me a coefficient / fit / residual / prediction table per group",
  rather than a single global fit.
version: 0.12.2
user-invocable: false
---

# Anofox Statistics — Batch / Per-Group Fitting Cheat Sheet

**Extension:** `anofox_statistics` v0.12.2 | **DuckDB:** v1.4.5 LTS / v1.5.x

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
  split      VARCHAR,   -- optional: column name of a train/test split; only rows where it is 'train' (or NULL) are trained on
  order_by   COLUMN     -- optional (by name): orders each group's rows; deterministic row alignment, and the feed order for RLS
) -> TABLE(<all source columns>, yhat, yhat_lower, yhat_upper, is_training)
```

Pass `split` and `order_by` by name: `ols_fit_predict_by('sales', category, revenue, [units], split := sp, order_by := t)`.

Rows with a NULL `y_col` are not used for fitting but still receive a prediction, so appending future rows with `y = NULL` gives you a forecast.

Available macros: `ols_fit_predict_by`, `huber_fit_predict_by`, `ransac_fit_predict_by`,
`theil_sen_fit_predict_by`, `ridge_fit_predict_by`, `elasticnet_fit_predict_by`,
`wls_fit_predict_by`, `rls_fit_predict_by`, `bls_fit_predict_by`, `alm_fit_predict_by`,
`poisson_fit_predict_by`, `pls_fit_predict_by`, `isotonic_fit_predict_by`,
`quantile_fit_predict_by`, `bayesian_ridge_fit_predict_by`, `ard_fit_predict_by` (posterior predictive band), `passive_aggressive_fit_predict_by`, `lm_dynamic_fit_predict_by` (sequential: pass `order_by := t`), and the GLM macros `binomial_fit_predict_by`,
`logistic_fit_predict_by`, `negbinom_fit_predict_by`, `gamma_fit_predict_by`,
`tweedie_fit_predict_by`. Related grouped macros: `glmm_fit_by(source, group_col, y_col, x_cols[, options])`,
`eb_shrink_by(source, estimate_col, se_col[, options])`, `aid_by`, `aid_anomaly_by`.

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
| `<model>_glance_by` | `(source, group_col, y_col, x_cols [, weight_col \| event_col] [, options := …])` | `model_type, metric, value` (one row per available fit statistic; OLS / WLS add `press`, `press_rmse`, `loo_r_squared`) |
| `ols_augment_by`, `<glm>_augment_by` | `(source, group_col, y_col, x_cols [, options := …, split := …, order_by := …])` | all source columns + `model_id, model_name, row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params` (+ `loo_residual, loo_fitted` for OLS / WLS) |
| `wls_augment_by` | `(source, group_col, y_col, x_cols, weight_col [, …])` | as above |
| `<model>_partial_residuals_by` | `(source, group_col, y_col, x_cols [, weight_col] [, options := …, names := […], order_by := …])` | `row_id, term_index, term, x, partial_residual, component, x_residual, y_residual` |
| `<model>_prediction_by` | `(source, group_col, x_col, y_col, x_cols [, weight_col] [, options := …, level := 0.95, split := …])` | `id, model_name, x, y, yhat, yhat_lower, yhat_upper, level, split` (`'train'` / `'test'` / `'future'`) |
| `<model>_effects_by` | `(source, group_col, y_col, x_cols [, weight_col] [, focal := 1, n_points := 50, at_values := NULL, options := …, level := 0.95, names := […]])` | `id, model_name, focal, x, y (NULL), yhat, yhat_lower, yhat_upper, level, split (NULL), se_fit, focal_name` (`names[focal]`, else `'x' \|\| focal` like `_tidy_by` terms) |
| `<model>_rolling_tidy_by` | `(source, group_col, order_col, y_col, x_cols [, weight_col] [, window_size := 30, min_obs := NULL, options := …, names := […]])` | `window_end` + the tidy columns (`index_name = 'window_end'`, `index_value` = window position) + `n_window` |
| `ridge_path_by`, `elasticnet_path_by`, `lars_path_by`, `quantile_process_by` | `(source, group_col, y_col, x_cols [, options := …, names := […]])` | the tidy columns, `index_name` `'lambda'` (`'alpha'` for LARS, `'tau'` for the quantile process) |
| `ridge_cv_by`, `elasticnet_cv_by` | `(source, group_col, y_col, x_cols [, options := …])` | `curve_type, series, x, y, y_low, y_high, threshold, weight, n_censor, lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds` |

Model coverage:
- `_tidy_by` / `_glance_by` (25): `ols`, `wls` (+`weight_col`), `ridge`, `elasticnet`, `lars`, `bayesian_ridge`, `ard`, `passive_aggressive` and `lm_dynamic` (both `(source, group_col, order_col, y_col, x_cols, …)`; `lm_dynamic_tidy_by` = per-row coefficients with `window_end`), `rls`, `huber`, `ransac`, `theil_sen`, `bls`, `nnls`, `pls`, `quantile`, `poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`, `alm`, `aft` (+`event_col`). Not `glmm`, `isotonic`.
- `_augment_by` and `_partial_residuals_by`: `ols`, `wls`, `poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`.
- `_prediction_by` (23): every `*_fit_predict_by` model; `pls`, `quantile`, `isotonic` have no band; `isotonic_prediction_by(source, group_col, x_col, y_col)` has no `x_cols`.
- `_effects_by` (9): `ols`, `wls`, `ridge` and the six GLMs (models whose `predict` has standard errors); GLM bands on the response scale.
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
`theil_sen`, `bls`, `alm`, `poisson`, `pls`, `isotonic`, `quantile`. A variant
`{model}_fit_predict_agg(y, x, split_col[, opts])` trains only on rows where `split_col = 'train'`.

```sql
SELECT category, unnest(ols_fit_predict_agg(revenue, [units, price])) AS p
FROM sales
GROUP BY category
LIMIT 3;
```

## Pattern 4 — Rolling / expanding `*_fit_predict` window functions

`{model}_fit_predict(y, [x…][, opts]) OVER (…)` fits on the window frame and returns
`STRUCT(yhat, yhat_lower, yhat_upper)` for the **last row of the frame**. Available for `ols`, `ridge`,
`elasticnet`, `wls` (extra weight arg), `rls`, `huber`, `ransac`, `theil_sen`.

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
