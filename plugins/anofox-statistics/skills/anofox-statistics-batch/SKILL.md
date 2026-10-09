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
  pls_fit_predict_by, isotonic_fit_predict_by, quantile_fit_predict_by)
  with split := and order_by :=, per-group aggregate fitting with GROUP BY
  plus predict / tidy / glance on the per-group model structs,
  the *_fit_predict_agg aggregates, rolling *_fit_predict window
  functions, the glmm_fit_by / eb_shrink_by /
  aid_by macros, and scaling patterns for fitting a separate model per
  segment / SKU / cohort over large panels. Use when the task is "fit one
  model per group" at scale rather than a single global fit.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Batch / Per-Group Fitting Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 | **DuckDB:** v1.4.5 LTS / v1.5.x

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
`quantile_fit_predict_by`, and the GLM macros `binomial_fit_predict_by`,
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
