---
name: anofox-statistics-batch
description: >
  AI-ready batch and per-group model fitting for the anofox_statistics DuckDB
  extension — training thousands of regression models in a single SQL query.
  Covers the *_fit_predict_by table macros (ols_fit_predict_by,
  huber_fit_predict_by, ransac_fit_predict_by, theil_sen_fit_predict_by,
  ridge_fit_predict_by, elasticnet_fit_predict_by, wls_fit_predict_by,
  rls_fit_predict_by, bls_fit_predict_by, alm_fit_predict_by,
  poisson_fit_predict_by, pls_fit_predict_by, isotonic_fit_predict_by,
  quantile_fit_predict_by), per-group aggregate fitting with GROUP BY,
  rolling/windowed *_fit_predict_agg, the glmm_fit_by / eb_shrink_by /
  aid_by macros, and scaling patterns for fitting a separate model per
  segment / SKU / cohort over large panels. Use when the task is "fit one
  model per group" at scale rather than a single global fit.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Batch / Per-Group Fitting Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 | **DuckDB:** v1.4.5 LTS / v1.5.4+

The extension's superpower for AI/analytics pipelines: **fit one model per group in a single SQL statement**, using DuckDB's `HASH_GROUP_BY` for parallelism — no Python loop, no per-group round-trips. Three patterns, from highest-level to lowest.

## Pattern 1 — `*_fit_predict_by` table macros (recommended)

Fit + predict per group with one call. The macro renames the target and adds prediction columns.

```sql
{model}_fit_predict_by(
  source     VARCHAR,   -- table name (quoted string, NOT a CTE/subquery)
  group_col  COLUMN,    -- segment identifier (unquoted)
  y_col      COLUMN,    -- target (unquoted)
  x_cols     LIST,      -- feature columns, e.g. [x1, x2]
  options    MAP        -- optional model options (defaults to NULL)
) -> TABLE(group_col, …, yhat)   -- predictions land in the `yhat` column
```

Available macros: `ols_fit_predict_by`, `huber_fit_predict_by`, `ransac_fit_predict_by`,
`theil_sen_fit_predict_by`, `ridge_fit_predict_by`, `elasticnet_fit_predict_by`,
`wls_fit_predict_by`, `rls_fit_predict_by`, `bls_fit_predict_by`, `alm_fit_predict_by`,
`poisson_fit_predict_by`, `pls_fit_predict_by`, `isotonic_fit_predict_by`,
`quantile_fit_predict_by`. Related grouped macros: `glmm_fit_by`, `eb_shrink_by`,
`aid_by`, `aid_anomaly_by`.

**Signature variants:** `wls_fit_predict_by` takes an extra `weight_col` before `options`
(`…, x_cols, weight_col, options`); `isotonic_fit_predict_by` takes a single `x_col`
(not a list). All others follow the 4-arg `(source, group_col, y_col, x_cols[, options])` shape.

```sql skip
-- One OLS model per product category, fit + in-sample predictions, in one query
SELECT category, yhat FROM ols_fit_predict_by('sales', category, revenue, [units, price]);
```

## Pattern 2 — `*_fit_agg` with `GROUP BY` (extract per-group coefficients)

When you want the fitted **model struct per group** (coefficients, R², inference) rather than predictions:

```sql skip
-- A separate model per category; keep coefficients + fit quality
SELECT
  category,
  (ols_fit_agg(revenue, [units, price], {'compute_inference': true})).r_squared        AS r2,
  (ols_fit_agg(revenue, [units, price], {'compute_inference': true})).coefficients     AS betas,
  (ols_fit_agg(revenue, [units, price], {'compute_inference': true})).p_values         AS pvals
FROM sales
GROUP BY category;
```

> Tip: compute the struct once in a subquery/CTE and unpack fields downstream to avoid repeating the `_agg` call — DuckDB evaluates each `(…_agg(...)).field` independently.

## Pattern 3 — Rolling / windowed `*_fit_predict_agg`

Use the aggregate as a window function for rolling regressions:

```sql skip
SELECT
  ds,
  ols_fit_predict_agg(y, [x]) OVER (
    PARTITION BY series ORDER BY ds ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) AS rolling_pred
FROM panel;
```

## Scaling & correctness notes

1. **`source` is a table name string, not a CTE.** `*_fit_predict_by` macros expand to a subselect internally — pass `'my_table'`, not `(SELECT …)`. Materialize a CTE to a temp table first if needed.
2. **Degenerate groups return NULL, not errors.** A group with fewer than `n_features + 1` rows (or a degenerate rolling window) yields `NULL` predictions rather than raising — filter with `WHERE yhat IS NOT NULL` if you need only fitted groups.
3. **Per-group option keys apply uniformly.** The `opts` MAP is applied to every group's fit; there's no per-group option override in one call.
4. **Parallelism is free.** Grouped fits scale across cores via DuckDB's `HASH_GROUP_BY` — the per-fit overhead is ~µs (see the extension's benchmark harness). Prefer one grouped query over N single-group queries.
5. **Choose the model per data regime.** For intermittent/lumpy segments classify with `aid_by` first (see **anofox-statistics-diagnostics**); use robust `*_fit_predict_by` (Huber/RANSAC/Theil-Sen) for outlier-heavy segments.

## Why "AI-ready"

An LLM/agent building a forecasting or scoring pipeline can express "train a model for every
segment and score it" as a single declarative SQL statement — no imperative loop, no data
egress from DuckDB, deterministic and reproducible. This is the batch surface to reach for
whenever the unit of work is *a group*, not *a row*.
