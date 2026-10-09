---
name: anofox-statistics-diagnostics
description: >
  Model diagnostics, residual analysis, multicollinearity, model-selection
  criteria, and demand-pattern classification in the anofox_statistics DuckDB
  extension. Covers Variance Inflation Factor (vif / vif_agg), AIC / BIC model
  selection, residual diagnostics (raw / standardized / studentized residuals,
  leverage) via residuals_diagnostics / residuals_diagnostics_agg, and AID
  demand-pattern and anomaly classification (aid_agg, aid_anomaly_agg, aid_by,
  aid_anomaly_by), plus the model report tools tidy (per-term estimates and
  inference) and glance (one-row fit summary) and model-aware predict for
  in-sample fitted values. Use when checking whether a fitted regression is sound —
  multicollinearity, influential points, residual adequacy — or classifying a
  series' demand regime before choosing a model.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Diagnostics Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 | **DuckDB:** v1.4.5 LTS / v1.5.x

Diagnostics answer "is this fit trustworthy?" — run them after fitting a model (see **anofox-statistics-regression**).

## Functions

| Function | Type | Purpose |
|---|---|---|
| `vif`, `vif_agg` | scalar / aggregate | Variance Inflation Factor per predictor — detect multicollinearity |
| `aic(rss, n, k)` | scalar | Akaike Information Criterion for model selection |
| `bic(rss, n, k)` | scalar | Bayesian Information Criterion for model selection |
| `jarque_bera(values[])`, `jarque_bera_agg(value)` | scalar / aggregate | Jarque-Bera normality test (`statistic`, `p_value`, `skewness`, `kurtosis`, `n`) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | scalar / aggregate | Residual analysis: `raw`, `standardized`, `studentized`, `leverage` arrays |
| `tidy(model[, names])` | scalar on a model STRUCT | Per-term table: `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` (intercept row first, `'(Intercept)'`) — `unnest(…, recursive := true)` |
| `glance(model)` | scalar on a model STRUCT | One-row summary of the model's scalar fields (`r_squared`, `aic`, `n_observations`, GLM `family`/`link`, …) — `unnest(glance(m))` |
| `predict(model, x)` | scalar on a model STRUCT | Fitted value for one row; use it to build `y_hat` for the residual diagnostics |
| `aid_agg`, `aid_by` | aggregate / table macro | Demand classification: `demand_type` is `regular` or `intermittent`, plus distribution, zero proportion, stockout / new / obsolete product and outlier counts |
| `aid_anomaly_agg`, `aid_anomaly_by` | aggregate / table macro | Per-observation flags: `stockout`, `new_product`, `obsolete_product`, `high_outlier`, `low_outlier` |

## Interpretation guidance

- **VIF:** > 5 warns of multicollinearity, > 10 is severe — consider dropping/combining collinear predictors or switching to Ridge/Elastic Net.
- **AIC / BIC:** lower is better; compare *nested or same-data* models. BIC penalizes complexity harder than AIC. GLM result structs already carry an `aic` field; AFT and ALM results carry both `aic` and `bic`.
- **Residuals:** `standardized`/`studentized` residuals with |value| > 2–3 flag potential outliers; high `leverage` flags influential x-positions. Structure in residuals ⇒ mis-specified model.
- **`tidy` inference columns** are NULL when the model has no inference (fit without `compute_inference`, or PLS/quantile/LARS); `ols_fit_agg` gives inference for slopes only, so its intercept row is NULL. Works per group: `SELECT g, unnest(tidy(ols_fit_agg(y, [x], {'compute_inference': true})), recursive := true) … GROUP BY g`.
- **AID demand classes:** use before forecasting/choosing a model — intermittent series (many zeros; threshold `intermittent_threshold`, default 0.3) need count/intermittent methods rather than plain OLS.

## Return fields

- `residuals_diagnostics_agg(y, y_hat[, x])` → STRUCT with `raw DOUBLE[]`, `standardized DOUBLE[]`, `studentized DOUBLE[]`, `leverage DOUBLE[]` (leverage needs the `x` argument). In the current build the aggregate returns NULL `standardized` / `studentized`; for those use the scalar full form `residuals_diagnostics(y[], y_hat[], x[][], residual_std_error, true)` (x column-major).
- `vif_agg([x1, x2, …])` → per-predictor VIF array (`vif(X)` is the scalar form on column-major arrays).

## Worked examples

```sql
-- Residual diagnostics from in-sample OLS predictions
WITH preds AS (
  SELECT
    unnest([120.0,155.0,190.0,225.0,265.0,300.0])::DOUBLE AS actual,
    unnest(predict(
      [[50.0,65.0,80.0,95.0,110.0,125.0]]::DOUBLE[][],
      (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
               [[50.0,65.0,80.0,95.0,110.0,125.0]])).coefficients,
      (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
               [[50.0,65.0,80.0,95.0,110.0,125.0]])).intercept
    )) AS yhat
)
SELECT (residuals_diagnostics_agg(actual, yhat)).raw AS raw_residuals FROM preds;
```

```sql
-- Same residual check with an aggregate fit: fitted values from predict(model, x)
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
  (50.0,120.0),(65.0,155.0),(80.0,190.0),(95.0,225.0),(110.0,265.0),(125.0,300.0)
) t(sqm, price);
WITH m AS (SELECT ols_fit_agg(price, [sqm]) AS fit FROM houses)
SELECT (residuals_diagnostics_agg(price, predict(fit, [sqm]))).raw AS raw_residuals
FROM houses, m;

-- Coefficient table with inference and a one-row model summary
SELECT unnest(tidy(ols_fit_agg(price, [sqm], {'compute_inference': true}), ['sqm']), recursive := true)
FROM houses;
SELECT unnest(glance(ols_fit_agg(price, [sqm]))) FROM houses;
```

```sql
-- Multicollinearity check (x3 is nearly x1 + x2)
SELECT vif_agg([x1, x2, x3]) AS vifs
FROM (SELECT i::DOUBLE AS x1, (i % 7)::DOUBLE AS x2, (i + (i % 7) + (i % 2) * 0.1)::DOUBLE AS x3
      FROM range(1, 51) t(i));

-- Model selection from RSS
SELECT aic(12.5, 100, 3) AS aic, bic(12.5, 100, 3) AS bic;

-- AID demand-pattern classification per SKU
SELECT sku, (aid_agg(demand)).demand_type AS demand_type
FROM (SELECT 'sku' || (i % 2) AS sku,
             CASE WHEN i % 2 = 0 THEN 10.0 + (i % 4) WHEN i % 5 = 0 THEN 3.0 ELSE 0.0 END AS demand
      FROM range(1, 61) t(i))
GROUP BY sku ORDER BY sku;
```
