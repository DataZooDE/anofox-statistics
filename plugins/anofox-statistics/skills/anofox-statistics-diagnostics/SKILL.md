---
name: anofox-statistics-diagnostics
description: >
  Model diagnostics, residual analysis, multicollinearity, model-selection
  criteria, and demand-pattern classification in the anofox_statistics DuckDB
  extension. Covers Variance Inflation Factor (vif / vif_agg), AIC / BIC model
  selection, residual diagnostics (raw / standardized / studentized residuals,
  leverage) via residuals_diagnostics / residuals_diagnostics_agg, and AID
  demand-pattern and anomaly classification (aid_agg, aid_anomaly_agg, aid_by,
  aid_anomaly_by). Use when checking whether a fitted regression is sound —
  multicollinearity, influential points, residual adequacy — or classifying a
  series' demand regime before choosing a model.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Diagnostics Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 | **DuckDB:** v1.4.5 LTS / v1.5.4+

Diagnostics answer "is this fit trustworthy?" — run them after fitting a model (see **anofox-statistics-regression**).

## Functions

| Function | Type | Purpose |
|---|---|---|
| `vif`, `vif_agg` | scalar / aggregate | Variance Inflation Factor per predictor — detect multicollinearity |
| `aic` | scalar | Akaike Information Criterion for model selection |
| `bic` | scalar | Bayesian Information Criterion for model selection |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | scalar / aggregate | Residual analysis: `raw`, `standardized`, `studentized`, `leverage` arrays |
| `aid_agg`, `aid_by` | aggregate / table macro | Demand-pattern classification (smooth / erratic / intermittent / lumpy) |
| `aid_anomaly_agg`, `aid_anomaly_by` | aggregate / table macro | Anomaly / influence detection on the series |

## Interpretation guidance

- **VIF:** > 5 warns of multicollinearity, > 10 is severe — consider dropping/combining collinear predictors or switching to Ridge/Elastic Net.
- **AIC / BIC:** lower is better; compare *nested or same-data* models. BIC penalizes complexity harder than AIC. GLM result structs already carry `aic` / `bic` fields.
- **Residuals:** `standardized`/`studentized` residuals with |value| > 2–3 flag potential outliers; high `leverage` flags influential x-positions. Structure in residuals ⇒ mis-specified model.
- **AID demand classes:** use before forecasting/choosing a model — intermittent/lumpy series need count/intermittent methods rather than plain OLS.

## Return fields

- `residuals_diagnostics_agg(actual, predicted)` → STRUCT with `raw DOUBLE[]`, `standardized DOUBLE[]`, `studentized DOUBLE[]`, `leverage DOUBLE[]`.
- `vif_agg(y, [x1, x2, …])` → per-predictor VIF array.

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

```sql skip
-- Multicollinearity check and model-selection comparison
SELECT vif_agg(y, [x1, x2, x3]) AS vifs FROM design;

-- AID demand-pattern classification per SKU
SELECT sku, (aid_agg(demand)).* FROM sales GROUP BY sku;
```
