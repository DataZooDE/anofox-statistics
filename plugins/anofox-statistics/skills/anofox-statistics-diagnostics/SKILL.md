---
name: anofox-statistics-diagnostics
description: >
  Model diagnostics, residual analysis, influence, multicollinearity,
  model-selection criteria, residual autocorrelation / heteroskedasticity
  checks, and demand-pattern classification in the anofox_statistics DuckDB
  extension. Covers per-row influence tables (ols_augment_by, wls_augment_by,
  poisson/binomial/logistic/negbinom/gamma/tweedie_augment_by: fitted,
  residual, std / studentized residuals, leverage, Cook's distance, DFFITS),
  partial residuals and added-variable data (<model>_partial_residuals_by,
  car::crPlots / avPlots), Variance Inflation Factor (vif / vif_agg), AIC /
  BIC, residual diagnostics (residuals_diagnostics /
  residuals_diagnostics_agg), residual ACF / PACF with the white-noise band
  (acf_agg, pacf_agg, acf_by, pacf_by), Ljung-Box / Box-Pierce, Durbin-Watson
  (with p-value), Breusch-Pagan and White heteroskedasticity tests, AID
  demand-pattern and anomaly classification (aid_agg, aid_anomaly_agg, aid_by,
  aid_anomaly_by), plus tidy / glance and model-aware predict for fitted
  values. Also leave-one-out diagnostics: loo_residual / loo_fitted in
  augment_by, PRESS, press_rmse and loo_r_squared in glance_by (compute_loo).
  Use when checking whether a fitted regression is sound — influential points,
  outliers, nonlinearity, multicollinearity, autocorrelated or heteroskedastic
  residuals — or classifying a series' demand regime before choosing a model.
version: 0.13.0
user-invocable: false
---

# Anofox Statistics — Diagnostics Cheat Sheet

**Extension:** `anofox_statistics` v0.13.0 | **DuckDB:** v1.5.6 / v2.0

Diagnostics answer "is this fit trustworthy?" — run them after fitting a model (see **anofox-statistics-regression**).

## Functions

| Function | Type | Purpose |
|---|---|---|
| `vif`, `vif_agg` | scalar / aggregate | Variance Inflation Factor per predictor — detect multicollinearity |
| `aic(rss, n, k)` | scalar | Akaike Information Criterion for model selection |
| `bic(rss, n, k)` | scalar | Bayesian Information Criterion for model selection |
| `jarque_bera(values[])`, `jarque_bera_agg(value)` | scalar / aggregate | Jarque-Bera normality test (`statistic`, `p_value`, `skewness`, `kurtosis`, `n`) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | scalar / aggregate | Residual analysis: `raw`, `standardized`, `studentized`, `leverage` arrays |
| `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` | table macro | broom `augment`: every source row + `model_id` / `model_name` (the method; no `id`, the group column is among the source columns), `row_id`, `fitted`, `residual`, `resid_type`, `std_residual` (R `rstandard`), `stud_residual` (`rstudent`), `leverage` (`hatvalues`), `cooks_d`, `dffits`, `weight`, `is_training`, `n_params` (`obs` schema); OLS / WLS also `loo_residual` (`residual / (1 − leverage)`, the PRESS residual) and `loo_fitted` (prediction without the row), NULL where leverage = 1 |
| `<model>_partial_residuals_by` | table macro | `ols`, `wls`, six GLMs: one row per source row and term — `term_index`, `term`, `x`, `component`, `partial_residual` (crPlots), `x_residual`, `y_residual` (avPlots; NULL for GLMs) |
| `acf_agg(x[, max_lag] ORDER BY t)`, `pacf_agg(…)` | aggregate | `LIST<STRUCT(lag, acf\|pacf, conf_low, conf_high)>`; band = ±qnorm(0.975)/√n; `max_lag` default `floor(10·log10(n))` |
| `acf_by`, `pacf_by` `(source, group_col, value_col, order_col[, max_lag := …])` | table macro | One ACF / PACF curve per group (`curve` schema, `curve_type` `'acf'` / `'pacf'`, `x` = lag, `y_low`/`y_high` = band) |
| `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg` | aggregate | Residual autocorrelation tests (unified test fields; see **anofox-statistics-tests**); `durbin_watson_agg(resid, x ORDER BY t)` adds the `dwtest` p-value |
| `breusch_pagan_agg(resid, x[, studentize])`, `white_test_agg(resid, x)` | aggregate | Heteroskedasticity tests (`lmtest::bptest`) |
| `tidy(model[, names])` | scalar on a model STRUCT | Per-term table: `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` (intercept row first, `'(Intercept)'`) — `unnest(…, recursive := true)` |
| `glance(model)` | scalar on a model STRUCT | One-row summary of the model's scalar fields (`r_squared`, `aic`, `n_observations`, GLM `family`/`link`, …) — `unnest(glance(m))` |
| `predict(model, x)` | scalar on a model STRUCT | Fitted value for one row; use it to build `y_hat` for the residual diagnostics |
| `aid_agg`, `aid_by` | aggregate / table macro | Demand classification: `demand_type` is `regular` or `intermittent`, plus distribution, zero proportion, stockout / new / obsolete product and outlier counts |
| `aid_anomaly_agg`, `aid_anomaly_by` | aggregate / table macro | Per-observation flags: `stockout`, `new_product`, `obsolete_product`, `high_outlier`, `low_outlier` |

## Interpretation guidance

- **VIF:** > 5 warns of multicollinearity, > 10 is severe — consider dropping/combining collinear predictors or switching to Ridge/Elastic Net.
- **AIC / BIC:** lower is better; compare *nested or same-data* models. BIC penalizes complexity harder than AIC. GLM result structs already carry an `aic` field; AFT and ALM results carry both `aic` and `bic`.
- **Residuals:** `standardized`/`studentized` residuals with |value| > 2–3 flag potential outliers; high `leverage` flags influential x-positions. Structure in residuals ⇒ mis-specified model.
- **`tidy` inference columns** are NULL when the model has no inference (fit without `compute_inference`, or PLS/LARS; quantile only with its `se` option). The intercept row (first) carries its own inference. Per group use `<model>_tidy_by` (batch skill).
- **Leave-one-out (OLS / WLS):** `ols_glance_by` / `wls_glance_by` report `press`, `press_rmse` and `loo_r_squared` (predicted R², Q²); a big gap to `r_squared` means the fit leans on individual rows, whose large `|loo_residual|` in `*_augment_by` point to them. Exact, from the leverage, no refits. A row with leverage 1 (single-observation dummy level) has NULL `loo_*` and makes the group's `press` metrics NULL.
- **Influence (`*_augment_by`):** `|stud_residual| > 3` flags outliers, `leverage > 2·n_params/n` high-leverage points, `cooks_d > 4/n` (or > 0.5–1) influential rows, `|dffits| > 2·sqrt(n_params/n)`. Rows outside training (`split` other than `'train'` / `'training'` in any case, or NULL; NULL y) get `fitted` but NULL diagnostics. GLMs: `fitted` on the response scale, `residual` is the deviance residual (`resid_type = 'deviance'`); penalized GLM fits (`glm_lambda`, `prior`) are rejected.
- **Partial residuals:** curvature of `partial_residual` against `x` ⇒ the term needs a transform; the slope of `y_residual` on `x_residual` through the origin equals the coefficient.
- **Autocorrelation:** ACF lags outside `[conf_low, conf_high]`, a small Ljung-Box p-value or Durbin-Watson well below 2 ⇒ residuals are autocorrelated (standard errors too small) — add lags / time terms or use a time-series model. `ORDER BY` goes **after** the constant arguments: `acf_agg(resid, 12 ORDER BY t)`.
- **Heteroskedasticity:** a small Breusch-Pagan / White p-value ⇒ use robust SEs (`'hc_type': 'hc3'` on OLS / WLS) or WLS. Pass the residuals of a model with an intercept and the same regressors.
- **AID demand classes:** use before forecasting/choosing a model — intermittent series (many zeros; threshold `intermittent_threshold`, default 0.3) need count/intermittent methods rather than plain OLS.

## Return fields

- `residuals_diagnostics_agg(y, y_hat[, x] ORDER BY …)` → STRUCT with `raw DOUBLE[]`, `standardized DOUBLE[]`, `studentized DOUBLE[]`, `leverage DOUBLE[]`, in `ORDER BY` order. Without `x`, `standardized`, `studentized` and `leverage` are NULL. For per-row diagnostics as a table with Cook's distance, prefer `ols_augment_by`.
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
-- Influential rows per group (Cook's distance), from a per-row augment table
CREATE OR REPLACE TABLE mt_demo AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS grp, i::DOUBLE AS x1,
       ((i * 3) % 7)::DOUBLE AS x2, 1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) AS y,
       round(exp(0.2 + 0.05 * i))::DOUBLE AS y_count
FROM range(1, 41) t(i);
SELECT grp, id, round(fitted, 3) AS fitted, round(stud_residual, 3) AS stud, round(leverage, 3) AS h, round(cooks_d, 3) AS cooks_d
FROM ols_augment_by('mt_demo', grp, y, [x1, x2])
QUALIFY row_number() OVER (PARTITION BY grp ORDER BY cooks_d DESC) <= 3
ORDER BY grp, cooks_d DESC;

-- Poisson: deviance residuals; the held-out rows get a prediction but no diagnostics
SELECT id, y_count, round(fitted, 2) AS fitted, round(residual, 3) AS dev_resid, is_training
FROM poisson_augment_by('mt_demo', grp, y_count, [x1], split := CASE WHEN id > 36 THEN 'test' ELSE 'train' END)
WHERE grp = 'A' AND id > 30 ORDER BY id;

-- Component-plus-residual data for one term
SELECT grp, row_id, x, round(partial_residual, 3) AS partial_residual, round(component, 3) AS component
FROM ols_partial_residuals_by('mt_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
WHERE term = 'cycle' ORDER BY grp, row_id LIMIT 5;
```

```sql
-- Residual autocorrelation and heteroskedasticity of a per-group fit
CREATE OR REPLACE TABLE resid_tbl AS
SELECT grp, id, x1, x2, y - fitted AS resid FROM ols_augment_by('mt_demo', grp, y, [x1, x2]);
SELECT grp,
       (ljung_box_agg(resid, 5 ORDER BY id)).p_value  AS ljung_box_p,
       (durbin_watson_agg(resid ORDER BY id)).statistic AS dw,
       (breusch_pagan_agg(resid, [x1, x2])).p_value     AS bp_p
FROM resid_tbl GROUP BY grp ORDER BY grp;
-- ACF with the white-noise band, one curve per group (curve schema)
SELECT model_id, x AS lag, round(y, 3) AS acf, round(y_high, 3) AS band
FROM acf_by('resid_tbl', grp, resid, id, max_lag := 4);
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

-- AID per-observation flags: unnest ONCE in a subquery, then read the struct fields
SELECT sku, t, flag.stockout, flag.high_outlier
FROM (SELECT sku,
             unnest(list(t ORDER BY t))                 AS t,
             unnest(aid_anomaly_agg(demand ORDER BY t)) AS flag
      FROM (SELECT 'sku' || (i % 2) AS sku, i AS t,
                   CASE WHEN i % 2 = 0 THEN 10.0 + (i % 4) WHEN i = 7 THEN 0.0 ELSE 5.0 END AS demand
            FROM range(1, 41) t(i))
      GROUP BY sku)
WHERE flag.stockout OR flag.high_outlier
ORDER BY sku, t;
```

Read `aid_anomaly_agg` results with the subquery pattern above (or use `aid_anomaly_by`),
not with one `(unnest(flags)).field` per field in the same `SELECT`: that form has
crashed or returned corrupted results in some ADBC-based clients (e.g. Bruin).
