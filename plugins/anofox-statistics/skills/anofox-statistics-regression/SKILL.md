---
name: anofox-statistics-regression
description: >
  Regression models and the fit / fit_agg / predict API surface of the
  anofox_statistics DuckDB extension. Covers OLS, robust estimators
  (Huber, RANSAC, Theil-Sen), penalized regression (Ridge, Elastic Net),
  WLS, recursive least squares (RLS), bounded/non-negative LS (BLS/NNLS),
  PLS, isotonic, quantile regression, GLMs (Poisson, Binomial/Logistic,
  Negative Binomial, Tweedie, Gamma), ALM, AFT survival, mixed-effects
  GLMM, and empirical-Bayes shrinkage — with the MAP-style option keys,
  return-struct field names (including GLM/AFT z_values), and the three
  call surfaces (scalar fit, aggregate fit_agg, window *_fit_predict,
  *_fit_predict_agg, predict), the model tools predict(model, x),
  linear_predict, tidy and glance, plus LARS (lars_fit_agg) and the AFT
  aft_cdf / aft_quantile helpers. Use when picking a regression method or
  writing ols_fit_agg / ridge_fit / ols_fit_predict / predict / tidy calls.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Regression Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 (Rust crate `anofox-regression`) | **DuckDB:** v1.4.5 LTS / v1.5.x | **API:** unprefixed, MAP-style options

Regression families expose up to five call surfaces:

| Surface | Signature | Use |
|---|---|---|
| **Scalar** `{model}_fit(y, X[, opts])` | `y DOUBLE[]`, `X DOUBLE[][]` (**column-major**), returns STRUCT | Fit on literal arrays / one series |
| **Aggregate** `{model}_fit_agg(y, [x1, x2, …][, opts])` | one row per observation, returns STRUCT | Fit across a table or per `GROUP BY` group |
| **Window** `{model}_fit_predict(y, [x1, …][, opts]) OVER (ORDER BY t ROWS BETWEEN … AND CURRENT ROW)` | returns `STRUCT(yhat, yhat_lower, yhat_upper)` for the **last row of the frame** | Expanding / rolling in-sample fits |
| **Fit-predict aggregate** `{model}_fit_predict_agg(y, [x1, …][, opts])` | returns `LIST<STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)>` | Fit once on non-NULL `y`, predict every row of the group |
| **Predict (model)** `predict(model, [x1, …][, {'type': 'response'\|'link'}])` | `model` = STRUCT from any `*_fit_agg` / `*_fit`, returns `DOUBLE` | Score one row; GLMs on the response scale by default, isotonic interpolates knots |
| **Predict (columns)** `predict(X_new, coefficients, intercept)` = `linear_predict(...)` | column-major `X_new`, returns `DOUBLE[]` | Score many points from raw coefficients |
| **Report** `tidy(model[, names])`, `glance(model)` | per-term LIST of structs / STRUCT of scalar fields | Coefficient table (`unnest(…, recursive := true)`) and one-row fit summary |

WLS takes the weight as a positional argument everywhere: `wls_fit_agg(y, x, weight[, opts])`, `wls_fit_predict(y, x, weight[, opts])`, `wls_fit_predict_agg(y, x, weight[, opts])`.

Window functions exist for: `ols`, `ridge`, `elasticnet`, `wls`, `rls`, `huber`, `ransac`, `theil_sen` (`*_fit_predict`).
Fit-predict aggregates exist for those eight plus `bls`, `alm`, `poisson`, `pls`, `isotonic`, `quantile` (`*_fit_predict_agg`). The fit-predict aggregates also accept a split column: `ols_fit_predict_agg(y, x, split_col[, opts])` trains only on rows where `split_col = 'train'`.

Prediction intervals (`yhat_lower`/`yhat_upper`) are leverage-aware, `yhat ± t(n−p)·s·sqrt(1 + x₀ᵀMx₀)`: exact for OLS (= R `predict.lm`) and WLS, ridge sandwich for Ridge, OLS-leverage approximation for Huber / RANSAC / Theil-Sen / RLS / BLS / ALM / Elastic Net. NULL when no interval exists (zero residual df, singular design); zero-width for an exact fit.

> For per-group **batch** fit+predict in one call (`*_fit_predict_by` table macros), see the **anofox-statistics-batch** skill.

## Critical gotchas

1. **Scalar `X` is column-major.** Each inner array is one *feature column* across all observations, not one row. Single feature `x` with 6 obs → `[[x1, x2, x3, x4, x5, x6]]`.
2. **Aggregate `X` is a list of features per row.** `ols_fit_agg(y, [x1, x2])` — the second arg is the feature vector for *that row*.
3. **No `anofox_stats_` prefix.** The prefix was removed in v0.10.0 (`ols_fit_agg`, `theil_sen_fit`, …). `theilsen` → `theil_sen`.
4. **Use `.r_squared`, never `.r2`.** And `.n_observations` / `.n_features` (not `.n_obs`).
5. **GLM & AFT report `z_values`, not `t_values`, and have no `r_squared`** — Wald statistics are asymptotically normal. Don't force z→t. GLMs report `pseudo_r_squared`, `iterations`, `converged`.
6. **Inference is opt-in.** `std_errors` / `t_values` / `p_values` / `ci_lower` / `ci_upper` (and OLS `f_statistic` / `f_pvalue`) only exist with `{'compute_inference': true}`.
7. **Option keys a function does not support raise `InvalidInputException` at bind time** — no silent ignore. Out-of-range values also raise.
8. **Too few rows → NULL, not an error.** A group or window frame with fewer than `n_features + 1` usable rows yields `NULL`.
9. **Constant / aliased columns get `NaN` coefficients** rather than failing.
10. **Window `*_fit_predict` predicts the last row of the frame.** Frames must end at `CURRENT ROW` over a unique `ORDER BY`. `*_fit_predict … AND 1 PRECEDING` is NOT a one-step-ahead forecast — for that use `predict((ols_fit_agg(y, [x]) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), [x])`. `OVER (PARTITION BY g)` without `ORDER BY` gives every row the same value; avoid `RANGE` frames with ties. For per-row predictions over a whole group use `*_fit_predict_agg` / `*_fit_predict_by`.
11. **Two `predict` forms.** `predict(model_struct, [x1, x2])` → one `DOUBLE`; `predict([[col1…], [col2…]], coefficients, intercept)` → `DOUBLE[]` (column-major, alias `linear_predict`). NULL model / NULL feature → NULL; wrong feature count → error. For GLMs prefer the model form — the 3-argument form returns the linear predictor, not the response.
12. **`tidy` intercept row has NULL inference for `ols_fit_agg`** (inference covers slopes only); inference columns are all NULL without `compute_inference`. Expand `glance(m)` with `unnest(glance(m))` — `glance(m).*` is a parser error.

## Models

### Linear / least squares

| Model | Functions | Notes |
|---|---|---|
| OLS | `ols_fit`, `ols_fit_agg` | Supports `compute_inference`, `hc_type` (`none`/`hc0`–`hc3`, default `none`), `solver` |
| WLS | `wls_fit`, `wls_fit_agg` | Weight is the 3rd positional argument |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive/online least squares; `forgetting_factor` (1.0), `initial_p_diagonal` (100) |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2; `alpha` (alias `lambda`), default `1.0`, must be ≥ 0. `alpha = 0` is accepted and reproduces OLS |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | L1+L2; `alpha` (alias `lambda`, default 1.0), `l1_ratio ∈ [0,1]` (default 0.5) |
| LARS | `lars_fit_agg` | Least Angle Regression; options `fit_intercept`, `alpha` (alias `lambda`, default 0.0); returns the 7 standard fields |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded (`lower_bound`, `upper_bound`) / non-negative coefficients |
| PLS | `pls_fit_agg`, `pls_fit_predict_agg` | Partial least squares; `n_components` (1), `fit_intercept`. Returns `coefficients`, `intercept`, `r_squared`, `n_components`, `n_observations`, `n_features` |
| Isotonic | `isotonic_fit_agg`, `isotonic_fit_predict_agg` | Monotonic regression on a single `x DOUBLE`; `increasing` (true). Returns knots `x[]`, `fitted[]`, `increasing`, `r_squared`, `n_observations`; `predict(m, [x])` interpolates, clamped at the ends |
| Quantile | `quantile_fit_agg`, `quantile_fit_predict_agg` | Quantile / median regression; `tau` (alias `quantile`, 0.5), `fit_intercept`, `max_iterations`, `tolerance`. Returns `coefficients`, `intercept`, `tau`, `n_observations`, `n_features` |

### Robust (outlier-resistant)

| Model | Functions | Key options / extra fields |
|---|---|---|
| Huber | `huber_fit`, `huber_fit_agg` | `epsilon` (default 1.35), `alpha` (1e-4), `max_iterations` (100); reports `scale`, `n_outliers` |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | `max_trials` (100), `residual_threshold` (default MAD of y), `min_samples` (default `n_features + 1`), `random_state`; reports `residual_threshold`, `n_inliers`, `n_trials` |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Spatial median over subsamples; `n_subsamples`, `max_subpopulation`, `random_state` |

### GLMs & survival (report `z_values`, no `r_squared`)

| Model | Function | For |
|---|---|---|
| Poisson | `poisson_fit_agg` | Count data (`link`: log / identity / sqrt) |
| Binomial | `binomial_fit_agg` | Success-rate data (`binomial_link`: logit / probit / cloglog) |
| Negative Binomial | `negbinom_fit_agg` | Overdispersed counts; `dispersion` field = estimated α |
| Tweedie | `tweedie_fit_agg` | Positive-skew continuous with zeros (`power`, default 1.5) |
| Gamma | `gamma_fit_agg` | Strictly-positive continuous (claims, durations) |
| Logistic | `logistic_fit_agg` | Binary classification; reports `accuracy` + `threshold` (default 0.5) |
| ALM | `alm_fit_agg` | Augmented Linear Model, 24 error distributions (`distribution`, `loss`); reports `t_values` |
| AFT survival | `aft_fit_agg` | Duration models with right censoring (`distribution`: weibull / lognormal / loglogistic / exponential) |
| AFT helpers | `aft_cdf(t, eta, scale, dist)`, `aft_quantile(p, eta, scale, dist)` | Scalar: `P(T ≤ t)` and the `p`-quantile of `T` given linear predictor `eta = intercept + x·β` and the fitted `scale` |
| Mixed effects | `glmm_fit_agg(y, x, group[, opts])` | Random intercepts / slopes; `family` option selects the GLM family |
| EB shrinkage | `eb_shrink_agg(estimate, se[, opts])` | Partial pooling of per-group estimates |

All GLMs accept `glm_lambda` (L2 penalty), `prior` + `feature_names` (coefficient priors), `max_iterations`, `tolerance`, `compute_inference`, `confidence_level`. There is no `family` key on the single-family GLM aggregates. Every GLM has a per-group table macro `{poisson,binomial,logistic,negbinom,gamma,tweedie}_fit_predict_by` (response-scale `yhat`; see the batch skill).

## Option-map keys (MAP literal)

```sql skip
{'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
```

| Key | Type | Default | Applies to |
|---|---|---|---|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | all regression |
| `compute_inference` | BOOLEAN | `false` | OLS, Ridge, WLS, robust, GLMs, ALM, AFT, GLMM |
| `confidence_level` | DOUBLE | `0.95` | inference and prediction intervals |
| `alpha` (alias `lambda`) | DOUBLE | `1.0` (Ridge/EN), `0.0` (LARS) | Ridge, Elastic Net, LARS — ≥ 0; `0` = unpenalized |
| `l1_ratio` | DOUBLE | `0.5` | Elastic Net — [0,1] |
| `lambda_scaling` | VARCHAR | `'raw'` | Ridge, Elastic Net (`'glmnet'` divides by n) |
| `solver` | VARCHAR | `'svd'` | OLS, WLS, Ridge (`qr`/`svd`/`cholesky`) |
| `hc_type` | VARCHAR | `'none'` | OLS, WLS robust SEs (`hc0`–`hc3`) |
| `max_iterations`, `tolerance` | INT / DOUBLE | function-specific | iterative solvers |
| `epsilon` | DOUBLE | `1.35` | Huber |
| `max_trials`, `residual_threshold`, `min_samples`, `random_state` | — | 100 / MAD / p+1 / — | RANSAC |
| `link` / `binomial_link` | VARCHAR | `'log'` / `'logit'` | Poisson / Binomial |
| `power` | DOUBLE | `1.5` | Tweedie |
| `family` | VARCHAR | `'gaussian'` | GLMM only |
| `distribution` | VARCHAR | — | ALM, AFT |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict`, `*_fit_predict_agg` (`'drop'` / `'drop_y_zero_x'`) |

Prediction intervals in the fit-predict functions use `confidence_level`; there is no `interval_type` option.

## Return-struct fields

**Linear (OLS, Ridge, WLS, Huber, RANSAC, Theil-Sen):**
`coefficients DOUBLE[]`, `intercept`, `r_squared`, `adj_r_squared`, `residual_std_error`,
`n_observations BIGINT`, `n_features BIGINT`; with inference also `std_errors[]`, `t_values[]`,
`p_values[]`, `ci_lower[]`, `ci_upper[]`, `f_statistic`, `f_pvalue`.
Elastic Net, LARS, RLS: the first seven fields only.

**GLM (Poisson/Binomial/Logistic/NegBinom/Gamma/Tweedie):** `coefficients`, `intercept`, `deviance`,
`null_deviance`, `pseudo_r_squared`, `aic`, `dispersion` (logistic: `accuracy`, `threshold`),
`n_observations`, `n_features`, `iterations`, `converged`; with inference `std_errors`, `z_values`,
`p_values`, `ci_lower`, `ci_upper`; always last: `family VARCHAR`, `link VARCHAR` (logistic reports `'binomial'` / `'logit'`). **No** `r_squared`, `log_likelihood` or `bic`.

**AFT survival:** `coefficients`, `intercept`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`,
`n_observations`, `n_events`, `n_censored`, `n_features`, `iterations`, `converged`; with inference `z_values` etc.

## Worked examples

```sql
-- Aggregate OLS with inference
CREATE TABLE houses AS SELECT * FROM (VALUES
  (50.0,120.0),(65.0,155.0),(80.0,190.0),(95.0,225.0),(110.0,265.0),(125.0,300.0)
) t(sqm, price);
SELECT
  round(fit.r_squared, 4) AS r2,
  fit.coefficients[1]     AS slope,
  fit.p_values[1]         AS slope_p
FROM (SELECT ols_fit_agg(price, [sqm], {'compute_inference': true}) AS fit FROM houses);
```

```sql
-- Scalar fit (column-major X) + predict on new points
SELECT unnest(predict(
  [[70.0, 100.0, 140.0]]::DOUBLE[][],
  (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
           [[50.0,65.0,80.0,95.0,110.0,125.0]])).coefficients,
  (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
           [[50.0,65.0,80.0,95.0,110.0,125.0]])).intercept
)) AS predicted;
```

```sql
-- Expanding-window OLS: in-sample fit on all rows up to and including the current one
SELECT sqm, price,
  (ols_fit_predict(price, [sqm]) OVER (
     ORDER BY sqm ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)).yhat AS yhat
FROM houses;
```

```sql
-- One-step-ahead forecast: fit on earlier rows only, then score the current row.
-- The model (and so the prediction) is NULL while the frame is too small.
SELECT sqm, price,
  predict((ols_fit_agg(price, [sqm]) OVER (
     ORDER BY sqm ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), [sqm]) AS yhat_next
FROM houses;
```

```sql
-- Model tools: score a new point, coefficient table, one-row summary
SELECT predict(ols_fit_agg(price, [sqm]), [140.0]) AS yhat_140 FROM houses;
SELECT unnest(tidy(ols_fit_agg(price, [sqm], {'compute_inference': true}), ['sqm']), recursive := true)
FROM houses;
SELECT unnest(glance(ols_fit_agg(price, [sqm]))) FROM houses;
```

```sql
-- GLM: predict on the response scale (default) or the link scale
SELECT predict(m, [3.0]) AS expected_count, predict(m, [3.0], {'type': 'link'}) AS log_rate
FROM (SELECT poisson_fit_agg(n, [x]) AS m
      FROM (VALUES (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (4.0, 8.0), (5.0, 12.0)) t(x, n));
```

```sql
-- Fit once on rows with known y, predict all rows (NULL y = rows to forecast)
SELECT unnest(ols_fit_predict_agg(price, [sqm])) AS p
FROM (SELECT * FROM houses UNION ALL SELECT 140.0, NULL);
```

```sql
-- AFT helpers: median survival time and P(T <= 5) for eta = 2, scale = 1
SELECT aft_quantile(0.5, 2.0, 1.0, 'weibull') AS median_t,
       aft_cdf(5.0, 2.0, 1.0, 'weibull')      AS p_by_5;
```

```sql skip
-- Ridge with L2 penalty; LARS; Poisson GLM reads z_values
SELECT (ridge_fit_agg(y, [x1, x2], {'alpha': 1.0})).coefficients FROM tbl;
SELECT (lars_fit_agg(y, [x1, x2, x3])).coefficients FROM tbl;
SELECT (poisson_fit_agg(events, [x1, x2], {'compute_inference': true})).z_values FROM counts;
```

## Errors

`InvalidInputException` — shape mismatch, all-non-finite input, unsupported option key, option out of range.
`InternalException` — numerical failure (singular matrix, convergence failure).
Messages are prefixed with the function name: `{function}: {problem}`.
