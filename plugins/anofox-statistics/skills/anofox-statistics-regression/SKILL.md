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
  call surfaces (scalar fit, aggregate fit_agg, predict). Use when picking
  a regression method or writing ols_fit_agg / ridge_fit / predict calls.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Regression Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 (Rust crate `anofox-regression`) | **DuckDB:** v1.4.5 LTS / v1.5.4+ | **API:** unprefixed, MAP-style options

Every regression family exposes up to three call surfaces:

| Surface | Signature | Use |
|---|---|---|
| **Scalar** `{model}_fit(y, X[, opts])` | `y DOUBLE[]`, `X DOUBLE[][]` (**column-major**), returns STRUCT | Fit on literal arrays / one series |
| **Aggregate** `{model}_fit_agg(y, [x1, x2, …][, opts])` | one row per observation, returns STRUCT | Fit across a table or per `GROUP BY` group |
| **Predict** `predict(X_new, coefficients, intercept)` | scalar, column-major `X_new` | Score new points from fitted coefficients |

> For per-group **batch** fit+predict in one call (`*_fit_predict_by` table macros), see the **anofox-statistics-batch** skill.

## Critical gotchas

1. **Scalar `X` is column-major.** Each inner array is one *feature column* across all observations, not one row. Single feature `x` with 6 obs → `[[x1, x2, x3, x4, x5, x6]]`.
2. **Aggregate `X` is a list of features per row.** `ols_fit_agg(y, [x1, x2])` — the second arg is the feature vector for *that row*.
3. **No `anofox_stats_` prefix.** All names are unprefixed as of v0.3.0 (`ols_fit_agg`, `theil_sen_fit`, …). `theilsen` → `theil_sen`.
4. **Use `.r_squared`, never `.r2`.** And `.n_observations` / `.n_features` (not `.n_obs`).
5. **GLM & AFT report `z_values`, not `t_values`, and have no `r_squared`** — Wald statistics are asymptotically normal. Don't force z→t.
6. **Inference is opt-in.** `std_errors` / `t_values` / `p_values` / `ci_lower` / `ci_upper` are only populated with `{'compute_inference': true}` (OLS/Ridge/WLS).
7. **Unknown option keys raise `InvalidInputException` at bind time** — no silent ignore. Values out of range (e.g. `l1_ratio > 1`) also raise immediately.

## Models

### Linear / least squares

| Model | Functions | Notes |
|---|---|---|
| OLS | `ols_fit`, `ols_fit_agg` | Ordinary least squares; supports `compute_inference`, `hc_type` robust SEs |
| WLS | `wls_fit`, `wls_fit_agg` | Weighted; pass `weight_col` (agg) |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive/online least squares |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2; `alpha` (alias `lambda`) > 0 |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | L1+L2; `alpha`, `l1_ratio ∈ [0,1]` (default 0.5) |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded / non-negative coefficients |
| PLS | `pls_fit_predict_agg` | Partial least squares |
| Isotonic | `isotonic_fit_predict_agg` | Monotonic regression |
| Quantile | `quantile_fit_predict_agg` | Quantile / median regression |

### Robust (outlier-resistant)

| Model | Functions | Key options |
|---|---|---|
| Huber | `huber_fit`, `huber_fit_agg` | `huber_epsilon` (default 1.35); reports MAD scale + outlier mask |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | `max_trials` (100), `residual_threshold`, `min_samples`; inlier mask + trial count |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Nonparametric spatial-median over OLS subsamples |

### GLMs & survival (report `z_values`, no `r_squared`)

| Model | Function | For |
|---|---|---|
| Poisson | `poisson_fit_agg` | Count data |
| Binomial | `binomial_fit_agg` | Success-rate data (`link`: logit / probit / cloglog) |
| Negative Binomial | `negbinom_fit_agg` | Overdispersed counts (dispersion α estimated) |
| Tweedie | `tweedie_fit_agg` | Positive-skew continuous |
| Gamma | `gamma_fit_agg` | Strictly-positive continuous (claims, durations) |
| Logistic | `logistic_fit_agg` | Binary classification; reports accuracy + threshold echo |
| ALM | `alm_fit_agg` | Additive linear models, 24 error distributions |
| AFT survival | `aft_fit_agg` | Duration models with right censoring (`distribution`) |
| Mixed effects | `glmm_fit_agg` | Random intercept over a grouping factor |
| EB shrinkage | `eb_shrink_agg` | Partial pooling of per-group estimates |

## Option-map keys (MAP literal)

```sql
{'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
```

| Key | Type | Default | Applies to |
|---|---|---|---|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | all regression |
| `compute_inference` | BOOLEAN | `false` | OLS, Ridge, WLS |
| `confidence_level` | DOUBLE | `0.95` | all with CIs — must be in (0,1) |
| `alpha` (alias `lambda`) | DOUBLE | — | Ridge, Elastic Net — > 0 |
| `l1_ratio` | DOUBLE | `0.5` | Elastic Net — [0,1] |
| `max_iterations`, `tolerance` | INT / DOUBLE | — | iterative solvers |
| `hc_type` | VARCHAR | `'HC3'` | OLS robust SEs |
| `weight_col` | VARCHAR | — | WLS |
| `huber_epsilon` | DOUBLE | `1.35` | Huber |
| `max_trials`, `residual_threshold`, `min_samples` | — | 100 / — / — | RANSAC |
| `link`, `family` | VARCHAR | — | GLM |
| `distribution` | VARCHAR | — | AFT |
| `interval_type` | VARCHAR | `'confidence'` | prediction — `'confidence'` \| `'prediction'` |

## Return-struct fields

**Standard (OLS/robust/penalized/WLS/RLS):**
`coefficients DOUBLE[]`, `intercept DOUBLE`, `std_errors[]`, `t_values[]`, `p_values[]`,
`r_squared`, `adj_r_squared`, `f_statistic`, `f_pvalue`, `residual_std_error`,
`n_observations BIGINT`, `n_features BIGINT`, `ci_lower[]`, `ci_upper[]`.

**GLM (Poisson/Logistic/Gamma/NegBinom):** replaces `t_values` with `z_values`; adds
`log_likelihood`, `deviance`, `null_deviance`, `aic`, `bic`, `n_iterations`; **no** `r_squared`.

**AFT survival:** `z_values`, `log_likelihood`, `aic`, `scale`; **no** `r_squared`.

## Worked examples

```sql
-- Aggregate OLS with inference
CREATE TABLE houses AS SELECT * FROM (VALUES
  (50.0,120.0),(65.0,155.0),(80.0,190.0),(95.0,225.0),(110.0,265.0),(125.0,300.0)
) t(sqm, price);
SELECT
  round((ols_fit_agg(price, [sqm], {'compute_inference': true})).r_squared, 4) AS r2,
  (ols_fit_agg(price, [sqm])).coefficients[1]                                   AS slope,
  (ols_fit_agg(price, [sqm])).intercept                                         AS intercept
FROM houses;
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

```sql skip
-- Ridge with L2 penalty; Poisson GLM reads z_values
SELECT (ridge_fit_agg(y, [x1, x2], {'alpha': 1.0})).coefficients FROM tbl;
SELECT (poisson_fit_agg(events, [x1, x2])).z_values FROM counts;
```

## Errors

`InvalidInputException` — shape mismatch, `n < n_features + 1`, all-non-finite input,
constant/zero-variance column, unknown option key, option out of range.
`FunctionException` — singular matrix, convergence failure, internal panic.
Message format: `{function}: {problem}; expected {shape} (got {actual})`.
