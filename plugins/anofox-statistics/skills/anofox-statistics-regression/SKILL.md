---
name: anofox-statistics-regression
description: >
  Regression models and the fit / fit_agg / predict API surface of the
  anofox_statistics DuckDB extension. Covers OLS, robust estimators (Huber,
  RANSAC, Theil-Sen), penalized regression (Ridge, Elastic Net, LARS),
  Bayesian Ridge and ARD (posterior predictive intervals), Passive-Aggressive
  online regression, lmDynamic time-varying coefficients, LOWESS smoothing
  (lowess_agg), WLS, recursive least squares (RLS), bounded/non-negative LS
  (BLS/NNLS), PLS, isotonic, quantile regression with standard errors (se
  option iid / nid / boot) and the quantile process (quantile_process_agg),
  GLMs (Poisson, Binomial/Logistic, Negative Binomial, Tweedie, Gamma), ALM,
  AFT survival, mixed-effects GLMM, and empirical-Bayes shrinkage — with the
  MAP-style option keys (including alias_tolerance for collinear columns in
  OLS / WLS), the stable return-struct fields (model_type, intercept
  inference, log_likelihood / aic / bic, vcov, df_residual, GLM/AFT z_values),
  the call surfaces (scalar fit, aggregate fit_agg, window *_fit_predict,
  *_fit_predict_agg), predict(model, x) with se_fit / confidence and
  prediction intervals, linear_predict, tidy and glance, regularisation paths
  and cross-validation (ridge_path_agg, elasticnet_path_agg, lars_path_agg,
  ridge_cv_agg, elasticnet_cv_agg, lambda.min / lambda.1se, group /
  leave-one-group-out / stratified / time-series folds via cv_strategy, nested
  CV with ridge_nested_cv_agg / elasticnet_nested_cv_agg), and the AFT aft_cdf
  / aft_quantile / aft_survival helpers. Also: GLMM Wald / Satterthwaite
  inference and BLUP SEs (ddf, glmm_tidy_by), Bayesian credible intervals,
  Theil-Sen exact / bootstrap CIs (ci_method), nested F tests (ols_anova_by),
  CV strategies (cv_strategy group / logo / stratified / time_series, nested
  CV) and leave-one-out PRESS (compute_loo). Use when picking a regression
  method, choosing a penalty, getting standard errors or intervals for
  predictions, or writing ols_fit_agg / ridge_fit / ols_fit_predict / predict
  / tidy calls. For per-group long tables (tidy_by, prediction_by, effects_by,
  rolling_tidy_by) see the batch and plotting skills.
version: 0.13.0
user-invocable: false
---

# Anofox Statistics — Regression Cheat Sheet

**Extension:** `anofox_statistics` v0.13.0 (Rust crate `anofox-regression`) | **DuckDB:** v1.4.5 LTS / v1.5.x / v2.0 | **API:** unprefixed, MAP-style options

Regression families expose up to five call surfaces:

| Surface | Signature | Use |
|---|---|---|
| **Scalar** `{model}_fit(y, X[, opts])` | `y DOUBLE[]`, `X DOUBLE[][]` (**column-major**), returns STRUCT | Fit on literal arrays / one series |
| **Aggregate** `{model}_fit_agg(y, [x1, x2, …][, opts])` | one row per observation, returns STRUCT | Fit across a table or per `GROUP BY` group |
| **Window** `{model}_fit_predict(y, [x1, …][, opts]) OVER (ORDER BY t ROWS BETWEEN … AND CURRENT ROW)` | returns `STRUCT(yhat, yhat_lower, yhat_upper)` for the **last row of the frame** | Expanding / rolling in-sample fits |
| **Fit-predict aggregate** `{model}_fit_predict_agg(y, [x1, …][, opts])` | returns `LIST<STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)>` | Fit once on non-NULL `y`, predict every row of the group |
| **Predict (model)** `predict(model, [x1, …][, {'type': 'response'\|'link'}])` | `model` = STRUCT from any `*_fit_agg` / `*_fit`, returns `DOUBLE` | Score one row; GLMs on the response scale by default, isotonic interpolates knots |
| **Predict with SE** `predict(model, [x1, …], {'se_fit': true, 'interval': 'confidence'\|'prediction', 'level': 0.95})` | returns `STRUCT(yhat, se_fit, yhat_lower, yhat_upper)` | OLS, WLS, Ridge, GLMs fitted with `compute_inference` (see below) |
| **Predict (columns)** `predict(X_new, coefficients, intercept)` = `linear_predict(...)` | column-major `X_new`, returns `DOUBLE[]` | Score many points from raw coefficients |
| **Report** `tidy(model[, names])`, `glance(model)` | per-term LIST of structs / STRUCT of scalar fields | Coefficient table (`unnest(…, recursive := true)`) and one-row fit summary |

WLS takes the weight as a positional argument everywhere: `wls_fit_agg(y, x, weight[, opts])`, `wls_fit_predict(y, x, weight[, opts])`, `wls_fit_predict_agg(y, x, weight[, opts])`.

Window functions exist for: `ols`, `ridge`, `elasticnet`, `wls`, `rls`, `huber`, `ransac`, `theil_sen` (`*_fit_predict`).
Fit-predict aggregates exist for those eight plus `bls`, `alm`, `poisson`, `pls`, `isotonic`, `quantile` (`*_fit_predict_agg`). The fit-predict aggregates also accept a split column: `ols_fit_predict_agg(y, x, split_col[, opts])` trains only on rows where `lower(split_col) IN ('train', 'training')` (a NULL split is a prediction row; the `*_fit_predict_by` macros' `split :=` follows the same rule).

Prediction intervals (`yhat_lower`/`yhat_upper`) are leverage-aware, `yhat ± t(n−p)·s·sqrt(1 + x₀ᵀMx₀)`: exact for OLS (= R `predict.lm`) and WLS, ridge sandwich for Ridge, OLS-leverage approximation for Huber / RANSAC / Theil-Sen / RLS / BLS / ALM / Elastic Net. NULL when no interval exists (zero residual df, singular design); zero-width for an exact fit.

> For per-group **batch** fit+predict in one call (`*_fit_predict_by`) and the per-group long tables (`<model>_tidy_by`, `_glance_by`, `_prediction_by`, `_effects_by`, `_rolling_tidy_by`, `_path_by`, `_cv_by`), see the **anofox-statistics-batch** skill; for which long table feeds which plot, the **anofox-statistics-plotting** skill.

## Critical gotchas

1. **Scalar `X` is column-major.** Each inner array is one *feature column* across all observations, not one row. Single feature `x` with 6 obs → `[[x1, x2, x3, x4, x5, x6]]`.
2. **Aggregate `X` is a list of features per row.** `ols_fit_agg(y, [x1, x2])` — the second arg is the feature vector for *that row*.
3. **No `anofox_stats_` prefix.** The prefix was removed in v0.10.0 (`ols_fit_agg`, `theil_sen_fit`, …). `theilsen` → `theil_sen`.
4. **Use `.r_squared`, never `.r2`.** And `.n_observations` / `.n_features` (not `.n_obs`).
5. **GLM & AFT report `z_values`, not `t_values`, and have no `r_squared`** — Wald statistics are asymptotically normal. Don't force z→t. GLMs report `pseudo_r_squared`, `iterations`, `converged`; every model reports `model_type` (`'ols'`, `'glm'`, `'aft'`, …), `family`, `link`.
6. **Inference is opt-in, the struct shape is not.** `std_errors` / `t_values` / `p_values` / `ci_lower` / `ci_upper` (also as `conf_low` / `conf_high`), OLS `f_statistic` / `f_pvalue`, the `intercept_*` inference and `vcov` are always present in the models that report them, and NULL unless `{'compute_inference': true}`.
7. **Option keys a function does not support raise `InvalidInputException` at bind time** — no silent ignore. Out-of-range values also raise.
8. **Too few rows → NULL, not an error.** A group or window frame with fewer than `n_features + 1` usable rows yields `NULL`. A row with a NULL or incomplete `x` gets a NULL prediction (never NaN): window `*_fit_predict` returns NULL for it, and `*_fit_predict_agg` keeps the row (one element per input row) with `yhat` NULL and `is_training = false`.
9. **Constant / aliased columns get `NaN` coefficients** rather than failing; OLS and WLS alias like R's `lm`, and `alias_tolerance` (default `1e-7`, `lm`'s `tol`) sets how close to collinear a column must be.
10. **Window `*_fit_predict` predicts the last row of the frame.** Frames must end at `CURRENT ROW` over a unique `ORDER BY`. `*_fit_predict … AND 1 PRECEDING` is NOT a one-step-ahead forecast — for that use `predict((ols_fit_agg(y, [x]) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), [x])`. `OVER (PARTITION BY g)` without `ORDER BY` gives every row the same value; avoid `RANGE` frames with ties. For per-row predictions over a whole group use `*_fit_predict_agg` / `*_fit_predict_by`.
11. **Two `predict` forms.** `predict(model_struct, [x1, x2])` → one `DOUBLE`; `predict([[col1…], [col2…]], coefficients, intercept)` → `DOUBLE[]` (column-major, alias `linear_predict`). NULL model / NULL feature → NULL; wrong feature count → error. For GLMs prefer the model form — the 3-argument form returns the linear predictor, not the response.
12. **`tidy` rows come intercept first, with the intercept's own inference**; inference columns are all NULL without `compute_inference`. Expand `glance(m)` with `unnest(glance(m))` — `glance(m).*` is a parser error.
13. **`predict` with `se_fit` / `interval` needs `vcov`**: fit an OLS, WLS, Ridge or GLM `*_fit_agg` with `{'compute_inference': true}`, otherwise it raises. `'prediction'` intervals are an error for GLMs. `vcov` is the classical covariance even with `hc_type`.
14. **`lambda` is a SQL keyword.** Quote the CV field as `c."lambda"` (or read `x`).

## Models

### Linear / least squares

| Model | Functions | Notes |
|---|---|---|
| OLS | `ols_fit`, `ols_fit_agg` | Supports `compute_inference`, `hc_type` (`none`/`hc0`–`hc3`, default `none`), `solver`, `alias_tolerance` |
| WLS | `wls_fit`, `wls_fit_agg` | Weight is the 3rd positional argument; `alias_tolerance` as OLS |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive/online least squares; `forgetting_factor` (1.0), `initial_p_diagonal` (100) |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2; `alpha` (alias `lambda`), default `1.0`, must be ≥ 0. `alpha = 0` is accepted and reproduces OLS |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | L1+L2; `alpha` (alias `lambda`, default 1.0), `l1_ratio ∈ [0,1]` (default 0.5) |
| LARS | `lars_fit_agg` | Least Angle Regression; options `fit_intercept`, `alpha` (alias `lambda`, default 0.0); returns the 7 standard fields |
| Bayesian Ridge / ARD | `bayesian_ridge_fit_agg`, `ard_fit_agg`, `*_fit_predict_agg`, `*_fit_predict_by` | scikit-learn `BayesianRidge` / `ARDRegression` (matches to ~1e-8): penalty from the evidence, no CV. Options `fit_intercept`, `max_iterations` (300), `tolerance` (1e-3), `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` (1e-6), ARD `threshold_lambda` (1e4). Returns `coefficients`, `intercept`, `r_squared`, `noise_precision` (sklearn `alpha_`; noise SD = 1/sqrt), `weight_precision` (BR) / `weight_precisions[]` (ARD, pruned ≥ threshold → coef exactly 0), `posterior_sd[]`, `iterations`, `log_marginal_likelihood` (BR). The inference fields hold the posterior: `std_errors` = `posterior_sd`, `conf_low`/`conf_high` and `intercept_conf_*` = central **credible** intervals at `confidence_level` (intercept SD `sqrt(1/(nα) + x̄ᵀΣx̄)`); `t_values`/`p_values` and `aic`/`bic` NULL. `*_tidy_by` carries them into `terms`. Fit-predict bands are posterior predictive intervals at `confidence_level` |
| Passive-Aggressive | `passive_aggressive_fit_agg(y, x[, opts] ORDER BY t)`, window `passive_aggressive_fit_predict`, `*_fit_predict_agg`, `*_fit_predict_by` | Online regression (sklearn `PassiveAggressiveRegressor`, matches to 1e-9): rows updated in **ORDER BY order** (no ORDER BY = arbitrary order). Options `c` (1.0), `epsilon` (0.1), `variant` (`'pa1'`/`'pa2'`), `fit_intercept`, `max_iterations` (passes, 1 = online `partial_fit`), `tolerance`. Returns `coefficients`, `intercept`, `r_squared`, `n_observations`, `n_features`, `iterations`; no intervals (bounds NULL) |
| lmDynamic | `lm_dynamic_fit_agg(y, x[, opts] ORDER BY t)`, `lm_dynamic_fit_predict_agg`, `_by` macros | Time-varying coefficients (greybox `lmDynamic` bruteforce, matches to 1e-8): all 2^p subsets (p ≤ 14) weighted per row by pointwise `ic` (`'aicc'`/`'aic'`/`'bic'`), weights smoothed with lowess `span` (0.3; 0 = none). Returns time-averaged `coefficients`/`intercept`, `n_models`, per-row `dynamic_intercept[]`, `dynamic_coefficients[][]`, `fitted_values[]`, greybox `log_likelihood` |
| LOWESS | `lowess_agg(y, x[, frac[, iterations]])`, `lowess_by` | R `stats::lowess` (matches to 1e-10): returns `LIST(STRUCT(x, yhat))`, x increasing; `frac` 2/3, `iterations` 3 (positional, not a MAP) |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded (`lower_bound`, `upper_bound`) / non-negative coefficients |
| PLS | `pls_fit_agg`, `pls_fit_predict_agg` | Partial least squares; `n_components` (1), `fit_intercept`. Returns `coefficients`, `intercept`, `r_squared`, `n_components`, `n_observations`, `n_features` |
| Isotonic | `isotonic_fit_agg`, `isotonic_fit_predict_agg` | Monotonic regression on a single `x DOUBLE`; `increasing` (true). Returns knots `x[]`, `fitted[]`, `increasing`, `r_squared`, `n_observations`; `predict(m, [x])` interpolates, clamped at the ends |
| Quantile | `quantile_fit_agg`, `quantile_fit_predict_agg` | Quantile / median regression; `tau` (alias `quantile`, 0.5), `fit_intercept`, `max_iterations`, `tolerance`. Returns `coefficients`, `intercept`, `tau`, `n_observations`, `n_features` + inference fields. Standard errors with `se` (`'iid'`, `'nid'`, `'boot'`; `compute_inference: true` = `'nid'`), `n_boot` (200), `seed` (42), as `quantreg::summary.rq` |
| Nested-model F test | `ols_anova_agg(y, x_reduced, x_full[, opts])`, `ols_anova_by(src, g, y, x_reduced := [...], x_full := [...])` | R `anova(lm(reduced), lm(full))` on the same rows (matches to 1e-9): does `x_full` beat `x_reduced`? `x_reduced` columns must appear in `x_full` (error otherwise); `[]` = intercept-only (overall F). Returns `test` fields `method` (`'nested F'`), `statistic`, `df1`, `df2`, `p_value` + `rss_reduced`, `rss_full`, `adj_r_squared_reduced`, `adj_r_squared_full`, `delta_adj_r_squared`, `n_observations`. Rank counted as R; `df1 = 0` or `df2 = 0` → NULL (not NaN) statistic and p-value. Option `fit_intercept` only. No type I/II per-term tables |
| Quantile process | `quantile_process_agg(y, x[, opts])` | One fit per tau as `terms` rows (`index_name = 'tau'`); `taus` (default 0.1…0.9), `se` (`'nid'`), `compute_inference` (false = estimates only), `feature_names` |

### Regularisation paths and cross-validation

Pick the penalty from a path or a CV curve instead of guessing `alpha`. Each aggregate returns a LIST of long rows; `unnest` it.

| Function | Returns |
|---|---|
| `ridge_path_agg(y, x[, opts])`, `elasticnet_path_agg(y, x[, opts])` | `terms` rows (`term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, index_value`), `index_name = 'lambda'`; inference columns NULL |
| `lars_path_agg(y, x[, opts])` | `terms` rows per knot, **`index_name = 'alpha'`** (upstream's LARS scale, not glmnet's lambda — don't overlay on a ridge path); `method` (`'lasso'` default / `'lar'`), `standardize` (true), `n_nonzero_coefs` |
| `ridge_cv_agg(y, x[, opts])`, `ridge_cv_agg(y, x, cv_key, opts)`, same for `elasticnet_cv_agg` | `curve` rows: `curve_type = 'lambda_cv'`, `x` (= lambda), `y` (= cv_mean), `y_low`/`y_high` (± cv_se), `threshold`, `lambda`, `cv_mean`, `cv_se`, `n_nonzero`, `is_min`, `is_1se`, `n_folds` |
| `ridge_nested_cv_agg(y, x[, cv_key], opts)`, same for `elasticnet_nested_cv_agg` | `curve` rows, `curve_type = 'nested_cv'`: per outer fold (`series = 'fold'`: `fold`, chosen `lambda`, `test_error`, `inner_cv_mean`, `inner_cv_se`, `n_train`, `n_test`, `n_inner_folds`) and a summary (`series = 'mean'`: `test_error` = weighted mean, `test_se`, `y_low`/`y_high`) |

Path options: `lambdas` (glmnet scale, default glmnet's grid), `n_lambda` (100), `lambda_min_ratio`, `l1_ratio` (EN, 0.5), `standardize` (false), `fit_intercept`, `feature_names`. CV adds `folds` (10, ≥ 3) and `seed` (42), `repeats`, `metric` (`'mse'` / `'mae'`). **CV strategies** (`cv_strategy`, the `cv_key` argument holds the key): `'kfold'` (default without key), `'fold_id'` (default with key: the key is the fold, ≥ 3 labels), `'group'` (a group never spans two folds, sklearn `GroupKFold`; several group variables as a STRUCT key `{'store': store, 'region': region}`), `'logo'`, `'stratified'` (key = stratum; without key, `n_bins` quantile bins of y), `'time_series'` (key = time, a number / date / timestamp: rolling origin with `test_size` (alias `horizon`), `gap`, `max_train_size` (sliding window), default 5 splits). **Nested CV** (`*_nested_cv_agg`) is the honest error of "pick lambda by CV, then refit": outer folds from the strategy (`folds`, default 5), inner folds (`inner_folds`, 5) built from each outer training set only with the same key, `rule` `'min'` / `'1se'`; the min of a plain CV curve is optimistic. The path `lambda` is glmnet's per-observation penalty, **not** the `alpha`/`lambda` option of `ridge_fit_agg` (raw scaling). Per-group table macros: `ridge_path_by`, `elasticnet_path_by`, `lars_path_by`, `ridge_cv_by`, `elasticnet_cv_by` (batch skill).

### Robust (outlier-resistant)

| Model | Functions | Key options / extra fields |
|---|---|---|
| Huber | `huber_fit`, `huber_fit_agg` | `epsilon` (default 1.35), `alpha` (1e-4), `max_iterations` (100); reports `scale`, `n_outliers` |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | `max_trials` (100), `residual_threshold` (default MAD of y), `min_samples` (default `n_features + 1`), `random_state`; reports `residual_threshold`, `n_inliers`, `n_trials` |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Median of pairwise slopes (1 feature) / spatial median over subsamples; `n_subsamples`, `max_subpopulation`, `random_state` (alias `seed`). `compute_inference`: `ci_method` `'exact'` (default for 1 feature: Sen's slope CI + Kendall z in `t_values` / p-value, = R `trend::sens.slope`; `std_errors` / intercept CI NULL) or `'bootstrap'` (default otherwise: pairs bootstrap SDs + percentile CIs incl. intercept, `n_bootstrap` 999, deterministic per seed; `t_values`/`p_values` NULL; costs ~`n_bootstrap` point fits per group, ~8 s/group at 500 rows × 3 features). `theil_sen_tidy_by` defaults to `'exact_or_none'` (multi-feature intervals NULL, no bootstrap); opt in with `options := {'ci_method': 'bootstrap'}` (199 resamples there). No F-test (NULL) |

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
| AFT survival curve | `aft_survival(model, x, times)` | `LIST<STRUCT(time, surv)>`: `S(t \| x)` of an `aft_fit_agg` model at each of `times` |
| Mixed effects | `glmm_fit_agg(y, x, group[, opts])` | Random intercepts / slopes; `family` option selects the GLM family. With `compute_inference`: Satterthwaite t tests for gaussian (lmerTest; `t_values` + `df`, `z_values` NULL), Wald z otherwise (`z_values`; `ddf` option); `ranef[].se` = BLUP conditional SEs. Long outputs: `glmm_tidy_by` / `glmm_glance_by` / `glmm_augment_by` (batch skill) |
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
| `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` | DOUBLE | `1e-6` | Bayesian Ridge / ARD gamma priors on the noise / weight precisions (not the ridge `alpha`) |
| `threshold_lambda` | DOUBLE | `10000` | ARD: prune a feature once its precision reaches this |
| `epsilon` | DOUBLE | `1.35` | Huber |
| `max_trials`, `residual_threshold`, `min_samples`, `random_state` | — | 100 / MAD / p+1 / — | RANSAC |
| `link` / `binomial_link` | VARCHAR | `'log'` / `'logit'` | Poisson / Binomial |
| `power` | DOUBLE | `1.5` | Tweedie |
| `family` | VARCHAR | `'gaussian'` | GLMM only |
| `ddf` | VARCHAR | `'satterthwaite'` (gaussian), `'wald'` (others) | GLMM only: degrees of freedom of the fixed-effect tests; `'satterthwaite'` is gaussian-only |
| `distribution` | VARCHAR | — | ALM, AFT |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict`, `*_fit_predict_agg` (`'drop'` / `'drop_y_zero_x'` / `'zero'` / `'ignore'`; `'zero'` fits NULL `y`/`x` as 0, `'ignore'` = `'drop'`) |
| `alias_tolerance` | DOUBLE | `1e-7` | OLS / WLS (`*_fit`, `*_fit_agg`, `*_fit_predict_agg`, `*_fit_predict`): collinearity tolerance (R `lm`'s `tol`); not accepted by Ridge etc. |
| `se`, `n_boot`, `seed` | VARCHAR / INT / BIGINT | — / 200 / 42 | Quantile inference (`'iid'`/`'nid'`/`'boot'`) |

Prediction intervals in the fit-predict functions use `confidence_level`; there is no `interval_type` option.

`predict(model, x, opts)` options: `type` (`'response'` default / `'link'`), `se_fit` (false), `interval` (`'none'` / `'confidence'` / `'prediction'`), `level` (0.95). OLS/WLS/Ridge follow R `predict.lm(se.fit = TRUE, interval = …)`; GLMs follow `predict.glm(se.fit = TRUE)` with the interval built on the link scale and back-transformed.

## Return-struct fields

Every model struct has a **stable shape**: the same fields whatever the options, NULL when not computed.

**Linear (OLS, Ridge, WLS, Huber, RANSAC, Theil-Sen, Elastic Net, LARS, RLS):**
`coefficients DOUBLE[]`, `intercept`, `r_squared`, `adj_r_squared`, `residual_std_error`,
`n_observations`, `n_features`, model extras (Huber `scale`, `n_outliers`; RANSAC `residual_threshold`,
`n_inliers`, `n_trials`), then `std_errors[]`, `t_values[]`, `p_values[]` (NULL without inference; always NULL
for Elastic Net, LARS, RLS) and, except for Elastic Net, LARS and RLS, `ci_lower[]`, `ci_upper[]`, `f_statistic`, `f_pvalue`.

**GLM (Poisson/Binomial/Logistic/NegBinom/Gamma/Tweedie):** `coefficients`, `intercept`, `deviance`,
`null_deviance`, `pseudo_r_squared`, `aic`, `dispersion` (logistic: `accuracy`, `threshold`),
`n_observations`, `n_features`, `iterations`, `converged`, `std_errors`, `z_values`,
`p_values`, `ci_lower`, `ci_upper`, `family`, `link` (logistic reports `'binomial'` / `'logit'`). **No** `r_squared`.

**AFT survival:** `coefficients`, `intercept`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`,
`n_observations`, `n_events`, `n_censored`, `n_features`, `iterations`, `converged`; inference as `z_values` etc.

**Trailing fields on every model** (`ols`, `wls`, `ridge`, `elasticnet`, `lars`, `bayesian_ridge`, `ard`, `passive_aggressive`, `lm_dynamic`, `rls`, `huber`, `ransac`,
`theil_sen`, the GLMs, `alm`, `aft`, `glmm`, `bls`/`nnls`, `pls`, `quantile`, `isotonic`):
`conf_low[]`, `conf_high[]`, `conf_level`, the intercept's inference `intercept_std_error`,
`intercept_statistic`, `intercept_p_value`, `intercept_conf_low`, `intercept_conf_high`,
`log_likelihood`, `aic`, `bic` (R's `logLik` / `AIC` / `BIC`; NULL without a likelihood),
`model_type` (`'ols'`, `'glm'`, `'aft'`, …), `family`, `link`. OLS, WLS, Ridge and the GLM `*_fit_agg`
structs end with `vcov DOUBLE[]` (row-major k × k, intercept first) and `df_residual`, NULL unless
`compute_inference`.
OLS and WLS (`*_fit_agg` and `ols_fit` / `wls_fit`) then add the leave-one-out statistics `press`
(Σ (eᵢ/(1−hᵢᵢ))², R `sum(rstandard(fit, type = "predictive")^2)`; WLS weighted), `press_rmse`
(√(press/n)) and `loo_r_squared` (predicted R² / Q² = 1 − press/SST, never above `r_squared`), NULL
unless `{'compute_loo': true}` (opt-in, one more QR; not implied by `compute_inference`; `ols_glance_by` /
`wls_glance_by` turn it on) and when a row has leverage 1. `glance()` carries them too (NULL for other models);
per-row `loo_residual` / `loo_fitted` are in `ols_augment_by` / `wls_augment_by`.

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

```sql
-- Standard error and a 90% prediction interval at a new point (needs compute_inference)
SELECT predict(ols_fit_agg(price, [sqm], {'compute_inference': true}), [140.0],
               {'se_fit': true, 'interval': 'prediction', 'level': 0.9}) AS p
FROM houses;
-- -> {'yhat': ..., 'se_fit': ..., 'yhat_lower': ..., 'yhat_upper': ...}

-- GLM: confidence interval for the expected count (link scale, back-transformed)
SELECT (p).yhat, (p).yhat_lower, (p).yhat_upper
FROM (SELECT predict(poisson_fit_agg(n, [x], {'compute_inference': true}), [3.0], {'interval': 'confidence'}) AS p
      FROM (VALUES (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (4.0, 8.0), (5.0, 12.0)) t(x, n));
```

```sql
-- Collinear design: the later column is aliased (NaN), as R lm(); alias_tolerance = lm's tol
SELECT (ols_fit_agg(price, [sqm, 2 * sqm + 1e-9 * price], {'alias_tolerance': 1e-7})).coefficients
FROM houses;
```

```sql
-- Nested-model F test: do x2 and x3 improve on x1? (R: anova(lm(y ~ x1), lm(y ~ x1 + x2 + x3)))
CREATE TABLE anova_demo AS
SELECT i::DOUBLE AS x1, ((i * 3) % 7)::DOUBLE AS x2, sin(i) AS x3, 1.0 + 0.5 * i + 0.3 * sin(i) + cos(3 * i) AS y,
       i % 2 AS g
FROM range(1, 41) t(i);
SELECT (r).statistic, (r).df1, (r).df2, (r).p_value, (r).delta_adj_r_squared
FROM (SELECT ols_anova_agg(y, [x1], [x1, x2, x3]) AS r FROM anova_demo);
SELECT g, statistic, p_value, adj_r_squared_reduced, adj_r_squared_full
FROM ols_anova_by('anova_demo', g, y, x_reduced := [x1], x_full := [x1, x2, x3]);
```

```sql
-- Choose the ridge penalty by 5-fold CV: lambda.min and lambda.1se ("lambda" must be quoted)
CREATE TABLE cv_demo AS
SELECT i::DOUBLE AS x1, ((i * 3) % 7)::DOUBLE AS x2, sin(i) AS x3,
       1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) + cos(3 * i) AS y, (i % 5)::BIGINT AS fold
FROM range(1, 81) t(i);
SELECT c."lambda", round(c.cv_mean, 4) AS cv_mean, c.is_min, c.is_1se
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2, x3], fold, {'n_lambda': 30})) AS c FROM cv_demo)
WHERE c.is_min OR c.is_1se;

-- Time-series CV (rolling origin over i, 4 test blocks of 8, 2-step embargo), then the nested estimate
SELECT c."lambda", c.is_min, c.is_1se
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2, x3], x1,
        {'cv_strategy': 'time_series', 'folds': 4, 'test_size': 8, 'gap': 2, 'n_lambda': 20})) AS c FROM cv_demo)
WHERE c.is_min OR c.is_1se;
SELECT round(n.test_error, 4) AS nested_mse, round(n.test_se, 4) AS se
FROM (SELECT unnest(ridge_nested_cv_agg(y, [x1, x2, x3], fold, {'inner_folds': 4, 'n_lambda': 20})) AS n
      FROM cv_demo)
WHERE n.series = 'mean';

-- The lasso path at three penalties (glmnet scale), as terms rows
SELECT e.index_value AS lambda_, e.term, round(e.estimate, 4) AS estimate
FROM (SELECT unnest(elasticnet_path_agg(y, [x1, x2, x3], {'l1_ratio': 1.0, 'lambdas': [1.0, 0.3, 0.01]})) AS e
      FROM cv_demo)
ORDER BY lambda_ DESC, e.term;
```

```sql
-- Median regression with Hendricks-Koenker ('nid') standard errors, and the tau process
SELECT unnest(tidy(quantile_fit_agg(y, [x1], {'tau': 0.5, 'se': 'nid'}), ['x1']), recursive := true)
FROM cv_demo;
SELECT e.index_value AS tau, e.term, round(e.estimate, 3) AS estimate, round(e.conf_low, 3) AS lo, round(e.conf_high, 3) AS hi
FROM (SELECT unnest(quantile_process_agg(y, [x1], {'taus': [0.1, 0.5, 0.9], 'feature_names': ['x1']})) AS e FROM cv_demo)
WHERE e.term = 'x1';
```

```sql
-- AFT survival curve S(t | x) at chosen times
SELECT unnest(aft_survival(m, [1.0], [1.0, 5.0, 10.0]), recursive := true)
FROM (SELECT aft_fit_agg(t, [x], event, {'distribution': 'weibull'}) AS m
      FROM (VALUES (2.0, 0.0, 1.0), (3.0, 1.0, 1.0), (5.0, 0.0, 0.0), (7.0, 1.0, 1.0), (4.0, 1.0, 1.0),
                   (9.0, 0.0, 1.0), (6.0, 1.0, 0.0), (8.0, 0.0, 1.0), (3.5, 1.0, 1.0), (10.0, 0.0, 0.0)) d(t, x, event));
```

```sql
-- Bayesian Ridge: data-driven penalty, posterior SDs, posterior predictive band.
-- ARD prunes x3 (coefficient exactly 0).
CREATE TABLE bayes AS SELECT sin(i * 0.37) AS x1, cos(i * 0.11) AS x2, ((i * 13) % 17) / 17.0 AS x3,
  1 + 2 * sin(i * 0.37) - cos(i * 0.11) + 0.2 * sin(i * 1.7) AS y FROM range(1, 101) t(i);
SELECT round(1 / sqrt(m.noise_precision), 3) AS noise_sd, m.posterior_sd
FROM (SELECT bayesian_ridge_fit_agg(y, [x1, x2, x3]) AS m FROM bayes);
SELECT (ard_fit_agg(y, [x1, x2, x3])).coefficients FROM bayes;
SELECT p.yhat, p.yhat_lower, p.yhat_upper
FROM (SELECT unnest(bayesian_ridge_fit_predict_agg(y, [x1, x2, x3], {'confidence_level': 0.9})) AS p FROM bayes) LIMIT 3;
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
