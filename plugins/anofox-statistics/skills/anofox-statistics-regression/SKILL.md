---
name: anofox-statistics-regression
description: >
  Regression models and the fit / fit_agg / predict API surface of the
  anofox_statistics DuckDB extension. Covers OLS, WLS, robust estimators
  (Huber incl. joint scale, RANSAC, Theil-Sen), penalized regression (Ridge
  incl. conditioning-aware lambda, Elastic Net, LARS, rms penalized OLS and
  pentrace), Bayesian Ridge / ARD, Passive-Aggressive, lmDynamic, LOWESS,
  P-splines (pspline_fit_agg), RLS, BLS/NNLS, PLS, isotonic, quantile
  regression (se iid / nid / boot, L1 alpha, quantile_process_agg), GLMs
  (Poisson, Binomial/Logistic, Negative Binomial, Tweedie, Gamma; prior
  weights, cluster-robust vcov, profile-likelihood CIs), multinomial logistic,
  ordinal (orm / polr cumulative link, continuation ratio), ALM, AFT and Cox
  proportional hazards survival (cox_fit_agg ties / strata / entry / robust,
  cox_zph_agg, cox_survival, cox_augment_agg), GLS with correlation / variance
  structures (nlme::gls), mixed-effects GLMM (n_agq), areg optimal
  transformations, and empirical-Bayes shrinkage. Also the Harrell / rms
  model-building workflow: spline bases (rcs, lsp, bspline, rcs_knots_agg,
  spline_spec, spline_basis, restricted_interaction), effects / contrast_agg /
  predict_grid_agg / nomogram_agg, Wald chunk tests (wald_anova) and fastbw,
  lr_test and glm_score_test_agg, validate_agg / calibrate_agg /
  calibrate_cph_agg / bootcov_agg, multiple imputation (areg_impute_agg) with
  pool_rubin_agg; HC and cluster-robust vcov (vcov follows hc_type), the
  stable return-struct fields, the call surfaces (scalar fit, aggregate
  fit_agg, window *_fit_predict, *_fit_predict_agg), predict(model, x) with
  se_fit / intervals (incl. stored multinomial / ordinal / pspline / areg
  models), tidy and glance, regularisation paths and cross-validation
  (ridge_cv_agg, cv_strategy, nested CV), nested F tests and ANOVA tables,
  and exact LOO PRESS. Use when picking a regression method, choosing a
  penalty, getting standard errors or intervals, modelling survival / ordinal
  / multiclass / longitudinal outcomes, or validating a prediction model in
  SQL. For per-group long tables see the batch skill; for VIF / condition
  indexes / separation / accuracy indexes the diagnostics skill; for t / TOST
  / ANOVA / Kaplan-Meier style tests the tests skill.
version: 0.16.0
user-invocable: false
---

# Anofox Statistics — Regression Cheat Sheet

**Extension:** `anofox_statistics` v0.16.0 (Rust crate `anofox-regression`) | **DuckDB:** v1.5.6 / v2.0 | **API:** unprefixed, MAP-style options

> **New in 0.15.0:** `ls_state_agg(y, x[, weight])` / `ls_state_combine_agg(state)` keep a compact, mergeable least-squares state; `ols_fit_from_state` / `wls_fit_from_state` / `ridge_fit_from_state` fit from it without the rows (incremental refits, one state for any ridge `alpha`; no `hc_type` / `compute_loo`). Options may be `MAP(VARCHAR, VARIANT)` and prepared-statement parameters.
>
> **New in 0.16.0:** Cox PH, ordinal / continuation-ratio, multinomial, GLS, P-splines, `areg`, rms penalized OLS / `pentrace`, cluster-robust GLM / AFT `vcov`, `lr_test` / score tests / `profile_ci`, `wald_anova` / `fastbw`, spline bases + effects, `validate` / `calibrate` / `bootcov`, `areg_impute` + `pool_rubin` (sections below). **Behaviour changes:** with `hc_type` the `vcov` of `ols_fit_agg` / `wls_fit_agg` is the HC covariance (was classical), so `predict(..., {'se_fit': true})` is robust too; `wls_fit_agg` + `hc_type` now honours the weights (`vcovHC(lm(weights = w))`); `aft_cdf` / `aft_quantile` / `aft_survival` give NaN for pieces that are not a model (e.g. `exponential` with scale ≠ 1).

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
Fit-predict aggregates exist for those eight plus `bls`, `alm`, `poisson`, `pls`, `isotonic`, `quantile`, `pspline` (`*_fit_predict_agg`). The fit-predict aggregates also accept a split column: `ols_fit_predict_agg(y, x, split_col[, opts])` trains only on rows where `lower(split_col) IN ('train', 'training')` (a NULL split is a prediction row; the `*_fit_predict_by` macros' `split :=` follows the same rule).

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
13. **`predict` with `se_fit` / `interval` needs `vcov`**: fit an OLS, WLS, Ridge or GLM `*_fit_agg` with `{'compute_inference': true}`, otherwise it raises. `'prediction'` intervals are an error for GLMs. **`vcov` follows `hc_type`:** with `hc_type` the `vcov` (and so `se_fit`) of `ols_fit_agg` / `wls_fit_agg` is the HC covariance (`sandwich::vcovHC`); with a cluster argument the cluster-robust one. Every consumer of `vcov` (`wald_anova`, `fastbw`, effects, `pool_rubin_agg`) inherits that choice.
14. **`lambda` is a SQL keyword.** Quote the CV field as `c."lambda"` (or read `x`).
15. **Positional extra columns are taken out of `x`.** Cox `strata` / `entry` / `cluster` / `offset`, ordinal and GLM `offset` are 1-based positions *into `x`*; the coefficients belong to the remaining columns.
16. **The GLM 3rd / 4th positional argument is overloaded.** `poisson` / `gamma` / `tweedie_fit_agg(y, x, w[, opts])` = prior weights (not with `glm_lambda` / priors); a cluster needs `f(y, x, cluster, {'vcov': 'cluster'})`. OLS / WLS / AFT take a cluster positionally without that key.
17. **Stored models predict through `predict(model, x)`** for ordinal (needs the `information` field, so refit models from earlier builds), continuation ratio, multinomial (class label or `{'type': 'prob'}`), P-spline (`[x]`, one value) and areg (`areg_predict`). Cox uses `cox_survival`, AFT `aft_survival`.

## Models

### Linear / least squares

| Model | Functions | Notes |
|---|---|---|
| OLS | `ols_fit`, `ols_fit_agg`, `ols_fit_agg(y, x, cluster, opts)` | Supports `compute_inference`, `hc_type` (`none`/`hc0`–`hc3`, default `none`), `solver`, `alias_tolerance`. Cluster form: `sandwich::vcovCL` (cluster any type, LIST / STRUCT = multiway; `hc_type` default `hc1`, `cadjust`), adds `vcov_type`, `n_clusters` |
| WLS | `wls_fit`, `wls_fit_agg`, `wls_fit_agg(y, x, w, cluster, opts)` | Weight is the 3rd positional argument; `alias_tolerance`, `hc_type` (weighted `vcovHC`), cluster form as OLS |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive/online least squares; `forgetting_factor` (1.0), `initial_p_diagonal` (100) |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2; `alpha` (alias `lambda`), default `1.0`, must be ≥ 0. `alpha = 0` is accepted and reproduces OLS. Conditioning-aware: `{'conditioning': true}` (or `lambda_small` 0.1 / `cond_threshold` 1000) uses `lambda_small` when κ(XᵀX) ≤ `cond_threshold`, `alpha` otherwise |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | L1+L2; `alpha` (alias `lambda`, default 1.0), `l1_ratio ∈ [0,1]` (default 0.5) |
| LARS | `lars_fit_agg` | Least Angle Regression; options `fit_intercept`, `alpha` (alias `lambda`, default 0.0); returns the 7 standard fields |
| Bayesian Ridge / ARD | `bayesian_ridge_fit_agg`, `ard_fit_agg`, `*_fit_predict_agg`, `*_fit_predict_by` | scikit-learn `BayesianRidge` / `ARDRegression` (matches to ~1e-8): penalty from the evidence, no CV. Options `fit_intercept`, `max_iterations` (300), `tolerance` (1e-3), `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` (1e-6), ARD `threshold_lambda` (1e4). Returns `coefficients`, `intercept`, `r_squared`, `noise_precision` (sklearn `alpha_`; noise SD = 1/sqrt), `weight_precision` (BR) / `weight_precisions[]` (ARD, pruned ≥ threshold → coef exactly 0), `posterior_sd[]`, `iterations`, `log_marginal_likelihood` (BR). The inference fields hold the posterior: `std_errors` = `posterior_sd`, `conf_low`/`conf_high` and `intercept_conf_*` = central **credible** intervals at `confidence_level` (intercept SD `sqrt(1/(nα) + x̄ᵀΣx̄)`); `t_values`/`p_values` and `aic`/`bic` NULL. `*_tidy_by` carries them into `terms`. Fit-predict bands are posterior predictive intervals at `confidence_level` |
| Passive-Aggressive | `passive_aggressive_fit_agg(y, x[, opts] ORDER BY t)`, window `passive_aggressive_fit_predict`, `*_fit_predict_agg`, `*_fit_predict_by` | Online regression (sklearn `PassiveAggressiveRegressor`, matches to 1e-9): rows updated in **ORDER BY order** (no ORDER BY = arbitrary order). Options `c` (1.0), `epsilon` (0.1), `variant` (`'pa1'`/`'pa2'`), `fit_intercept`, `max_iterations` (passes, 1 = online `partial_fit`), `tolerance`. Returns `coefficients`, `intercept`, `r_squared`, `n_observations`, `n_features`, `iterations`; no intervals (bounds NULL) |
| lmDynamic | `lm_dynamic_fit_agg(y, x[, opts] ORDER BY t)`, `lm_dynamic_fit_predict_agg`, `_by` macros | Time-varying coefficients (greybox `lmDynamic` bruteforce, matches to 1e-8): all 2^p subsets (p ≤ 14) weighted per row by pointwise `ic` (`'aicc'`/`'aic'`/`'bic'`), weights smoothed with lowess `span` (0.3; 0 = none). Returns time-averaged `coefficients`/`intercept`, `n_models`, per-row `dynamic_intercept[]`, `dynamic_coefficients[][]`, `fitted_values[]`, greybox `log_likelihood` |
| LOWESS | `lowess_agg(y, x[, frac[, iterations]])`, `lowess_by` | R `stats::lowess` (matches to 1e-10): returns `LIST(STRUCT(x, yhat))`, x increasing; `frac` 2/3, `iterations` 3 (positional, not a MAP) |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded (`lower_bound`, `upper_bound`) / non-negative coefficients |
| PLS | `pls_fit_agg`, `pls_fit_predict_agg` | Partial least squares; `n_components` (1), `fit_intercept`. Returns `coefficients`, `intercept`, `r_squared`, `n_components`, `n_observations`, `n_features` |
| Isotonic | `isotonic_fit_agg`, `isotonic_fit_predict_agg` | Monotonic regression on a single `x DOUBLE`; `increasing` (true). Returns knots `x[]`, `fitted[]`, `increasing`, `r_squared`, `n_observations`; `predict(m, [x])` interpolates, clamped at the ends |
| Quantile | `quantile_fit_agg`, `quantile_fit_predict_agg` | Quantile / median regression; `tau` (alias `quantile`, 0.5), `fit_intercept`, `max_iterations`, `tolerance`, `alpha` (L1 penalty on the slopes, sklearn `QuantileRegressor(alpha)`, default 0; no inference with `alpha > 0`). Returns `coefficients`, `intercept`, `tau`, `n_observations`, `n_features` + inference fields. Standard errors with `se` (`'iid'`, `'nid'`, `'boot'`; `compute_inference: true` = `'nid'`), `n_boot` (200), `seed` (42), as `quantreg::summary.rq` |
| Nested-model F test | `ols_anova_agg(y, x_reduced, x_full[, opts])`, `ols_anova_by(src, g, y, x_reduced := [...], x_full := [...])` | R `anova(lm(reduced), lm(full))` on the same rows (matches to 1e-9): does `x_full` beat `x_reduced`? `x_reduced` columns must appear in `x_full` (error otherwise); `[]` = intercept-only (overall F). Returns `test` fields `method` (`'nested F'`), `statistic`, `df1`, `df2`, `p_value` + `rss_reduced`, `rss_full`, `adj_r_squared_reduced`, `adj_r_squared_full`, `delta_adj_r_squared`, `n_observations`. Rank counted as R; `df1 = 0` or `df2 = 0` → NULL (not NaN) statistic and p-value. Option `fit_intercept` only |
| Per-term ANOVA table | `ols_anova_table_agg(y, x[, opts])` → `STRUCT[]`, `ols_anova_table_by(src, g, y, [x...], options := {...})` | R `anova(lm(...))` (type I, sequential in column order, default) or `car::Anova(fit, type = 2)` (`{'type': 'II'}`, each term adjusted for all others); matches to 1e-9. One row per term: `term`, `method`, `statistic` (F), `df1` (term df), `df2` (residual df), `p_value`, `sum_sq`, `mean_sq`, `residual_sum_sq`, `residual_mean_sq`, `n_observations`. Terms `x1`, `x2`, … or `'terms': [...]` labels (one per column; equal labels = one term, e.g. factor dummies). Aliased term → `df1 = 0`, NULL F/p. Options `type`, `terms`, `fit_intercept`; no interactions |
| Quantile process | `quantile_process_agg(y, x[, opts])` | One fit per tau as `terms` rows (`index_name = 'tau'`); `taus` (default 0.1…0.9), `se` (`'nid'`), `compute_inference` (false = estimates only), `feature_names` |
| GLS | `gls_fit_agg(y, x, subject, time[, var_cov], opts)`, window `gls_augment_agg`, `gls_fit_by` / `_tidy_by` / `_glance_by` / `_augment_by` | `nlme::gls`: errors correlated within `subject`. `correlation` `independent` / `compound_symmetry` (`cs`) / `ar1` (integer times) / `car1` / `exponential` / `gaussian` / `linear` / `spherical` (+ `nugget`) / `unstructured`; `variance` `constant` / `ident` (stratum) / `power` / `exp` of the 5th argument; `method` `'reml'` (default) / `'ml'` (use ML to compare fixed effects); `variogram: k`. Returns t tests on N−p df, `sigma`, `correlation_params` (phi / rho / range), `variance_params`, logLik / AIC / BIC, `vcov`, var-cov intervals with `compute_inference`. Pick the structure by REML AIC with the same fixed effects |
| P-spline | `pspline_fit_agg(y, x[, opts])`, `pspline_fit_predict_agg(y, x[, split][, opts])`, `pspline_fit_predict_by`, `pspline_prediction_by` | mgcv `s(x, bs = "ps")`, GCV-chosen `lambda` unless given; `x` is a single DOUBLE. Options `n_basis` (alias `k`; 0 = n/4+4 in [6, 40]; mgcv default is 10), `penalty_order` (2), `lambda`; fit-predict `interval` (`'prediction'` default / `'confidence'` / `'none'`). Returns `x[]`, `fitted[]`, `se_fit[]`, `edf`, `sigma2`, `knots`, `covariance`, logLik / AIC / BIC. `predict(m, [x], {'interval': …})` evaluates a stored fit, linear beyond the data. NULL under 6 rows |
| areg | `areg_fit_agg(y, x[, opts])`, `areg_predict(m, x[, {'type': 'fitted'}])`, `areg_transform(m, y)`, `areg_inverse(m, t)` | `Hmisc::areg`: optimal (spline) transformations of both sides; `nk` (4), `x_types` (`s` / `l` / `c`, one or per predictor), `y_type`. `areg_predict` = linear predictor on the transformed-y scale, `'fitted'` back on the y scale (NULL if not monotone). NULL under 6 rows. Reference: `docs/api/statistics/reduction.md` |
| Penalized OLS (rms) | `penalized_ols_fit_agg(y, x[, opts])`, `pentrace_agg(y, x, opts)`, `penalized_ols_fit_by`, `pentrace_by` | `rms::ols(penalty =)` / `rms::pentrace`: quadratic penalty on **standardised** coefficients, separate by column type via `terms`, `nonlinear`, `interactions`, `categorical` (as `wald_anova`). `penalty` / `penalties` a number / list or `{'simple', 'nonlinear', 'interaction', 'nonlinear_interaction'}`; `var_penalty` `'simple'` / `'sandwich'`. `pentrace_agg` traces `model` `'ols'` / `'logistic'` / `'ordinal'` (+ `link`), one row per penalty with `df` (effective), `lr_chisq`, `aic` / `bic` / `aic_c` (chi-square scale: **larger is better**), `is_best` by `criterion` (`'aic_c'`). Penalty choice without CV |

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
| Huber | `huber_fit`, `huber_fit_agg` | `epsilon` (default 1.35), `alpha` (1e-4), `max_iterations` (100), `scale_method` (`'mad'` default = `MASS::rlm`-style; `'joint'` = sklearn `HuberRegressor` joint coefficient / scale estimate); reports `scale`, `n_outliers` |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | `max_trials` (100), `residual_threshold` (default MAD of y), `min_samples` (default `n_features + 1`), `random_state`; reports `residual_threshold`, `n_inliers`, `n_trials` |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Median of pairwise slopes (1 feature) / spatial median over subsamples; `n_subsamples`, `max_subpopulation`, `random_state` (alias `seed`). `compute_inference`: `ci_method` `'exact'` (default for 1 feature: Sen's slope CI + Kendall z in `t_values` / p-value, = R `trend::sens.slope`; `std_errors` / intercept CI NULL) or `'bootstrap'` (default otherwise: pairs bootstrap SDs + percentile CIs incl. intercept, `n_bootstrap` 999, deterministic per seed; `t_values`/`p_values` NULL; costs ~`n_bootstrap` point fits per group, ~8 s/group at 500 rows × 3 features). `theil_sen_tidy_by` defaults to `'exact_or_none'` (multi-feature intervals NULL, no bootstrap); opt in with `options := {'ci_method': 'bootstrap'}` (199 resamples there). No F-test (NULL) |

### GLMs, ordinal, multiclass & survival (report `z_values`, no `r_squared`)

| Model | Function | For |
|---|---|---|
| Poisson | `poisson_fit_agg` | Count data (`link`: log / identity / sqrt) |
| Binomial | `binomial_fit_agg` | Success-rate data (`binomial_link`: logit / probit / cloglog) |
| Negative Binomial | `negbinom_fit_agg` | Overdispersed counts; `dispersion` field = estimated α |
| Tweedie | `tweedie_fit_agg` | Positive-skew continuous with zeros (`power`, default 1.5) |
| Gamma | `gamma_fit_agg` | Strictly-positive continuous (claims, durations) |
| Logistic | `logistic_fit_agg` | Binary classification; reports `accuracy` + `threshold` (default 0.5) |
| ALM | `alm_fit_agg` | Augmented Linear Model, 24 error distributions (`distribution`, `loss`); reports `t_values` |
| AFT survival | `aft_fit_agg(time, x, event[, cluster, opts])` | Duration models with right censoring (`distribution`: weibull / lognormal / loglogistic / exponential). Cluster form = `survreg(cluster =)` (`cadjust` default false). Residuals: window `aft_augment_agg` / `aft_augment_by` (linear predictor, standardized, Cox-Snell, score) |
| AFT helpers | `aft_cdf(t, eta, scale, dist)`, `aft_quantile(p, eta, scale, dist)` | Scalar: `P(T ≤ t)` and the `p`-quantile of `T` given linear predictor `eta = intercept + x·β` and the fitted `scale` |
| AFT survival curve | `aft_survival(model, x, times)` | `LIST<STRUCT(time, surv)>`: `S(t \| x)` of an `aft_fit_agg` model at each of `times` |
| Cox PH | `cox_fit_agg(time, x, event[, weight][, opts])` | `survival::coxph`; `x` has **no intercept**. `ties` `'efron'` (default) / `'breslow'`; `compute_inference` **on by default**; `robust`, and 1-based positions in `x` of `cluster`, `strata`, `entry` (counting process `(entry, time]`: left truncation / time-varying covariates), `offset`; `ridge`, `timefix`. Returns log-HR `coefficients`, `hazard_ratios` + `hr_conf_low/high`, `naive_std_errors`, `lr_` / `wald_` / `score_statistic` (+ `_p_value`), `concordance` (Harrell's C), `pseudo_r_squared` (rms R2), `baseline` cumulative hazard per stratum. Fractional weights do **not** auto-switch to robust (unlike R): pass `'robust': true` |
| Cox helpers | `cox_zph_agg(...)` (same args), `cox_survival(m, x, times[, stratum])`, window `cox_augment_agg`, `cox_fit_by` / `_tidy_by` / `_glance_by` / `_augment_by` | `cox.zph` PH test per covariate + `GLOBAL` (`transform` `'km'` / `'rank'` / `'identity'` / `'log'`); `S(t \| x)` point estimate (no interval, offset not added; stratified model needs the stratum); per-row martingale / deviance / score / dfbeta(s) / (scaled) Schoenfeld residuals. Calibration: `calibrate_cph_agg` (below) |
| Ordinal (cumulative link) | `ordinal_fit_agg(y, x[, weight][, opts])`, `ordinal_fit_predict_by` | `rms::orm` / `MASS::polr`: `link` `logistic` / `probit` / `loglog` / `cloglog` / `cauchit`, `penalty`, `offset` (position in `x`), `po_test`. Returns `levels`, `intercepts` (orm's α_j; **polr's ζ_j = −α_j**), slopes with Wald z, `lr_chisq`, AIC, proportional-odds score test `po_test_statistic / _df / _p_value`, `information` (needed by predict). A continuous `y` = orm's semiparametric model (one intercept per value) |
| Continuation ratio | `continuation_ratio_fit_agg(y, x[, weight][, {'cohort_specific': [j]}])`, `continuation_ratio_fit_predict_by` | `rms::cr.setup` + `lrm`, ≥ 3 levels. Coefficients: θ₁, cohort effects, slopes, then cohort-specific slope differences |
| Ordinal predict | `ordinal_predict(m, x[, {'type': …}])` = `predict(m, x, …)` | `'class'` (default), `'prob'` (P(Y = y_j)), `'exceedance'` (P(Y ≥ y_{j+1})), `'mean'`, `'quantile'` (cumulative link only: `q` 0.5, `level`, `method` `'interpolated'` / `'discrete'` → `STRUCT(yhat, yhat_lower, yhat_upper)`, `rms::Quantile.orm`) |
| Multinomial | `multinomial_fit_agg(y, x[, weight][, opts])`, `multinomial_predict(m, x[, {'type': 'prob'}])` = `predict`, `multinomial_fit_predict_by` | Softmax logistic, sklearn `LogisticRegression` objective (penalized: matches sklearn; `lambda` 0: `nnet::multinom`). `y` VARCHAR or integer (**DOUBLE rejected**). `C` (1.0) **or** `lambda` (= 1/C; 0 = none), `l1_ratio` (0 ridge … 1 lasso). Returns `classes`, `coefficients` (one row per class; **a single log-odds row for two classes**; centred over classes when unpenalized — `coefficients[k] − coefficients[1]` = multinom's baseline log-odds), `intercepts`, `accuracy`, `converged`. No standard errors: for binary inference use `logistic_fit_agg` |
| Mixed effects | `glmm_fit_agg(y, x, group[, opts])` | Random intercepts / slopes; `family` option selects the GLM family; `n_agq` (lme4 `nAGQ`: 0 default = `glmer(nAGQ = 0)`, 1 = glmer's Laplace default, k > 1 adaptive Gauss-Hermite, scalar random intercept only, ≤ 25). With `compute_inference`: Satterthwaite t tests for gaussian (lmerTest; `t_values` + `df`, `z_values` NULL), Wald z otherwise (`z_values`; `ddf` option); `ranef[].se` = BLUP conditional SEs. Long outputs: `glmm_tidy_by` / `glmm_glance_by` / `glmm_augment_by` (batch skill) |
| EB shrinkage | `eb_shrink_agg(estimate, se[, opts])` | Partial pooling of per-group estimates |

All GLMs accept `glm_lambda` (L2 penalty), `prior` + `feature_names` (coefficient priors), `max_iterations`, `tolerance`, `compute_inference`, `confidence_level`, `profile_ci` (profile-likelihood `conf_low/high` = R `confint(glm)`; implies `compute_inference`, z / p stay Wald; not with `glm_lambda` / priors; ~20 IRLS fits per coefficient; unreached bound = NULL). There is no `family` key on the single-family GLM aggregates. **Prior weights:** `poisson_fit_agg` / `gamma_fit_agg` / `tweedie_fit_agg(y, x, w[, opts])` = `glm(weights = w)`. **Cluster-robust:** `{poisson,binomial,logistic,negbinom,gamma,tweedie}_fit_agg(y, x, cluster, {'vcov': 'cluster', …})` = `sandwich::vcovCL(glm)` (`hc_type` `'hc0'` default / `'hc1'`, HC2/3 error; `cadjust` default true), appends `vcov_type`, `n_clusters`. Every GLM has a per-group table macro `{poisson,binomial,logistic,negbinom,gamma,tweedie}_fit_predict_by` (response-scale `yhat`; see the batch skill).

## Model building, inference and validation (rms workflow)

These take an ordinary `DOUBLE[]` design, or a fitted model STRUCT with `vcov` (fit with `{'compute_inference': true}`; a `vcov` NULL raises).

| Task | Functions | Notes |
|---|---|---|
| Spline bases | `rcs(x, knots[, {'norm'}])`, `lsp(x, knots)`, `bspline(x, knots[, degree[, {'extrapolation', 'include_bias'}]])`, `restricted_interaction(a, b[, a_nl, b_nl])` (`%ia%`) | Expand one value into a basis row for any `*_fit_agg`. `rcs` with k knots → k−1 columns (`x` first, then nonlinear), linear beyond the outer knots. NULL / NaN `x` → NULL row (skipped by fits) |
| Knots / spec | `rcs_knots_agg(x[, nk])` (default 5, = `rms::rcs` default knots), `bspline_knots_agg(x[, n[, {'knots': 'quantile'}]])`, `default_knot_quantiles(nk)`, `spline_spec(kind, knots[, opts])`, `spline_basis(x, spec)` | `spline_spec` returns `{kind, knots, …, n_columns, nonlinear[]}`; **keep the spec next to the model** and expand new data with `spline_basis` so the knots match. `rcs_knots_agg` may return fewer knots than asked |
| Effects | `effects_term(name, 'linear' \| 'categorical'[, levels])` / `effects_term(name, spline_spec(...))`, `effects_interaction(a, b[, restricted])`, `effects_design(x, terms)` | Describe how each predictor enters; fit any `*_fit_agg` on `effects_design([raw predictors], terms)` (main effects in term order, then interactions). Categorical = treatment dummies; give `levels`. No B-spline terms |
| Effect summaries | `effects_summary_agg(x, m, terms)`, `contrast_agg(x, m, terms, a, b)`, `double_contrast_agg(…, a, b, a2, b2)`, `predict_grid_agg(x, m, terms, 'term'[, {'values', 'simultaneous'}])`, `nomogram_agg(x, lp, m, terms[, {'fun': 'plogis'}])`, `predict_grid_by` | `rms::summary` (IQR / level effects; `ratio` for logit / log links), `contrast`, `Predict`, `nomogram`. Aggregates over the **data rows** (datadist: quartiles, median / modal level). Settings are constant STRUCTs `{'age': 60}`. GLM results on the link scale. Every level of a categorical term must occur in the data (per group!) |
| Wald chunk tests | `wald_anova(m[, terms[, {'nonlinear', 'interactions', 'test'}]])` | `rms::anova`: per term all its columns incl. interactions, ` Nonlinear`, interaction chunks, `TOTAL …`. `terms` one label per coefficient (equal labels = one term, NULL = left out). F + partial SS for OLS / WLS / Ridge, chi-square otherwise. `chi_square_minus_df` = importance. Labels keep rms's leading space: `trim(label)` |
| Step-down | `fastbw(m[, terms[, {'rule': 'aic' \| 'p', 'sls', 'type'}]])` | `rms::fastbw`: `deleted[]`, `kept_terms`, approximate reduced `coefficients` (0 = deleted), reduced `vcov`, `kept_parameters`. NULL-labelled terms are forced in. Ignores hierarchy; report the full model's inference, not the selected one |
| LR / score tests | `lr_test(m_full, m_reduced)`, `lr_test(m)` (vs null model; not OLS), `glm_score_test_agg(y, x_reduced, x_add[, {'family', 'link', 'offset'}])`, `glm_score_test_by` | `anova(test = "LRT")` / `lmtest::lrtest`; both fits from the same function on the **same rows** (checked: type, family, n); nesting is your responsibility. GLMM / GLS: compare ML fits. Score test fits only the reduced model (`anova(test = "Rao")`). Profile CIs: `profile_ci` (GLMs above) |
| Validation | `validate_agg(y, x[, {'model': 'ols' \| 'logistic', 'method': 'boot' \| 'crossvalidation' \| '.632', 'B', 'seed'}])`, `calibrate_agg` (`predy` grid), `calibrate_cph_agg(time, x, event, {'u': …})`, `validate_by`, `calibrate_by`, `calibrate_cph_by` | `rms::validate` / `calibrate`: optimism-corrected Dxy, R2, slope, Brier …; the model is refitted internally on `x` (OLS or logistic, or Cox for `calibrate_cph`), so pass the full prespecified design. Seeded, order / thread independent, not R's draws |
| Bootstrap covariance | `bootcov_agg(y, x[, cluster], opts)`, `bootcov_by` | `rms::bootcov`: `terms`-like rows with bootstrap SE, percentile CI, `vcov` row; cluster bootstrap resamples whole clusters (options required with a cluster) |
| Multiple imputation | `areg_impute_agg(id, x[, opts])` / `areg_impute_by(src, g, id, [cols])`, `pool_rubin_agg(model[, opts])` | `Hmisc::aregImpute` (same seed replays R): `x` = **all** analysis variables incl. the outcome, NULL = missing; `n_impute` (5), `nk`, `categorical` / `linear` (1-based), `seed`. Fit one model per `imputation` (with `compute_inference`), then `pool_rubin_agg(fit)`: pooled coefficients, Barnard-Rubin `df`, `fmi`, pooled `vcov`, D1 test (`f_statistic`, `f_pvalue`); `tidy()` works on it |

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
| `hc_type` | VARCHAR | `'none'` (`'hc1'` OLS/WLS cluster, `'hc0'` GLM/AFT cluster) | OLS, WLS robust SEs (`hc0`–`hc3`; `vcov` follows it); GLM / AFT cluster `hc0` / `hc1` |
| `vcov`, `cadjust` | VARCHAR / BOOLEAN | — / true (GLM, OLS) false (AFT) | GLM cluster form needs `'vcov': 'cluster'`; `cadjust` = G/(G−1) |
| `profile_ci` | BOOLEAN | `false` | GLM aggregates: profile-likelihood intervals |
| `conditioning`, `lambda_small`, `cond_threshold` | BOOLEAN / DOUBLE | false / 0.1 / 1000 | Ridge conditioning-aware penalty |
| `scale_method` (alias `scale`) | VARCHAR | `'mad'` | Huber (`'joint'` = sklearn) |
| `n_agq` (alias `nagq`) | INTEGER | `0` | GLMM non-gaussian families (lme4 `nAGQ`) |
| `ties`, `robust`, `strata`, `entry`, `cluster`, `offset`, `ridge` | — | `'efron'`, false, none… | Cox (`strata` / `entry` / `cluster` / `offset` = 1-based positions in `x`) |
| `link`, `penalty`, `po_test` | VARCHAR / DOUBLE / BOOLEAN | `'logistic'`, 0, true (≤ 50 levels) | Ordinal |
| `C` / `lambda`, `l1_ratio` | DOUBLE | 1.0, 0.0 | Multinomial (give `C` or `lambda`, not both) |
| `correlation`, `variance`, `method`, `nugget`, `variogram` | — | `'independent'`, `'constant'`, `'reml'`, false, 0 | GLS |
| `n_basis` (`k`), `penalty_order`, `lambda`, `interval` | — | 0 (auto), 2, GCV, `'prediction'` | P-spline |
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
| `alpha` (quantile) | DOUBLE | `0` | Quantile: L1 penalty on the slopes (no inference when > 0) |

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

**Cox, ordinal, continuation ratio, multinomial, GLS, P-spline, areg, penalized OLS:** model-specific structs, fields listed in their rows above (`model_type` `'cox'`, `'ordinal'`, `'continuation_ratio'`, `'multinomial_logistic'`, `'gls'`, `'pspline'`, `'areg'`, `'penalized_ols'`). `wald_anova`, `fastbw` and the effects functions are for OLS / WLS / Ridge / GLM structs with `vcov`.

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
OLS, WLS and Ridge (`*_fit_agg` and `ols_fit` / `wls_fit` / `ridge_fit`) then add the leave-one-out statistics `press`
(Σ (eᵢ/(1−hᵢᵢ))², R `sum(rstandard(fit, type = "predictive")^2)`; WLS weighted), `press_rmse`
(√(press/n)) and `loo_r_squared` (predicted R² / Q² = 1 − press/SST, never above `r_squared`), NULL
unless `{'compute_loo': true}` (opt-in, one more QR / SVD; not implied by `compute_inference`; `ols_glance_by` /
`wls_glance_by` / `ridge_glance_by` turn it on) and when a row has leverage 1. `glance()` carries them too (NULL for other models);
per-row `loo_residual` / `loo_fitted` are in `ols_augment_by` / `wls_augment_by` / `ridge_augment_by`. Ridge is exact
too: hat matrix `X(XᵀX + λI)⁻¹Xᵀ`, intercept unpenalised, same λ (glmnet scaling: `alpha·n` of the full fit held
fixed), so PRESS over an `alpha` grid is exact LOO-CV. GLMs have no exact LOO; their `_augment_by` has the one-step
approximation `approx_loo_fitted` / `approx_loo_residual` instead (not the refit).

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
-- Per-term tables: sequential (R anova(fit)) and partial (car::Anova(fit, type = 2))
SELECT (r).term, (r).df1, (r).sum_sq, (r).statistic, (r).p_value
FROM (SELECT unnest(ols_anova_table_agg(y, [x1, x2, x3])) AS r FROM anova_demo);
SELECT g, term, statistic, p_value
FROM ols_anova_table_by('anova_demo', g, y, [x1, x2, x3], options := {'type': 'II', 'terms': ['x1', 'x2', 'x3']});
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

```sql
-- Cox PH: hazard ratios, C index, PH test, predicted survival; stratified + delayed entry
CREATE TABLE surv AS
SELECT i AS id, (40 + (i * 7) % 35)::DOUBLE AS age, (i % 2)::DOUBLE AS treated, (i % 3)::DOUBLE AS site,
       ((i * 11) % 20)::DOUBLE AS entry_day,
       least(25 + 300 * exp(-0.03 * ((40 + (i * 7) % 35) - 55) + 0.6 * (i % 2)) * (0.05 + ((i * 37) % 101) / 101.0), 400) AS days,
       (25 + 300 * exp(-0.03 * ((40 + (i * 7) % 35) - 55) + 0.6 * (i % 2)) * (0.05 + ((i * 37) % 101) / 101.0) <= 400)::INTEGER::DOUBLE AS died
FROM range(200) t(i);
SELECT round(m.hazard_ratios[2], 3) AS hr_treated, round(m.hr_conf_low[2], 3) AS lo, round(m.hr_conf_high[2], 3) AS hi,
       round(m.concordance, 3) AS c, m.n_events
FROM (SELECT cox_fit_agg(days, [age, treated], died, {'ties': 'efron'}) AS m FROM surv);
SELECT unnest(cox_zph_agg(days, [age, treated], died), recursive := true) FROM surv;
SELECT (cox_fit_agg(days, [age, treated, site, entry_day], died, {'strata': 3, 'entry': 4, 'robust': true})).coefficients FROM surv;
SELECT unnest(cox_survival(m, [55.0, 1.0], [90, 180, 365]), recursive := true)
FROM (SELECT cox_fit_agg(days, [age, treated], died) AS m FROM surv);
```

```sql
-- Ordinal (proportional odds), predicted probabilities / median; continuation ratio; multinomial
CREATE TABLE ord AS
SELECT i, (30 + (i * 37) % 41)::DOUBLE AS age, ((i * 7) % 3)::DOUBLE AS dose,
       (CASE WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 1.5 THEN 1
             WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 3.0 THEN 2
             WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 4.5 THEN 3 ELSE 4 END)::DOUBLE AS rating,
       ['a', 'b', 'c'][1 + (CASE WHEN ((i * 37) % 41) / 10.0 + sin(i) < 1.5 THEN 0 WHEN ((i * 37) % 41) / 10.0 + sin(i) < 3 THEN 1 ELSE 2 END)] AS cls
FROM range(1, 201) t(i);
WITH f AS (SELECT ordinal_fit_agg(rating, [age, dose]) AS m FROM ord)
SELECT m.coefficients, round(m.po_test_p_value, 4) AS po_p,
       predict(m, [50, 1]) AS most_probable,
       ordinal_predict(m, [50, 1], {'type': 'prob'}) AS probs,
       ordinal_predict(m, [50, 1], {'type': 'quantile', 'q': 0.5}) AS median_ci
FROM f;
SELECT (continuation_ratio_fit_agg(rating, [age, dose])).coefficients FROM ord;
WITH f AS (SELECT multinomial_fit_agg(cls, [age, dose], {'C': 1.0}) AS m FROM ord)
SELECT m.classes, round(m.accuracy, 3) AS acc, predict(m, [50, 1]) AS cls_hat,
       multinomial_predict(m, [50, 1], {'type': 'prob'}) AS probs
FROM f;
```

```sql
-- GLS with AR(1) errors within subject; P-spline smooth evaluated from the stored model
CREATE TABLE visits AS
SELECT 'p' || (i // 6) AS patient, (i % 6 + 1) AS week, 0.5 + ((i * 13) % 11) / 4.0 AS dose,
       2 + 0.8 * (0.5 + ((i * 13) % 11) / 4.0) - 0.2 * (i % 6 + 1) + ((i // 6) % 5) * 0.3
         + (((i * 37) % 17) - 8) / 8.0 AS score
FROM range(120) t(i);
SELECT r.coefficients, round(r.correlation_params[1], 3) AS phi, round(r.aic, 2) AS aic
FROM (SELECT gls_fit_agg(score, [dose, week], patient, week, {'correlation': 'ar1'}) AS r FROM visits);
WITH fit AS (SELECT pspline_fit_agg(sin(i / 10.0) + ((i * 7919) % 97 / 97.0 - 0.5) * 0.4, i / 10.0, {'n_basis': 20}) AS m
             FROM range(1, 101) t(i))
SELECT round(m.edf, 2) AS edf, round(predict(m, [5.05]), 3) AS yhat,
       predict(m, [5.05], {'interval': 'confidence'}) AS band
FROM fit;
```

```sql
-- rms workflow: rcs spline, Wald chunk tests, fastbw, LR test, cluster-robust GLM, profile CIs
CREATE TABLE rms_demo AS
SELECT i, (20 + (i * 37) % 50)::DOUBLE AS age, (100 + (i * 13) % 40)::DOUBLE AS bp,
       (i % 3 = 0)::INTEGER::DOUBLE AS treated, sin(i * 0.7) AS noise, i % 10 AS clinic,
       0.04 * (20 + (i * 37) % 50) + 0.002 * ((20 + (i * 37) % 50) - 45) ^ 2 + 0.03 * (100 + (i * 13) % 40)
         + 0.8 * (i % 3 = 0)::INTEGER + sin(i * 1.3) AS outcome,
       ((i * 7919) % 100 < (20 + (i * 37) % 50) * 1.2 - 15)::INTEGER::DOUBLE AS event
FROM range(300) t(i);
CREATE TABLE age_spec AS SELECT spline_spec('rcs', rcs_knots_agg(age, 4)) AS spec FROM rms_demo;
-- Does age matter, and is it nonlinear? (anova.rms)
SELECT trim(r.label) AS row, r.df1, round(r.statistic, 2) AS f, round(r.p_value, 4) AS p
FROM (SELECT unnest(wald_anova(
        ols_fit_agg(outcome, list_concat(spline_basis(age, s.spec), [bp, treated, noise]), {'compute_inference': true}),
        ['age', 'age', 'age', 'bp', 'treated', 'noise'],
        {'nonlinear': [false, true, true, false, false, false]})) AS r
      FROM rms_demo, age_spec s);
SELECT (fastbw(logistic_fit_agg(event, [age, bp, noise], {'compute_inference': true}), ['age', 'bp', 'noise'])).kept_terms
FROM rms_demo;
SELECT round(t.statistic, 3) AS lr, t.df1, round(t.p_value, 4) AS p
FROM (SELECT lr_test(logistic_fit_agg(event, [age, bp, noise]), logistic_fit_agg(event, [age])) AS t FROM rms_demo);
SELECT m.std_errors, m.vcov_type, m.n_clusters
FROM (SELECT logistic_fit_agg(event, [age, bp], clinic, {'vcov': 'cluster'}) AS m FROM rms_demo);
SELECT (logistic_fit_agg(event, [age, bp], {'profile_ci': true})).conf_low FROM rms_demo;
-- Choose an rms penalty by corrected AIC (larger is better), nonlinear spline columns penalised
SELECT r.simple AS penalty, round(r.df, 2) AS eff_df, round(r.aic_c, 2) AS aic_c, r.is_best
FROM (SELECT unnest(pentrace_agg(outcome, list_concat(spline_basis(age, s.spec), [bp, treated]),
        {'terms': ['age', 'age', 'age', 'bp', 'treated'], 'nonlinear': [false, true, true, false, false],
         'penalties': [0, 1, 4, 16]})) AS r
      FROM rms_demo, age_spec s);
```

```sql
-- Effects on the covariate scale: IQR effect and a contrast for rcs(age) + treated
CREATE TABLE fx_terms AS
SELECT [effects_term('age', spline_spec('rcs', rcs_knots_agg(age, 4))),
        effects_term('treated', 'categorical', [0, 1])] AS terms FROM rms_demo;
CREATE TABLE fx_fit AS
SELECT ols_fit_agg(outcome, effects_design([age, treated], t.terms), {'compute_inference': true}) AS m,
       any_value(t.terms) AS terms
FROM rms_demo, fx_terms t;
SELECT r.label, round(r.estimate, 3) AS effect, round(r.conf_low, 3) AS lo, round(r.conf_high, 3) AS hi
FROM (SELECT unnest(effects_summary_agg([age, treated], f.m, f.terms)) AS r FROM rms_demo, fx_fit f);
SELECT round(c.estimate, 3) AS age60_vs_40, round(c.p_value, 4) AS p
FROM (SELECT contrast_agg([age, treated], f.m, f.terms, {'age': 60}, {'age': 40}) AS c FROM rms_demo, fx_fit f);
```

```sql
-- Validation (optimism bootstrap), bootstrap covariance by cluster, multiple imputation + Rubin pooling
SELECT r.index_name, round(r.index_orig, 3) AS apparent, round(r.index_corrected, 3) AS corrected
FROM (SELECT unnest(validate_agg(event, [age, bp], {'model': 'logistic', 'B': 50, 'seed': 1})) AS r FROM rms_demo)
WHERE r.index_name IN ('Dxy', 'Slope', 'B');
SELECT r.term, round(r.std_error, 4) AS boot_se
FROM (SELECT unnest(bootcov_agg(outcome, [age, bp], clinic, {'B': 50, 'feature_names': ['age', 'bp']})) AS r FROM rms_demo);
CREATE TABLE imp_demo AS
SELECT i AS id, 1 + 0.5 * (i % 7) + 0.3 * (i % 5) + ((i * 37) % 11) * 0.1 AS y,
       CASE WHEN i % 6 = 0 THEN NULL ELSE (i % 7)::DOUBLE + ((i * 13) % 5) * 0.2 END AS x1,
       CASE WHEN i % 8 = 3 THEN NULL ELSE (i % 5)::DOUBLE END AS x2
FROM range(1, 81) t(i);
WITH imp AS (SELECT * FROM areg_impute_by('imp_demo', 1, id, [y, x1, x2], options := {'n_impute': 5, 'seed': 2026})),
     fits AS (SELECT imputation, ols_fit_agg(x[1], [x[2], x[3]], {'compute_inference': true}) AS fit FROM imp GROUP BY imputation)
SELECT unnest(tidy(pool_rubin_agg(fit), ['x1', 'x2']), recursive := true) FROM fits;
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
