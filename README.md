# Anofox Statistics — DuckDB Extension

A DuckDB extension for regression, generalized linear models, hypothesis tests and model diagnostics in SQL. Models are fitted inside DuckDB's aggregation engine, so you can fit one model per group, per window frame or per table without moving data out of the database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.5%20LTS%20%7C%20v1.5.x-brightgreen.svg)](https://duckdb.org)
[![Build & WASM tests](https://github.com/DataZooDE/anofox-statistics/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main)](https://github.com/DataZooDE/anofox-statistics/actions/workflows/MainDistributionPipeline.yml)
[![Claude Code Plugin](https://img.shields.io/badge/Claude_Code-plugin-8A2BE2)](#claude-code-plugin)

> [!TIP]
> The repository includes a **[Claude Code plugin](#claude-code-plugin)** with five skills that document the SQL API (function signatures, option keys, result fields) for the Claude Code assistant.

> [!IMPORTANT]
> The extension is pre-1.0: minor releases may contain breaking changes (listed in [CHANGELOG.md](CHANGELOG.md)).
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Plot-Ready Outputs](#plot-ready-outputs)
- [Claude Code Plugin](#claude-code-plugin)
- [API Reference](#api-reference)
- [Development](#development)
- [Support](#support)
- [Citation](#citation)
- [License](#license)

---

## Key Features

Function names have no prefix (`ols_fit_agg`, `theil_sen_fit`, …), and model options are passed as a MAP/STRUCT literal (`{'fit_intercept': true, 'compute_inference': true}`). Regression aggregates take `fn(y, [x1, x2, …][, options])`; WLS adds a positional weight (`wls_fit_agg(y, x, weight[, options])`). Most model families are available in several forms:

| Form | Example | Use |
|------|---------|-----|
| Scalar fit | `ols_fit(y_array, x_columns[, options])` | Fit on literal arrays (`x` is column-major) |
| Aggregate fit | `ols_fit_agg(y, [x1, x2][, options])` | One model per table or per `GROUP BY` group |
| Window fit-predict | `ols_fit_predict(y, [x1, x2][, options]) OVER (ORDER BY t ROWS BETWEEN ... AND CURRENT ROW)` | Expanding / rolling in-sample fits; prediction for the last row of each frame |
| Fit-predict aggregate | `ols_fit_predict_agg(y, [x1, x2][, options])` | Fit on rows with non-NULL `y`, predict every row of the group |
| Table macro | `ols_fit_predict_by('table', group, y, [x1, x2][, options])` | Per-group fit + predict in one call |

### Regression Methods

| Method | Functions | Description | Docs |
|--------|-----------|-------------|------|
| OLS | `ols_fit`, `ols_fit_agg` | Ordinary Least Squares; collinear columns are aliased (`NaN` coefficient) as in R's `lm`, with the `alias_tolerance` option as `lm`'s `tol` | [ols](docs/api/regression/ols.md) |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2 regularization | [ridge](docs/api/regression/ridge.md) |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | Combined L1+L2 regularization | [elasticnet](docs/api/regression/elasticnet.md) |
| LARS / Lasso | `lars_fit_agg` | Least Angle Regression | [lars](docs/api/regression/lars.md) |
| WLS | `wls_fit`, `wls_fit_agg` | Weighted Least Squares (also takes `alias_tolerance`) | [wls](docs/api/regression/wls.md) |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive Least Squares (online) | [rls](docs/api/regression/rls.md) |
| Huber | `huber_fit`, `huber_fit_agg` | Robust M-estimator (reports robust `scale` and `n_outliers`) | [huber](docs/api/regression/huber.md) |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | Robust consensus regression (reports `n_inliers` and `n_trials`) | [ransac](docs/api/regression/ransac.md) |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Robust nonparametric regression via spatial median over subsamples | [theil_sen](docs/api/regression/theil_sen.md) |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded / non-negative least squares | [bls](docs/api/regression/bls.md) |
| PLS | `pls_fit_agg`, `pls_fit_predict_agg` | Partial Least Squares | [pls](docs/api/regression/pls.md) |
| Isotonic | `isotonic_fit_agg`, `isotonic_fit_predict_agg` | Monotonic regression | [isotonic](docs/api/regression/isotonic.md) |
| Quantile | `quantile_fit_agg`, `quantile_fit_predict_agg`, `quantile_process_agg` | Quantile / median regression; standard errors with `{'se': 'iid' \| 'nid' \| 'boot'}` (as `quantreg::summary.rq`); the regression-quantile process over a grid of `taus` | [quantile](docs/api/regression/quantile.md) |
| Paths & CV | `ridge_path_agg`, `elasticnet_path_agg`, `lars_path_agg`, `ridge_cv_agg`, `elasticnet_cv_agg` | Coefficient paths over a penalty grid and k-fold CV curves (glmnet's objective, grid and `cv.glmnet` `lambda.min` / `lambda.1se`) | [paths](docs/api/regression/paths.md) |

### Generalized Linear, Survival and Hierarchical Models

| Method | Functions | Description | Docs |
|--------|-----------|-------------|------|
| Poisson | `poisson_fit_agg`, `poisson_fit_predict_agg`, `poisson_fit_predict_by` | GLM for count data | [poisson](docs/api/glm/poisson.md) |
| Binomial | `binomial_fit_agg`, `binomial_fit_predict_by` | GLM for success-rate data (logit / probit / cloglog links) | [binomial](docs/api/glm/binomial.md) |
| Logistic | `logistic_fit_agg`, `logistic_fit_predict_by` | Binary classification (logit link); reports accuracy and the threshold used | [logistic](docs/api/glm/logistic.md) |
| Negative Binomial | `negbinom_fit_agg`, `negbinom_fit_predict_by` | GLM for overdispersed counts (dispersion estimated jointly) | [negbinom](docs/api/glm/negbinom.md) |
| Gamma | `gamma_fit_agg`, `gamma_fit_predict_by` | GLM for strictly positive continuous outcomes | [gamma](docs/api/glm/gamma.md) |
| Tweedie | `tweedie_fit_agg`, `tweedie_fit_predict_by` | GLM for zero-inflated positive-skew outcomes | [tweedie](docs/api/glm/tweedie.md) |
| GLM priors | `prior` option | Normal / Laplace priors on GLM coefficients | [priors](docs/api/glm/priors.md) |
| ALM | `alm_fit_agg`, `alm_fit_predict_agg`, `alm_fit_predict_by` | Augmented Linear Model with 25 error distributions | [alm](docs/api/glm/alm.md) |
| AFT survival | `aft_fit_agg`, `aft_cdf`, `aft_quantile`, `aft_survival` | Accelerated failure time models with right censoring; CDF, quantile and survival-curve helpers | [aft](docs/api/survival/aft.md) |
| Survival curves | `kaplan_meier_agg`, `kaplan_meier_by`, `log_rank_test_agg` | Kaplan-Meier curves with confidence bands per stratum, log-rank test | [survival_curves](docs/api/survival/survival_curves.md) |
| GLM results | `family`, `link` fields | Every GLM result struct ends with its `family` and `link`, so `predict` returns response-scale predictions | [model_tools](docs/api/regression/model_tools.md) |
| Mixed effects | `glmm_fit_agg`, `glmm_fit_by` | Random intercepts / slopes over grouping factors | [glmm](docs/api/glm/glmm.md) |
| EB shrinkage | `eb_shrink_agg`, `eb_shrink_by` | Empirical-Bayes partial pooling of per-group estimates | [eb_shrink](docs/api/glm/eb_shrink.md) |

### Prediction

| Function | Description | Docs |
|----------|-------------|------|
| `predict(model, x[, {'type': 'response'\|'link'}])` | Score one row with the STRUCT returned by any `*_fit_agg` / `*_fit`; GLMs predict on the response scale (or the link scale), isotonic models interpolate between knots | [model_tools](docs/api/regression/model_tools.md) |
| `predict(model, x, {'se_fit': true, 'interval': 'confidence'\|'prediction', 'level': 0.95})` | `STRUCT(yhat, se_fit, yhat_lower, yhat_upper)` as R's `predict.lm` / `predict.glm(se.fit = TRUE)`, for OLS, WLS, Ridge and the GLMs fitted with `compute_inference` (they carry `vcov` and `df_residual`) | [model_tools](docs/api/regression/model_tools.md#standard-errors-and-intervals) |
| `predict(x_new, coefficients, intercept)`, `linear_predict(...)` | Score new points from fitted coefficients (column-major `x_new`); returns a LIST | [model_tools](docs/api/regression/model_tools.md) |
| `tidy(model[, names])` | One row per term (intercept first, with its own inference): estimate, std. error, statistic, p-value, confidence interval — use with `unnest(..., recursive := true)` | [model_tools](docs/api/regression/model_tools.md) |
| `glance(model)` | One-row summary with the same 19 fields for every model (`model_type`, R², log-likelihood, AIC/BIC, deviance, …; NULL where a model has no such value) | [model_tools](docs/api/regression/model_tools.md) |
| `ols_fit_predict`, `ridge_fit_predict`, `elasticnet_fit_predict`, `wls_fit_predict`, `rls_fit_predict`, `huber_fit_predict`, `ransac_fit_predict`, `theil_sen_fit_predict` | Window aggregates (`OVER (...)`): fit on the frame and return `yhat`, `yhat_lower`, `yhat_upper` for the last row of the frame (use frames ending at `CURRENT ROW`) | [fit_predict_window](docs/api/regression/fit_predict_window.md) |
| `ols_fit_predict_agg`, `ridge_fit_predict_agg`, `elasticnet_fit_predict_agg`, `wls_fit_predict_agg`, `rls_fit_predict_agg`, `huber_fit_predict_agg`, `ransac_fit_predict_agg`, `theil_sen_fit_predict_agg`, `bls_fit_predict_agg`, `alm_fit_predict_agg`, `poisson_fit_predict_agg`, `pls_fit_predict_agg`, `isotonic_fit_predict_agg`, `quantile_fit_predict_agg` | Fit on rows with non-NULL `y`, return a LIST of per-row predictions (`y`, `yhat`, `yhat_lower`, `yhat_upper`, `is_training`) | [fit_predict_agg](docs/api/regression/fit_predict_agg.md) |

### Model Outputs per Group (long tables)

These table macros fit one model per group of a table and return long tables in the fixed column sets of the [integration contract](#plot-ready-outputs), ready to filter, join or plot. Arguments follow `*_fit_predict_by`: `(source, group_col, y_col, x_cols[, options := ..., names := [...]])`; WLS adds `weight_col` after `x_cols`, AFT adds `event_col`.

| Macro | Models | Returns | Docs |
|-------|--------|---------|------|
| `<model>_tidy_by` | 21: `ols`, `wls`, `ridge`, `elasticnet`, `lars`, `rls`, `huber`, `ransac`, `theil_sen`, `bls`, `nnls`, `pls`, `quantile`, the six GLMs, `alm`, `aft` | One row per group and term: `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` (`terms`) | [model_tools](docs/api/regression/model_tools.md#tidy_by-and-glance_by) |
| `<model>_glance_by` | the same 21 | One row per group and fit statistic: `model_type`, `metric`, `value` (`summary`) | [model_tools](docs/api/regression/model_tools.md#tidy_by-and-glance_by) |
| `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` | OLS, WLS, the six GLMs | Every source row plus `fitted`, `residual`, `std_residual`, `stud_residual`, `leverage`, `cooks_d`, `dffits` (`obs`, as broom's `augment`) | [model_tools](docs/api/regression/model_tools.md#augment_by) |
| `<model>_partial_residuals_by` | OLS, WLS, the six GLMs | Component-plus-residual (`car::crPlots`) and, for OLS/WLS, added-variable (`car::avPlots`) coordinates per row and term | [model_tools](docs/api/regression/model_tools.md#partial_residuals_by) |
| `<model>_prediction_by` | 19 (every `*_fit_predict_by` model) | Fitted values with their band and `split` (`train` / `test` / `future`), keyed by a plot position `x_col` (`prediction`) | [model_tools](docs/api/regression/model_tools.md#prediction_by-and-effects_by) |
| `<model>_effects_by` | OLS, WLS, Ridge, the six GLMs | Marginal-effect grid over one focal feature, other features at their means, with a confidence band (`prediction`) | [model_tools](docs/api/regression/model_tools.md#prediction_by-and-effects_by) |
| `<model>_rolling_tidy_by` | OLS, WLS, Ridge, Elastic Net, Huber, Poisson | Coefficients of every trailing window of `window_size` rows ordered by `order_col` (`terms`, `index_name = 'window_end'`) | [model_tools](docs/api/regression/model_tools.md#rolling_tidy_by) |
| `ridge_path_by`, `elasticnet_path_by`, `lars_path_by` | — | Coefficient path per group (`terms`, `index_name = 'lambda'`; `'alpha'` for LARS) | [paths](docs/api/regression/paths.md) |
| `ridge_cv_by`, `elasticnet_cv_by` | — | k-fold CV curve per group with `is_min` / `is_1se` (`curve`, `curve_type = 'lambda_cv'`) | [paths](docs/api/regression/paths.md) |
| `quantile_process_by` | — | Coefficients and intervals over a grid of `taus` (`terms`, `index_name = 'tau'`) | [quantile](docs/api/regression/quantile.md#quantile_process_agg) |

The prediction and rolling macros take a position column: `<model>_prediction_by(source, group_col, x_col, y_col, x_cols[, level := 0.95, split := ...])` and `<model>_rolling_tidy_by(source, group_col, order_col, y_col, x_cols[, window_size := 30, min_obs := ...])`.

### Fit-Predict Table Macros (`*_fit_predict_by`)

Per-group model fitting and prediction in a single call. `source` is a table name given as a string; the result is every source row plus `yhat`, `yhat_lower`, `yhat_upper` and `is_training`. Optional named arguments: `options := {...}`, `split := col` (rows whose value is neither `'train'` nor NULL are predicted but not used for fitting) and `order_by := col` (deterministic row alignment within each group; the binomial, logistic, negative binomial, gamma and Tweedie macros score rows directly and have no `order_by`). Prediction intervals are leverage-aware (see [METHODOLOGY](docs/METHODOLOGY.md#prediction-intervals)). See [table_macros](docs/api/macros/table_macros.md).

| Macro | Description |
|-------|-------------|
| `ols_fit_predict_by` | OLS per-group fit + predict |
| `ridge_fit_predict_by` | Ridge per-group fit + predict |
| `elasticnet_fit_predict_by` | Elastic Net per-group fit + predict |
| `wls_fit_predict_by` | WLS per-group fit + predict (extra `weight_col` argument) |
| `rls_fit_predict_by` | RLS per-group fit + predict |
| `huber_fit_predict_by` | Huber robust per-group fit + predict |
| `ransac_fit_predict_by` | RANSAC robust per-group fit + predict |
| `theil_sen_fit_predict_by` | Theil-Sen robust per-group fit + predict |
| `bls_fit_predict_by` | Bounded LS per-group fit + predict |
| `alm_fit_predict_by` | ALM per-group fit + predict |
| `poisson_fit_predict_by` | Poisson GLM per-group fit + predict |
| `binomial_fit_predict_by`, `logistic_fit_predict_by`, `negbinom_fit_predict_by`, `gamma_fit_predict_by`, `tweedie_fit_predict_by` | GLM per-group fit + predict; `yhat` on the response scale, `yhat_lower`/`yhat_upper` NULL |
| `pls_fit_predict_by` | PLS per-group fit + predict |
| `isotonic_fit_predict_by` | Isotonic per-group fit + predict (single `x_col`) |
| `quantile_fit_predict_by` | Quantile per-group fit + predict |
| `glmm_fit_by` | One mixed model across groups; returns per-group random effects ([glmm](docs/api/glm/glmm.md)) |
| `eb_shrink_by` | Empirical-Bayes shrinkage of existing per-group estimates ([eb_shrink](docs/api/glm/eb_shrink.md)) |
| `aid_by`, `aid_anomaly_by` | Per-group demand classification / per-row anomaly flags ([aid](docs/api/aid/aid.md)) |
| `roc_curve_by`, `pr_curve_by`, `calibration_curve_by` | Per-group classifier curves in the contract `curve` schema ([classification](docs/api/statistics/classification.md)) |

### Statistical Hypothesis Tests

| Category | Functions | Docs |
|----------|-----------|------|
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg`, `dagostino_k2_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Parametric | `t_test_agg`, `one_way_anova_agg`, `yuen_agg`, `brown_forsythe_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`, `permutation_t_test_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Post-hoc (pairwise) | `pairwise_t_test_agg`, `tukey_hsd_agg`, `dunn_test_agg` and the `_by` macros `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by`; `p_adjust` holm / hochberg / hommel / bonferroni / BH / BY / none | [hypothesis](docs/api/statistics/hypothesis.md#pairwise-post-hoc-tests) |
| Equivalence | `tost_t_test_agg`, `tost_paired_agg`, `tost_correlation_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Distribution | `energy_distance_agg`, `mmd_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Forecast | `diebold_mariano_agg`, `clark_west_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Residual autocorrelation | `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg` (with the regressors `x`, the `lmtest::dwtest` p-value) | [timeseries](docs/api/diagnostics/timeseries.md) |
| Heteroskedasticity | `breusch_pagan_agg`, `white_test_agg` | [timeseries](docs/api/diagnostics/timeseries.md) |
| Survival | `log_rank_test_agg` | [survival_curves](docs/api/survival/survival_curves.md) |
| Correlation | `pearson_agg`, `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg` | [correlation](docs/api/statistics/correlation.md) |
| Categorical | `chisq_test_agg`, `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg` | [categorical](docs/api/statistics/categorical.md) |
| Effect size | `cramers_v_agg`, `phi_coefficient_agg`, `contingency_coef_agg`, `cohen_kappa_agg` | [categorical](docs/api/statistics/categorical.md) |
| Proportion | `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg` | [categorical](docs/api/statistics/categorical.md) |
| Classifier evaluation | `roc_agg`, `pr_agg`, `calibration_agg`, `auc_agg`, `brier_agg`, `log_loss_agg` | [classification](docs/api/statistics/classification.md) |

Every test returns the [unified result fields](docs/api/statistics/hypothesis.md#unified-result-fields) (`method`, `statistic`, `p_value`, `estimate`, `conf_low`, `conf_high`, `effect_size`, …). The older names `ci_lower` / `ci_upper`, `df` and the ANOVA `f_statistic`, `df_between`, `df_within` are deprecated aliases and will be removed in the next minor release.

The time-series functions (`acf_agg`, `pacf_agg`, `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg`) read rows in the order of their own `ORDER BY`, which goes after the constant arguments: `ljung_box_agg(resid, 10 ORDER BY t)`, `durbin_watson_agg(resid, [x1, x2] ORDER BY t)`.

### Diagnostics & Utilities

| Function | Description | Docs |
|----------|-------------|------|
| `vif`, `vif_agg` | Variance Inflation Factor | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `aic`, `bic` | Model selection criteria from RSS, n and k | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `jarque_bera`, `jarque_bera_agg` | Jarque-Bera normality test (scalar on an array, or aggregate) | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | Raw, standardized, studentized residuals and leverage | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `acf_agg`, `pacf_agg`, `acf_by`, `pacf_by` | Residual (partial) autocorrelation with the white-noise band | [timeseries](docs/api/diagnostics/timeseries.md) |
| `aid_agg`, `aid_anomaly_agg` | Demand classification (regular / intermittent) and per-row anomaly flags | [aid](docs/api/aid/aid.md) |

### Behaviour

- **Options are validated**: an option key the function does not support raises an error at bind time instead of being ignored, and out-of-range values are rejected.
- **Input validation**: dimension mismatches, too few rows, all-non-finite input and similar problems raise a descriptive error before any computation.
- **Stable model structs**: every fit returns the same fields whatever the options (inference fields are NULL without `compute_inference`), including the intercept's inference, `log_likelihood`, `aic`, `bic` and `model_type`, `family`, `link`. Linear models share `r_squared`, `residual_std_error`, `n_observations`, `n_features`; GLM and AFT results report `z_values` (Wald z) instead of `t_values`.
- **Unified test results**: every test aggregate returns `method`, `alternative`, `statistic`, `df1`, `df2`, `p_value`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name` and `n`, NULL where they do not apply.
- **Documented NULL handling**: see [docs/NULL_SEMANTICS.md](docs/NULL_SEMANTICS.md). The statistical methods are described in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

The numerical core is the [anofox-regression](https://github.com/DataZooDE/anofox-regression) Rust crate, whose test suite compares results with R (`lm()`, `glmnet` and other standard packages); the SQL test suite in this repository adds reference checks against R for the tests and models exposed here.

### Performance

Model fitting runs inside DuckDB's parallel aggregation, so per-group fits are spread across all cores. Example end-to-end query times from `bash scripts/bench.sh` on a multi-core desktop (they vary with hardware):

| Workload | Scale | Query time |
|----------|-------|------------|
| `ols_fit_agg` per group (3 features) | 10K groups / 1M rows | ~0.3 s |
| `ols_fit_agg` with `compute_inference: true` (1 feature) | 500 groups / 50K rows | ~0.01 s |

Profiling shows that most of the query time is spent in DuckDB's `GROUP BY` machinery rather than in the model fits. See [bench/README.md](bench/README.md) and [bench/PROFILING.md](bench/PROFILING.md).

---

## Installation

Supported DuckDB versions: **v1.4.5 LTS** and **v1.5.x** (release binaries are built against v1.4.5 and v1.5.6).

### From the DataZoo extension repository (recommended)

The binaries hosted at `get.erpl.io` are not signed by the DuckDB Foundation, so start DuckDB with `-unsigned` (or set `allow_unsigned_extensions` in your client):

```bash
duckdb -unsigned
```

```sql skip
INSTALL anofox_statistics FROM 'https://get.erpl.io';
LOAD anofox_statistics;
```

Binaries are available for Linux (amd64/arm64), macOS (amd64/arm64) and Windows (amd64). No build toolchain is required.

### From the DuckDB community extension repository

The community build is signed by DuckDB, so no `-unsigned` flag is needed. It is updated less often than the DataZoo repository and may lag behind the latest release.

```sql skip
INSTALL anofox_statistics FROM community;
LOAD anofox_statistics;
```

### Telemetry

The extension sends **pseudonymous** usage telemetry (extension load events and per-session function call counts; never SQL text, table or column names, or data). It is enabled by default and can be turned off with either:

```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

```sql skip
SET anofox_telemetry_enabled = false;
```

See [TELEMETRY.md](TELEMETRY.md) for the complete list of collected properties.

---

## Quick Start

Function names have no prefix and options are passed as a MAP/STRUCT literal. Upgrading from v0.9.x, where functions carried an `anofox_stats_` prefix, is covered in [docs/MIGRATION.md](docs/MIGRATION.md); [docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md) is the naming and options reference.

### Step 1 — Fit an OLS model

```sql
-- House size (sqm) and sale price (kEUR)
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

-- price_keur ~ sqm, fitted with the aggregate form
SELECT
    round(fit.r_squared, 4)       AS r_squared,
    round(fit.coefficients[1], 4) AS slope,
    round(fit.intercept, 4)       AS intercept
FROM (SELECT ols_fit_agg(price_keur, [sqm]) AS fit FROM houses);
-- r_squared = 0.9995, slope = 2.4095, intercept = -1.6667
```

### Step 2 — Predict, summarize

The STRUCT returned by `ols_fit_agg` is the fitted model. `predict(model, x)` scores one row with it:

```sql
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

SELECT new_houses.sqm, round(predict(m.fit, [new_houses.sqm]), 1) AS predicted_keur
FROM (SELECT ols_fit_agg(price_keur, [sqm]) AS fit FROM houses) m,
     (VALUES (70.0), (100.0), (140.0)) new_houses(sqm)
ORDER BY new_houses.sqm;
-- 70 → 167.0, 100 → 239.3, 140 → 335.7
```

`tidy` returns one row per term and `glance` a one-row summary of the model's scalar fields:

```sql
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

-- Columns: term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, index_value
-- (Intercept): estimate -1.6667, std_error 2.3514, p = 0.518
-- sqm:         estimate  2.4095, std_error 0.0258, t = 93.4, p = 7.9e-08
SELECT unnest(tidy(ols_fit_agg(price_keur, [sqm], {'compute_inference': true}), ['sqm']), recursive := true)
FROM houses;

-- The same 19 columns for every model: model_type, family, link, n_observations, r_squared,
-- log_likelihood, aic, bic, deviance, ... (NULL where the model has no such value)
-- model_type = 'ols', r_squared = 0.99954, residual_std_error = 1.618, aic = 26.37
SELECT unnest(glance(ols_fit_agg(price_keur, [sqm]))) FROM houses;
```

The scalar `ols_fit(y, X)` fits literal arrays. `X` is **column-major**: one inner list per feature, holding that feature's values for all observations. `linear_predict(X_new, coefficients, intercept)` (also available as the three-argument `predict`) scores new points in the same layout and returns a LIST:

```sql
WITH m AS (
    SELECT ols_fit([120.0, 155.0, 190.0, 225.0, 265.0, 300.0],
                   [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]]) AS fit
)
SELECT round(unnest(linear_predict([[70.0, 100.0, 140.0]], fit.coefficients, fit.intercept)), 1) AS predicted_keur
FROM m;
-- 167.0, 239.3, 335.7
```

### Step 3 — Inspect residuals

`residuals_diagnostics_agg(y, yhat, x)` returns raw, standardized and studentized residuals plus leverage, in the order given by `ORDER BY`:

```sql
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

WITH fitted AS (
    SELECT sqm, price_keur,
           predict((SELECT ols_fit_agg(price_keur, [sqm]) FROM houses), [sqm]) AS yhat
    FROM houses
)
SELECT
    round(unnest(d.raw), 3)         AS raw,
    round(unnest(d.standardized), 3) AS standardized,
    round(unnest(d.studentized), 3)  AS studentized,
    round(unnest(d.leverage), 3)     AS leverage
FROM (SELECT residuals_diagnostics_agg(price_keur, yhat, [sqm] ORDER BY sqm) AS d FROM fitted);
-- raw: 1.19, 0.048, -1.095, -2.238, 1.619, 0.476; leverage: 0.524, 0.295, 0.181, 0.181, 0.295, 0.524
```

### Per-group models with `GROUP BY`

Every `*_fit_agg` function works with `GROUP BY`, fitting one model per group:

```sql
-- Two product categories with different unit prices (3 and 5) plus a small periodic term
CREATE OR REPLACE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    i::DOUBLE                                 AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

SELECT
    category,
    fit.n_observations            AS n,
    round(fit.r_squared, 4)       AS r_squared,
    round(fit.coefficients[1], 3) AS slope,
    round(fit.std_errors[1], 4)   AS slope_se
FROM (
    SELECT category, ols_fit_agg(revenue, [units_sold], {'compute_inference': true}) AS fit
    FROM sales_data
    GROUP BY category
)
ORDER BY category;
-- A: n = 20, r_squared = 0.9995, slope = 2.997, slope_se = 0.0164
-- B: n = 20, r_squared = 0.9998, slope = 4.997, slope_se = 0.0164
```

### Fit and predict per group in one call

`*_fit_predict_by` fits one model per group and returns every source row with `yhat`, `yhat_lower`, `yhat_upper` (95% prediction interval) and `is_training`:

```sql
CREATE OR REPLACE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    i::DOUBLE                                 AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

SELECT category, units_sold, revenue,
       round(yhat, 2) AS yhat, round(yhat_lower, 2) AS yhat_lower, round(yhat_upper, 2) AS yhat_upper
FROM ols_fit_predict_by('sales_data', category, revenue, [units_sold], order_by := units_sold)
ORDER BY category, units_sold
LIMIT 3;
-- A, 2.0,  8.0,  7.10,  5.16,  9.04
-- A, 4.0, 13.0, 13.09, 11.18, 15.01
-- A, 6.0, 18.0, 19.09, 17.19, 20.99
```

### Rolling regression with a window function

`*_fit_predict(...) OVER (...)` fits on the window frame and returns the prediction for the **last row of the frame**. Use frames that end at `CURRENT ROW` over a unique ordering:

```sql
CREATE OR REPLACE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    i::DOUBLE                                 AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

-- Expanding window: each row's in-sample fit uses all rows up to and including it
SELECT
    units_sold,
    revenue,
    round((ols_fit_predict(revenue, [units_sold]) OVER (
        PARTITION BY category ORDER BY units_sold
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).yhat, 2) AS yhat_expanding
FROM sales_data
WHERE category = 'A'
ORDER BY units_sold
LIMIT 5;
-- yhat_expanding: NULL, NULL, 18.0, 25.1, 31.0
-- (a model with an intercept and one feature needs at least 3 rows)
```

A frame that ends before the current row (`... AND 1 PRECEDING`) does not turn `*_fit_predict` into a one-step-ahead forecast, and `OVER (PARTITION BY g)` without `ORDER BY` gives every row the same prediction. For one prediction per row of a whole group, use `ols_fit_predict_agg` or `ols_fit_predict_by`; see [fit_predict_window](docs/api/regression/fit_predict_window.md).

For a one-step-ahead forecast, fit the aggregate on the strictly earlier rows and score the current row with `predict`:

```sql
CREATE OR REPLACE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    i::DOUBLE                                 AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

SELECT
    units_sold,
    revenue,
    round(predict((ols_fit_agg(revenue, [units_sold]) OVER (
        PARTITION BY category ORDER BY units_sold
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )), [units_sold]), 2) AS yhat_next
FROM sales_data
WHERE category = 'A'
ORDER BY units_sold
LIMIT 5;
-- yhat_next: NULL, NULL, NULL, 23.0, 31.0
-- (NULL until the frame holds 3 earlier rows)
```

---

## Plot-Ready Outputs

Model STRUCTs are convenient in SQL but awkward to chart. The `_by` table macros therefore also return **long tables with fixed column sets**, the six schemas of the anofox integration contract. A table has one row per mark: a coefficient, an observation, a curve point, a test. Optional columns are always present (NULL when not computed), and your group column is kept. The macros that fit one of the extension's models (`_tidy_by`, `_glance_by`, `_partial_residuals_by`, `_rolling_tidy_by`, `_path_by`, `_cv_by`, `quantile_process_by`, `_prediction_by`, `_effects_by`) return the group column, `id` (the group as text), then `model_id` and `model_name` (the method: `'ols'`, `'poisson'`, …) before the schema columns; `_augment_by`, which keeps all source columns, has `model_id` and `model_name` but no `id`. So the tables of several models stack with `UNION ALL BY NAME`. In the curve and test macros whose groups are the things being compared (`roc_curve_by`, `pr_curve_by`, `calibration_curve_by`, `acf_by`, `pacf_by`, `kaplan_meier_by`, `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by`), `model_id` is the group value. Other anofox extensions such as anofox-forecast use the same schemas.

| Schema | Main columns | Produced by |
|--------|--------------|-------------|
| `terms` | `model_id, term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, index_value` | `<model>_tidy_by`, `<model>_rolling_tidy_by`, `ridge_path_by` / `elasticnet_path_by` / `lars_path_by`, `quantile_process_by` |
| `summary` | `model_id, model_type, metric, value` | `<model>_glance_by` |
| `obs` | `model_id, model_name, row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params` | `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` |
| `prediction` | `id, model_id, model_name, x, y, yhat, yhat_lower, yhat_upper, level, split` | `<model>_prediction_by`, `<model>_effects_by` |
| `curve` | `model_id, series, curve_type, x, y, y_low, y_high, threshold, weight, n_censor` | `roc_curve_by`, `pr_curve_by`, `calibration_curve_by`, `ridge_cv_by`, `elasticnet_cv_by`, `kaplan_meier_by`, `acf_by`, `pacf_by` |
| `test` | `model_id, test_id, method, statistic, p_value, group1, group2, p_adj, df1, df2, estimate, conf_low, conf_high, …` | `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by`; the test aggregates return the same fields as a STRUCT |

```sql
CREATE OR REPLACE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    i::DOUBLE                                 AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

-- terms: one row per category and term, with 95% intervals
SELECT category, term, round(estimate, 3) AS estimate, round(conf_low, 3) AS conf_low, round(conf_high, 3) AS conf_high
FROM ols_tidy_by('sales_data', category, revenue, [units_sold], names := ['units_sold'])
ORDER BY category, term;
-- A, (Intercept), 1.105, 0.278, 1.932
-- A, units_sold,  2.997, 2.963, 3.032   (B: slope 4.997)

-- prediction: fitted values with a 90% band; x_col (here units_sold) is the plot position
SELECT id, x, y, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS yhat_lower, round(yhat_upper, 2) AS yhat_upper, split
FROM ols_prediction_by('sales_data', category, units_sold, revenue, [units_sold], level := 0.9)
ORDER BY id, x
LIMIT 3;
-- A, 2.0,  8.0,  7.10,  5.50,  8.70, train
-- A, 4.0, 13.0, 13.09, 11.51, 14.68, train
```

Every producer is tagged in `duckdb_functions()`, so a client can find them and check the contract version:

```sql
SELECT anofox_contract_version();   -- '1'

SELECT DISTINCT function_name, tags['anofox.output'] AS output
FROM duckdb_functions()
WHERE tags['anofox.contract'] = '1'
ORDER BY output, function_name;
```

[anofox-visualization](https://github.com/DataZooDE/anofox-visualization) is the DuckDB extension that draws these tables. It has one plot macro per schema, which takes whole rows and returns an SVG, so a contract table goes straight into a chart without reshaping:

```sql skip
LOAD anofox_visualization;

-- Coefficient forest comparing OLS and Huber for category A, coloured by model_id
SELECT anofox_plot_terms(t)
FROM (SELECT * FROM ols_tidy_by('sales_data', category, revenue, [units_sold], names := ['units_sold'])
      UNION ALL BY NAME
      SELECT * FROM huber_tidy_by('sales_data', category, revenue, [units_sold], names := ['units_sold'])) t
WHERE category = 'A';

-- Observed points, fitted line and 90% band, one panel per category
SELECT anofox_plot_prediction(p)
FROM ols_prediction_by('sales_data', category, units_sold, revenue, [units_sold], level := 0.9) p;
```

See the [anofox-visualization documentation](https://github.com/DataZooDE/anofox-visualization) for its plot macros (terms, prediction, curve, summary and residual diagnostics) and their options.

---

## Claude Code Plugin

The repository ships a [Claude Code](https://claude.com/claude-code) plugin with five skills. Each skill documents a part of the SQL API — function signatures, option keys and result fields — so that Claude Code can write queries against this extension from a plain-language request such as *"fit a robust regression per store and flag outliers"*.

Install it in a Claude Code session:

```
/plugin marketplace add DataZooDE/anofox-statistics
/plugin install anofox-statistics@anofox-statistics
```

| Skill | Covers |
|-------|--------|
| `anofox-statistics-regression` | OLS, robust (Huber/RANSAC/Theil-Sen), Ridge/Elastic Net, WLS/RLS, BLS/NNLS, PLS, isotonic, quantile (with inference and the quantile process), GLMs, ALM, AFT, GLMM, EB — options (including `alias_tolerance`), result fields, `predict` with standard errors and intervals, regularisation paths and CV |
| `anofox-statistics-tests` | Normality, parametric and nonparametric tests, pairwise post-hoc tests, correlation, categorical/contingency tests, effect sizes, TOST equivalence, distribution comparison, forecast-evaluation, time-series and survival tests, classifier evaluation (ROC/PR/calibration, AUC, Brier, log loss); the unified result fields |
| `anofox-statistics-diagnostics` | VIF, AIC/BIC, residual diagnostics, `*_augment_by` influence measures, partial residuals, Breusch-Pagan/White, ACF/PACF, Ljung-Box, Durbin-Watson, AID demand-pattern classification |
| `anofox-statistics-batch` | Fitting many models in one query: `*_fit_predict_by`, `*_fit_agg` with `GROUP BY`, `*_fit_predict_agg`, window functions, and the per-group `_tidy_by` / `_glance_by` / `_augment_by` / `_prediction_by` / `_effects_by` / `_rolling_tidy_by` / test and curve macros |
| `anofox-statistics-plotting` | Plot-ready outputs: which integration-contract table (`terms`, `summary`, `obs`, `prediction`, `curve`, `test`) answers a question, its producers and columns, discovery via `duckdb_functions()` tags, and the hand-off to anofox-visualization |

The skills live in [`plugins/anofox-statistics/`](plugins/anofox-statistics/). To use a local checkout instead of the marketplace, start Claude Code with `claude --plugin-dir ./plugins/anofox-statistics`.

---

## API Reference

- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — function signatures, option keys, and return-struct fields for all functions.
- **[docs/api/](docs/api/)** — per-family reference pages (linked from the tables above).
- **[docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md)** — naming convention, MAP-option keys, return-struct field names and error taxonomy.
- **[docs/MIGRATION.md](docs/MIGRATION.md)** — upgrading from v0.9.x to v0.10.0 (prefix removal, renames).
- **[docs/NULL_SEMANTICS.md](docs/NULL_SEMANTICS.md)** — how NULL and NaN inputs are handled.
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — standard errors, p-values, IRLS convergence and prediction intervals.
- **[CHANGELOG.md](CHANGELOG.md)** — release history.

User-facing guides are in the [`guides/`](guides/) directory:

- **[guides/01_quick_start.md](guides/01_quick_start.md)** — getting started with worked examples
- **[guides/02_technical_guide.md](guides/02_technical_guide.md)** — architecture and implementation details
- **[guides/03_business_guide.md](guides/03_business_guide.md)** — real-world business use cases
- **[guides/04_advanced_use_cases.md](guides/04_advanced_use_cases.md)** — complex analytical workflows

---

## Development

### Building from source

**Prerequisites:** Rust stable toolchain, a C++17 compiler, and CMake:

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-statistics.git
cd anofox-statistics
make release
```

This produces `build/release/duckdb` (the CLI) and `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension`.

**Rust unit tests** (the core regression library):

```bash
cargo test
```

**DuckDB SQL test suite**:

```bash
make test
```

**Documentation SQL validation**: CI runs every fenced `sql` block in this README, `guides/*.md`, `docs/*.md`, `docs/api/**/*.md` and the Claude Code skills (`plugins/anofox-statistics/skills/*/SKILL.md`) against the built extension; the blocks of one file share a session. Blocks marked `sql skip` (such as network installs) are excluded:

```bash
python3 scripts/validate_docs_sql.py                     # all documentation files
python3 scripts/validate_docs_sql.py --file README.md    # a single file
```

**Benchmark harness**:

```bash
bash scripts/bench.sh          # default workloads, ~1 s total
bash scripts/bench.sh --full   # adds a 1M-group workload (~8 GB RAM, ~160 s)
```

Results are written to `bench/results/` as diffable markdown files. See [bench/README.md](bench/README.md) for details.

### Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for the build, test and documentation workflow. Useful areas include additional statistical tests, documentation and examples, bug reports and fixes, and performance work.

---

## Support

- **API docs**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) and [docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md)
- **Guides**: [guides/](guides/)
- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-statistics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DataZooDE/anofox-statistics/discussions)
- **Email**: contact@datazoo.de

If a fit fails or a result looks wrong, please open an issue and describe the shape of your data (row count, number of features, NULLs, constant columns); real data often triggers cases that synthetic tests miss. Error messages from the fit and predict functions include the link to the issue tracker.

When the extension is loaded in an interactive terminal, it prints a short banner with the same request at most once a day. The banner is not shown when output is piped, in notebooks or in CI. Turn it off with `SET datazoo_banner = false;` or the environment variable `DATAZOO_NO_BANNER=1`.

---

## Citation

If you use this extension in research, please cite:

```bibtex
@software{anofox_statistics,
  title   = {Anofox Statistics: Statistical Analysis Extension for DuckDB},
  author  = {{DataZoo GmbH}},
  year    = {2026},
  url     = {https://github.com/DataZooDE/anofox-statistics},
  version = {0.11.0}
}
```

---

## License

This project is licensed under the **Business Source License 1.1** (BSL 1.1). The [LICENSE](LICENSE) file is authoritative; in summary:

- **Licensor:** DataZoo GmbH
- **Additional Use Grant:** you may use the Licensed Work for production purposes, but you may not offer the Licensed Work to third parties on a hosted or embedded basis.
- **Change Date:** five years from the date of first publication of each version.
- **Change License:** Mozilla Public License 2.0 (MPL 2.0). Five years after a version is first published, that version becomes available under the MPL 2.0.

The license permits copying, modifying, creating derivative works, redistributing and non-production use; production use is permitted under the Additional Use Grant. For other uses — for example offering the extension to third parties as a hosted or embedded service — contact contact@datazoo.de for a commercial license.
