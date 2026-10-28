# Anofox Statistics Extension - API Reference

**Version:** 0.13.0
**DuckDB versions:** v1.5.6 and v2.0
**Backend:** Rust (anofox-regression 0.5, anofox-statistics 0.4, faer)

## Overview

The Anofox Statistics extension brings regression, generalized linear models,
survival models, mixed models, demand classification and statistical hypothesis
tests to DuckDB. The numerical work is done in Rust; every function is exposed
as plain SQL.

This page is the complete function reference. Each family also has a dedicated
page under [`docs/api/`](#detailed-documentation) with longer explanations.

> **Function names:** as of v0.10.0 all functions use short names such as
> `ols_fit_agg`. The `anofox_stats_` prefix was removed in v0.10.0; see
> [MIGRATION.md](MIGRATION.md) for upgrading from v0.9.

## Contents

- [Conventions](#conventions)
- [Quick Reference](#quick-reference)
- [Sample Data](#sample-data)
- [Linear Regression](#linear-regression)
- [Robust Regression](#robust-regression)
- [Constrained Regression (BLS/NNLS)](#constrained-regression-blsnnls)
- [PLS, Isotonic and Quantile Regression](#pls-isotonic-and-quantile-regression)
- [Generalized Linear Models](#generalized-linear-models)
- [ALM](#alm)
- [AFT Survival Regression](#aft-survival-regression)
- [Mixed-Effects GLMs](#mixed-effects-glms)
- [Empirical-Bayes Shrinkage](#empirical-bayes-shrinkage)
- [Causal Mediation Analysis](#causal-mediation-analysis)
- [AID Demand Classification](#aid-demand-classification)
- [Statistical Hypothesis Tests](#statistical-hypothesis-tests)
- [Classifier Evaluation](#classifier-evaluation)
- [Fit-Predict Window Functions](#fit-predict-window-functions)
- [Fit-Predict Aggregate Functions](#fit-predict-aggregate-functions)
- [Fit-Predict Table Macros](#fit-predict-table-macros)
- [Predict and Diagnostics](#predict-and-diagnostics)
- [Common Options](#common-options)
- [Return Types](#return-types)
- [Error and NULL Handling](#error-and-null-handling)
- [Detailed Documentation](#detailed-documentation)

---

## Conventions

**Calling convention.** Every regression and GLM fit takes the response first,
the features as a list second, and an optional options map last:

```sql skip
-- Aggregate (one row per observation)
<method>_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options])

-- Scalar (whole arrays in one call; x is a list of feature columns)
<method>_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options])
```

The only extra positional arguments are data columns: the weight column for WLS
(`wls_fit_agg(y, x, weight)`), the event indicator for AFT
(`aft_fit_agg(time, x, event)`) and the grouping key for GLMMs
(`glmm_fit_agg(y, x, group)`). Hyperparameters such as `alpha` or
`fit_intercept` are always passed through `options`, never positionally.

**Options.** Write options as a struct literal, `{'alpha': 0.5, 'fit_intercept': false}`.
Keys are case-insensitive. Option keys a function does not support raise an
error rather than being silently ignored. Option values must be constants.

**Results.** Fits return a `STRUCT`. Read a field with `(result).field`; expand all fields into columns with `unnest(result)`.

---

## Quick Reference

### Regression

| Method | Scalar | Aggregate | Window | Fit-predict agg | Table macro | Page |
|--------|--------|-----------|--------|-----------------|-------------|------|
| OLS | `ols_fit` | `ols_fit_agg` | `ols_fit_predict` | `ols_fit_predict_agg` | `ols_fit_predict_by` | [OLS](api/regression/ols.md) |
| Ridge | `ridge_fit` | `ridge_fit_agg` | `ridge_fit_predict` | `ridge_fit_predict_agg` | `ridge_fit_predict_by` | [Ridge](api/regression/ridge.md) |
| Elastic Net | `elasticnet_fit` | `elasticnet_fit_agg` | `elasticnet_fit_predict` | `elasticnet_fit_predict_agg` | `elasticnet_fit_predict_by` | [Elastic Net](api/regression/elasticnet.md) |
| WLS | `wls_fit` | `wls_fit_agg` | `wls_fit_predict` | `wls_fit_predict_agg` | `wls_fit_predict_by` | [WLS](api/regression/wls.md) |
| RLS | `rls_fit` | `rls_fit_agg` | `rls_fit_predict` | `rls_fit_predict_agg` | `rls_fit_predict_by` | [RLS](api/regression/rls.md) |
| Huber | `huber_fit` | `huber_fit_agg` | `huber_fit_predict` | `huber_fit_predict_agg` | `huber_fit_predict_by` | [Huber](api/regression/huber.md) |
| RANSAC | `ransac_fit` | `ransac_fit_agg` | `ransac_fit_predict` | `ransac_fit_predict_agg` | `ransac_fit_predict_by` | [RANSAC](api/regression/ransac.md) |
| Theil-Sen | `theil_sen_fit` | `theil_sen_fit_agg` | `theil_sen_fit_predict` | `theil_sen_fit_predict_agg` | `theil_sen_fit_predict_by` | [Theil-Sen](api/regression/theil_sen.md) |
| LARS | - | `lars_fit_agg` | - | - | - | [LARS](api/regression/lars.md) |
| Bayesian Ridge | - | `bayesian_ridge_fit_agg` | - | `bayesian_ridge_fit_predict_agg` | `bayesian_ridge_fit_predict_by` | [Bayesian](api/regression/bayesian.md) |
| ARD | - | `ard_fit_agg` | - | `ard_fit_predict_agg` | `ard_fit_predict_by` | [Bayesian](api/regression/bayesian.md) |
| BLS | - | `bls_fit_agg` | - | `bls_fit_predict_agg` | `bls_fit_predict_by` | [BLS/NNLS](api/regression/bls.md) |
| NNLS | - | `nnls_fit_agg` | - | - | - | [BLS/NNLS](api/regression/bls.md) |
| PLS | - | `pls_fit_agg` | - | `pls_fit_predict_agg` | `pls_fit_predict_by` | [PLS](api/regression/pls.md) |
| Isotonic | - | `isotonic_fit_agg` | - | `isotonic_fit_predict_agg` | `isotonic_fit_predict_by` | [Isotonic](api/regression/isotonic.md) |
| Passive-Aggressive | - | `passive_aggressive_fit_agg` | `passive_aggressive_fit_predict` | `passive_aggressive_fit_predict_agg` | `passive_aggressive_fit_predict_by` | [Online / dynamic](api/regression/online_dynamic.md) |
| lmDynamic | - | `lm_dynamic_fit_agg` | - | `lm_dynamic_fit_predict_agg` | `lm_dynamic_fit_predict_by` | [Online / dynamic](api/regression/online_dynamic.md) |
| LOWESS | - | `lowess_agg` | - | - | `lowess_by` | [Online / dynamic](api/regression/online_dynamic.md) |
| Quantile | - | `quantile_fit_agg` | - | `quantile_fit_predict_agg` | `quantile_fit_predict_by` | [Quantile](api/regression/quantile.md) |
| Nested-model F test (OLS) | - | `ols_anova_agg` | - | - | `ols_anova_by` | [OLS](api/regression/ols.md#nested-model-f-test-ols_anova_agg--ols_anova_by) |
| Per-term ANOVA table (OLS) | - | `ols_anova_table_agg` | - | - | `ols_anova_table_by` | [OLS](api/regression/ols.md#per-term-anova-tables-ols_anova_table_agg--ols_anova_table_by) |

Overview pages: [window functions](api/regression/fit_predict_window.md),
[fit-predict aggregates](api/regression/fit_predict_agg.md),
[table macros](api/macros/table_macros.md),
[model tools](api/regression/model_tools.md) (`predict`, `tidy`, `glance` on any
fitted model struct).

### GLMs and Related Models

| Method | Aggregate | Other forms | Page |
|--------|-----------|-------------|------|
| Poisson | `poisson_fit_agg` | `poisson_fit_predict_agg`, `poisson_fit_predict_by` | [Poisson](api/glm/poisson.md) |
| Binomial | `binomial_fit_agg` | `binomial_fit_predict_by` | [Binomial](api/glm/binomial.md) |
| Logistic | `logistic_fit_agg` | `logistic_fit_predict_by` | [Logistic](api/glm/logistic.md) |
| Negative Binomial | `negbinom_fit_agg` | `negbinom_fit_predict_by` | [Negative Binomial](api/glm/negbinom.md) |
| Gamma | `gamma_fit_agg` | `gamma_fit_predict_by` | [Gamma](api/glm/gamma.md) |
| Tweedie | `tweedie_fit_agg` | `tweedie_fit_predict_by` | [Tweedie](api/glm/tweedie.md) |
| ALM (24 distributions) | `alm_fit_agg` | `alm_fit_predict_agg`, `alm_fit_predict_by` | [ALM](api/glm/alm.md) |
| AFT survival | `aft_fit_agg` | scalars `aft_cdf`, `aft_quantile`, `aft_survival` | [AFT](api/survival/aft.md) |
| Survival curves | `kaplan_meier_agg`, `log_rank_test_agg` | `kaplan_meier_by` (`curve`), scalar `aft_survival` | [Survival curves](api/survival/survival_curves.md) |
| Mixed effects | `glmm_fit_agg`, `glmm_augment_agg` | `glmm_fit_by`, `glmm_tidy_by` (`terms`), `glmm_glance_by` (`summary`), `glmm_augment_by` (`obs`) | [Mixed-effects GLMs](api/glm/glmm.md) |
| Empirical-Bayes shrinkage | `eb_shrink_agg` | `eb_shrink_by` | [EB shrinkage](api/glm/eb_shrink.md) |
| Explicit priors | options on the GLM and AFT aggregates | - | [Priors](api/glm/priors.md) |

### Statistical Tests

| Category | Functions | Page |
|----------|-----------|------|
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg`, `dagostino_k2_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Parametric | `t_test_agg`, `one_way_anova_agg`, `yuen_agg`, `brown_forsythe_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Nested models (OLS) | `ols_anova_agg`, `ols_anova_by` | [OLS](api/regression/ols.md#nested-model-f-test-ols_anova_agg--ols_anova_by) |
| ANOVA tables (OLS, type I / II) | `ols_anova_table_agg`, `ols_anova_table_by` | [OLS](api/regression/ols.md#per-term-anova-tables-ols_anova_table_agg--ols_anova_table_by) |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`, `permutation_t_test_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Distribution comparison | `energy_distance_agg`, `mmd_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Data drift | `ks_test_agg`, `ad_test_agg`, `cvm_test_agg`, `epps_singleton_agg`, `wasserstein_agg`, `psi_agg`, `kl_divergence_agg`, `jensen_shannon_agg`, `hellinger_agg`, `tvd_agg` | [Hypothesis tests](api/statistics/hypothesis.md#data-drift) |
| Equivalence (TOST) | `tost_t_test_agg`, `tost_paired_agg`, `tost_correlation_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Forecast comparison | `diebold_mariano_agg`, `clark_west_agg` | [Hypothesis tests](api/statistics/hypothesis.md) |
| Residual autocorrelation | `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg` | [Time-series diagnostics](api/diagnostics/timeseries.md) |
| Heteroskedasticity | `breusch_pagan_agg`, `white_test_agg` | [Time-series diagnostics](api/diagnostics/timeseries.md) |
| Survival | `log_rank_test_agg` | [Survival curves](api/survival/survival_curves.md) |
| Correlation | `pearson_agg`, `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg` | [Correlation](api/statistics/correlation.md) |
| Categorical | `chisq_test_agg`, `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg` | [Categorical](api/statistics/categorical.md) |
| Pairwise post-hoc | `pairwise_t_test_agg`, `tukey_hsd_agg`, `dunn_test_agg`, `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by` | [Hypothesis tests](api/statistics/hypothesis.md#pairwise-post-hoc-tests) |
| Multiple testing | `p_adjust_by`, `p_adjust_agg`, `p_adjust_list` | [Hypothesis tests](api/statistics/hypothesis.md#multiple-testing) |
| Effect sizes | `cramers_v_agg`, `phi_coefficient_agg`, `contingency_coef_agg`, `cohen_kappa_agg` | [Categorical](api/statistics/categorical.md) |
| Proportions | `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg` | [Categorical](api/statistics/categorical.md) |
| Classifier evaluation | `roc_agg`, `pr_agg`, `calibration_agg`, `auc_agg`, `brier_agg`, `log_loss_agg`, `roc_curve_by`, `pr_curve_by`, `calibration_curve_by` | [Classification](api/statistics/classification.md) |
| Causal mediation | `mediation_agg`, `mediation_by` (ACME, ADE, total effect, proportion mediated; bootstrap, Sobel or quasi-Bayesian) | [Mediation](api/statistics/mediation.md) |

### Diagnostics and Utilities

| Function | Description | Page |
|----------|-------------|------|
| `predict` | Prediction from any fitted model struct (`predict(model, x)`), or column-layout linear prediction (`predict(x, coefficients, intercept)`) | [Model tools](api/regression/model_tools.md) |
| `linear_predict` | Column-layout linear prediction (same as the 3-argument `predict`) | [Model tools](api/regression/model_tools.md) |
| `tidy` | Per-term table (estimate, std. error, statistic, p-value, CI) of a fitted model | [Model tools](api/regression/model_tools.md) |
| `glance` | One-row summary of a fitted model, the same fields for every model | [Model tools](api/regression/model_tools.md) |
| `<model>_tidy_by`, `<model>_glance_by` | Per-group coefficient table / fit statistics in long form | [Model output macros](#model-output-table-macros) |
| `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` | Per-row fitted values, residuals, leverage, Cook's distance, DFFITS (`obs`) | [Per-observation diagnostics](#per-observation-diagnostics-ols_augment_by--wls_augment_by) |
| `predict(model, x, {'se_fit', 'interval', 'level'})` | Prediction with standard error and confidence / prediction interval from the model's `vcov` (OLS, WLS, Ridge, GLMs) | [Model tools](api/regression/model_tools.md#standard-errors-and-intervals) |
| `<model>_partial_residuals_by`, `<model>_partial_residuals_agg` | Partial-residual (crPlots) and added-variable (avPlots) coordinates for OLS, WLS and the GLMs | [Model tools](api/regression/model_tools.md#partial_residuals_by) |
| `ridge_path_agg`, `elasticnet_path_agg`, `lars_path_agg`, `<model>_path_by` | Coefficient path over lambda (`terms`, `index_name = 'lambda'`; LARS: `'alpha'`) | [Regularisation paths](api/regression/paths.md) |
| `ridge_cv_agg`, `elasticnet_cv_agg`, `<model>_cv_by` | CV curve over lambda (`curve`, `curve_type = 'lambda_cv'`): k-fold, repeated, group, leave-one-group-out, stratified or time-series folds (`cv_strategy`, `cv_key`) | [Regularisation paths](api/regression/paths.md#cv-strategies) |
| `ridge_nested_cv_agg`, `elasticnet_nested_cv_agg`, `<model>_nested_cv_by` | Nested CV: lambda chosen by an inner CV on each outer training set, error on the outer test rows (`curve`, `curve_type = 'nested_cv'`) | [Regularisation paths](api/regression/paths.md#ridge_nested_cv_agg--elasticnet_nested_cv_agg) |
| `quantile_process_agg`, `quantile_process_by` | Quantile process: one quantile fit per tau with inference (`terms`, `index_name = 'tau'`) | [Quantile](api/regression/quantile.md#quantile_process_agg) |
| `vif`, `vif_agg` | Variance inflation factors | [Diagnostics](api/diagnostics/diagnostics.md) |
| `aic`, `bic` | Information criteria from RSS | [Diagnostics](api/diagnostics/diagnostics.md) |
| `jarque_bera`, `jarque_bera_agg` | Jarque-Bera normality test | [Diagnostics](api/diagnostics/diagnostics.md) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | Raw, standardized, studentized residuals, leverage, Cook's distance | [Diagnostics](api/diagnostics/diagnostics.md) |
| `acf_agg`, `pacf_agg`, `acf_by`, `pacf_by` | Residual (partial) autocorrelation with the white-noise band (`curve`) | [Time-series diagnostics](api/diagnostics/timeseries.md) |
| `aid_agg`, `aid_anomaly_agg`, `aid_by`, `aid_anomaly_by` | Demand classification and anomaly flags | [AID](api/aid/aid.md) |
| `anofox_contract_version` | Version of the anofox integration contract; functions are tagged in `duckdb_functions()` | [Discovery](#anofox_contract_version--function-tags) |

---

## Sample Data

The examples on this page are runnable. They use the tables created here:
`reg_data` (60 rows of regression, count, binary, positive and survival
responses) and `demand` (weekly demand for three SKUs, with zeros).

```sql
CREATE OR REPLACE TABLE reg_data AS
SELECT
    i AS id,
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS category,
    (i % 7)::DOUBLE + i * 0.1 AS x1,
    ((i * 3) % 11)::DOUBLE AS x2,
    ((i * 5) % 13)::DOUBLE / 2 AS x3,
    (1 + i % 3)::DOUBLE AS weight,
    2.0 + 1.5 * x1 - 0.8 * x2 + 0.3 * x3 + sin(i) AS y,
    round(exp(0.3 + 0.15 * x1 - 0.05 * x2) + i % 3)::DOUBLE AS y_count,
    (CASE WHEN sin(i * 1.7) + 0.3 * x1 - 0.2 * x2 > 0 THEN 1 ELSE 0 END)::DOUBLE AS y_binary,
    exp(0.5 + 0.1 * x1 + 0.2 * abs(sin(i))) AS y_positive,
    exp(1.0 + 0.1 * x1 + 0.3 * sin(i)) AS duration,
    (CASE WHEN i % 5 = 0 THEN 0 ELSE 1 END)::DOUBLE AS event,
    'store_' || (i % 6) AS store,
    (i % 2)::INTEGER AS grp2,
    (i % 3)::INTEGER AS grp3
FROM range(1, 61) t(i);

CREATE OR REPLACE TABLE demand AS
SELECT
    sku,
    week,
    CASE
        WHEN sku = 'steady' THEN 10.0 + (week % 4)
        WHEN sku = 'sparse' THEN CASE WHEN week % 3 = 0 THEN 5.0 ELSE 0.0 END
        ELSE CASE WHEN week <= 4 THEN 0.0 ELSE 8.0 + (week % 2) END   -- new product
    END AS qty
FROM (VALUES ('steady'), ('sparse'), ('launch')) s(sku), range(1, 21) w(week);
```

---

## Linear Regression

All linear fits return the [FitResult](#fitresult-structure) struct.

### ols_fit / ols_fit_agg

Ordinary least squares.

**Signatures:**

```sql skip
ols_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
ols_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include an intercept term |
| compute_inference | BOOLEAN | false | Add standard errors, t-tests, p-values, confidence intervals, F-test |
| confidence_level | DOUBLE | 0.95 | Confidence level for the intervals |
| solver | VARCHAR | 'svd' | `'svd'`, `'qr'` or `'cholesky'` (see [solver](#solver)) |
| hc_type | VARCHAR | 'none' | Heteroscedasticity-consistent standard errors (see [hc_type](#hc_type)) |
| alias_tolerance | DOUBLE | 1e-7 | Collinearity (aliasing) tolerance, R `lm`'s `tol` (> 0); aliased columns get a NaN coefficient |

**Examples:**

```sql
-- Scalar: y = 2.1 x + 0.8
SELECT ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.5],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'compute_inference': true}
) AS fit;

-- Aggregate: one model per category
SELECT category, (ols_fit_agg(y, [x1, x2])).r_squared AS r2
FROM reg_data
GROUP BY category
ORDER BY category;

-- Rolling coefficient via the aggregate used as a window function
SELECT id,
       (ols_fit_agg(y, [x1]) OVER (ORDER BY id ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)).coefficients[1] AS rolling_beta
FROM reg_data
ORDER BY id
LIMIT 5;
```

### ols_anova_agg / ols_anova_by

Nested-model F test: does the full OLS model fit significantly better than the
reduced one? R's `anova(lm(y ~ x_reduced), lm(y ~ x_full))`, both fits on the
same rows. Returns the `test` fields (`method = 'nested F'`, `statistic`,
`df1`, `df2`, `p_value`) plus `rss_reduced`, `rss_full`,
`adj_r_squared_reduced`, `adj_r_squared_full`, `delta_adj_r_squared` and
`n_observations`. The `x_reduced` columns must appear in `x_full`, and
`x_reduced := []` is the intercept-only model (the overall F test). Rank is
counted as in R; when `df1 = 0` or `df2 = 0` the statistic and p-value are NULL (R's `NA`, never NaN). The only option is
`fit_intercept`. For per-term (type I / II) tables see
[`ols_anova_table_agg`](#ols_anova_table_agg--ols_anova_table_by).

**Signatures:**

```sql skip
ols_anova_agg(y DOUBLE, x_reduced LIST(DOUBLE), x_full LIST(DOUBLE) [, options]) -> STRUCT
ols_anova_by(source VARCHAR, group_col, y_col, x_reduced := [], x_full := [...] [, options := ...]) -> TABLE
```

**Examples:**

```sql
-- Do x2 and x3 add to x1, per category?
SELECT category, (ols_anova_agg(y, [x1], [x1, x2, x3])).p_value < 0.05 AS improves
FROM reg_data
GROUP BY category
ORDER BY category;

SELECT category, df1, df2, round(adj_r_squared_reduced, 3) AS before, round(adj_r_squared_full, 3) AS after
FROM ols_anova_by('reg_data', category, y, x_reduced := [x1], x_full := [x1, x2, x3]);
```

### ols_anova_table_agg / ols_anova_table_by

Per-term ANOVA table of one OLS fit: R's `anova(lm(...))` (type I, sequential,
the default) or `car::Anova(fit, type = 2)` (`{'type': 'II'}`, each term
adjusted for all others). One row per term in the `test` schema: `term`,
`method`, `statistic` (F), `df1` (term df), `df2` (residual df), `p_value`,
`sum_sq`, `mean_sq`, `residual_sum_sq`, `residual_mean_sq`, `n_observations`.
Terms are `x1`, `x2`, ... or the labels of the `terms` option (one per column;
equal labels form one term, e.g. factor dummies). An aliased term has
`df1 = 0` and NULL statistic / p-value. Options: `type`, `terms`,
`fit_intercept`.

**Signatures:**

```sql skip
ols_anova_table_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> LIST(STRUCT)
ols_anova_table_by(source VARCHAR, group_col, y_col, x_cols [, options := ...]) -> TABLE
```

**Examples:**

```sql
SELECT category, (r).term, (r).df1, round((r).statistic, 2) AS f
FROM (SELECT category, unnest(ols_anova_table_agg(y, [x1, x2, x3])) AS r FROM reg_data GROUP BY category)
ORDER BY category, (r).term;

SELECT category, term, round(sum_sq, 2) AS sum_sq, p_value < 0.05 AS significant
FROM ols_anova_table_by('reg_data', category, y, [x1, x2, x3], options := {'type': 'II'});
```

### ridge_fit / ridge_fit_agg

Ridge regression (L2 penalty).

**Signatures:**

```sql skip
ridge_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
ridge_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Options:** all OLS options except `hc_type`, plus:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha (alias `lambda`) | DOUBLE | 1.0 | L2 penalty strength (>= 0) |
| lambda_scaling | VARCHAR | 'raw' | `'raw'` or `'glmnet'` (see [lambda_scaling](#lambda_scaling)) |
| compute_loo | BOOLEAN | false | Exact leave-one-out `press`, `press_rmse`, `loo_r_squared` from the ridge hat matrix (one SVD; see [model_tools](api/regression/model_tools.md#ridge)) |

```sql
SELECT ridge_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'alpha': 0.1}
) AS fit;

SELECT (ridge_fit_agg(y, [x1, x2, x3], {'alpha': 0.5})).coefficients AS coefficients
FROM reg_data;
```

### elasticnet_fit / elasticnet_fit_agg

Elastic Net (combined L1/L2 penalty, coordinate descent). The inference fields (`std_errors`, `t_values`, `p_values`) are always NULL.

**Signatures:**

```sql skip
elasticnet_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
elasticnet_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha (alias `lambda`) | DOUBLE | 1.0 | Overall penalty strength |
| l1_ratio | DOUBLE | 0.5 | Share of L1 penalty: 0 = ridge, 1 = lasso |
| fit_intercept | BOOLEAN | true | Include an intercept term |
| max_iterations | INTEGER | 1000 | Coordinate-descent iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |
| lambda_scaling | VARCHAR | 'raw' | `'raw'` or `'glmnet'` |

```sql
SELECT elasticnet_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'alpha': 0.1, 'l1_ratio': 0.5}
) AS fit;

SELECT (elasticnet_fit_agg(y, [x1, x2, x3], {'alpha': 0.1, 'l1_ratio': 0.9})).coefficients AS coefficients
FROM reg_data;
```

### wls_fit / wls_fit_agg

Weighted least squares. The weight is a data argument, not an option.

**Signatures:**

```sql skip
wls_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)), weights LIST(DOUBLE) [, options]) -> STRUCT
wls_fit_agg(y DOUBLE, x LIST(DOUBLE), weight DOUBLE [, options]) -> STRUCT
```

**Options:** same as OLS (`fit_intercept`, `compute_inference`, `confidence_level`, `solver`, `hc_type`, `alias_tolerance`).

```sql
SELECT wls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.5],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    [1.0, 2.0, 3.0, 2.0, 1.0]
) AS fit;

SELECT (wls_fit_agg(y, [x1, x2], weight, {'compute_inference': true})).p_values AS p_values
FROM reg_data;
```

### rls_fit / rls_fit_agg

Recursive least squares with optional exponential forgetting. The inference fields (`std_errors`, `t_values`, `p_values`) are always NULL.

**Signatures:**

```sql skip
rls_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
rls_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| forgetting_factor | DOUBLE | 1.0 | 1.0 = no forgetting; 0.95-0.99 adapts to recent data |
| initial_p_diagonal (alias `p_diagonal`) | DOUBLE | 100.0 | Initial covariance diagonal |
| fit_intercept | BOOLEAN | true | Include an intercept term |

```sql
SELECT rls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'forgetting_factor': 0.99}
) AS fit;

SELECT (rls_fit_agg(y, [x1, x2], {'forgetting_factor': 0.95})).coefficients AS coefficients
FROM reg_data;
```

### lars_fit_agg

Least Angle Regression. Returns the [FitResult](#fitresult-structure) struct
without inference fields. Aggregate only.

```sql skip
lars_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include an intercept term |

```sql
SELECT (lars_fit_agg(y, [x1, x2, x3])).coefficients AS coefficients
FROM reg_data;
```

---

## Robust Regression

Huber, RANSAC and Theil-Sen resist outliers. They have scalar, aggregate,
window, fit-predict aggregate and table-macro forms with the same calling
convention as OLS. All accept `fit_intercept`, `compute_inference` and
`confidence_level`, and return the [FitResult](#fitresult-structure) fields plus
the method-specific fields listed below.

### huber_fit / huber_fit_agg

Huber M-estimator.

```sql skip
huber_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
huber_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| epsilon | DOUBLE | 1.35 | Huber threshold; smaller is more robust |
| alpha | DOUBLE | 0.0001 | L2 regularization strength |
| max_iterations | INTEGER | 100 | IRLS iterations |
| tolerance | DOUBLE | 1e-5 | Convergence tolerance |

Extra result fields: `scale DOUBLE` (robust scale estimate), `n_outliers BIGINT`.

```sql
SELECT unnest(huber_fit_agg(y, [x1, x2], {'epsilon': 1.35})) FROM reg_data;
```

### ransac_fit / ransac_fit_agg

RANSAC: repeatedly fits random subsets and keeps the model with most inliers.

```sql skip
ransac_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
ransac_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| max_trials | INTEGER | 100 | Maximum random subsets |
| min_samples | INTEGER | automatic | Rows per subset |
| residual_threshold | DOUBLE | automatic | Absolute residual below which a row is an inlier |
| stop_probability | DOUBLE | 0.99 | Stop once this confidence of an outlier-free subset is reached |
| stop_n_inliers | INTEGER | unset | Stop once this many inliers are found |
| random_state (alias `seed`) | INTEGER | 0 | Random seed (results are reproducible) |

Extra result fields: `residual_threshold DOUBLE`, `n_inliers BIGINT`, `n_trials BIGINT`.

```sql
SELECT (ransac_fit_agg(y, [x1, x2], {'max_trials': 200, 'random_state': 42})).n_inliers AS n_inliers
FROM reg_data;
```

### theil_sen_fit / theil_sen_fit_agg

Theil-Sen estimator (median of subset slopes).

```sql skip
theil_sen_fit(y LIST(DOUBLE), x LIST(LIST(DOUBLE)) [, options]) -> STRUCT
theil_sen_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| max_subpopulation | INTEGER | 10000 | Maximum number of subsets considered |
| n_subsamples | INTEGER | automatic | Rows per subset |
| max_iterations | INTEGER | 300 | Iterations of the spatial-median solver |
| tolerance | DOUBLE | 1e-3 | Convergence tolerance |
| random_state (alias `seed`) | INTEGER | 0 | Random seed (subsets and bootstrap) |
| compute_inference | BOOLEAN | false | Coefficient confidence intervals |
| confidence_level | DOUBLE | 0.95 | Level of the intervals |
| ci_method | VARCHAR | `'exact'` (one feature), else `'bootstrap'`; `theil_sen_tidy_by`: `'exact_or_none'` (exact where it applies, otherwise NULL intervals) | `'exact'`: Sen's distribution-free slope interval with Kendall's z / p-value (`trend::sens.slope`); `'bootstrap'`: pairs bootstrap SDs and percentile intervals, also for the intercept |
| n_bootstrap (alias `n_boot`) | INTEGER | 999 (`theil_sen_tidy_by`: 199) | Bootstrap resamples; the bootstrap costs about this many point fits per group |

With `compute_inference` the exact method fills `conf_low` / `conf_high`,
`t_values` (Kendall's z) and `p_values` for the slope (no standard error, no
intercept interval: `std_errors` and the `intercept_*` fields are NULL); the
bootstrap fills `std_errors`, `conf_low` / `conf_high` and the `intercept_*` SD
and interval (`t_values` / `p_values` NULL). There is no F-test (`f_statistic` /
`f_pvalue` NULL). See [Theil-Sen](api/regression/theil_sen.md#inference).

```sql
SELECT theil_sen_fit(
    [1.0, 2.1, 2.9, 4.2, 25.0, 6.1],
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
) AS fit;
```

---

## Constrained Regression (BLS/NNLS)

### bls_fit_agg

Bounded least squares: box constraints on the coefficients. Without any bound
option the lower bound defaults to 0, which makes it equivalent to NNLS.

```sql skip
bls_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| lower_bound (alias `lower`) | DOUBLE | 0 when no bound is given | Lower bound for every coefficient |
| upper_bound (alias `upper`) | DOUBLE | unbounded | Upper bound for every coefficient |
| fit_intercept | BOOLEAN | false | Include an (unconstrained) intercept |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-10 | Convergence tolerance |

**Returns:** the [BlsFitResult](#blsfitresult-structure) struct.

```sql
SELECT unnest(bls_fit_agg(y, [x1, x2, x3], {'lower_bound': 0.0, 'upper_bound': 1.0})) FROM reg_data;
```

### nnls_fit_agg

Non-negative least squares: every coefficient is constrained to be >= 0.
Options: `fit_intercept` (default false), `max_iterations`, `tolerance`.
Returns the [BlsFitResult](#blsfitresult-structure) struct.

```sql
SELECT category, (nnls_fit_agg(y, [x1, x2])).coefficients AS coefficients
FROM reg_data
GROUP BY category
ORDER BY category;
```

---

## PLS, Isotonic and Quantile Regression

These three methods have a `*_fit_agg` aggregate that returns the fitted model,
fit-predict aggregates (`pls_fit_predict_agg`, `isotonic_fit_predict_agg`,
`quantile_fit_predict_agg`) and table macros (`*_fit_predict_by`); see
[Fit-Predict Aggregate Functions](#fit-predict-aggregate-functions).

| Method | Options (default) | Notes |
|--------|-------------------|-------|
| PLS | `n_components` (1), `fit_intercept` (true) | SIMPLS latent components; for collinear or wide data |
| Isotonic | `increasing` (true) | Monotone fit (PAVA); `x` is a single `DOUBLE`, not a list |
| Quantile | `tau` (alias `quantile`, 0.5), `fit_intercept` (true), `max_iterations` (1000), `tolerance` (1e-6), `se` (`'iid'`, `'nid'`, `'boot'`), `compute_inference`, `confidence_level`, `n_boot`, `seed` | Conditional quantile; `tau = 0.5` is median regression; with `se` the fit carries `quantreg::summary.rq` standard errors |

```sql
SELECT
    (p).y,
    round((p).yhat, 3) AS yhat
FROM (
    SELECT unnest(quantile_fit_predict_agg(y, [x1, x2], {'tau': 0.9})) AS p
    FROM reg_data
)
LIMIT 3;
```

**Model aggregates.**

```text
pls_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options])
    -> STRUCT(coefficients, intercept, r_squared, n_components, n_observations, n_features)
quantile_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options])
    -> STRUCT(coefficients, intercept, tau, n_observations, n_features)
isotonic_fit_agg(y DOUBLE, x DOUBLE [, options])
    -> STRUCT(x DOUBLE[], fitted DOUBLE[], increasing, r_squared, n_observations)
```

The isotonic model is a monotone function given by its knots (`x`) and the
fitted value at each knot (`fitted`); `predict(model, [x])` interpolates
between knots and clamps outside the training range. All three models work
with [`predict`](#predict); PLS and quantile models also work with
[`tidy`](#tidy) (PLS: estimates only; quantile: with inference when fitted
with the `se` or `compute_inference` option). A group with too few usable
rows returns NULL.

```sql
SELECT unnest(pls_fit_agg(y, [x1, x2, x3], {'n_components': 2})) FROM reg_data;
SELECT unnest(quantile_fit_agg(y, [x1, x2], {'tau': 0.5})) FROM reg_data;
SELECT (isotonic_fit_agg(y, x1, {'increasing': true})).r_squared AS r2 FROM reg_data;

-- Median prediction for a new row
SELECT round(predict(quantile_fit_agg(y, [x1, x2], {'tau': 0.5}), [5.0, 3.0]), 3) AS median_yhat
FROM reg_data;
```

---

## Generalized Linear Models

`poisson_fit_agg`, `binomial_fit_agg`, `logistic_fit_agg`, `negbinom_fit_agg`,
`gamma_fit_agg` and `tweedie_fit_agg` are fitted by IRLS and share one
signature and most options:

```sql skip
<family>_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

**Shared options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include an intercept term |
| max_iterations | INTEGER | 100 | Maximum IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Add standard errors, z-tests, p-values, confidence intervals |
| confidence_level | DOUBLE | 0.95 | Confidence level for the intervals |
| glm_lambda | DOUBLE | 0.0 | L2 regularization (see [glm_lambda](#glm_lambda)) |
| offset | INTEGER | none | 1-based index into `x` of an offset column; added to the linear predictor with coefficient 1 and removed from the design. Used as-is, so take logs upstream when the link needs it. |
| feature_names, prior, vcov | | | Explicit priors and interval type; see [Priors](api/glm/priors.md) |

**Family-specific options:**

| Function | Key | Default | Description |
|----------|-----|---------|-------------|
| `poisson_fit_agg` | link (alias `poisson_link`) | 'log' | `'log'`, `'identity'`, `'sqrt'` |
| `binomial_fit_agg` | binomial_link | 'logit' | `'logit'`, `'probit'`, `'cloglog'` |
| `logistic_fit_agg` | threshold | 0.5 | Classification threshold used for `accuracy` |
| `negbinom_fit_agg` | theta (aliases `nb_theta`, `dispersion`) | estimated | Fix the NB dispersion instead of estimating it |
| `tweedie_fit_agg` | power (alias `tweedie_power`) | 1.5 | Variance power, typically between 1 and 2 |
| `gamma_fit_agg` | - | - | Log link, variance power fixed at 2 |

**Returns:** the [GlmFitResult](#glmfitresult-structure) struct;
`logistic_fit_agg` replaces `dispersion` with `accuracy` and `threshold`. Every
GLM struct ends with `family` and `link` (VARCHAR), so
[`predict(model, x)`](#predict) can map predictions to the response scale.

**Per-group prediction.** `poisson_fit_predict_by`, `binomial_fit_predict_by`,
`logistic_fit_predict_by`, `negbinom_fit_predict_by`, `gamma_fit_predict_by` and
`tweedie_fit_predict_by` fit one model per group and append `yhat` (response
scale) to every source row; see [Fit-Predict Table Macros](#fit-predict-table-macros).

**Examples:**

```sql
-- Count data
SELECT unnest(poisson_fit_agg(y_count, [x1, x2], {'compute_inference': true})) FROM reg_data;

-- Binary outcome with a probit link
SELECT (binomial_fit_agg(y_binary, [x1, x2], {'binomial_link': 'probit'})).coefficients AS coefficients
FROM reg_data;

-- Logistic regression with in-sample accuracy
SELECT (logistic_fit_agg(y_binary, [x1, x2])).accuracy AS accuracy FROM reg_data;

-- Over-dispersed counts
SELECT (negbinom_fit_agg(y_count, [x1, x2])).dispersion AS dispersion FROM reg_data;

-- Positive continuous outcomes
SELECT (gamma_fit_agg(y_positive, [x1, x2])).coefficients AS gamma_coef,
       (tweedie_fit_agg(y_positive, [x1, x2], {'power': 1.5})).coefficients AS tweedie_coef
FROM reg_data;

-- Predicted probability (response scale) and log-odds (link scale) for a new row
WITH fit AS (SELECT logistic_fit_agg(y_binary, [x1, x2]) AS m FROM reg_data)
SELECT m.family, m.link,
       round(predict(m, [4.0, 5.0]), 4) AS probability,
       round(predict(m, [4.0, 5.0], {'type': 'link'}), 4) AS log_odds
FROM fit;
```

---

## ALM

### alm_fit_agg

Augmented Linear Model: likelihood-based regression with a choice of 24 error
distributions and several loss functions.

```sql skip
alm_fit_agg(y DOUBLE, x LIST(DOUBLE) [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| distribution (alias `dist`) | VARCHAR | 'normal' | Error distribution (table below) |
| loss | VARCHAR | 'likelihood' | `'likelihood'`, `'mse'`, `'mae'`, `'ham'`, `'role'` |
| quantile | DOUBLE | 0.5 | Quantile for `asymmetric_laplace` |
| role_trim | DOUBLE | 0.05 | Trim share for the ROLE loss |
| fit_intercept | BOOLEAN | true | Include an intercept term |
| max_iterations | INTEGER | 100 | Maximum iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Add standard errors, t-tests, p-values, intervals |
| confidence_level | DOUBLE | 0.95 | Confidence level for the intervals |

| Category | Distributions |
|----------|--------------|
| Continuous (unbounded) | `normal`, `laplace`, `student_t`, `logistic`, `asymmetric_laplace`, `generalised_normal`, `s` |
| Continuous (positive) | `log_normal`, `log_laplace`, `log_s`, `log_generalised_normal`, `gamma`, `inverse_gaussian`, `exponential` |
| Continuous (bounded) | `folded_normal`, `rectified_normal`, `box_cox_normal`, `beta`, `logit_normal` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |
| Ordinal | `cumulative_logistic`, `cumulative_normal` |

**Returns:** the [AlmFitResult](#almfitresult-structure) struct.

```sql
-- Laplace errors (median regression, robust to outliers)
SELECT unnest(alm_fit_agg(y, [x1, x2], {'distribution': 'laplace'})) FROM reg_data;

-- 75th percentile via the asymmetric Laplace distribution
SELECT (alm_fit_agg(y, [x1, x2], {'distribution': 'asymmetric_laplace', 'quantile': 0.75})).coefficients AS coefficients
FROM reg_data;
```

---

## AFT Survival Regression

### aft_fit_agg

Accelerated failure time model for right-censored durations.

```sql skip
aft_fit_agg(time DOUBLE, x LIST(DOUBLE), event DOUBLE [, options]) -> STRUCT
```

`event` is 1 when the event was observed and 0 when the row is right-censored.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| distribution (alias `dist`) | VARCHAR | 'weibull' | `'weibull'`, `'lognormal'`, `'loglogistic'`, `'exponential'` |
| fit_intercept | BOOLEAN | true | Include an intercept term |
| max_iterations | INTEGER | 100 | Newton iterations |
| tolerance | DOUBLE | see [AFT](api/survival/aft.md) | Convergence tolerance |
| compute_inference | BOOLEAN | false | Add standard errors, z-tests, intervals |
| confidence_level | DOUBLE | 0.95 | Confidence level for the intervals |
| feature_names, prior, vcov | | | See [Priors](api/glm/priors.md) |

**Returns:**

```text
STRUCT(coefficients DOUBLE[], intercept DOUBLE, scale DOUBLE,
       log_likelihood DOUBLE, null_log_likelihood DOUBLE, aic DOUBLE, bic DOUBLE,
       n_observations BIGINT, n_events BIGINT, n_censored BIGINT,
       n_features BIGINT, iterations INTEGER, converged BOOLEAN
     [, std_errors DOUBLE[], z_values DOUBLE[], p_values DOUBLE[],
        ci_lower DOUBLE[], ci_upper DOUBLE[],
        intercept_std_error DOUBLE, log_scale_std_error DOUBLE])
```

Coefficients are on the log-time scale.

```sql
SELECT unnest(aft_fit_agg(duration, [x1, x2], event, {'dist': 'weibull'})) FROM reg_data;
```

### aft_cdf / aft_quantile

Stateless scalar helpers for a fitted AFT model.

```sql skip
aft_cdf(t DOUBLE, eta DOUBLE, scale DOUBLE, distribution VARCHAR) -> DOUBLE       -- P(T <= t)
aft_quantile(p DOUBLE, eta DOUBLE, scale DOUBLE, distribution VARCHAR) -> DOUBLE  -- p-quantile of T
```

`eta` is the linear predictor `intercept + x'beta` and `scale` the fitted
`scale`. Any `NULL` argument gives `NULL`.

```sql
WITH fit AS (
    SELECT aft_fit_agg(duration, [x1, x2], event, {'dist': 'weibull'}) AS f FROM reg_data
)
SELECT
    aft_cdf(5.0, f.intercept + f.coefficients[1] * 3.0 + f.coefficients[2] * 2.0, f.scale, 'weibull') AS p_within_5,
    aft_quantile(0.5, f.intercept + f.coefficients[1] * 3.0 + f.coefficients[2] * 2.0, f.scale, 'weibull') AS median_time
FROM fit;
```

---

## Mixed-Effects GLMs

### glmm_fit_agg

One model fitted jointly across groups with a random intercept per group.

```sql skip
glmm_fit_agg(y DOUBLE, x LIST(DOUBLE), group ANY [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| family | VARCHAR | 'gaussian' | `'gaussian'`, `'poisson'`, `'binomial'` (see [GLMM](api/glm/glmm.md) for others) |
| reml | BOOLEAN | true | REML rather than ML for Gaussian variance components |
| random (alias `random_slopes`) | INTEGER[] | none | 1-based indices into `x` that also get a random slope |
| groups (alias `crossed`) | INTEGER[] | none | 1-based indices into `x` of extra crossed grouping columns |
| fit_intercept | BOOLEAN | true | Include a fixed intercept |
| max_iterations | INTEGER | 100 | Inner iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Fixed-effect standard errors, tests, intervals |
| ddf | VARCHAR | family default | `'satterthwaite'` (t tests with lmerTest's Satterthwaite df; gaussian only, its default) or `'wald'` (z tests; the default for the other families) |
| confidence_level | DOUBLE | 0.95 | Confidence level for the intervals |

**Returns:**

```text
STRUCT(coefficients DOUBLE[], intercept DOUBLE,
       var_group DOUBLE, var_residual DOUBLE, icc DOUBLE,
       log_likelihood DOUBLE, aic DOUBLE, bic DOUBLE, deviance DOUBLE,
       n_observations BIGINT, n_groups BIGINT, n_features BIGINT,
       iterations INTEGER, converged BOOLEAN,
       random_cov DOUBLE[], random_dim INTEGER,
       factors STRUCT(n_levels BIGINT, var DOUBLE)[]
     [, std_errors DOUBLE[], z_values DOUBLE[], p_values DOUBLE[],
        ci_lower DOUBLE[], ci_upper DOUBLE[], intercept_std_error DOUBLE],
       ranef STRUCT("group" VARCHAR, intercept DOUBLE, se DOUBLE, n BIGINT,
                    effects DOUBLE[], effects_se DOUBLE[],
                    effects_conf_low DOUBLE[], effects_conf_high DOUBLE[])[],
       nb_theta DOUBLE, df DOUBLE[], intercept_df DOUBLE, df_residual DOUBLE, t_values DOUBLE[], ...)
```

`ranef[].se` / `effects_se` are the BLUPs' conditional standard errors
(lme4 `ranef(condVar = TRUE)`); `df` / `intercept_df` the Satterthwaite degrees of
freedom (NULL for Wald). Under Satterthwaite the statistics are in `t_values`
(`z_values` NULL), under Wald in `z_values` (`t_values` NULL). See [GLMM](api/glm/glmm.md#inference).

```sql
SELECT (glmm_fit_agg(y, [x1, x2], store)).icc AS icc FROM reg_data;
```

### glmm_fit_by

Table macro: fits one GLMM over the whole table and returns one row per group
with its random effect.

```sql skip
glmm_fit_by(source VARCHAR, group_col, y_col, x_cols [, options]) -> TABLE
```

Output columns: `group`, `ranef`, `ranef_se`, `n`, `fixed_intercept`,
`fixed_coefficients`, `var_group`, `var_residual`, `icc`.

```sql
SELECT * FROM glmm_fit_by('reg_data', store, y, [x1, x2]);
```

### glmm_tidy_by, glmm_glance_by, glmm_augment_by

Integration-contract outputs of one GLMM per `group_col` with random effects over
`re_group` (broom.mixed `tidy` / `glance` / `augment`); `model_id` = `model_name` =
`'glmm'`. See [GLMM](api/glm/glmm.md#glmm_tidy_by-glmm_glance_by-glmm_augment_by).

```sql skip
glmm_tidy_by(source, group_col, y_col, x_cols, re_group [, options, names, re_names]) -> TABLE  -- terms
glmm_glance_by(source, group_col, y_col, x_cols, re_group [, options]) -> TABLE                 -- summary
glmm_augment_by(source, group_col, y_col, x_cols, re_group [, options, order_by]) -> TABLE      -- obs
```

`glmm_tidy_by` adds `effect` (`fixed`, `ran_pars`, `ran_vals`), `re_group`,
`re_level` and `df` after the `terms` columns; `glmm_augment_by` adds
`fitted_population` and `pearson_residual` after the `obs` columns.

```sql
CREATE OR REPLACE TABLE reg_scope AS SELECT 'all' AS scope, * FROM reg_data;
SELECT effect, term, estimate, std_error, df
FROM glmm_tidy_by('reg_scope', scope, y, [x1, x2], store) WHERE effect <> 'ran_vals';
```

---

## Empirical-Bayes Shrinkage

### eb_shrink_agg

Shrinks a set of per-group estimates toward their precision-weighted mean.

```sql skip
eb_shrink_agg(estimate DOUBLE, se DOUBLE [, options]) -> STRUCT
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| tau_squared (alias `tau2`) | DOUBLE | estimated | Fix the between-group variance |
| tau_method (alias `shrinkage`) | VARCHAR | 'dl' | `'dl'` (DerSimonian-Laird) or `'none'` (complete pooling) |

**Returns:**

```text
STRUCT(mu DOUBLE, mu_se DOUBLE, tau_squared DOUBLE, i_squared DOUBLE, q DOUBLE,
       n_groups BIGINT,
       shrunken STRUCT(estimate DOUBLE, se DOUBLE, shrunken DOUBLE,
                       shrunken_se DOUBLE, weight DOUBLE)[])   -- input order
```

```sql
CREATE OR REPLACE TABLE store_slopes AS
SELECT store,
       (ols_fit_agg(y, [x1], {'compute_inference': true})).coefficients[1] AS est,
       (ols_fit_agg(y, [x1], {'compute_inference': true})).std_errors[1] AS se
FROM reg_data
GROUP BY store;

SELECT (eb_shrink_agg(est, se)).tau_squared AS tau_squared FROM store_slopes;
```

### eb_shrink_by

Table macro over a table of estimates. Returns every source column plus
`shrunken`, `shrunken_se`, `weight`, `mu` and `tau_squared`.

```sql skip
eb_shrink_by(source VARCHAR, estimate_col, se_col [, options]) -> TABLE
```

```sql
SELECT store, est, shrunken FROM eb_shrink_by('store_slopes', est, se) ORDER BY store;
```

---

## Causal Mediation Analysis

Mediation analysis as R's `mediation::mediate` for linear models: per group, the
mediator model `m ~ x + covariates` (path `a`) and the outcome model
`y ~ x + m + covariates` (`b`, direct path `c'`) are fitted by OLS on the same
complete rows. See [Mediation](api/statistics/mediation.md) for the inference
details.

### mediation_by

```sql skip
mediation_by(source VARCHAR, group_col, treatment, mediator, outcome
             [, covariates := [c1, ...]] [, options := {...}]) -> TABLE
```

One row per group and effect (`term` = `acme` `a·b·Δ`, `ade` `c'·Δ`, `total`,
`prop_mediated`, with `Δ = treat_value − control_value`) in the contract `terms`
schema (`id` = group, `model_id = model_name = 'mediation'`), plus `method`,
`n_obs` and `n_sims`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| method | VARCHAR | 'bootstrap' | `'bootstrap'` (case resampling), `'sobel'` (delta method) or `'quasi_bayesian'` (`mediate`'s Monte Carlo) |
| ci_type | VARCHAR | 'percentile' | Bootstrap interval: `'percentile'` or `'bca'` |
| n_bootstrap (aliases `n_sims`, `sims`) | INTEGER | 1000 | Resamples / draws |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the draws; results are reproducible |
| confidence_level | DOUBLE | 0.95 | Interval level |
| treat_value, control_value | DOUBLE | 1, 0 | Treatment contrast |

```sql
CREATE OR REPLACE TABLE med_data AS
SELECT i, 'g' || (i % 2) AS grp, ((i // 2) % 2)::DOUBLE AS x, sin(i * 1.3) AS c1,
       0.5 + 0.8 * ((i // 2) % 2) + 0.3 * sin(i * 1.3) + sin(i * 2.1 + 0.4) * 0.9 AS m
FROM range(120) t(i);
CREATE OR REPLACE TABLE med_data AS
SELECT *, 1 + 0.4 * x + 0.6 * m - 0.2 * c1 + cos(i * 1.7 + 0.2) * 0.8 AS y FROM med_data;

SELECT grp, term, round(estimate, 3) AS estimate, round(conf_low, 3) AS conf_low, round(conf_high, 3) AS conf_high
FROM mediation_by('med_data', grp, treatment := x, mediator := m, outcome := y, covariates := [c1]);
```

### mediation_agg

```sql skip
mediation_agg(y DOUBLE, m DOUBLE, x DOUBLE [, covariates DOUBLE[] [, options]]) -> LIST(STRUCT)
```

The aggregate behind `mediation_by`; returns the four effect rows
`STRUCT(term, estimate, std_error, statistic, p_value, conf_low, conf_high,
conf_level, method, n_obs, n_sims)`, or NULL for a degenerate group.

```sql
SELECT grp, (mediation_agg(y, m, x, [c1], {'method': 'sobel'}))[1].p_value AS acme_p
FROM med_data GROUP BY grp ORDER BY grp;
```

---

## AID Demand Classification

AID (Automatic Identification of Demand) classifies demand series as regular or
intermittent, picks a best-fit distribution and flags stockouts, product
launches, obsolescence and outliers.

**Options (all AID functions):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero share at or above which a series is intermittent |
| outlier_method | VARCHAR | 'zscore' | `'zscore'` (mean ± 3 sd) or `'iqr'` (1.5 × IQR) |

### aid_agg

```sql skip
aid_agg(y DOUBLE [, options]) -> STRUCT
```

**Returns:**

```text
STRUCT(demand_type VARCHAR, is_intermittent BOOLEAN, distribution VARCHAR,
       mean DOUBLE, variance DOUBLE, zero_proportion DOUBLE, n_observations BIGINT,
       has_stockouts BOOLEAN, is_new_product BOOLEAN, is_obsolete_product BOOLEAN,
       stockout_count BIGINT, new_product_count BIGINT, obsolete_product_count BIGINT,
       high_outlier_count BIGINT, low_outlier_count BIGINT)
```

```sql
SELECT sku, unnest(aid_agg(qty ORDER BY week))
FROM demand
GROUP BY sku
ORDER BY sku;
```

### aid_anomaly_agg

Per-observation anomaly flags, in input order (use `ORDER BY` inside the call).

```sql skip
aid_anomaly_agg(y DOUBLE [, options]) -> STRUCT(stockout BOOLEAN, new_product BOOLEAN,
    obsolete_product BOOLEAN, high_outlier BOOLEAN, low_outlier BOOLEAN)[]
```

- **Stockout:** a zero between non-zero values.
- **New product:** the leading run of zeros.
- **Obsolete product:** the trailing run of zeros.
- **High / low outlier:** beyond the `outlier_method` limits.

```sql
SELECT aid_anomaly_agg(demand ORDER BY t)
FROM (VALUES (1, 0.0), (2, 0.0), (3, 5.0), (4, 0.0), (5, 8.0), (6, 0.0), (7, 0.0)) AS v(t, demand);
```

### aid_by

Table macro: one row per group with the `aid_agg` fields as columns (the group
column keeps its name).

```sql skip
aid_by(source VARCHAR, group_col, y_col [, options]) -> TABLE
```

```sql
SELECT sku, demand_type, is_new_product FROM aid_by('demand', sku, qty);
```

### aid_anomaly_by

Table macro: one row per observation with the five anomaly flags, ordered by
`order_col` within each group.

```sql skip
aid_anomaly_by(source VARCHAR, group_col, order_col, y_col [, options]) -> TABLE
```

```sql
SELECT * FROM aid_anomaly_by('demand', sku, week, qty) WHERE sku = 'launch' LIMIT 6;
```

---

## Statistical Hypothesis Tests

All tests are aggregates. Two-sample tests take a value and an `INTEGER` group
indicator with two distinct values; paired tests take two `DOUBLE` columns.
Unless noted, tests return `statistic`, `p_value` and a `method` label.

> **Unified result fields (#159).** Every test aggregate also returns the
> integration contract's `test` fields: `method, alternative, statistic, df1, df2,
> p_value, estimate, conf_low, conf_high, conf_level, effect_size, effect_name, n`.
> They are always present and NULL when they do not apply (for example `df2` of a
> t-test, or `conf_low` of a normality test). A test keeps its own fields in their
> existing order; the unified fields it did not already have are appended after
> them. The values come from the upstream `anofox-statistics` `TestResult`: `df1`
> / `df2` are the (numerator / denominator) degrees of freedom, `estimate` the
> point estimate (for two-sample tests `group1 - group2`), and `conf_level` the
> level of `conf_low` / `conf_high`.
>
> Deprecated aliases (removed in the next minor release): `ci_lower` / `ci_upper`
> (use `conf_low` / `conf_high`), `df` (use `df1`), and in `one_way_anova_agg`
> `f_statistic` (use `statistic`) and `df_between` / `df_within` (use `df1` / `df2`).

```sql
SELECT r.statistic, r.df1, r.estimate, r.conf_low, r.conf_high, r.effect_name
FROM (SELECT t_test_agg(y, grp2) AS r FROM reg_data);
```

> **Option syntax.** Most tests read options from a struct literal
> (`{'alternative': 'less'}`). The functions marked **MAP** in the tables below
> currently read options only from a `MAP` literal (`MAP {'trim': 0.1}`).

### Normality

| Function | Signature | Returns |
|----------|-----------|---------|
| `shapiro_wilk_agg` | `(value DOUBLE)` | `statistic, p_value, n, method` |
| `jarque_bera_agg` | `(value DOUBLE)` | `statistic, p_value, skewness, kurtosis, n` |
| `dagostino_k2_agg` | `(value DOUBLE)` | `statistic, p_value, n, method` |

```sql
SELECT (shapiro_wilk_agg(y)).p_value AS sw_p,
       (jarque_bera_agg(y)).p_value AS jb_p,
       (dagostino_k2_agg(y)).p_value AS k2_p
FROM reg_data;
```

### Parametric Tests

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `t_test_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `alternative` ('two_sided'), `confidence_level` (0.95), `kind` ('welch'; or 'student', alias `var_equal`), `mu` (0) | `statistic, p_value, df, effect_size, ci_lower, ci_upper, n1, n2, method` |
| `one_way_anova_agg` | `(value DOUBLE, group BIGINT \| VARCHAR)` | - | `f_statistic, p_value, df_between, df_within, ss_between, ss_within, n_groups, n, method` |
| `yuen_agg` **MAP** | `(value DOUBLE, group_id INTEGER [, options])` | `trim` (0.2), `alternative`, `confidence_level` (0.95) | `statistic, p_value, df, effect_size, ci_lower, ci_upper, n1, n2, method` |
| `brown_forsythe_agg` | `(value DOUBLE, group BIGINT \| VARCHAR)` | - | `statistic, p_value, df, n, method` |

```sql
SELECT (t_test_agg(y, grp2)).p_value AS welch_p,
       (t_test_agg(y, grp2, {'kind': 'student', 'alternative': 'less'})).p_value AS student_less_p,
       (one_way_anova_agg(y, grp3)).p_value AS anova_p,
       (yuen_agg(y, grp2, MAP {'trim': 0.1})).p_value AS yuen_p,
       (brown_forsythe_agg(y, grp3)).p_value AS bf_p
FROM reg_data;
```

### Nonparametric Tests

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `mann_whitney_u_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `alternative`, `confidence_level`, `continuity_correction` (alias `correction`) | `statistic, p_value, effect_size, ci_lower, ci_upper, n1, n2, method` |
| `kruskal_wallis_agg` | `(value DOUBLE, group BIGINT \| VARCHAR)` | - | `statistic, p_value, df, n, method` |
| `wilcoxon_signed_rank_agg` | `(x DOUBLE, y DOUBLE [, options])` | `alternative`, `confidence_level`, `continuity_correction` | `statistic, p_value, ci_lower, ci_upper, n, method` |
| `brunner_munzel_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `alternative`, `confidence_level` | `statistic, p_value, df, effect_size, ci_lower, ci_upper, n1, n2, method` |
| `permutation_t_test_agg` **MAP** | `(value DOUBLE, group_id INTEGER [, options])` | `alternative`, `n_permutations` (10000), `seed` (42) | `statistic, p_value, n1, n2, method` |

```sql
SELECT (mann_whitney_u_agg(y, grp2, {'alternative': 'greater'})).p_value AS mw_p,
       (kruskal_wallis_agg(y, grp3)).p_value AS kw_p,
       (wilcoxon_signed_rank_agg(y, x1)).p_value AS wsr_p,
       (brunner_munzel_agg(y, grp2)).p_value AS bm_p,
       (permutation_t_test_agg(y, grp2, MAP {'n_permutations': 2000})).p_value AS perm_p
FROM reg_data;
```

### Distribution Comparison

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `energy_distance_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `n_permutations` (alias `permutations`), `seed` (42) | `statistic, p_value, n1, n2, method` |
| `mmd_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `n_permutations` (alias `permutations`), `seed` (42) | `statistic, p_value, n1, n2, method` |

```sql
SELECT (energy_distance_agg(y, grp2, {'n_permutations': 199})).p_value AS energy_p,
       (mmd_agg(y, grp2, {'n_permutations': 199})).p_value AS mmd_p
FROM reg_data;
```

### Data Drift

Current sample against a reference sample (evidently's drift tests). `group` is BIGINT or
VARCHAR with two labels (`ad_test_agg`: two or more); the reference is the smaller label
unless `reference` names it. Every result also carries the unified test fields.

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `ks_test_agg` | `(value DOUBLE, group [, options])` | `reference`, `alternative`, `exact` | `statistic, p_value, n1, n2, method, exact` |
| `ad_test_agg` | `(value DOUBLE, group [, options])` | `midrank` (true) | `statistic, ad, p_value, k, n, method` |
| `cvm_test_agg` | `(value DOUBLE, group [, options])` | `reference`, `method` (`auto` / `exact` / `asymptotic`) | `statistic, p_value, n1, n2, method, exact` |
| `epps_singleton_agg` | `(value DOUBLE, group [, options])` | `reference`, `t` ([0.4, 0.8]) | `statistic, p_value, df, n1, n2, method` |
| `wasserstein_agg` | `(value DOUBLE, group [, options])` | `reference`, `p` (1), `n_permutations` (0), `seed` (42) | `distance, distance_normed, p_value, n1, n2, method` |
| `psi_agg` | `(value DOUBLE \| VARCHAR, group [, options])` | `reference`, `binning`, `n_bins`, `threshold` (0.1) | `psi, n_bins, n1, n2, drift, method` |
| `kl_divergence_agg` | `(value DOUBLE \| VARCHAR, group [, options])` | as `psi_agg`, plus `base` | `kl, n_bins, n1, n2, drift, method` |
| `jensen_shannon_agg` | `(value DOUBLE \| VARCHAR, group [, options])` | as `psi_agg`, plus `base` | `distance, divergence, n_bins, n1, n2, drift, method` |
| `hellinger_agg` | `(value DOUBLE \| VARCHAR, group [, options])` | as `psi_agg` | `distance, n_bins, n1, n2, drift, method` |
| `tvd_agg` | `(category DOUBLE \| VARCHAR, group [, options])` | `reference`, `n_permutations` (1000), `seed` (42) | `statistic, p_value, n_categories, n1, n2, method` |

```sql
SELECT (ks_test_agg(y, grp2)).p_value AS ks_p,
       (psi_agg(y, grp2)).psi AS psi,
       (tvd_agg(grp3::VARCHAR, grp2)).statistic AS tvd
FROM reg_data;
```

### Equivalence Tests (TOST)

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `tost_t_test_agg` | `(value DOUBLE, group_id INTEGER [, options])` | `delta` (symmetric bound), `bound_lower`, `bound_upper`, `confidence_level`, `kind` | `t_lower, t_upper, p_lower, p_upper, p_value, df, estimate, ci_lower, ci_upper, bound_lower, bound_upper, equivalent, n, method` |
| `tost_paired_agg` **MAP** | `(x DOUBLE, y DOUBLE [, options])` | `delta`, `bound_lower` (-0.5), `bound_upper` (0.5), `alpha` (0.05) | `estimate, ci_lower, ci_upper, p_value, equivalent, n, method` |
| `tost_correlation_agg` **MAP** | `(x DOUBLE, y DOUBLE [, options])` | `rho_null` (0), `delta`, `bound_lower` (-0.1), `bound_upper` (0.1), `alpha`, `method` ('pearson' or 'spearman') | `estimate, ci_lower, ci_upper, p_value, equivalent, n, method` |

```sql
SELECT (tost_t_test_agg(y, grp2, {'delta': 2.0})).equivalent AS equivalent_means,
       (tost_paired_agg(y, x1, MAP {'delta': 5.0})).p_value AS paired_p,
       (tost_correlation_agg(x1, y, MAP {'delta': 0.3})).equivalent AS equivalent_corr
FROM reg_data;
```

### Forecast Comparison

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `diebold_mariano_agg` **MAP** | `(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE [, options])` | `loss` (squared; `'absolute'`), `horizon` (1), `var_estimator` (`'bartlett'`), `alternative` | `statistic, p_value, n, method` |
| `clark_west_agg` **MAP** | `(actual DOUBLE, forecast_restricted DOUBLE, forecast_unrestricted DOUBLE [, options])` | `horizon` (1) | `statistic, p_value, n, method` |

```sql
SELECT (diebold_mariano_agg(y, y + sin(id), y + 2 * cos(id))).p_value AS dm_p,
       (clark_west_agg(y, y + sin(id), y + 0.5 * cos(id))).p_value AS cw_p
FROM reg_data;
```

### Correlation Tests

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `pearson_agg` | `(x DOUBLE, y DOUBLE [, options])` | `confidence_level` (0.95) | `r, statistic, p_value, ci_lower, ci_upper, n, method` |
| `spearman_agg` | `(x DOUBLE, y DOUBLE [, options])` | `confidence_level` | `r, statistic, p_value, ci_lower, ci_upper, n, method` |
| `kendall_agg` | `(x DOUBLE, y DOUBLE [, options])` | `confidence_level`, `variant` ('tau_a', 'tau_b', 'tau_c') | `tau, statistic, p_value, ci_lower, ci_upper, n, method` |
| `distance_cor_agg` **MAP** | `(x DOUBLE, y DOUBLE [, options])` | `n_permutations` (1000), `seed` (42) | `dcor, statistic, p_value, n, method` |
| `icc_agg` **MAP** | `(value DOUBLE, subject_id BIGINT, rater_id BIGINT [, options])` | `type` ('single' or 'average') | `icc, f_statistic, ci_lower, ci_upper, n_subjects, n_raters, method` |

```sql
SELECT (pearson_agg(x1, y)).r AS pearson_r,
       (spearman_agg(x1, y, {'confidence_level': 0.99})).ci_lower AS spearman_ci_lower,
       (kendall_agg(x1, y)).tau AS kendall_tau,
       (distance_cor_agg(x1, y, MAP {'n_permutations': 199})).dcor AS dcor
FROM reg_data;

-- ICC: 10 subjects rated by 3 raters
SELECT (icc_agg(score, subject, rater)).icc AS icc
FROM (
    SELECT s::BIGINT AS subject, r::BIGINT AS rater, s * 1.0 + r * 0.2 + sin(s * r) AS score
    FROM range(1, 11) a(s), range(1, 4) b(r)
);
```

### Categorical Tests

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `chisq_test_agg` | `(row_var INTEGER, col_var INTEGER [, options])` | `continuity_correction` (aliases `correction`, `yates`) | `statistic, p_value, df, method` |
| `chisq_gof_agg` | `(observed BIGINT, expected_prob DOUBLE)` | - | `statistic, p_value, df, method` |
| `g_test_agg` | `(row_var BIGINT, col_var BIGINT)` | - | `statistic, p_value, df, method` |
| `fisher_exact_agg` | `(row_var INTEGER, col_var INTEGER [, options])` | `alternative` | `statistic, p_value, odds_ratio, ci_lower, ci_upper, n, method` |
| `mcnemar_agg` **MAP** | `(var1 BIGINT, var2 BIGINT [, options])` | `correction` (true) | `statistic, p_value, df, method` |

```sql
SELECT (chisq_test_agg(grp2, grp3)).p_value AS chisq_p,
       (g_test_agg(grp2::BIGINT, grp3::BIGINT)).p_value AS g_p,
       (fisher_exact_agg(grp2, (y_binary)::INTEGER)).odds_ratio AS odds_ratio,
       (mcnemar_agg(grp2::BIGINT, y_binary::BIGINT)).p_value AS mcnemar_p
FROM reg_data;

-- Goodness of fit: observed counts against expected proportions
SELECT unnest(chisq_gof_agg(observed, expected_prob))
FROM (VALUES (18::BIGINT, 0.25), (22::BIGINT, 0.25), (31::BIGINT, 0.25), (29::BIGINT, 0.25)) AS v(observed, expected_prob);
```

### Effect Sizes

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `cramers_v_agg` | `(row_var BIGINT, col_var BIGINT)` | - | `DOUBLE` (0 to 1) |
| `phi_coefficient_agg` | `(row_var BIGINT, col_var BIGINT)` | - | `DOUBLE` (-1 to 1) |
| `contingency_coef_agg` | `(row_var BIGINT, col_var BIGINT)` | - | `DOUBLE` |
| `cohen_kappa_agg` **MAP** | `(rater1 BIGINT, rater2 BIGINT [, options])` | `weighted` (false) | `kappa, se, ci_lower, ci_upper, z, p_value` |

```sql
SELECT cramers_v_agg(grp2::BIGINT, grp3::BIGINT) AS cramers_v,
       phi_coefficient_agg(grp2::BIGINT, y_binary::BIGINT) AS phi,
       contingency_coef_agg(grp2::BIGINT, grp3::BIGINT) AS contingency,
       (cohen_kappa_agg(grp3::BIGINT, ((id + id % 2) % 3)::BIGINT)).kappa AS kappa
FROM reg_data;
```

### Proportion Tests

The input is a 0/1 success indicator, one row per trial.

| Function | Signature | Options | Returns |
|----------|-----------|---------|---------|
| `prop_test_one_agg` **MAP** | `(value BIGINT [, options])` | `p0` (alias `p`; 0.5), `alternative` | `statistic, p_value, estimate, ci_lower, ci_upper, n, method` |
| `prop_test_two_agg` **MAP** | `(value BIGINT, group_id BIGINT [, options])` | `alternative`, `correction` (true) | `statistic, p_value, estimate, ci_lower, ci_upper, n, method` |
| `binom_test_agg` **MAP** | `(value BIGINT [, options])` | `p0` (alias `p`; 0.5), `alternative` | `statistic, p_value, estimate, ci_lower, ci_upper, n, method` |

```sql
SELECT (prop_test_one_agg(y_binary::BIGINT, MAP {'p0': 0.4})).p_value AS prop_p,
       (prop_test_two_agg(y_binary::BIGINT, grp2::BIGINT)).p_value AS prop2_p,
       (binom_test_agg(y_binary::BIGINT)).estimate AS share
FROM reg_data;
```

---

### Pairwise Post-hoc Tests

Pairwise comparisons between all levels of a group column (#159). Each returns
`LIST(STRUCT(group1, group2, estimate, statistic, p_value, p_adj, conf_low,
conf_high, method, df1, alternative, conf_level, p_adjust_method))`, one element
per pair. Groups are the sorted distinct labels (`VARCHAR` or integer);
`group1` sorts before `group2` and `estimate` is `group2 - group1`. Rows follow R's
order (`B-A, C-A, C-B, ...` for `TukeyHSD`).

| Function | Signature | Options | Matches |
|----------|-----------|---------|---------|
| `pairwise_t_test_agg` | `(value DOUBLE, group VARCHAR \| BIGINT [, options])` | `p_adjust` ('holm'; 'hochberg', 'hommel', 'bonferroni', 'BH'/'fdr', 'BY', 'none'), `pool_sd` (true), `alternative` ('two_sided'), `confidence_level` (0.95; interval per comparison for 'none', else Bonferroni) | R `pairwise.t.test`, intervals as `emmeans` |
| `tukey_hsd_agg` | `(value DOUBLE, group VARCHAR \| BIGINT [, options])` | `confidence_level` (0.95) | R `TukeyHSD(aov(value ~ group))` |
| `dunn_test_agg` | `(value DOUBLE, group VARCHAR \| BIGINT [, options])` | `p_adjust` ('holm') | `FSA::dunnTest` |

`p_value` is the unadjusted p-value and `p_adj` the adjusted one (for Tukey HSD both
are the family-wise p-value). Tukey HSD alone has a simultaneous interval
(`conf_low`, `conf_high`); Dunn's z has no `df1`. Fewer than two groups give NULL.

```sql
SELECT unnest(pairwise_t_test_agg(y, grp3, {'p_adjust': 'BH'}), recursive := true)
FROM reg_data;

SELECT unnest(tukey_hsd_agg(y, grp3), recursive := true) FROM reg_data;
```

The `_by` table macros run the test within each group of a table and return the
`test` schema in long form, one row per pair: the group column, `model_id`,
`test_id` (`'<group2>-<group1>'`), `method`, `statistic`, `p_value`, `group1`,
`group2`, `p_adj`, `df1`, `df2`, `estimate`, `conf_low`, `conf_high`,
`conf_level`, `effect_size`, `effect_name`, `alternative`, `p_adjust_method`.

| Macro | Signature |
|-------|-----------|
| `pairwise_t_test_by` | `(source, group_col, value_col, factor_col, options := NULL)` |
| `tukey_hsd_by` | `(source, group_col, value_col, factor_col, options := NULL)` |
| `dunn_test_by` | `(source, group_col, value_col, factor_col, options := NULL)` |

```sql
SELECT category, group1, group2, estimate, p_adj
FROM tukey_hsd_by('reg_data', category, y, grp3);
```

### Multiple Testing

Adjust p-values from any tests for multiple comparisons, as R
`p.adjust(p, method, n)` (#219). `method`: 'holm' (default), 'hochberg', 'hommel',
'bonferroni', 'BH' (alias 'fdr'), 'BY', 'none'; case-insensitive, an unknown
method is an error. NULL p-values stay NULL and are not counted. `n` (number of
hypotheses) defaults to the number of non-NULL p-values and may not be smaller.

| Function | Signature | Returns |
|----------|-----------|---------|
| `p_adjust_by` | `(source, group_col, p_col, method := 'holm', n := NULL)` | every source row, in source order, with `p_adjusted` (one family per group) |
| `p_adjust_agg` | `(p DOUBLE [, method VARCHAR [, n BIGINT]])` | `DOUBLE[]` in input order (use `ORDER BY` inside the call); constant `method`/`n` |
| `p_adjust_list` | `(p DOUBLE[] [, method VARCHAR [, n BIGINT]])` | `DOUBLE[]` in list order |
| `p_adjust` | `(p DOUBLE [, method VARCHAR [, n BIGINT]]) OVER (...)` | `DOUBLE`, the row's adjusted p-value within its window frame (the partition without `ORDER BY`); window-only, **DuckDB 2.0+** |

```sql
CREATE OR REPLACE TABLE results AS SELECT * FROM (VALUES
    ('weight', 'dose_a', 0.012), ('weight', 'dose_b', 0.030), ('bp', 'dose_a', 0.001)) t(endpoint, test, p_value);
SELECT * FROM p_adjust_by('results', endpoint, p_value, method := 'BH');
SELECT endpoint, p_adjust_agg(p_value, 'holm' ORDER BY test) FROM results GROUP BY endpoint;
SELECT p_adjust_list([0.01, 0.02, 0.04], 'BY');
```

On DuckDB 2.0, `p_adjust(p_value, 'BH') OVER (PARTITION BY endpoint)` gives the
row-aligned result inline (#224). It is not registered on DuckDB 1.5, whose window
aggregates cannot tell which row of a whole-partition frame they compute; there
`p_adjust_by` gives the row-aligned result, and Bonferroni, Holm and BH can also be
written with plain window functions; see
[Multiple Testing](api/statistics/hypothesis.md#row-aligned-with-plain-window-functions).

## Classifier Evaluation

ROC, precision-recall and calibration curves, ROC AUC, Brier score and log loss
of a binary classifier (#153), following scikit-learn and validated against it
and R `pROC`. `y_true` is BOOLEAN or a numeric 0/1; rows with a NULL `y_true` or
score are skipped. See [Classification](api/statistics/classification.md) for
ties, constant scores and single-class input.

| Function | Signature | Returns |
|----------|-----------|---------|
| `roc_agg` | `(y_true, score DOUBLE [, drop_intermediate BOOLEAN])` | `LIST(STRUCT(threshold, fpr, tpr))`, as sklearn `roc_curve` |
| `pr_agg` | `(y_true, score DOUBLE)` | `LIST(STRUCT(threshold, recall, precision))`, as sklearn `precision_recall_curve` |
| `calibration_agg` | `(y_true, prob DOUBLE, n_bins BIGINT [, conf_level DOUBLE])` | `LIST(STRUCT(bin, mean_pred, frac_pos, n, conf_low, conf_high))`, as sklearn `calibration_curve` with Wilson intervals |
| `auc_agg` | `(y_true, score DOUBLE)` | `DOUBLE`, sklearn `roc_auc_score` / pROC `auc`; NULL with one class |
| `brier_agg` | `(y_true, prob DOUBLE)` | `DOUBLE`, sklearn `brier_score_loss` |
| `log_loss_agg` | `(y_true, prob DOUBLE)` | `DOUBLE`, sklearn `log_loss` |

```sql
SELECT category, auc_agg(y_binary = 1, x1) AS auc FROM reg_data GROUP BY category;
```

The table macros `roc_curve_by(source, group_col, y_true_col, score_col,
drop_intermediate := true)`, `pr_curve_by(source, group_col, y_true_col,
score_col)` and `calibration_curve_by(source, group_col, y_true_col, prob_col,
n_bins := 10, conf_level := 0.95)` return the integration contract's `curve`
schema, one row per point: the group column, `model_id`, `series`, `curve_type`
(`'roc'`, `'pr'`, `'calibration'`), `x`, `y`, `y_low`, `y_high`, `threshold`,
`weight`, `n_censor`.

```sql
SELECT category, x AS fpr, y AS tpr FROM roc_curve_by('reg_data', category, y_binary, x1);
```

## Fit-Predict Window Functions

Eight window aggregates fit a model over the window frame and return a
prediction for the current row:

`ols_fit_predict`, `ridge_fit_predict`, `elasticnet_fit_predict`,
`wls_fit_predict`, `rls_fit_predict`, `huber_fit_predict`,
`ransac_fit_predict`, `theil_sen_fit_predict`.

```sql skip
<method>_fit_predict(y DOUBLE, x LIST(DOUBLE) [, options]) OVER (window) -> STRUCT
wls_fit_predict(y DOUBLE, x LIST(DOUBLE), weight DOUBLE [, options]) OVER (window) -> STRUCT
```

**Returns:** `STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE)`. The
result is `NULL` while the frame holds too few training rows.

**Semantics.** The model is fitted on the rows of the window frame and the
prediction is made for the **last row of the frame**. Use these functions for
rolling or expanding in-sample fits whose frame ends at `CURRENT ROW` over a
unique ordering, for example
`OVER (ORDER BY t ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)`.

Not supported (the result is not a prediction for the current row):

- frames that end before the current row, such as `... AND 1 PRECEDING`;
- `OVER (PARTITION BY g)` without `ORDER BY`, where every row gets the same prediction;
- `RANGE` frames over an ordering with ties.

For a prediction for every row of a group, use the
[fit-predict aggregates](#fit-predict-aggregate-functions) or
[table macros](#fit-predict-table-macros) instead.

**Options:** the method's own options (see its section above) plus
`confidence_level` (0.95, prediction interval level) and
[`null_policy`](#null_policy).

See [window functions](api/regression/fit_predict_window.md) for details.

```sql
-- Rolling 30-row in-sample fit
SELECT id, y,
       round((ols_fit_predict(y, [x1, x2]) OVER (
           ORDER BY id ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
       )).yhat, 3) AS yhat
FROM reg_data
ORDER BY id
LIMIT 8;

-- Rolling 20-row window, per category, with a robust method
SELECT category, id,
       (huber_fit_predict(y, [x1, x2]) OVER (
           PARTITION BY category ORDER BY id ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
       )).yhat AS yhat
FROM reg_data
ORDER BY category, id
LIMIT 5;

-- Weighted variant, expanding window ending at the current row
SELECT id, (wls_fit_predict(y, [x1], weight) OVER (
           ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       )).yhat AS yhat
FROM reg_data
ORDER BY id
LIMIT 5;
```

**One-step-ahead forecasts.** To predict each row from a model trained only on
earlier rows, fit with `ols_fit_agg` over a frame ending at `1 PRECEDING` and
evaluate the fitted model on the current row with the model-aware
[`predict(model, x)`](#predict). While the frame is too small to fit, the model
and the prediction are NULL.

```sql
SELECT id, y,
       round(predict((ols_fit_agg(y, [x1, x2]) OVER (
           ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
       )), [x1, x2]), 3) AS yhat_next
FROM reg_data
ORDER BY id
LIMIT 8;
```

**Prediction intervals.** `yhat_lower`/`yhat_upper` are leverage-aware
prediction intervals for the frame's last row (see
[Methodology](METHODOLOGY.md#prediction-intervals)); they are NULL when no
interval exists (zero residual degrees of freedom, singular design).

---

## Fit-Predict Aggregate Functions

These aggregates fit once on the training rows and return a prediction for
**every** row, including rows whose `y` is `NULL` (out-of-sample rows):

`ols_fit_predict_agg`, `ridge_fit_predict_agg`, `elasticnet_fit_predict_agg`,
`wls_fit_predict_agg`, `rls_fit_predict_agg`, `huber_fit_predict_agg`,
`ransac_fit_predict_agg`, `theil_sen_fit_predict_agg`, `bls_fit_predict_agg`,
`alm_fit_predict_agg`, `poisson_fit_predict_agg`, `pls_fit_predict_agg`,
`isotonic_fit_predict_agg`, `quantile_fit_predict_agg`.

```sql skip
<method>_fit_predict_agg(y DOUBLE, x LIST(DOUBLE) [, split_col VARCHAR] [, options]) -> LIST(STRUCT)
wls_fit_predict_agg(y DOUBLE, x LIST(DOUBLE), weights DOUBLE [, split_col VARCHAR] [, options]) -> LIST(STRUCT)
isotonic_fit_predict_agg(y DOUBLE, x DOUBLE [, split_col VARCHAR] [, options]) -> LIST(STRUCT)
```

**Training rows.** By default a row trains the model when `y` is not `NULL`.
With `split_col`, a row trains only when its split value is `'train'` or
`'training'` (case-insensitive) and `y` is not `NULL`; every other row is
predicted.

**Returns:** one struct per input row, in input order:

| Function group | Struct fields |
|----------------|---------------|
| ols, ridge, elasticnet, wls, rls, huber, ransac, theil_sen, bls, alm, poisson | `y, yhat, yhat_lower, yhat_upper, is_training` |
| pls, isotonic, quantile | `y, yhat, is_training` |

**Options:** the method's options from its section above, plus
`confidence_level` (0.95) and [`null_policy`](#null_policy) where intervals are
produced. PLS takes `n_components` and `fit_intercept`; isotonic takes
`increasing`; quantile takes `tau` and `fit_intercept`.

See [fit-predict aggregates](api/regression/fit_predict_agg.md) for details.

```sql
-- Hold out the last 10 rows by nulling y; they still get predictions
SELECT (p).y, round((p).yhat, 3) AS yhat, (p).is_training
FROM (
    SELECT unnest(ols_fit_predict_agg(CASE WHEN id <= 50 THEN y END, [x1, x2])) AS p
    FROM reg_data
)
WHERE NOT (p).is_training;

-- Explicit split column
SELECT count(*) FILTER (WHERE (p).is_training) AS n_train,
       count(*) FILTER (WHERE NOT (p).is_training) AS n_predicted
FROM (
    SELECT unnest(ridge_fit_predict_agg(y, [x1, x2], CASE WHEN id % 4 = 0 THEN 'test' ELSE 'train' END,
                                        {'alpha': 0.5})) AS p
    FROM reg_data
);

-- Per group, with other methods
SELECT category, unnest(poisson_fit_predict_agg(y_count, [x1, x2])) AS p
FROM reg_data
GROUP BY category
LIMIT 3;

SELECT unnest(bls_fit_predict_agg(y, [x1, x3], {'lower_bound': 0.0})) AS p FROM reg_data LIMIT 2;
SELECT unnest(alm_fit_predict_agg(y, [x1, x2], {'distribution': 'laplace'})) AS p FROM reg_data LIMIT 2;
SELECT unnest(isotonic_fit_predict_agg(y, x1)) AS p FROM reg_data LIMIT 2;
SELECT unnest(pls_fit_predict_agg(y, [x1, x2, x3], {'n_components': 2})) AS p FROM reg_data LIMIT 2;
```

---

## Fit-Predict Table Macros

Table macros wrap the fit-predict aggregates for per-group fitting. All source
columns are passed through and prediction columns are appended.

| Macro | Signature | Appended columns |
|-------|-----------|------------------|
| `ols_fit_predict_by`, `ridge_fit_predict_by`, `elasticnet_fit_predict_by`, `rls_fit_predict_by`, `huber_fit_predict_by`, `ransac_fit_predict_by`, `theil_sen_fit_predict_by`, `bls_fit_predict_by`, `alm_fit_predict_by`, `poisson_fit_predict_by` | `(source, group_col, y_col, x_cols [, options] [, split] [, order_by])` | `yhat, yhat_lower, yhat_upper, is_training` |
| `wls_fit_predict_by` | `(source, group_col, y_col, x_cols, weight_col [, options] [, split] [, order_by])` | `yhat, yhat_lower, yhat_upper, is_training` |
| `pls_fit_predict_by`, `quantile_fit_predict_by` | `(source, group_col, y_col, x_cols [, options] [, split] [, order_by])` | `yhat, is_training` |
| `isotonic_fit_predict_by` | `(source, group_col, y_col, x_col [, options] [, split] [, order_by])` | `yhat, is_training` |
| `binomial_fit_predict_by`, `logistic_fit_predict_by`, `negbinom_fit_predict_by`, `gamma_fit_predict_by`, `tweedie_fit_predict_by` | `(source, group_col, y_col, x_cols [, options] [, split])` | `yhat` (response scale), `yhat_lower`, `yhat_upper` (both NULL), `is_training` |

- `source` is the table name as a string; the other arguments are column references.
- `options` takes the same keys as the underlying aggregate.
- `split` is an optional expression; rows whose value is neither `'train'` nor
  NULL are predicted but not used for training. Pass it by name: `split := ...`.
- `order_by` (by name: `order_by := col`) orders the rows of each group, so the
  alignment of predictions to rows is deterministic; it also sets the row order
  for order-dependent fits such as RLS. The GLM macros (binomial, logistic,
  negbinom, gamma, tweedie) fit each group's model and apply
  [`predict`](#predict) to every row, so they do not take `order_by`.
- Rows with a `NULL` `y` are predicted but not used for training.
- Prediction intervals are leverage-aware (see
  [Methodology](METHODOLOGY.md#prediction-intervals)) and NULL where no
  interval exists.

Other table macros on this page: [`glmm_fit_by`](#glmm_fit_by),
[`eb_shrink_by`](#eb_shrink_by), [`ols_anova_by`](#ols_anova_agg--ols_anova_by),
[`ols_anova_table_by`](#ols_anova_table_agg--ols_anova_table_by), [`aid_by`](#aid_by),
[`glmm_tidy_by`, `glmm_glance_by`, `glmm_augment_by`](#glmm_tidy_by-glmm_glance_by-glmm_augment_by),
[`aid_anomaly_by`](#aid_anomaly_by). See also [Table Macros](api/macros/table_macros.md).

```sql
-- Per-category OLS with 99% prediction intervals
SELECT category, id, y, yhat, yhat_lower, yhat_upper
FROM ols_fit_predict_by('reg_data', category, y, [x1, x2], {'confidence_level': 0.99})
LIMIT 3;

-- Train/test split
SELECT category, count(*) AS n_test
FROM ols_fit_predict_by('reg_data', category, y, [x1, x2],
                        split := CASE WHEN id > 50 THEN 'test' ELSE 'train' END)
WHERE NOT is_training
GROUP BY category
ORDER BY category;

-- Weighted, regularized and robust variants
SELECT count(*) AS n FROM wls_fit_predict_by('reg_data', category, y, [x1, x2], weight);
SELECT count(*) AS n FROM ridge_fit_predict_by('reg_data', category, y, [x1, x2], {'alpha': 0.5});
SELECT count(*) AS n FROM huber_fit_predict_by('reg_data', category, y, [x1, x2]);
SELECT count(*) AS n FROM poisson_fit_predict_by('reg_data', category, y_count, [x1, x2]);
SELECT count(*) AS n FROM isotonic_fit_predict_by('reg_data', category, y, x1);

-- Order-dependent RLS, rows fed in id order within each category
SELECT count(*) AS n FROM rls_fit_predict_by('reg_data', category, y, [x1, x2], order_by := id);

-- GLM macros: predictions on the response scale
SELECT category, id, y_binary, round(yhat, 3) AS p_hat
FROM logistic_fit_predict_by('reg_data', category, y_binary, [x1, x2])
ORDER BY category, id
LIMIT 3;
SELECT count(*) AS n FROM gamma_fit_predict_by('reg_data', category, y_positive, [x1, x2]);
```

---

## Predict and Diagnostics

### predict

Two forms; see [Model tools](api/regression/model_tools.md) for details.

```sql skip
predict(model STRUCT, x LIST(DOUBLE) [, {'type': 'response' | 'link'}]) -> DOUBLE
predict(x LIST(LIST(DOUBLE)), coefficients LIST(DOUBLE), intercept DOUBLE) -> LIST(DOUBLE)
linear_predict(x LIST(LIST(DOUBLE)), coefficients LIST(DOUBLE), intercept DOUBLE) -> LIST(DOUBLE)
```

**Model-aware form.** `model` is the struct returned by any `*_fit_agg` or
`*_fit` function and `x` holds one row's features. Linear models give
`intercept + coefficients · x`. GLMs map the linear predictor through their
`link` to the response scale (default) or return it unchanged with
`{'type': 'link'}`. Isotonic models interpolate between knots and clamp at the
ends. A NULL model or NULL feature gives NULL; a feature count that does not
match the coefficients raises an error.

**Column-layout form.** `x` is a list of feature columns, like the scalar fit
functions; returns one prediction per position. Also available as
`linear_predict`.

```sql
-- Score every row with its category's model
WITH models AS (
    SELECT category, ols_fit_agg(y, [x1, x2]) AS m FROM reg_data GROUP BY category
)
SELECT r.id, r.category, round(predict(m.m, [r.x1, r.x2]), 3) AS yhat
FROM reg_data r JOIN models m USING (category)
ORDER BY r.id
LIMIT 4;

-- Column layout
WITH model AS (
    SELECT ols_fit([3.0, 5.0, 7.0, 9.0, 11.0], [[1.0, 2.0, 3.0, 4.0, 5.0]]) AS fit
)
SELECT predict([[6.0, 7.0, 8.0]], fit.coefficients, fit.intercept) AS predictions,
       linear_predict([[6.0, 7.0, 8.0]], fit.coefficients, fit.intercept) AS same
FROM model;
```

### tidy

```sql skip
tidy(model STRUCT [, names LIST(VARCHAR)])
    -> LIST(STRUCT(term, estimate, std_error, statistic, p_value, conf_low, conf_high,
                   conf_level, index_name, index_value))
```

One entry per term, intercept first (`'(Intercept)'`), then the slopes named
`x1 .. xk` or by `names`. The inference columns come from the model's
inference fields (the intercept's from its `intercept_*` fields) and are NULL
when the model has none (e.g. no `compute_inference`). `index_name` and
`index_value` identify a point on a coefficient path or process (lambda, tau,
...) and are NULL for a single fit.

```sql
SELECT category,
       unnest(tidy(ols_fit_agg(y, [x1, x2], {'compute_inference': true}), ['x1', 'x2']),
              recursive := true)
FROM reg_data
GROUP BY category
ORDER BY category;
```

### glance

```sql skip
glance(model STRUCT) -> STRUCT(model_type, family, link, n_observations, n_features,
    r_squared, adj_r_squared, residual_std_error, f_statistic, f_pvalue, log_likelihood,
    aic, bic, deviance, null_deviance, pseudo_r_squared, dispersion, iterations, converged,
    press, press_rmse, loo_r_squared)
```

The same fields for every model; those a model does not have are NULL.
Model-specific values (Huber's `scale`, a GLM's `theta`, ...) stay on the model
struct. Expand with `unnest(glance(...))`.

```sql
SELECT category, unnest(glance(poisson_fit_agg(y_count, [x1, x2])))
FROM reg_data
GROUP BY category
ORDER BY category;
```

### Model output table macros

```sql skip
<model>_tidy_by(source, group_col, y_col, x_cols [, weight_col]
                [, options := ..., names := [...]])
<model>_glance_by(source, group_col, y_col, x_cols [, weight_col] [, options := ...])
rls_tidy_by(source, group_col, y_col, x_cols [, options := ..., names := [...], order_by := ...])
rls_glance_by(source, group_col, y_col, x_cols [, options := ..., order_by := ...])
```

RLS is order dependent: pass `order_by := <column>` to fit each group's rows
in that order (without it the order, and so the fit, is undefined).

For `ols`, `wls` (with `weight_col`), `ridge`, `elasticnet`, `huber`, `ransac`,
`theil_sen`, `rls`, `lars`, `bayesian_ridge`, `ard`, `bls`, `nnls`, `pls`, `quantile`, `poisson`,
`binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`, `alm` and `aft` (with
`event_col`). They fit one model per group with
`<model>_fit_agg` and return long tables in the anofox integration contract's
`terms` and `summary` schemas:

- `_tidy_by`: the group column, `id` (the group as text), `model_id` and `model_name` (the method, e.g. `'ols'`), then `term, estimate, std_error,
  statistic, p_value, conf_low, conf_high, conf_level, index_name,
  index_value`, one row per term. Inference is computed by default for models
  that have it (`compute_inference` is merged under the user's `options`, so
  passing other options keeps it; `{'compute_inference': false}` turns it off).
  For `bayesian_ridge` and `ard` `std_error` is the posterior SD and
  `conf_low` / `conf_high` the central Bayesian credible interval at
  `confidence_level` (not a confidence interval; `statistic` and `p_value` are
  NULL).
- `_glance_by`: the group column, `id` (the group as text), `model_id` and `model_name` (the method, e.g. `'ols'`), `model_type, metric, value`, one
  row per available numeric metric, so several models compare with a `UNION`.

```sql
SELECT * FROM ols_tidy_by('reg_data', category, y, [x1, x2], names := ['x1', 'x2']);

SELECT * FROM ols_glance_by('reg_data', category, y, [x1, x2])
UNION ALL
SELECT * FROM huber_glance_by('reg_data', category, y, [x1, x2])
ORDER BY category, metric, model_type;
```

### Per-observation diagnostics: ols_augment_by / wls_augment_by

```sql skip
ols_augment_by(source, group_col, y_col, x_cols [, options := ..., split := ..., order_by := ...])
wls_augment_by(source, group_col, y_col, x_cols, weight_col [, options := ..., split := ..., order_by := ...])
ols_augment_agg(y, x [, options]) OVER (...)           -> LIST(STRUCT(...)), one entry per frame row
wls_augment_agg(y, x, weight [, options]) OVER (...)
poisson_augment_by(source, group_col, y_col, x_cols [, options := ..., split := ..., order_by := ...])
-- likewise binomial_, logistic_, negbinom_, gamma_, tweedie_augment_by / _augment_agg
```

broom `augment` for OLS and WLS: one model per group, every source row returned
with the integration contract's `obs` columns appended (tagged
`anofox.output = 'obs'`):

| Column | Type | Meaning |
|--------|------|---------|
| `model_id`, `model_name` | VARCHAR | the method (`'ols'`, `'poisson'`, ...); the group column is among the source columns |
| `row_id` | BIGINT | source row number |
| `fitted` | DOUBLE | fitted value; the prediction for non-training rows |
| `residual` | DOUBLE | response residual `y - fitted` (GLMs: deviance residual) |
| `resid_type` | VARCHAR | `'response'` (GLMs: `'deviance'`) |
| `std_residual` | DOUBLE | standardised residual (R `rstandard`) |
| `stud_residual` | DOUBLE | externally studentised residual (R `rstudent`) |
| `leverage` | DOUBLE | hat value (R `hatvalues`) |
| `cooks_d` | DOUBLE | Cook's distance (R `cooks.distance`) |
| `dffits` | DOUBLE | DFFITS (R `dffits`) |
| `weight` | DOUBLE | prior weight (`weight_col` for WLS, 1 for OLS) |
| `is_training` | BOOLEAN | the row was used to fit |
| `n_params` | INTEGER | estimated coefficients, intercept included |
| `loo_residual` | DOUBLE | leave-one-out (PRESS) residual `residual / (1 - leverage)`, `y` minus the prediction of the fit without the row (R `rstandard(fit, type = "predictive")` for OLS); OLS / WLS / Ridge only, NULL where `leverage = 1` |
| `loo_fitted` | DOUBLE | `y - loo_residual`, the leave-one-out prediction; OLS / WLS / Ridge only |
| `approx_loo_fitted` | DOUBLE | GLMs only: one-step (Pregibon) approximation of the leave-one-out fitted mean, `linkinv(eta - h z / (1 - h))` with `z` the working residual (R `influence(fit)$coefficients` applied to the row); an approximation, not the refit |
| `approx_loo_residual` | DOUBLE | GLMs only: `y - approx_loo_fitted` |

Training rows have `y_col` NOT NULL, every feature present, `split` omitted or
`'train'` / `'training'` in any case (a NULL split value is not a training row) and, for WLS, a positive weight. Other rows get the prediction as
`fitted` and NULL diagnostics. Options: `fit_intercept`, `null_policy`, `solver`.
The measures come from `anofox_regression::diagnostics::augment` and match R
`lm()` (with `weights =` for WLS). A group with fewer than two training rows
gives NULL columns.

**Leave-one-out (#220).** `ols_glance_by` / `wls_glance_by` (and the
`ols_fit_agg` / `wls_fit_agg` / `ols_fit` / `wls_fit` STRUCT with the opt-in
option `{'compute_loo': true}`, one more QR of the design; so `glance()`)
add `press = sum(weight * loo_residual^2)`, `press_rmse = sqrt(press / n)` and
the predicted R² `loo_r_squared = 1 - press / SST` (SST as for `r_squared`).
For WLS `loo_residual` is on the response scale and `press` is weighted by the
prior weights, which equals R's `sum(rstandard(fit, type = "predictive")^2)`.
A row with leverage 1 (e.g. the only observation of a dummy level) has NULL
leave-one-out columns and makes the group's `press` metrics NULL. See
[model_tools](api/regression/model_tools.md#leave-one-out-diagnostics).

**Ridge (#235).** `ridge_augment_by(source, group_col, y_col, x_cols [, options,
split, order_by])` returns the same `obs` columns with `leverage` the diagonal of
the ridge hat matrix `X(XᵀX + λI)⁻¹Xᵀ` (intercept unpenalised) and the exact
leave-one-out `loo_residual` / `loo_fitted` at the same λ; `std_residual`,
`stud_residual`, `cooks_d` and `dffits` are NULL. Options: `alpha` / `lambda`,
`lambda_scaling`, `fit_intercept`, `null_policy`, `solver`. `ridge_glance_by`
reports `press`, `press_rmse`, `loo_r_squared`, and `ridge_fit_agg` / `ridge_fit`
take `compute_loo`. With `'lambda_scaling': 'glmnet'` the penalty `alpha · n` of
the full fit is held fixed. See [model_tools](api/regression/model_tools.md#ridge).

**GLMs.** `poisson_augment_by`, `binomial_augment_by`, `logistic_augment_by`,
`negbinom_augment_by`, `gamma_augment_by` and `tweedie_augment_by` (same
arguments as `ols_augment_by`; window aggregates `<family>_augment_agg(y, x [, options])`)
return the same columns for the matching GLM:

- `fitted` is the mean on the response scale (the prediction for non-training rows);
- `residual` is the **deviance** residual and `resid_type` is `'deviance'` (as R / broom);
- `std_residual` is the deviance residual standardised by `sqrt(phi * (1 - h))`
  (R `rstandard(fit, type = "deviance")`), `stud_residual` is R `rstudent`;
- `leverage` is the hat value of the IRLS-weighted design, `weight` is 1;
- the dispersion `phi` is 1 for Poisson, binomial, logistic and Negative Binomial
  (as `MASS::glm.nb`), and the Pearson estimate for Gamma and Tweedie;
- instead of the exact `loo_residual` / `loo_fitted` they have `approx_loo_fitted`
  and `approx_loo_residual`, the one-step approximate leave-one-out prediction
  (#235; see [model_tools](api/regression/model_tools.md#approximate-leave-one-out-for-the-glms)).

Options are those of the matching `<family>_fit_agg` that change the fit:
`fit_intercept`, `max_iterations`, `tolerance`, `offset` (1-based index of the
offset column in `x_cols`), plus `link` (Poisson, binomial), `theta` (Negative
Binomial) and `power` (Tweedie). Penalized fits (`glm_lambda > 0` or a `prior`)
are rejected with an error: their leverage and Cook's distance have no R
counterpart. The values come from `anofox_regression::diagnostics::augment_glm`
and match R `glm()`, `MASS::glm.nb()` and `statmod::tweedie`.

```sql
-- Poisson: the rows with the largest Cook's distance per category
SELECT category, row_id, y_count, fitted, residual, leverage, cooks_d
FROM poisson_augment_by('reg_data', category, y_count, [x1, x2])
QUALIFY row_number() OVER (PARTITION BY category ORDER BY cooks_d DESC) <= 2
ORDER BY category, cooks_d DESC;
```

```sql
SELECT category, row_id, fitted, residual, leverage, cooks_d
FROM ols_augment_by('reg_data', category, y, [x1, x2])
ORDER BY cooks_d DESC
LIMIT 5;
```

### vif / vif_agg

Variance inflation factor per feature. Rule of thumb: above 5 is a warning,
above 10 is severe multicollinearity.

```sql skip
vif(x LIST(LIST(DOUBLE))) -> LIST(DOUBLE)    -- x is a list of feature columns
vif_agg(x LIST(DOUBLE)) -> LIST(DOUBLE)      -- one row per observation
```

```sql
SELECT vif([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 6.0]]) AS vif_values;
SELECT vif_agg([x1, x2, x3]) AS vif_values FROM reg_data;
```

### aic / bic

Information criteria from the residual sum of squares.

```sql skip
aic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
bic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

`k` counts all parameters, including the intercept.

```sql
SELECT aic(100.0, 50, 3) AS aic_value, bic(100.0, 50, 3) AS bic_value;
```

### jarque_bera / jarque_bera_agg

Jarque-Bera normality test on an array or a column.

```sql skip
jarque_bera(values LIST(DOUBLE)) -> STRUCT(statistic DOUBLE, p_value DOUBLE, skewness DOUBLE, kurtosis DOUBLE, n BIGINT)
jarque_bera_agg(value DOUBLE) -> STRUCT(...)   -- same fields
```

```sql
SELECT (jarque_bera([1.0, 2.0, 3.5, 2.2, 4.1, 0.3, 2.9])).p_value AS p_value;
SELECT unnest(jarque_bera_agg(y)) FROM reg_data;
```

### residuals_diagnostics / residuals_diagnostics_agg

Raw, standardized and studentized residuals and leverage.

```sql skip
residuals_diagnostics(y LIST(DOUBLE), y_hat LIST(DOUBLE)) -> STRUCT
residuals_diagnostics(y LIST(DOUBLE), y_hat LIST(DOUBLE), x LIST(LIST(DOUBLE)),
                      residual_std_error DOUBLE, include_studentized BOOLEAN) -> STRUCT
residuals_diagnostics_agg(y DOUBLE, y_hat DOUBLE [, x LIST(DOUBLE)]) -> STRUCT
```

**Returns:** `STRUCT(raw DOUBLE[], standardized DOUBLE[], studentized DOUBLE[], leverage DOUBLE[])`.
Fields that need `x` (leverage, studentized residuals) are `NULL` without it.

```sql
SELECT residuals_diagnostics([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8]) AS diagnostics;

WITH fit AS (SELECT ols_fit_agg(y, [x1, x2]) AS f FROM reg_data)
SELECT (residuals_diagnostics_agg(y, f.intercept + f.coefficients[1] * x1 + f.coefficients[2] * x2, [x1, x2])).leverage[1:3] AS leverage
FROM reg_data, fit;
```

### anofox_contract_version / function tags

The anofox extensions (statistics, forecast, visualization, ...) share an
integration contract: a set of long output schemas (`terms`, `obs`,
`prediction`, `curve`, `summary`, `test`) that plotting and other tools consume
directly. `anofox_contract_version()` returns the contract version this
extension implements (`'1'`); the first anofox extension loaded registers it.

Every function and table macro of this extension carries the tag
`anofox.family = 'statistics'` in `duckdb_functions().tags`. Functions whose
output follows a contract schema additionally carry `anofox.contract` (the
version) and `anofox.output` (the schema name).

```sql
SELECT anofox_contract_version() AS contract_version;

SELECT function_type, count(DISTINCT function_name) AS functions
FROM duckdb_functions()
WHERE tags['anofox.family'] = 'statistics'
GROUP BY function_type
ORDER BY function_type;

-- Contract producers across all loaded anofox extensions
SELECT DISTINCT function_name, tags['anofox.output'] AS output
FROM duckdb_functions()
WHERE tags['anofox.contract'] IS NOT NULL
ORDER BY function_name;
```

---

## Common Options

### null_policy

Used by the window and fit-predict functions.

| Value | Training rows | Predictions |
|-------|---------------|-------------|
| `'drop'` (default) | Rows where `y` is not `NULL` | Every row |
| `'drop_y_zero_x'` | Rows where `y` is not `NULL` and no feature is 0 | Every row |
| `'zero'` | Every row; a `NULL` `y` or `NULL` feature is fitted as 0 | Every row; a `NULL` feature counts as 0 |
| `'ignore'` | Same as `'drop'` (alias for input with no `NULL`s) | Every row |

Under `'zero'`, a `NULL` `y` no longer marks a row to predict, so every row is a
training row; use a `split_col` to keep future rows out of the fit. A `NULL` `x`
list and `NaN` values are not zero-filled. See
[NULL_SEMANTICS.md](NULL_SEMANTICS.md#the-null_policy-option).

### solver

Matrix decomposition for OLS, WLS and Ridge.

| Value | Description |
|-------|-------------|
| `'svd'` (default) | Most robust; handles rank-deficient designs |
| `'qr'` | Faster for well-conditioned problems |
| `'cholesky'` | Fastest when `X'X` is positive definite |

### hc_type

Heteroscedasticity-consistent standard errors for OLS and WLS; needs
`compute_inference: true`. See [METHODOLOGY.md](METHODOLOGY.md).

| Value | Description |
|-------|-------------|
| `'none'` (default) | Classical standard errors |
| `'hc0'` | White's estimator |
| `'hc1'` | HC0 with a degrees-of-freedom correction |
| `'hc2'` | HC0 with a leverage adjustment |
| `'hc3'` | HC0 with a squared-leverage adjustment (most conservative) |

```sql
SELECT (ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': 'hc3'})).std_errors AS robust_se
FROM reg_data;
```

### lambda_scaling

Penalty convention for Ridge and Elastic Net.

| Value | Description |
|-------|-------------|
| `'raw'` (default) | The penalty is used as given |
| `'glmnet'` | The penalty is scaled to match R's glmnet convention |

### glm_lambda

L2 regularization for the GLM aggregates. 0.0 (default) means none.

### Priors

The GLM and AFT aggregates accept `feature_names`, `prior` and `vcov` for
explicit coefficient priors and Laplace intervals. See [Priors](api/glm/priors.md).

---

## Return Types

### FitResult Structure

Returned by the linear and robust fits (`ols`, `ridge`, `elasticnet`, `wls`,
`rls`, `lars`, `huber`, `ransac`, `theil_sen`; scalar and aggregate forms).
The shape does not depend on the options: fields that were not computed are
NULL.

```text
STRUCT(
    coefficients DOUBLE[],      -- one per feature, intercept excluded
    intercept DOUBLE,           -- NaN when fit_intercept = false
    r_squared DOUBLE,
    adj_r_squared DOUBLE,
    residual_std_error DOUBLE,
    n_observations BIGINT,
    n_features BIGINT,
    -- huber adds: scale DOUBLE, n_outliers BIGINT
    -- ransac adds: residual_threshold DOUBLE, n_inliers BIGINT, n_trials BIGINT
    -- coefficient inference, NULL without compute_inference = true
    -- (and always NULL for elasticnet, rls, lars and ridge with alpha > 0):
    std_errors DOUBLE[],
    t_values DOUBLE[],
    p_values DOUBLE[],
    ci_lower DOUBLE[],          -- deprecated alias of conf_low
    ci_upper DOUBLE[],          -- deprecated alias of conf_high
    f_statistic DOUBLE,         -- not for elasticnet, rls, lars
    f_pvalue DOUBLE,
    -- the same for every model:
    conf_low DOUBLE[],          -- confidence intervals of the coefficients
    conf_high DOUBLE[],
    conf_level DOUBLE,
    intercept_std_error DOUBLE, -- the intercept's inference
    intercept_statistic DOUBLE,
    intercept_p_value DOUBLE,
    intercept_conf_low DOUBLE,
    intercept_conf_high DOUBLE,
    log_likelihood DOUBLE,      -- NULL for estimators without a likelihood
    aic DOUBLE,                 --   (huber, ransac, theil_sen)
    bic DOUBLE,
    model_type VARCHAR,         -- 'ols', 'wls', 'ridge', 'elastic_net', 'lars', 'rls',
                                -- 'huber', 'ransac', 'theil_sen'
    family VARCHAR,             -- 'gaussian', or NULL without a likelihood
    link VARCHAR                -- 'identity'
)
```

Log-likelihood, AIC and BIC follow R (`logLik`, `AIC`, `BIC` of `lm`, which
count the residual variance as a parameter).

### GlmFitResult Structure

Returned by `poisson_fit_agg`, `binomial_fit_agg`, `negbinom_fit_agg`,
`gamma_fit_agg`, `tweedie_fit_agg` and `logistic_fit_agg`. The shape does not
depend on the options: fields that were not computed are NULL.

```text
STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    deviance DOUBLE,
    null_deviance DOUBLE,
    pseudo_r_squared DOUBLE,
    aic DOUBLE,
    dispersion DOUBLE,          -- logistic_fit_agg: accuracy DOUBLE, threshold DOUBLE instead
    n_observations BIGINT,
    n_features BIGINT,
    iterations INTEGER,
    converged BOOLEAN,          -- whether IRLS reached the tolerance
    -- coefficient inference, NULL without compute_inference = true:
    std_errors DOUBLE[],
    z_values DOUBLE[],
    p_values DOUBLE[],
    ci_lower DOUBLE[],          -- deprecated alias of conf_low
    ci_upper DOUBLE[],          -- deprecated alias of conf_high
    family VARCHAR,             -- 'poisson', 'binomial', 'negative_binomial', 'gamma', 'tweedie'
    link VARCHAR,               -- e.g. 'log', 'logit', 'probit', 'cloglog', 'sqrt', 'identity'
    -- the same for every model (see FitResult):
    conf_low DOUBLE[], conf_high DOUBLE[], conf_level DOUBLE,
    intercept_std_error DOUBLE, intercept_statistic DOUBLE, intercept_p_value DOUBLE,
    intercept_conf_low DOUBLE, intercept_conf_high DOUBLE,
    log_likelihood DOUBLE, bic DOUBLE,
    model_type VARCHAR          -- 'glm'
)
```

`logistic_fit_agg` reports `family = 'binomial'` and `link = 'logit'`. A Tweedie
power of 1 or 2 reports `'poisson'` / `'gamma'`. Log-likelihood, AIC and BIC
follow R's `logLik`, `AIC`, `BIC` of `glm` (`MASS::glm.nb` for the negative
binomial); the intervals are Wald intervals, as `confint.default`.

When an `offset` column is given it is removed from the design, so
`coefficients` and `n_features` count one fewer than the input feature list.

The AFT, ALM, GLMM, BLS/NNLS, PLS, quantile and isotonic structs end with the
same shared fields. `model_type` is `'aft'` (family = the distribution, link
`'log'`), `'alm'` (family = the distribution), `'glmm'`, `'bls'` / `'nnls'`,
`'pls'`, `'quantile'` or `'isotonic'`. Quantile and isotonic regression have
no likelihood: `family`, `log_likelihood`, `aic` and `bic` are NULL. The GLMM's
AIC and BIC count parameters as lme4's `logLik` (fixed effects, covariance
parameters, and the dispersion, including a fixed negative-binomial θ).

### AlmFitResult Structure

Returned by `alm_fit_agg`.

```text
STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    log_likelihood DOUBLE,
    aic DOUBLE,
    bic DOUBLE,
    scale DOUBLE,
    n_observations BIGINT,
    n_features BIGINT,
    iterations INTEGER,
    -- with compute_inference = true:
    std_errors DOUBLE[],
    t_values DOUBLE[],
    p_values DOUBLE[],
    ci_lower DOUBLE[],
    ci_upper DOUBLE[]
)
```

### BlsFitResult Structure

Returned by `bls_fit_agg` and `nnls_fit_agg`.

```text
STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    ssr DOUBLE,                    -- residual sum of squares
    r_squared DOUBLE,
    n_observations BIGINT,
    n_features BIGINT,
    n_active_constraints BIGINT,   -- coefficients sitting on a bound
    at_lower_bound BOOLEAN[],
    at_upper_bound BOOLEAN[]
)
```

### Accessing Results

```sql
-- Individual fields
SELECT (fit).r_squared, (fit).coefficients[1] AS beta1, (fit).coefficients[2] AS beta2
FROM (SELECT ols_fit_agg(y, [x1, x2]) AS fit FROM reg_data);

-- All fields as columns
SELECT unnest(ols_fit_agg(y, [x1, x2])) FROM reg_data;
```

---

## Error and NULL Handling

- **Invalid options** (unsupported keys, bad values such as an unknown `solver`)
  raise an error when the query is bound.
- **Degenerate data** (too few rows for the number of parameters, a singular
  design, a model that cannot be identified) returns `NULL` instead of failing
  the query.
- **NULL inputs:** rows with a `NULL` response or feature are skipped by the fit
  aggregates. The window and fit-predict functions use `NULL` responses to mark
  rows to predict; see [`null_policy`](#null_policy) and
  [NULL_SEMANTICS.md](NULL_SEMANTICS.md).

```sql
-- One row is not enough to fit an intercept and a slope: the result is NULL
SELECT ols_fit_agg(y, [x]) IS NULL AS is_null FROM (VALUES (1.0, 2.0)) AS t(y, x);
```

---

## Detailed Documentation

- **Regression:** [OLS](api/regression/ols.md) | [Ridge](api/regression/ridge.md) | [Elastic Net](api/regression/elasticnet.md) | [WLS](api/regression/wls.md) | [RLS](api/regression/rls.md) | [Huber](api/regression/huber.md) | [RANSAC](api/regression/ransac.md) | [Theil-Sen](api/regression/theil_sen.md) | [LARS](api/regression/lars.md) | [Bayesian Ridge / ARD](api/regression/bayesian.md) | [Passive-Aggressive / lmDynamic / LOWESS](api/regression/online_dynamic.md) | [BLS/NNLS](api/regression/bls.md) | [PLS](api/regression/pls.md) | [Isotonic](api/regression/isotonic.md) | [Quantile](api/regression/quantile.md)
- **Prediction:** [Window functions](api/regression/fit_predict_window.md) | [Fit-predict aggregates](api/regression/fit_predict_agg.md) | [Table macros](api/macros/table_macros.md) | [Model tools (predict, tidy, glance)](api/regression/model_tools.md)
- **GLM:** [Poisson](api/glm/poisson.md) | [Binomial](api/glm/binomial.md) | [Logistic](api/glm/logistic.md) | [Negative Binomial](api/glm/negbinom.md) | [Gamma](api/glm/gamma.md) | [Tweedie](api/glm/tweedie.md) | [ALM](api/glm/alm.md) | [Priors](api/glm/priors.md) | [GLMM](api/glm/glmm.md) | [EB shrinkage](api/glm/eb_shrink.md)
- **Survival:** [AFT](api/survival/aft.md)
- **Statistics:** [Hypothesis tests](api/statistics/hypothesis.md) | [Correlation](api/statistics/correlation.md) | [Categorical](api/statistics/categorical.md) | [Mediation](api/statistics/mediation.md)
- **Demand:** [AID](api/aid/aid.md)
- **Diagnostics:** [Model diagnostics](api/diagnostics/diagnostics.md)
- **Background:** [API conventions](API_CONVENTIONS.md) | [Methodology](METHODOLOGY.md) | [NULL semantics](NULL_SEMANTICS.md) | [Migration from v0.9](MIGRATION.md)

For the release history see [CHANGELOG.md](../CHANGELOG.md).
