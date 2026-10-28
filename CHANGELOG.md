# Changelog

All notable changes to the Anofox Statistics DuckDB extension are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is below 1.0, minor releases may contain breaking changes.
Breaking changes are called out explicitly.

## [Unreleased]

### Fixed

- **`aid_by` takes `order_by := <time column>` (#271).** AID reads each series in order:
  leading zeros mark a new product, trailing zeros an obsolete one. `aid_by` called
  `aid_agg` without an `ORDER BY`, so the new / obsolete / stockout flags depended on the
  scan order and could change with the thread count. Pass `order_by :=` the time column.
  Without it the behaviour is unchanged (row arrival order).
- **Determinism test for the order-dependent aggregates (#271).** With `ORDER BY` inside
  the aggregate, `rls_fit_agg`, `passive_aggressive_fit_agg`, `lm_dynamic_fit_agg`,
  `aid_agg`, `acf_agg` / `pacf_agg`, the time-series tests, `diebold_mariano_agg`,
  `clark_west_agg`, `rls_fit_predict_agg`, `ols_augment_agg`, `residuals_diagnostics_agg`
  and `p_adjust_agg` are bitwise identical for 1 and 8 threads and shuffled input
  (`test/sql/issues/issue_271.test`). METHODOLOGY documents this and the cost of the
  ordering.

### Added

- **Options from prepared-statement parameters (#274).** Options and the other constant
  arguments (time-series `lag`, `p_adjust` method / `n`, classifier-curve and survival
  constants, `predict` options) may now be `?` / `$1` / `$name` parameters. Bind throws
  `ParameterNotResolvedException`, so DuckDB re-binds the statement with the values at
  `EXECUTE`. Before, `PREPARE ... {'threshold': ?}` failed with "options must be a constant".
  Options from SQL variables (`getvariable('x')`, `$x` on DuckDB 2.0) already worked and are
  now documented (API_CONVENTIONS §2, batch skill).
- **Benchmarks: DuckDB v1.5.6 vs v2.0 (#269).** New workloads: `_by` macros on Parquet,
  tests per VARCHAR segment, and a drift monitor over Hive-partitioned Parquet. A new
  `scripts/bench_compare.py` reports median timings for several DuckDB CLIs, and the results
  are in `docs/PERFORMANCE.md`. Main finding: `*_fit_predict_by` runs 2× faster on 2.0 (DuckDB's
  window operator), while the aggregate-bound workloads are within ±15%.

### Fixed

- **Empty `{}` and NULL options mean "no options" everywhere (#276).** On DuckDB 2.0,
  `options := {}` parses, and every aggregate, scalar, window function and table macro gives
  the same result for `{}` as for no options (checked for all of them). The list-input scalar
  fits (`ols_fit`, `wls_fit`, `ridge_fit`, `rls_fit`, `ransac_fit`, `theil_sen_fit`,
  `huber_fit`, `elasticnet_fit`) used to return NULL when the options argument was NULL. They
  now treat it as no options, like the aggregates and macros already did. A NULL `y`, `X` or
  weights row still gives a NULL result.

## [0.14.0] - 2026-10-28

### Added

- **Drift tests on anofox-statistics 0.6.0** (upstream sipemu/anofox-statistics-rs#42-#45):
  - `psi_agg` / `kl_divergence_agg`: new option `epsilon`, a constant zero fill in place
    of evidently's rule (KL renormalizes afterwards).
  - `wasserstein_agg`: `distance_normed` and the permutation p-value now work for any
    order `p` (W_p / population SD of the reference). Previously they were p = 1 only.
  - `jensen_shannon_agg`: `divergence` now comes from upstream (`philentropy::JSD`).
    It was previously computed in the extension as `distance²`.
  - `hellinger_agg`: `binning = 'auto'` now bins per value when the reference has at most
    20 distinct values, as the other divergences do. Integer category codes no longer need
    `binning = 'categorical'`. This is a behaviour change for numeric data with ≤ 20
    distinct values.
- **Data-drift tests (#245-#255): evidently's drift statistics as aggregates.** They come
  from upstream anofox-statistics 0.5.0 and are validated against R
  (`validation/generators/make_reference_drift.R`, `test/sql/reference/drift_reference.test`).
  All take `(value, group [, options])`. The group is BIGINT or VARCHAR. The reference
  sample is the smaller label unless the new `reference` option names it.
  - `ks_test_agg` (#245): two-sample Kolmogorov-Smirnov test as R `ks.test`. The p-value
    is exact (also with ties) when n1·n2 < 10000, otherwise R's limiting distribution.
    Options: `alternative`, `exact`.
  - `ad_test_agg` (#246): k-sample Anderson-Darling test as `kSamples::ad.test`, for k ≥ 2.
    Its asymptotic p-value is not capped like scipy's. Option: `midrank`.
  - `cvm_test_agg` (#247): two-sample Cramér-von Mises test. The p-value is exact when
    max(n1, n2) ≤ 20, otherwise asymptotic. Option: `method`.
  - `epps_singleton_agg` (#248): Epps-Singleton test as scipy. Option: `t`.
  - `wasserstein_agg` (#249): 1-D Wasserstein-p distance, plus evidently's
    `distance_normed` (W₁ / population SD of the reference) and an optional seeded
    permutation p-value.
  - `psi_agg` (#250), `kl_divergence_agg` (#251), `jensen_shannon_agg` (#252) and
    `hellinger_agg` (#253): binned divergences for numeric or VARCHAR values. Binning is
    evidently's `auto` rule (Sturges bins when the reference has more than 20 distinct
    values, otherwise one bin per value), `sturges`, `quantile` (reference quantiles,
    `n_bins`) or `categorical`. Each returns a `drift` flag for `value ≥ threshold`
    (default 0.1). KL and JS also take a logarithm `base`.
  - `tvd_agg` (#254): total variation distance between category distributions, with a
    seeded permutation p-value (1000 permutations, seed 42).
- Dependency: anofox-statistics 0.4.11 → 0.6.0.
- **Exact leave-one-out diagnostics for Ridge, approximate leave-one-out for the GLMs (#235).**
  Computed upstream in anofox-regression 0.5.37 (`FittedRidge::loo`,
  `Augment::approx_loo_fitted` / `approx_loo_residual`); the extension maps fields.
  - Ridge: `ridge_glance_by` adds `press`, `press_rmse` and `loo_r_squared` (on by default,
    as for OLS); `ridge_fit_agg` / `ridge_fit` take `compute_loo` (default false: one SVD of
    the design) and fill the same STRUCT fields (so `glance()` has them). The leverage is the
    diagonal of the ridge hat matrix `X(XᵀX + λI)⁻¹Xᵀ` with the intercept unpenalised, and
    `press = Σ (residual / (1 − leverage))²` is exactly the error of the fits without each
    row at the same λ (with `'lambda_scaling': 'glmnet'`, `alpha · n` of the full fit held
    fixed). Over an `alpha` grid this is exact leave-one-out cross-validation.
  - New `ridge_augment_by` / `ridge_augment_agg`: the contract `obs` columns for Ridge, with
    the ridge `leverage` and exact `loo_residual` / `loo_fitted`; `std_residual`,
    `stud_residual`, `cooks_d` and `dffits` are NULL (no ridge definition in R). Options:
    `alpha` / `lambda`, `lambda_scaling`, `fit_intercept`, `null_policy`, `solver`.
  - GLMs: `poisson_augment_by`, `binomial_augment_by`, `logistic_augment_by`,
    `negbinom_augment_by`, `gamma_augment_by` and `tweedie_augment_by` (and their
    `_augment_agg`) add `approx_loo_fitted` / `approx_loo_residual`: the one-step (Pregibon)
    approximation of the leave-one-out prediction on the response scale,
    `linkinv(eta − h z / (1 − h))`, R's `influence(fit)$coefficients` applied to the row.
    Named `approx_*` because it is not the refit; there is no GLM PRESS.
  - Validation: `test/sql/macros/test_ridge_loo.test` checks every ridge LOO value against an
    explicit refit without the row (in SQL, to 1e-9) and R's explicitly built hat matrix, and
    the GLM approximation against R 4.6.1 `influence()`.
- **`p_adjust(p [, method [, n]]) OVER (...)` window function on DuckDB 2.0 (#224).**
  Returns each row's p-value adjusted for multiple testing (R `p.adjust`, the same
  methods and `n` as `p_adjust_agg`) within its window frame, i.e. within the partition
  when `OVER` has no `ORDER BY`, so `p_adjust(p_value, 'BH') OVER (PARTITION BY
  endpoint)` gives the row-aligned result inline. NULL p-values and rows removed by
  `FILTER` stay NULL and are not counted; a row outside its own frame is NULL. It is
  window-only (no plain aggregate form) and uses DuckDB 2.0's batched window callback,
  detected in `duckdb_compat.hpp`; each distinct frame is adjusted once. On DuckDB 1.5
  it is not registered: use `p_adjust_by` or the plain-window recipe there. Matches R
  and `p_adjust_by` (`test/sql/reference/p_adjust_window.test`).
- **`ols_anova_table_agg` / `ols_anova_table_by`: per-term ANOVA tables (#231).** One
  OLS fit, one row per term with its sum of squares and F test, as R's `anova(lm(...))`
  (type I, sequential sums of squares in the column order, the default) or
  `car::Anova(fit, type = 2)` (`{'type': 'II'}`, each term adjusted for all others, the
  drop-one F test). `ols_anova_table_agg(y, x[, options])` returns a LIST of STRUCTs in
  the contract `test` schema: `term`, `method` (`'type I ANOVA'` / `'type II ANOVA'`),
  `statistic`, `df1` (term df), `df2` (residual df), `p_value`, `sum_sq`, `mean_sq`,
  `residual_sum_sq`, `residual_mean_sq`, `n_observations` and the remaining unified test
  fields. `ols_anova_table_by(source, group_col, y_col, x_cols, options := NULL)` returns
  one row per group and term with `id` and `model_id` / `model_name` = `'ols'`. Terms are
  `x1`, `x2`, ... unless the `terms` option labels the columns; equal labels form one term
  (factor dummies). Degrees of freedom count the rank: an aliased term has `df1 = 0` and
  NULL statistic / p-value. Options: `type`, `terms`, `fit_intercept`. The table comes from
  anofox-regression 0.5.36 (`inference::anova_table`, validated against R `anova()` and
  car 3.1.5 `Anova(type = 2)` to 1e-9).

### Fixed

- **Bitwise-reproducible results with more than one thread (#226).** The one-sample
  and paired tests, the correlation aggregates, the regression `*_fit_agg` and
  `*_fit_predict_agg` aggregates and the `*_fit_predict` window functions summed their
  rows in the order DuckDB's threads delivered them, so with `threads > 1` the same
  data could give results that differed in the last bits from run to run. Their
  finalizers now sort the buffered rows into a canonical order (lexicographic on the
  full row, with a total order on doubles) before computing, as #225 did for the
  group tests; `*_fit_predict_agg` fits on the sorted training rows and still returns
  each row's own prediction in input order. `glmm_fit_agg` and `log_rank_test_agg`
  number their groups by sorted label and `icc_agg` its subjects and raters by sorted
  id, instead of first-seen order, so `glmm_fit_agg`'s random effects are now listed
  in label order. `eb_shrink_agg` keeps its per-group list in input order. Results can
  move in the last bits compared with earlier releases. The aggregates that depend on
  the row order by design (RLS, Passive-Aggressive, lmDynamic, AID, the time-series
  tests) and the per-row `*_augment_agg` / `residuals_diagnostics_agg` are unchanged
  and documented (Methodology, "Reproducibility across threads"). Sorting adds
  O(n log n) work: on 2M rows with 8 threads, `ols_fit_agg` (3 features) took about 5%
  longer (20% with 100 groups), `t_test_agg` about 10%, and the cheap O(n) correlation
  aggregates up to about 4× (`pearson_agg` 0.08 s → 0.3 s).

### Changed

- **CI: PR fast lane (#243).** Pull requests now build and test linux_amd64 only, on both
  DuckDB lines, plus the required Windows MSVC suite. The full platform matrix (macOS,
  arm64, musl, WASM and the WASM runtime test) runs on main, on tags, on
  `workflow_dispatch` and on PRs labelled `full-ci`. Adding the label re-runs the pipeline.
- The augment STRUCT of `<model>_augment_agg` has two more fields (`approx_loo_fitted`,
  `approx_loo_residual`; NULL for OLS / WLS / Ridge), and the GLM `_augment_by` macros two
  more `obs` columns.

- **Supported DuckDB versions are now v1.5.6 and v2.0.** CI no longer builds,
  smoke-tests or deploys the v1.4.5 LTS line or the older v1.5.4 / v1.5.5 patch lines,
  and the vendored LTS distribution workflow is removed. Binaries already published
  for those versions on get.erpl.io stay in place; v0.13.0 is the last release built
  for them.

## [0.13.0] - 2026-10-23

### Added

- **Credible intervals for Bayesian Ridge / ARD (#222).** `bayesian_ridge_fit_agg` and
  `ard_fit_agg` now fill the shared inference fields from the posterior: `std_errors`
  is the posterior SD (`posterior_sd`, 0 for features ARD pruned), `conf_low` /
  `conf_high` the central credible interval `coefficient ∓ z · sd` at
  `confidence_level` (now accepted by the fit aggregates, default 0.95) and
  `conf_level` that level. The intercept gets a flat prior: `intercept_std_error` is
  `sqrt(1/(n·α) + x̄ᵀ Σ x̄)` with its interval in `intercept_conf_low` /
  `intercept_conf_high`. There is no test, so `t_values`, `p_values` and the intercept
  statistic / p-value stay NULL. `bayesian_ridge_tidy_by` / `ard_tidy_by` carry the
  values into the `terms` rows; they are Bayesian credible intervals, not confidence
  intervals (documented; the `terms` schema is unchanged). Validated against
  scikit-learn 1.9.1's `sigma_` to 1e-7 (`test/sql/reference/regression_bayesian.test`;
  the intercept SD against the inverted full posterior precision). Previously these
  fields were NULL. Requires anofox-regression 0.5.34.

- **Theil-Sen confidence intervals (#222).** New options `ci_method` (`'exact'` |
  `'bootstrap'` | `'exact_or_none'`) and `n_bootstrap` (alias `n_boot`, default 999;
  the existing `seed` /
  `random_state` seeds it) for `theil_sen_fit` / `theil_sen_fit_agg` with
  `compute_inference`. `'exact'` (the default with one feature) is Sen's (1968)
  distribution-free slope interval with Kendall's tie-corrected variance, plus
  Kendall's continuity-corrected z in `t_values` and its p-value; it matches R's
  `trend::sens.slope` to 1e-12. `'bootstrap'` (the default with several features)
  is the pairs bootstrap: bootstrap standard errors and percentile intervals for
  every coefficient and the intercept, deterministic for a seed whatever the row
  order or thread count; it costs about `n_bootstrap` point fits per group (about
  8 s per group at 500 rows × 3 features). `theil_sen_tidy_by` carries the
  intervals into `terms`; it defaults to `'exact_or_none'` (one feature: exact;
  several features: NULL intervals and no bootstrap, so existing multi-feature
  calls stay as fast as before) and bootstraps only with `options :=
  {'ci_method': 'bootstrap'}` (199 resamples by default there). Previously
  these fields were NaN / NULL (see Fixed). `n_bootstrap`
  is also an alias of `quantile_fit_agg`'s `n_boot`.

- **Multiple-testing correction for any p-values (#219).** R's
  `p.adjust(p, method, n)`, from upstream anofox-statistics 0.4.11 (which adds the
  explicit `n`), in three forms:
  - `p_adjust_by(source, group_col, p_col, method := 'holm', n := NULL)` returns every
    source row, in source order, with `p_adjusted`; each group is one family.
  - `p_adjust_agg(p [, method [, n]])` returns the adjusted p-values of a group as a
    `DOUBLE[]` in input order (`ORDER BY` inside the call).
  - `p_adjust_list(p DOUBLE[] [, method [, n]])` adjusts a list.

  Methods: `holm` (default), `hochberg`, `hommel`, `bonferroni`, `BH` (alias `fdr`),
  `BY`, `none`; case-insensitive. An unknown method is an error at bind time. NULL
  p-values stay NULL and are not counted. `n` (the number of hypotheses) defaults to the
  number of non-NULL p-values; a smaller `n` is an error. All methods, with and without
  `n`, match R 4.6.1 to 1e-14 (`test/sql/reference/p_adjust_reference.test`). A
  `p_adjust(p) OVER (PARTITION BY ...)` window function is not possible on DuckDB 1.x:
  an extension's window aggregate is not told which row of a whole-partition frame it is
  computing (follow-up for DuckDB 2.0: #224).

- **`mediation_by` and `mediation_agg`: causal mediation analysis per group (#223).**
  `mediation_by(source, group_col, treatment, mediator, outcome, covariates := [...],
  options := {...})` fits the mediator model `m ~ x + covariates` and the outcome model
  `y ~ x + m + covariates` by OLS on the same complete rows (rows with a NULL anywhere
  are dropped from both) and returns, per group, the ACME (indirect effect `a·b`), the
  ADE (direct effect `c'`), the total effect and the proportion mediated for the
  contrast `treat_value − control_value`, in the contract `terms` schema (`id` = group,
  `model_id = 'mediation'`) plus `method`, `n_obs` and `n_sims`. Inference by a
  nonparametric case-resampling bootstrap (default: `n_bootstrap` 1000, `seed` 42,
  percentile or `mediate`'s BCa interval, `mediate`'s p-value), the Sobel / delta
  method (`method := 'sobel'`) or `mediate`'s quasi-Bayesian Monte Carlo. Bootstrap
  results are reproducible: the rows are put in a canonical order before resampling,
  so neither the thread count nor the input order changes them. `mediation_agg(y, m, x
  [, covariates [, options]])` is the aggregate building block. The computation is
  upstream (anofox-regression 0.5.31, `solvers::mediation`), validated against R
  `mediation::mediate`, `lavaan` and `bda::mediation.test`. Linear models only for now:
  binary outcomes or mediators, a treatment × mediator interaction, multiple mediators
  and the sensitivity analysis are not supported. See
  [Mediation](docs/api/statistics/mediation.md).

- **`ols_anova_agg` / `ols_anova_by`: nested-model F test (#221).** Tests whether a full
  OLS model fits significantly better than a reduced one, as R's
  `anova(lm(y ~ x_reduced), lm(y ~ x_full))`, with both fits on the same rows (a row is
  dropped when `y` or any list element is NULL). `ols_anova_agg(y, x_reduced, x_full[, options])`
  returns the contract `test` fields (`method = 'nested F'`, `statistic`, `df1`, `df2`,
  `p_value`, ...) plus `rss_reduced`, `rss_full`, the adjusted R² of both models
  (`adj_r_squared_reduced` / `adj_r_squared_full`), `delta_adj_r_squared` and
  `n_observations`. `ols_anova_by(source, group_col, y_col, x_reduced := [...], x_full := [...])`
  returns one row per group with `id` = the group and `model_id = 'ols'` (#187), and both are
  tagged as `test` producers. The `x_reduced` columns must lie in the span of `x_full`
  (an error otherwise), and `x_reduced := []` (the macro's default) is the intercept-only model,
  i.e. the overall F test. Degrees of freedom count the numerical rank as R does, so aliased
  columns add none; when `df1 = 0` or `df2 = 0` the statistic and p-value are NULL (R's `NA`;
  upstream's NaN is never passed through, so `avg()` over the results is not poisoned). The fits and the test come from
  anofox-regression 0.5.32 (`inference::nested_f_test`), which is validated against R, including
  `df1 = 1` (F = t²), the intercept-only and no-intercept models and a rank-deficient design.
  Sequential (type I) and partial (type II) per-term ANOVA tables are not included.

- **GLMM inference and contract macros (#218).** `glmm_fit_agg` with
  `compute_inference := true` now reports, per the new `ddf` option, lmerTest's
  **Satterthwaite** t tests (`'satterthwaite'`, the gaussian default: t statistics,
  t p-values and t intervals, the statistics in the new `t_values` field and the
  degrees of freedom in the new `df` / `intercept_df` fields) or Wald z tests (`'wald'`, the default for the other
  families; `df` is NULL). `compute_inference` stays opt-in. The BLUPs' conditional
  standard errors are now filled in (`ranef[].se`, previously NaN; lme4
  `ranef(condVar = TRUE)`), and every `ranef` entry also carries its full
  random-effect vector `effects` with `effects_se` and a normal interval
  (`effects_conf_low` / `effects_conf_high`). New field `df_residual` (lme4
  `df.residual()`). Crossed fits (`groups`) now fill `ranef` for the positional
  `group` factor (it was empty).

- **`glmm_tidy_by`, `glmm_glance_by`, `glmm_augment_by`** (integration contract v1
  `terms`, `summary` and `obs`, as broom.mixed `tidy` / `glance` / `augment`): one
  GLMM per `group_col` with random effects over `re_group`; `model_id` =
  `model_name` = `'glmm'`, `id` the group (not in `_augment_by`). `glmm_tidy_by`
  returns fixed effects (with `df`), variance components (`ran_pars`: SDs,
  correlations, residual SD) and BLUPs (`ran_vals`: conditional SEs and intervals),
  told apart by the extra `effect`, `re_group` and `re_level` columns.
  `glmm_augment_by` adds `fitted_population` and `pearson_residual`. The window
  aggregate `glmm_augment_agg` returns the per-row values.

- Index-list options (`random`, `groups`) also accept a list rendered as text
  (`'[1, 2]'`), the form they take when a macro passes options through a
  `MAP(VARCHAR, VARCHAR)`.

- **Leave-one-out diagnostics for OLS and WLS (#220).** Exact, from the hat-matrix
  diagonal, without refitting (computed upstream in anofox-regression,
  `diagnostics::press` and `Augment::loo_residual` / `loo_fitted`):
  - `ols_augment_by` / `wls_augment_by` (and `ols_augment_agg` / `wls_augment_agg`)
    add `loo_residual = residual / (1 - leverage)`, the error of the fit without the
    row (R `rstandard(fit, type = "predictive")` for OLS), and `loo_fitted = y -
    loo_residual`, the prediction from the fit without the row.
  - `ols_glance_by` / `wls_glance_by` add the metrics `press` (Σ loo_residual², weighted
    by the prior weights for WLS), `press_rmse = sqrt(press / n)` and `loo_r_squared`, the
    predicted R² (Q²) `1 - press / SST` with the SST of `r_squared`. They come from new
    fields of the `ols_fit_agg`, `wls_fit_agg`, `ols_fit` and `wls_fit` STRUCT, filled
    with the new opt-in option `compute_loo` (default false: one more QR of the design;
    independent of `compute_inference`), which the glance macros turn on; `glance()` has
    them for every model (NULL where not computed).
  - A row with leverage 1 (e.g. the only observation of a dummy level, or a fit with zero
    residual degrees of freedom) has NULL `loo_residual` / `loo_fitted`, and the group's
    `press`, `press_rmse` and `loo_r_squared` are NULL; rows outside training have NULL
    leave-one-out columns. Validated against R `lm()` (`residuals / (1 - hatvalues)`,
    `rstandard(type = "predictive")`, explicit refits).
  - Additive to the integration contract v1 (`obs`, `summary`). Ridge and the GLMs have no
    leave-one-out columns yet.

- **Cross-validation strategies and nested CV for ridge / elastic net (#204).** The fold
  generators, explicit splits and the nested loop are anofox-regression 0.5.30's
  `solvers::cv` (validated against scikit-learn); the extension maps arguments.
  - `ridge_cv_agg` / `elasticnet_cv_agg` take a `cv_key` argument of any type and a
    `cv_strategy` option: `'kfold'` (default without a key), `'fold_id'` (default with a
    key: the key is the fold, as the previous BIGINT `fold_id` form, which keeps working),
    `'group'` (group k-fold balanced by rows, sklearn `GroupKFold`), `'logo'`
    (leave one group out), `'stratified'` (stratum key, or without a key `n_bins` quantile
    bins of y) and `'time_series'` (rolling origin over a numeric / date / timestamp key
    with `test_size` (alias `horizon`), `gap`, `window`, `max_train_size`; the splits are
    over the distinct times, so all series of a date stay on one side). Several group
    variables form one key as a STRUCT (`{'store': store, 'region': region}`), hashed in
    the extension. New options `repeats` (repeated k-fold / stratified) and `metric`
    (`'mse'`, `'mae'`).
  - `ridge_nested_cv_agg` / `elasticnet_nested_cv_agg` (and `ridge_nested_cv_by` /
    `elasticnet_nested_cv_by`): lambda is chosen by an inner CV (`inner_folds`, built from
    each outer training set only with the same key; `rule` `'min'` or `'1se'`; per-fold
    default grids) and scored on the outer test rows. They return `curve` rows
    (`curve_type = 'nested_cv'`): one per outer fold (`series = 'fold'`, with `lambda`,
    `test_error`, `inner_cv_mean`, `inner_cv_se`, `n_train`, `n_test`, `n_inner_folds`) and
    a summary row (`series = 'mean'`, the test-size weighted mean and `test_se`).
  - `ridge_cv_by` / `elasticnet_cv_by` and the nested macros take `cv_key := column`.

### Changed

- **Behaviour change: the Monte Carlo tests are seeded by default (#214).**
  `permutation_t_test_agg`, `energy_distance_agg`, `mmd_agg` and `distance_cor_agg`
  drew a fresh random set of permutations, and so a different p-value, on every
  call unless `{'seed': ...}` was passed. Without a seed they now use the fixed
  default seed 42, so repeated calls return the same p-value; an explicit `seed`
  (alias `random_state`) still overrides it. Their samples are always put in a
  canonical order first, so the seeded result is also independent of the thread
  count. To get a different draw, pass a different seed.

- **Behaviour change: window `*_fit_predict` returns NULL, not NaN, for a row with an
  incomplete `x`** (#208). When the last row of the frame has an `x` list with a NULL,
  NaN or ±Inf element, `ols_fit_predict`, `ridge_fit_predict`, `wls_fit_predict`,
  `rls_fit_predict`, `elasticnet_fit_predict`, `huber_fit_predict`,
  `ransac_fit_predict` and `theil_sen_fit_predict` used to return `yhat = NaN`; they
  now return NULL, as `passive_aggressive_fit_predict` and a NULL `x` list already did.
  NaN propagated silently through `avg()` and similar aggregates; NULL is skipped.
  Code that tested `isnan(yhat)` for these rows should test `yhat IS NULL`. Rows with a
  complete `x` are unchanged, and under `null_policy = 'zero'` a NULL element still
  counts as 0.

- CI builds, smoke-tests and deploys the extension for **DuckDB v2.0.0** on every
  default platform (Linux amd64 and arm64, macOS arm64, WASM; `osx_amd64` and
  musl are opt-in in DuckDB 2.0's CI tooling and are not built) and
  publishes it to get.erpl.io, like the 1.5.x lines. This replaces the
  non-blocking 2.0-dev job, which built only linux_amd64 and wasm_eh and deployed
  nothing. The smoke test downloads DuckDB 2.0's renamed `.tar.gz` CLI assets.

- Random-slope and crossed GLMM fits converge more tightly (anofox-regression
  0.5.33 polishes the variance parameters with Newton steps after Nelder–Mead), so
  their estimates can move in the 5th–6th digit towards lme4's.

- anofox-regression 0.5.33.

- **Behaviour change: the `split` argument of the `*_fit_predict_by`, `*_augment_by` and
  `*_prediction_by` macros follows the `*_fit_predict_agg` `split_col` rule (#205).** A row
  whose split value is `'train'` or `'training'` in any case (`'Train'`, `'TRAINING'`, ...)
  is a training row; any other value is a prediction row. A per-row NULL split value is now
  a prediction row (the macros trained on it before). Calls that omit `split` (or pass
  `split := NULL`) are unchanged: every row with a non-NULL `y` trains.

### Fixed

- **Bitwise-reproducible group tests (#214).** `brown_forsythe_agg`,
  `kruskal_wallis_agg` and `one_way_anova_agg` could differ in their last bits
  between identical runs, even with `threads = 1` and ordered input: the FFI split
  the rows into groups through a Rust `HashMap`, whose per-instance random seed
  changed the group (and hence summation) order on every call. The groups are now
  ordered by label and the values within each group are sorted before the test
  runs. The two-sample tests (`t_test_agg`, `mann_whitney_u_agg`, `yuen_agg`,
  `brunner_munzel_agg`, `tost_t_test_agg`) and the post-hoc tests (`tukey_hsd_agg`,
  `pairwise_t_test_agg`, `dunn_test_agg`) sort their samples the same way. The same
  rows now give the bitwise-identical result whatever the thread count, the row
  order or the order in which the group labels first appear. Statistics can change
  in the last bits relative to earlier versions.

- **Theil-Sen `compute_inference` now populates the inference fields (#222).**
  Previously they were returned but always NaN / NULL: the upstream solver had no
  inference, so the documented "standard errors, t-values, ... and the F-test" were
  never computed (they were never OLS-style values). They now hold the exact or
  bootstrap intervals described under Added. Field names are unchanged; what each
  holds depends on the method: with `'exact'` `t_values` is Kendall's z (not a t
  statistic) and `std_errors` is NULL; with `'bootstrap'` `t_values` / `p_values`
  are NULL. `f_statistic` / `f_pvalue` are NULL: Theil-Sen has no F-test. Statistics
  a method does not define are NULL, not NaN. Note the
  cost: multi-feature fits with `compute_inference` (and `theil_sen_tidy_by`, which bootstraps
  only on request) refit the model `n_bootstrap` times per group: on 500 rows ×
  3 features about 8 s of CPU per group with the aggregate's 999 resamples and
  1.4 s with tidy_by's 199. Lower `n_bootstrap`, use `'exact_or_none'` or pass
  `{'compute_inference': false}` where that is too slow.

- **Window `*_fit_predict` in long partitions** (#208). When the frame's last row had a
  NULL `x` list and the frame was assembled from several pre-aggregated blocks, all nine
  window `*_fit_predict` functions could return the prediction for an earlier row's `x`
  instead of NULL. The frame now always ends at its own last row.

- **`docs/NULL_SEMANTICS.md` described the NULL-`x` behaviour wrongly** (#208). Every
  `*_fit_predict_agg` keeps a row whose `x` list is NULL, in input order, with `yhat`
  NULL and `is_training = false`; the doc said the row was left out. The doc,
  `fit_predict_agg.md`, `fit_predict_window.md`, the isotonic and quantile pages and the
  regression skill now describe the actual behaviour. No aggregate changed.

- `*_fit_predict_by` and the other split-taking macros treated only an exact `'train'` as
  training, so `split = 'Training'` or `'TRAIN'` silently held every row out, and they fitted
  rows with a NULL split value that the aggregates predict. A macro and its aggregate called
  with the same split column now train on the same rows (#205).

- `rls_tidy_by` / `rls_glance_by` without `order_by` fit the rows in scan order,
  which is not deterministic under parallel scans, so the coefficients could
  change between runs (#206). Pass `order_by := t` for a reproducible fit; the
  docs now say the order is undefined without it.

### Behaviour change

- **Gaussian `glmm_fit_agg` inference uses Satterthwaite t tests by default.** With
  `compute_inference := true`, a gaussian GLMM's `p_values`, `ci_lower` / `ci_upper`
  and the intercept's p-value and interval were normal (z); they now follow
  lmerTest's Satterthwaite t distribution. The statistics move to the new
  `t_values` field and `z_values` is NULL (a non-NULL `df` also marks t tests);
  under Wald inference `z_values` is filled and `t_values` is NULL. The standard
  errors are unchanged. `{'ddf': 'wald'}` restores the previous values.

- `rls_tidy_by` and `rls_glance_by` take an optional named `order_by := <column>`
  (#206). Recursive least squares depends on the row order; with `order_by` each
  group's rows are fed to `rls_fit_agg` in that order (`ORDER BY order_by`), as
  `rls_fit_predict_by` already did. Existing calls are unchanged.

## [0.12.2] - 2026-10-22

### Fixed

- **macOS build against DuckDB v2.0.0.** DuckDB 2.0's bundled HTTP client uses the
  macOS keychain, so on macOS the extension must link the `Security` framework (next to
  `CoreFoundation`). Without it the `osx_arm64` build on v2.0.0 failed to link with
  undefined `_Sec*` symbols; the community-extensions registry builds macOS only after
  merge, and one failed platform skips the deploy for every platform. The 2.0 CI job is
  pinned to the v2.0.0 commit (`d54cbbe`). No SQL or numeric changes.

## [0.12.1] - 2026-10-16

### Fixed

- The DuckDB 2.0 WebAssembly build in the community-extensions registry failed
  to link (`wasm-opt: Unknown option '--enable-bulk-memory-opt'`). The registry
  builds with Rust 1.97, whose wasm output (including the precompiled
  `wasm32-unknown-emscripten` std) uses target features the pinned emsdk 3.1.71
  cannot optimise. A `rust-toolchain.toml` now pins Rust 1.86.0 for all builds,
  the version extension-ci-tools v1.4/v1.5 already use for wasm. There are no
  functional code changes. CI now also builds `wasm_eh` in the DuckDB 2.0-dev job.
- The Linux musl build is now chosen from the CI's `DUCKDB_PLATFORM` (or a musl
  libc for local builds) instead of from whether the musl Rust target happens to
  be installed, which the toolchain pin now does on every platform.

## [0.12.0] - 2026-10-16

### Added

- Bayesian Ridge and ARD regression (#61), scikit-learn's `BayesianRidge` /
  `ARDRegression` from anofox-regression 0.5.29 (results agree with scikit-learn
  1.9.1 to about 1e-8): `bayesian_ridge_fit_agg`, `ard_fit_agg`,
  `bayesian_ridge_fit_predict_agg`, `ard_fit_predict_agg`,
  `bayesian_ridge_fit_predict_by`, `ard_fit_predict_by` and the contract
  producers `bayesian_ridge_` / `ard_tidy_by`, `_glance_by` and `_prediction_by`.
  The noise precision, weight precision(s) (`alpha_`, `lambda_`) and the penalty
  are estimated from the evidence; the model STRUCT has `noise_precision`,
  `weight_precision` (ARD: `weight_precisions[]`, pruned features at or above
  `threshold_lambda` get a coefficient of exactly 0), `posterior_sd[]`,
  `iterations` and, for Bayesian Ridge, `log_marginal_likelihood`; the
  frequentist inference fields and `log_likelihood` / `aic` / `bic` are NULL.
  The fit-predict bands are posterior predictive intervals at
  `confidence_level` (scikit-learn's `predict(return_std=True)`), with no
  bootstrap. Options: `fit_intercept`, `max_iterations`, `tolerance`,
  `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, `threshold_lambda` (ARD).
  See [docs/api/regression/bayesian.md](docs/api/regression/bayesian.md).
- Passive-Aggressive regression, lmDynamic and LOWESS (#63), from
  anofox-regression 0.5.29 (see [docs/api/regression/online_dynamic.md](docs/api/regression/online_dynamic.md)):
  - `passive_aggressive_fit_agg(y, x[, options] ORDER BY t)`, the window
    `passive_aggressive_fit_predict`, `passive_aggressive_fit_predict_agg` /
    `_fit_predict_by` / `_prediction_by`, and `passive_aggressive_tidy_by` /
    `_glance_by(source, group_col, order_col, y_col, x_cols, ...)`: online
    regression updated row by row in the given order (scikit-learn
    `PassiveAggressiveRegressor`; one pass equals `partial_fit`, validated to
    1e-9). Options `c`, `epsilon`, `variant` (`'pa1'` | `'pa2'`),
    `fit_intercept`, `max_iterations` (passes, default 1), `tolerance`.
  - `lm_dynamic_fit_agg(y, x[, options] ORDER BY t)`, `lm_dynamic_fit_predict_agg`
    / `_fit_predict_by` / `_prediction_by`, `lm_dynamic_glance_by` and
    `lm_dynamic_tidy_by(source, group_col, order_col, y_col, x_cols, ...)`, which
    returns the coefficients of every row as `terms` with `index_name =
    'window_end'` (the shape of `rolling_tidy_by`): time-varying coefficients
    from greybox's `lmDynamic(bruteforce = TRUE)` (normal distribution, validated
    against greybox 2.0.9 to 1e-8). Options `ic` (`'aicc'` | `'aic'` | `'bic'`),
    `span` (lowess smoothing of the weights, 0 = none), `fit_intercept`.
  - `lowess_agg(y, x[, frac[, iterations]])` returns R's `stats::lowess` curve
    as `LIST(STRUCT(x, yhat))` (validated to 1e-10), and `lowess_by` returns it
    per group in the contract's `curve` schema with `curve_type = 'lowess'`.
- `<model>_effects_by` returns `focal_name VARCHAR` (last column), the name of the
  focal feature, for labelling marginal-effect plots (#197). It is `names[focal]`
  with the new `names := [...]` parameter (as `<model>_tidy_by`), else
  `'x' || focal` (`x1`, `x2`, …), the same labels as the default `_tidy_by` terms.
  `focal` (the 1-based index) is unchanged.
- `pairwise_t_test_agg` / `pairwise_t_test_by` fill `conf_low`, `conf_high` and
  `conf_level` for each difference (#198). The interval uses the same standard error
  and degrees of freedom as the t test (pooled SD or Welch) and is one-sided for a
  one-sided `alternative`. It is per comparison with `p_adjust` `'none'`, and
  Bonferroni-adjusted otherwise, as `emmeans::confint(pairs(...), adjust = )`.
  New option `confidence_level` (default 0.95). Validated against `emmeans` and
  `t.test` (anofox-statistics 0.4.10).
- `null_policy` accepts `'zero'` and `'ignore'` (#8), in every function that reads
  it: the `*_fit_predict_agg` aggregates and `*_fit_predict_by` macros (OLS, Ridge,
  WLS, RLS, Elastic Net, Huber, RANSAC, Theil-Sen, BLS, ALM, Poisson, Bayesian Ridge,
  ARD, Passive-Aggressive, lmDynamic), the window
  `*_fit_predict` functions and `ols_augment_agg` / `wls_augment_agg`. `'zero'` fits
  a NULL `y` or a NULL element of `x` as 0, the same result as fitting
  `COALESCE(..., 0)`-filled input. A NULL `y` then no longer marks a row to predict,
  so use a split column to hold out future rows. A NULL `x` list and `NaN` values
  are not filled. `'ignore'` is an alias for `'drop'`, for input with no NULLs. See
  `docs/NULL_SEMANTICS.md`. `passive_aggressive_fit_predict_agg` and
  `lm_dynamic_fit_predict_agg` now accept `null_policy`. The `*_fit_predict_by`
  macros now pass `split :=` to the aggregate's split argument instead of NULLing `y`
  for held-out rows (`ols_augment_by` / `wls_augment_by` pass NaN), so held-out rows
  stay out of the fit under `'zero'`. Results under the other policies are unchanged.
- Docs: "Fit per series, score new rows per series" in
  `docs/api/regression/model_tools.md` and the batch skill shows two ways to score new
  rows with per-series models: `*_fit_predict_by` with a NULL `y`, or a `GROUP BY` fit
  table joined to the new rows with `predict(model, x)`, and what INNER, LEFT and FULL
  joins do with series missing on one side (#7).
- Docs: `docs/api/aid/aid.md` and the diagnostics skill recommend unnesting
  `aid_anomaly_agg` once in a subquery and then reading the struct fields, instead
  of repeating `(unnest(flags)).field`, which has crashed some ADBC clients (#27).

### Changed

- The extension builds against DuckDB 2.0-dev (duckdb `b15358c`) as well as
  v1.5.x and the v1.4 LTS. A small compatibility header
  (`src/include/duckdb_compat.hpp`) detects the changed C++ APIs (struct child
  names as `Identifier`, batched aggregate initialize, bind inputs, read-only
  `FlatVector::GetData`, accessor-only function fields); no behaviour changes.
  CI gains a non-blocking DuckDB 2.0-dev build-and-test job.

- **Behaviour change:** `prop_test_two_agg`'s `estimate` is now the difference of
  proportions p₁ − p₂, the quantity that its confidence interval, statistic and
  p-value are for (#195). It used to be p₁, the proportion of group 1
  (`group_id = 0`), which could lie outside its own interval. To get p₁, compute it
  directly, e.g. `avg(value) FILTER (WHERE group_id = 0)`. The other fields are
  unchanged.

### Fixed

- The scalar `ols_fit`, `ridge_fit`, `wls_fit`, `elasticnet_fit`, `huber_fit`,
  `ransac_fit`, `rls_fit`, `theil_sen_fit`, `vif`, `residuals_diagnostics` and
  `jarque_bera` returned NULL (or failed) for every row after the first when one
  list argument was a constant and another varied per row, and mixing both shapes
  in one SELECT could crash DuckDB (#210). List arguments are now read per row
  for constant, per-row and dictionary (join / filter) inputs alike.

- `lars_fit_agg` and `lars_glance_by` reported `log_likelihood`, `aic` and
  `bic` as 0, so LARS looked like the best model by AIC/BIC (#196). They are now the
  Gaussian log-likelihood and information criteria with the LARS degrees of
  freedom (active coefficients plus intercept), equal to R's `lm` on the full LARS
  path (anofox-regression 0.5.27).

## [0.11.0] - 2026-10-14

### Added

- Contract `prediction` producers (#183): `<model>_prediction_by(source, group_col,
  x_col, y_col, x_cols, level := 0.95)` (19 models) returns fitted values with their
  band as `prediction` rows (`id, model_id, model_name, x, y, yhat, yhat_lower,
  yhat_upper, level, split`), aligned with anofox-forecast; `<model>_effects_by`
  (OLS, WLS, Ridge, GLMs) returns marginal-effect grids with confidence bands.
- Rolling coefficients (#177): `<model>_rolling_tidy_by(source, group_col,
  order_col, y_col, x_cols, window_size := 30, min_obs := NULL)` for `ols`, `wls`,
  `ridge`, `elasticnet`, `huber` and `poisson` returns the coefficients of every
  trailing window as contract `terms` rows (`index_name = 'window_end'`).
- Durbin-Watson p-value (#177): `durbin_watson_agg(resid, x [, alternative [,
  fit_intercept]] ORDER BY t)` returns the `lmtest::dwtest` p-value (exact via
  Imhof for n < 100, else the normal approximation; anofox-statistics 0.4.9).
  The residual-only form keeps its NULL p-value. Validated against R.
- `alias_tolerance` option (#164) for `ols_fit`, `ols_fit_agg`, `wls_fit`,
  `wls_fit_agg`, the OLS / WLS `*_fit_predict_agg` aggregates and
  `*_fit_predict` window functions: the tolerance of the collinearity test
  (R `lm`'s `tol`, default `1e-7`) that decides when a column is aliased (NaN
  coefficient). Validated against `lm(..., tol =)`.
- Regularisation paths and cross-validation (#154): `ridge_path_agg`,
  `elasticnet_path_agg` and `lars_path_agg` return the coefficient path as
  `terms` rows (`index_name = 'lambda'`; `'alpha'` on upstream's LARS scale for
  `lars_path_agg`), glmnet's Gaussian objective and default
  grid (`lambdas`, `n_lambda`, `lambda_min_ratio`, `standardize`, `l1_ratio`);
  `ridge_cv_agg` / `elasticnet_cv_agg` return the k-fold CV curve as `curve` rows
  (`curve_type = 'lambda_cv'`: `cv_mean`, `cv_se`, `is_min`, `is_1se`, as
  `cv.glmnet`) with seeded folds or a fold-id column; table macros
  `ridge_path_by`, `elasticnet_path_by`, `lars_path_by`, `ridge_cv_by`,
  `elasticnet_cv_by` (contract producers). Validated against glmnet's documented
  objective / grid / `cv.glmnet` aggregates and grpreg.
- Quantile-regression inference (#157): `quantile_fit_agg` fills `std_errors`,
  `t_values`, `p_values`, `conf_low`, `conf_high` and the intercept inference with
  `{'se': 'iid' | 'nid' | 'boot'}` or `{'compute_inference': true}` (`'nid'`), as
  `quantreg::summary.rq`; `quantile_tidy_by` computes it by default.
  `quantile_process_agg` / `quantile_process_by` return the regression-quantile
  process (`terms`, `index_name = 'tau'`). Validated against quantreg.
- Predictions with standard errors (#158): `predict(model, x, {'se_fit': true,
  'interval': 'confidence' | 'prediction', 'level': 0.95})` returns
  `STRUCT(yhat, se_fit, yhat_lower, yhat_upper)` for OLS, WLS, Ridge and the GLMs
  (GLM intervals on the link scale, transformed); calls without these options
  are unchanged. The OLS, WLS, Ridge and GLM `*_fit_agg` structs gain `vcov` and
  `df_residual` (NULL unless `compute_inference`). Validated against R
  `predict.lm` / `predict.glm(se.fit = TRUE)`.
- Partial residuals (#158): `<model>_partial_residuals_by` and
  `<model>_partial_residuals_agg` for `ols`, `wls` and the six GLMs return
  component-plus-residual (`car::crPlots`) and, for OLS / WLS, added-variable
  (`car::avPlots`) coordinates per row and term. Validated against R
  `residuals(type = "partial")` and `lsfit`.
- Time-series regression diagnostics (#155): `acf_agg` / `pacf_agg`
  (`agg(x [, max_lag] ORDER BY t)` -> `[{lag, acf|pacf, conf_low, conf_high}]` with
  the 95% white-noise band), `ljung_box_agg` / `box_pierce_agg` (`lag`, `fitdf`),
  `durbin_watson_agg` (statistic only), and the heteroskedasticity tests
  `breusch_pagan_agg(resid, x [, studentize])` and `white_test_agg(resid, x)`, all
  returning the unified test fields; `acf_by` / `pacf_by` table macros emit the
  contract's `curve` schema (`curve_type` 'acf' / 'pacf'). Validated against R
  `acf`, `pacf`, `Box.test`, `lmtest::dwtest` and `lmtest::bptest`.
- Survival curves (#156): `kaplan_meier_agg(time, event [, conf_level [, conf_type]])`
  (Greenwood standard errors, log / log-log / plain intervals), the
  `kaplan_meier_by` table macro (one `curve` per stratum, `curve_type` 'km'),
  `log_rank_test_agg(time, event, group)` (unified test fields) and
  `aft_survival(model, x, times)` for the model-based survival curve of an
  `aft_fit_agg` fit. Validated against R `survival::survfit`, `survdiff` and
  `survreg` / `psurvreg`.

- Unified hypothesis-test results (#159): every test aggregate (`t_test_agg`,
  `yuen_agg`, `mann_whitney_u_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`,
  `permutation_t_test_agg`, `kruskal_wallis_agg`, `one_way_anova_agg`,
  `brown_forsythe_agg`, `shapiro_wilk_agg`, `jarque_bera_agg` (and the scalar
  `jarque_bera`), `dagostino_k2_agg`, `energy_distance_agg`, `mmd_agg`, `pearson_agg`,
  `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg`, `chisq_test_agg`,
  `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg`, `cohen_kappa_agg`,
  `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg`, `tost_t_test_agg`,
  `tost_paired_agg`, `tost_correlation_agg`, `diebold_mariano_agg`, `clark_west_agg`)
  now returns the integration contract's `test` fields `method, alternative,
  statistic, df1, df2, p_value, estimate, conf_low, conf_high, conf_level,
  effect_size, effect_name, n`, always present and NULL when not applicable, taken
  from upstream `TestResult`. The fields a test lacked are appended after its
  existing ones. Validated against R `t.test`, `wilcox.test`, `kruskal.test`,
  `oneway.test`, `chisq.test`, `fisher.test` and `cor.test`. Requires
  anofox-statistics 0.4.7.
- Pairwise post-hoc tests (#159): `pairwise_t_test_agg` (R `pairwise.t.test`, with
  `p_adjust` holm/hochberg/hommel/bonferroni/BH/BY/none, `pool_sd`, `alternative`),
  `tukey_hsd_agg` (R `TukeyHSD`) and `dunn_test_agg` (`FSA::dunnTest`), returning one
  `{group1, group2, estimate, statistic, p_value, p_adj, conf_low, conf_high, method,
  ...}` per pair, and the table macros `pairwise_t_test_by`, `tukey_hsd_by`,
  `dunn_test_by` emitting the long `test` schema. Validated against R
  `pairwise.t.test` (none, holm, bonferroni, BH; pooled and Welch) and `TukeyHSD`.
- All hypothesis-test aggregates and the post-hoc aggregates and macros are tagged
  `anofox.output = 'test'` in `duckdb_functions()`.

- Stable model outputs for every fit (#152): the linear and robust fits (`ols`, `wls`,
  `ridge`, `elasticnet`, `lars`, `rls`, `huber`, `ransac`, `theil_sen`, aggregate and
  scalar), the GLMs (`poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`),
  `alm`, `aft`, `glmm`, `bls`/`nnls`, `pls`, `quantile` and `isotonic`:
  every model struct now ends with `conf_low`, `conf_high`, `conf_level`, the
  intercept's inference (`intercept_std_error`, `intercept_statistic`,
  `intercept_p_value`, `intercept_conf_low`, `intercept_conf_high`), `log_likelihood`,
  `aic`, `bic` (as R's `logLik`/`AIC`/`BIC`; NULL for estimators without a likelihood)
  and `model_type`, `family`, `link` (from upstream `HasModelInfo`; GLMs report
  `model_type = 'glm'`). Validated against R `lm` (OLS, WLS, HC3), `glm` (Poisson,
  logistic), `survival::survreg` (AFT) and lme4 (GLMM). Requires anofox-regression 0.5.24.
- `<model>_tidy_by` and `<model>_glance_by` table macros for 21 models (not `glmm`,
  `isotonic`; `wls` takes `weight_col`, `aft` `event_col`): per-group
  coefficient tables and long-form fit statistics in the integration contract's `terms`
  and `summary` schemas, tagged `anofox.output = 'terms' | 'summary'`.

- `ols_augment_by` and `wls_augment_by` table macros (#151), broom `augment` style:
  every source row with `model_id, row_id, fitted, residual, resid_type, std_residual,
  stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params` in the
  integration contract's `obs` schema (tagged `anofox.output = 'obs'`). Non-training
  rows (`split` other than `'train'`, NULL y) get the prediction and NULL diagnostics.
  Backed by the window aggregates `ols_augment_agg` / `wls_augment_agg` and computed by
  `anofox_regression::diagnostics::augment`; validated against R `lm` (`rstandard`,
  `rstudent`, `hatvalues`, `cooks.distance`, `dffits`).
- GLM augment macros (#151): `poisson_augment_by`, `binomial_augment_by`,
  `logistic_augment_by`, `negbinom_augment_by`, `gamma_augment_by` and
  `tweedie_augment_by` (window aggregates `<family>_augment_agg`), with the same `obs`
  columns as `ols_augment_by`. `fitted` is on the response scale, `residual` is the
  deviance residual (`resid_type = 'deviance'`); non-training rows get the
  response-scale prediction. Computed by
  `anofox_regression::diagnostics::augment_glm` (anofox-regression 0.5.25) and
  validated against R `glm`, `MASS::glm.nb` and `statmod::tweedie`. Penalized fits
  (`glm_lambda > 0` or a `prior`) are rejected.

- anofox integration contract scaffolding (#160): `anofox_contract_version()` returns
  `'1'` (registered only if no other anofox extension has registered it yet), and every
  function and table macro is tagged `anofox.family = 'statistics'` in
  `duckdb_functions().tags`. Contract producers will also carry `anofox.contract` and
  `anofox.output`. Every overload now has a description and an example, including the
  `VARCHAR`-group overloads of the two-sample tests, `aft_fit_agg`, `aft_cdf`,
  `aft_quantile`, `eb_shrink_agg`, `glmm_fit_agg` and `predict(model, x, options)`.

- `glmm_fit_agg` families `negbinom` (θ estimated as `lme4::glmer.nb`, or fixed with
  `'theta'`), `gamma` and `tweedie` (`'power'`), all with the log link; the `offset`
  option; random slopes (`'random'`) together with crossed `'groups'` factors, applied to
  the positional grouping factor. The result struct gains a trailing `nb_theta` field.
  Validated against `lme4` with `nAGQ = 0` (`test/sql/reference/glm_glmm.test`).

- `pls_fit_agg`, `quantile_fit_agg` and `isotonic_fit_agg`: the fitted models of these
  methods can now be retrieved (previously only `*_fit_predict_agg` / `*_fit_predict_by`).
- Model-aware `predict(model, x[, {'type': 'response' | 'link'}])` for the STRUCT returned
  by any `*_fit_agg` / `*_fit` function: GLMs are mapped to the response scale through
  their link, isotonic models are interpolated. `linear_predict` is an unambiguous name
  for the existing column-layout `predict(x, coefficients, intercept)`.
- `tidy(model[, names])` (one row per term: estimate, std_error, statistic, p_value,
  conf_low, conf_high) and `glance(model)` (the model's scalar fields).
- GLM result structs (`poisson`, `binomial`, `logistic`, `negbinom`, `gamma`, `tweedie`)
  gain trailing `family` and `link` fields.
- `binomial_fit_predict_by`, `logistic_fit_predict_by`, `negbinom_fit_predict_by`,
  `gamma_fit_predict_by`, `tweedie_fit_predict_by` (predictions on the response scale).
- `order_by` parameter on every `*_fit_predict_by` macro. Predictions are matched to rows
  by a total ordering (the key plus a row id) instead of relying on two windows happening
  to share a sort.
- Two-sample tests accept VARCHAR group labels.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/MIGRATION.md`, `docs/NULL_SEMANTICS.md`,
  `docs/METHODOLOGY.md`. New reference pages for Huber, RANSAC, Theil-Sen, LARS,
  the binomial, logistic, gamma and Tweedie GLMs, the window and fit-predict
  aggregates.
- The doc-SQL validation now covers every page under `docs/` and `docs/api/`.
- A Claude Code plugin (`plugins/anofox-statistics`) with skills for regression,
  tests, diagnostics and batch use.
- Option aliases (old keys keep working): `link` sets the link of the fitted
  family (`poisson_fit_agg`, `poisson_fit_predict_agg`, `binomial_fit_agg`);
  `lambda`/`alpha` alias `glm_lambda` on the GLMs (poisson, binomial, logistic,
  negbinom, gamma, tweedie); `tau` aliases `quantile` on ALM and `quantile`
  aliases `tau` on quantile regression; `lambda` aliases `forgetting_factor` on
  RLS; `lambda` aliases `alpha` on `ridge_fit_predict_agg` /
  `elasticnet_fit_predict_agg`; `seed`/`random_state` on every seeded function.
- TOST: `tost_t_test_agg`, `tost_paired_agg` and `tost_correlation_agg` all accept
  both `alpha` and `confidence_level` (alpha = 1 - confidence_level; supplying
  inconsistent values is an error) and `lower`/`upper` as aliases of
  `bound_lower`/`bound_upper`.
- Seeds for `permutation_t_test_agg`, `energy_distance_agg`, `mmd_agg` and
  `distance_cor_agg` (`seed` / `random_state`).
- `mann_whitney_u_agg` options `exact` and `mu`; `fisher_exact_agg` option
  `confidence_level` and fields `conditional_odds_ratio`, `conditional_ci_lower`,
  `conditional_ci_upper` (R `fisher.test` estimate and exact interval);
  `confidence_level` for `binom_test_agg`, `prop_test_one_agg`,
  `prop_test_two_agg`; `mcnemar_agg` option `exact`; `lars_fit_agg` options
  `method` (`'lar'` | `'lasso'`), `n_nonzero_coefs` and `standardize`.
- Hypothesis-test results now all carry `statistic`, `p_value`, `n` (total
  sample size), `method` and `alternative` (`'two_sided'` | `'less'` |
  `'greater'`, NULL where not applicable). Missing fields were appended at the
  end of each struct, e.g. `cohen_kappa_agg` gains `statistic` (= z), `n`,
  `method`; `icc_agg` gains `statistic` (= F), `p_value` (one-way F test of
  ICC = 0), `n`; `tost_*_agg` gain `statistic`; `jarque_bera`/`jarque_bera_agg`
  gain `method`.

### Changed

- **Breaking:** `model_id` in the model-fitting `_by` macros (`<model>_tidy_by`,
  `_glance_by`, `_augment_by`, `_rolling_tidy_by`, `_partial_residuals_by`, the
  path / CV / quantile-process macros) is now the method (`'ols'`, ...) instead of
  the group value. A new `model_name` column holds the same, and a new `id` column
  holds the group as text (except in `_augment_by`, which keeps the source
  columns). Outputs of several methods now stack with `UNION ALL`, as
  `_prediction_by` and anofox-forecast do. The test and curve macros keep the group
  as `model_id`.
- Hypothesis-test result structs gain the unified `test` fields (#159) after their
  existing fields; existing field names, positions and values are unchanged. Code that
  compares whole result structs with `=` now also compares the new fields (for TOST,
  `conf_level` is `1 - 2 * alpha` as computed upstream).

- **Model structs have a stable shape (#152).** The coefficient inference fields
  (`std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper`, `f_statistic`,
  `f_pvalue`) are always present and NULL without `compute_inference`, instead of being
  added only when it is set. Elastic Net, LARS and RLS gain `std_errors`, `t_values`,
  `p_values` (always NULL). Existing fields keep their names and order.
- **`glance(model)` returns one fixed set of fields for every model** (`model_type`,
  `family`, `link`, `n_observations`, `n_features`, `r_squared`, `adj_r_squared`,
  `residual_std_error`, `f_statistic`, `f_pvalue`, `log_likelihood`, `aic`, `bic`,
  `deviance`, `null_deviance`, `pseudo_r_squared`, `dispersion`, `iterations`,
  `converged`), NULL where a model has no such value. Model-specific scalars such as
  Huber's `scale` are no longer in `glance`; read them from the model struct.
- `tidy(model)` rows gain `conf_level`, `index_name` and `index_value`, and the intercept
  row carries the intercept's inference where the model reports it.
- `negbinom_fit_agg` reports `family = 'negative_binomial'` (was `'negbinom'`), the
  upstream / R name. `family` and `link` of every GLM now come from upstream.
- `glmm_fit_agg`'s `aic` / `bic` come from upstream (`FittedGlmm::aic` / `bic`, lme4's
  parameter count) instead of being computed in the extension; the values are unchanged.


- **Behaviour change (Poisson inference).** `poisson_fit_agg` and the other Poisson
  aggregates compute standard errors at dispersion 1, exactly R's `glm(family = poisson)` /
  `summary.glm`. Previously the covariance was scaled by `max(1, Pearson χ²/df)`
  (floored quasi-Poisson), so on overdispersed data `std_errors`, `z_values`, `p_values`
  and confidence intervals are now smaller by `√(χ²/df)` and `dispersion` is 1.0.
  Coefficients are unchanged. Use `negbinom_fit_agg` for overdispersed counts.
- **Behaviour change (Gamma fit statistics).** `gamma_fit_agg` evaluates the
  log-likelihood (and so `aic` / `bic`) at the dispersion `deviance / n`, exactly R's
  `logLik.glm` / `AIC`. Coefficients, standard errors and the reported (Pearson)
  `dispersion` are unchanged.
- **Behaviour change (null deviance).** For every GLM family, with an offset the
  `null_deviance` is that of the intercept-only model fitted with the offset, and without
  an intercept it is the deviance at `μ = linkinv(offset)` (`linkinv(0)` without an
  offset), as R's `glm`. `null_deviance` and `pseudo_r_squared` change for offset and
  no-intercept models only.
- The AID demand classification (`aid_agg`, `aid_anomaly_agg`) and empirical-Bayes
  shrinkage (`eb_shrink_agg`, `eb_shrink_by`) are delegated to `anofox-regression`
  (`solvers::aid::heuristic`, `solvers::eb_shrink`); results are bit-identical.
- `glmm_fit_agg` `aic` / `bic` count every random-effect covariance parameter
  (`q(q+1)/2` with random slopes, previously 1), as the df of lme4's `logLik`.
- Dependency: `anofox-regression` 0.5.21 -> 0.5.22. The negative binomial θ estimation
  (`negbinom_fit_agg` and friends, `glmm_fit_agg` with the `negbinom` family) is much
  faster (sipemu/anofox-regression#72); results are unchanged.
- The penalized GLM engine (Poisson, Binomial, Negative Binomial, Tweedie, Gamma, Logistic,
  priors, Laplace curvature) and the AFT survival model are now delegated to
  `anofox_regression::solvers::{penalized_glm, aft}`; the extension keeps only option
  conversion and error mapping. No change in results, error messages or NULL policy.
- `glmm_fit_agg` z-values, p-values and Wald intervals come from `anofox-regression`
  (match lme4 `summary()` / `confint(method = "Wald")`); negative binomial standard
  errors always use dispersion 1 like `MASS::glm.nb`; Theil-Sen aliases collinear
  feature columns (NaN coefficient, as R's `lm`) instead of returning huge values.
- Dependency: `anofox-regression` 0.5.15 -> 0.5.21. OLS / WLS alias rank-deficient
  designs like R's `lm` with every solver, including the default `'svd'` (#148): the
  later column of a collinear set gets a `NaN` coefficient instead of a minimum-norm
  split of the coefficient. `lars_fit_agg` reports `adj_r_squared` and
  `residual_std_error` (#147; equal to R's `summary(lm())` on the active set at the
  end of the LARS path); LARS aliases collinear columns deterministically.
- Dependency: `anofox-regression` 0.5.13 -> 0.5.15. More regression code moves
  upstream: prediction-interval variance factors and leverage
  (`anofox_interval_matrix` / `anofox_predict_with_interval_matrix`,
  `anofox_t_critical`, `anofox_predict_with_interval`; C ABI unchanged), Huber
  MASS-style standard errors, RANSAC's final inlier OLS fit, RLS (now
  `RlsRegressor`), constant-column / no-intercept handling of the linear models,
  the Negative Binomial y = 0 deviance and ML theta, and the GLMM log-likelihood.
  Behaviour inherited from upstream, now matching R:
  - logLik / AIC / BIC of the linear models use R's convention (RSS/n); Poisson,
    Negative Binomial and GLMM report the full log-likelihood (lme4's
    `logLik(glmer)`), so their AIC/BIC values change.
  - Tweedie and Gamma dispersion is the Pearson estimate.
  - Univariate Theil-Sen is the median of pairwise slopes. With
    `compute_inference := true` its inference lists now hold one NULL per
    feature (Theil-Sen has no inference) instead of being empty.
  - Quantile regression returns the exact LP optimum.
  - RANSAC: inliers are `|r| <= threshold` (a constant response now fits), and
    R², adjusted R², sigma and inference are those of the OLS fit on the inliers;
    collinear inliers give an aliased (NaN) coefficient.
  - Ridge with `alpha > 0` and `compute_inference := true` reports the sandwich
    standard errors `sigma * sqrt(diag(A X'X A))`, `A = (X'X + lambda P)^-1`
    (previously NULL); t, p and CI stay NULL.
  - WLS sigma is on the weighted scale, as `summary(lm(weights = w))$sigma`.
  - No-intercept models: R², adjusted R² and F use the uncentered TSS and a
    constant column (e.g. all ones) is kept as the intercept, computed upstream.
    Constant-column detection is scale-relative with tolerance 1e-10 (was 1e-12).
  - GLMs of every family drop an intercept-collinear constant column (NaN
    coefficient), as Poisson did; previously binomial/negbinom/gamma/tweedie/
    logistic could alias the intercept instead.
  - RLS fit statistics come from upstream (R² clamped to [0, 1]); RLS needs at
    least two rows.
- Dependency: `anofox-statistics` 0.4.4 -> 0.4.6: `mann_whitney_u_agg` and
  `wilcoxon_signed_rank_agg` no longer exhaust memory on large groups (the
  Hodges-Lehmann estimate and interval are computed as in R, O(n) memory; a group
  of 125,000 rows previously killed the process); one-sided tests report
  one-sided intervals as R does; Kendall, distance correlation, MMD and the 1-D
  energy test use O(n) memory; `mcnemar_agg` exact p-values no longer become NaN
  for large discordant counts.
- Dependency: `anofox-statistics` 0.4.3 -> 0.4.4: no panics on non-finite input;
  ±Inf are ranked like R in rank tests; NaN input and non-finite data in
  permutation / energy / MMD / forecast tests are reported as a "non-finite
  value" error instead of a fabricated p-value.
- Dependency: `anofox-statistics` 0.4.2 -> 0.4.6. The extension no longer carries
  its own implementations of statistical methods; it delegates to the library:
  Shapiro-Wilk (R `swilk.c` port), Kendall's tie-corrected Var(S), the exact
  binomial test with Clopper-Pearson interval, the Wilson / two-sample proportion
  intervals at the requested confidence level, Fisher's conditional MLE and exact
  conditional interval (`fisher_exact_conditional`) and the Woolf interval at the
  requested level, Mann-Whitney / Wilcoxon p-values at the centre and with all
  values tied, the rank-biserial correlation, Cohen's d for `t_test`, Jarque-Bera
  and the percentile bootstrap of the mean. Results are unchanged.
- Non-finite input to `jarque_bera` and the Pearson / Spearman / Kendall
  correlations is treated as degenerate data (NULL result) instead of raising an
  error (Jarque-Bera previously returned NaN).

- **Breaking:** every regression-family function (aggregates, table/scalar
  fits, window functions, fit-predict aggregates) declares the option keys it
  reads; any other key raises an error naming the function and listing its
  supported keys. Keys that were silently ignored before, e.g.
  `ols_fit_agg(y, x, {'alpha': 1})` or `compute_inference` on fit-predict
  functions, now error. The same holds for the hypothesis-test aggregates;
  `mmd_agg` rejects `bandwidth`/`sigma` (never used) and `tost_t_test_agg`
  rejects `alternative`, `paired` and `mu` (never used).
- **Breaking:** options must be a constant expression; a per-row expression
  used to be skipped silently and now errors.
- **Breaking:** `confidence_level` must lie strictly inside (0, 1) everywhere;
  integer options reject negative, fractional and overflowing values instead of
  truncating them.
- `lars_fit_agg`: `alpha` > 0 without a `method` now selects the LassoLars path
  (plain LAR ignores alpha, so it used to have no effect); `{'method': 'lar',
  'alpha': > 0}` is an error.
- `fisher_exact_agg` returns a result for any non-empty table (n < 4 used to
  return NULL).
- RANSAC and Theil-Sen (aggregates, fit-predict aggregates and window
  functions) fit on a canonical row order, so a given `random_state` yields the
  same model whatever the thread count or input order. Results can differ from
  earlier versions for the same seed.
- Prediction intervals of all `*_fit_predict_agg` and window `*_fit_predict` functions are
  leverage-aware: `yhat ± t·s·sqrt(1 + x₀ᵀMx₀)` with M = (XᵀX)⁻¹ (OLS), (XᵀWX)⁻¹ (WLS),
  the ridge sandwich (ridge), and the OLS leverage of the training rows as an
  approximation for the robust and sparse estimators. The old `s·sqrt(1 + 1/n)` was only
  correct at the centroid of x and too narrow when extrapolating. Bounds are NULL (not
  `yhat`) when no interval can be formed, e.g. zero residual degrees of freedom.
- Window `*_fit_predict` functions are documented as predicting the **last row of the
  frame** (SQL window aggregates cannot see the current row); use frames ending at
  `CURRENT ROW`, or `predict(*_fit_agg(...) OVER (...), x)` for one-step-ahead forecasts.
- The dummy `unused` parameter of `glmm_fit_by` / `eb_shrink_by` was removed.
- **Error policy (behaviour change).** All aggregates, scalar, table and window
  functions now treat FFI failures the same way: invalid input or options
  (bad option values, mismatched lengths, out-of-domain values such as negative
  Poisson responses or non-positive AFT times, expected probabilities that do
  not sum to 1) raise an `Invalid Input Error` naming the function; data that is
  valid but degenerate (too few rows, no valid rows, zero variance, a single
  group, a singular design, a numerically broken fit) returns `NULL`. Previously
  most aggregates returned `NULL` for both, while `ols_fit`, `ridge_fit`,
  `wls_fit`, `elasticnet_fit` & co. raised on insufficient data; those now
  return `NULL` like their `_agg` counterparts.
- GLM (and AFT) non-convergence is no longer an error: the fit is returned with
  `converged = false`.
- `ridge_fit` / `ridge_fit_agg` with `alpha > 0` return `NULL` for
  `std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper`, `f_statistic`
  and `f_pvalue`: classical tests on shrunken coefficients are not valid.
  `alpha = 0` keeps the OLS inference.
- OLS with a requested `hc_type` no longer silently falls back to classical
  standard errors when the HC estimator is unavailable; the coefficient
  inference is NaN instead.
- RANSAC inference (`compute_inference`) is the OLS inference of the final
  inlier-only refit and ignores the inlier selection, so it is optimistic.
  Values are unchanged; treat them as descriptive.
- `negbinom_fit_agg` estimates theta by maximum likelihood like
  `MASS::glm.nb` (was a moment estimate); results now match R.
- `rls_fit_agg`, `rls_fit_predict_agg`, `diebold_mariano_agg`, `clark_west_agg`,
  `aid_agg` and `aid_anomaly_agg` declare themselves order dependent; use
  `agg(... ORDER BY t)` for a deterministic sequential result.
- License metadata is consistent with `LICENSE`: BSL 1.1 that converts to MPL 2.0
  five years after each version is first published. The Cargo workspace license
  is now `BUSL-1.1`.
- **Behaviour change:** R², adjusted R² and the F statistic of `ols_fit_agg` /
  `wls_fit_agg` (and their scalar/window variants) fitted **without an intercept** now use
  R's `summary.lm` convention: the uncentered total sum of squares (`R² = 1 - RSS/Σw·y²`,
  adjusted with `n/(n - p)`, F on `p` and `n - p` df). Previously the centered TSS was
  used, which compares the model with an intercept-only model it does not nest. Other
  estimators (ridge, elastic net, ...) are unchanged.

### Deprecated

- Hypothesis-test fields superseded by the unified `test` fields (#159): `ci_lower` /
  `ci_upper` (use `conf_low` / `conf_high`), `df` (use `df1`), and in
  `one_way_anova_agg` `f_statistic` (use `statistic`), `df_between` / `df_within` (use
  `df1` / `df2`). The aliases are removed in the next minor release.

- `ci_lower` / `ci_upper` in the model structs: use `conf_low` / `conf_high`. The aliases are removed in the next minor release.

### Fixed

- `one_way_anova_agg`, `kruskal_wallis_agg` and `brown_forsythe_agg` accept any
  group label (integer or VARCHAR), as the two-sample and post-hoc tests do;
  they used to bind only INTEGER group ids (#186).
- `<model>_tidy_by` / `_rolling_tidy_by` kept inference only while `options` was
  omitted; `compute_inference` is now merged under the user's options.
- `quantile_fit_agg` (and `quantile_fit_predict_agg`) now honours
  `fit_intercept = false` (previously ignored: it fitted with an intercept and
  dropped it).
- `t_test`, `one_way_anova`, `brown_forsythe` and `yuen` on constant data no longer
  abort with an internal error (fixed upstream in `anofox-statistics` 0.4.3); they
  return a NaN p-value.
- GLM weighted least-squares solve (`solve_weighted_ls_qr`) built the full n x n
  `Q` of the QR factorisation, which exhausted memory on large inputs; it now uses
  the thin factorisation.
- Hypothesis-test aggregates with hand-written option parsing (yuen, permutation
  t-test, TOST paired/correlation, Diebold-Mariano, Clark-West, binomial and
  proportion tests, ICC, McNemar, Cohen's kappa, distance correlation) ignored
  STRUCT literals `{'k': v}` and silently fell back to defaults on invalid enum
  values (e.g. `alternative: 'sideways'`); they now accept MAP and STRUCT, and
  reject unknown keys and invalid values.
- `MAP {...}` options to the shared test-option parsers (`t_test_agg`,
  `mann_whitney_u_agg`, ...) failed with "Invalid MAP structure".
- `binomial_fit_agg(..., {'link': 'probit'})` failed with a Poisson-link error;
  `quantile_fit_predict_agg` ignored `quantile`; GLMs ignored `lambda`.
- `t_test_agg` silently ignored `paired: true` (it has no pairing information);
  it now raises an error pointing to `tost_paired_agg` /
  `wilcoxon_signed_rank_agg`.
- `fisher_exact_agg` silently dropped values other than 0/1; it now errors.
- Seeded tests and RANSAC/Theil-Sen gave different results depending on the
  number of threads.
- Statistical accuracy (all checked against R / scipy):
  - `shapiro_wilk_agg`: the p-value for n <= 11 used a wrong approximation (0.070 vs R's
    0.161 on a 10-value sample); W and p are now a port of R's `swilk.c` (Royston 1995).
  - `kendall_agg`: the z statistic / p-value now use the tie-corrected variance of S
    (R `cor.test(method = "kendall", exact = FALSE)`).
  - `brunner_munzel_agg`: the confidence interval was a ~5% interval (the confidence
    level was passed where alpha is expected); it now matches lawstat.
  - `binom_test_agg`: exact Clopper-Pearson interval (was an approximate beta quantile)
    and R's relative tolerance in the two-sided p-value (was floored for large n);
    one-sided alternatives give one-sided intervals, as R `binom.test`.
  - `prop_test_one_agg` / `prop_test_two_agg`: intervals use `qnorm` instead of a
    hard-coded 1.96; `prop_test_two_agg` with the (default) continuity correction reports
    R's continuity-corrected interval; one-sided alternatives give one-sided intervals.
  - `glmm_fit_agg` (poisson / binomial): `log_likelihood`, `aic`, `bic` include the
    saturated-model term, matching lme4 `logLik(glmer)` (was `-deviance/2`).
  - `chisq_test_agg`: category codes were used as table indices, so codes not starting
    at 0 added empty rows/columns (wrong df and statistic) and negative codes were merged
    into 0; codes are now compacted to the observed levels. A 1 x K table (df = 0)
    returns NULL, like `g_test_agg` (R's `chisq.test` would switch to a goodness-of-fit
    test; use `chisq_gof_agg`).
  - `mann_whitney_u_agg`: all observations tied returned p = 0; now p = 1. Two-sided
    `mann_whitney_u_agg` / `wilcoxon_signed_rank_agg` apply no continuity correction when
    the statistic equals its null expectation (R: `sign(z) * 0.5`), so p = 1 there.
  - `aft_quantile`: `p` is no longer clamped to [1e-12, 1 - 1e-12]; p = 0 gives 0, p = 1
    gives +Inf, p outside [0, 1] or a non-positive scale gives NaN.
  - `ols_fit_agg`, `bls_fit_agg` and the window `*_fit_predict` functions counted constant
    (e.g. all-zero) feature columns in their minimum-row check and returned NULL for fits
    the core supports (aliased columns get NaN coefficients).
  - `wls_fit_agg` intercept-only fits (all features constant) divided the residual
    variance by n instead of n - 1 (R `lm(y ~ 1, weights = w)`), giving too-narrow
    prediction intervals.
- Window frames larger than a few rows silently fitted on too few rows: aggregate
  Combine moved data out of segment-tree nodes that DuckDB reuses (it now copies unless
  DuckDB allows destructive combines).
- A NULL element inside the `x` list (e.g. `[1.0, NULL]`) was read as an arbitrary value
  by 21 aggregates/window functions; such rows are now skipped.
- `t_test_agg`, `mann_whitney_u_agg`, `brunner_munzel_agg`, `yuen_agg`,
  `permutation_t_test_agg` treated group 0 as group 1 and every other value as group 2:
  data coded 1/2 returned NULL and a third group was merged silently. Any two labels now
  work (the smaller is group 1); a third label is an error.
- `phi_coefficient_agg` could flip sign depending on row order (parallel execution).
- Aggregate states no longer contain self-referential members (a crash risk because
  DuckDB relocates states with memcpy): `glmm_fit_agg` group index.
- Rust FFI: every exported function is guarded so a Rust panic can no longer abort
  DuckDB; all feature columns are length-checked; constant columns are kept when there
  is no intercept; RLS covariance update read already-overwritten rows (wrong
  coefficients with more than one parameter); Fisher exact honours `confidence_level`;
  Mann-Whitney reports the rank-biserial correlation as `effect_size`.
- `*_fit_predict_agg` dropped rows whose whole `x` list is NULL, so the output
  LIST was shorter than the group. Such rows are now kept with `yhat` (and
  bounds) NULL and `is_training = false`.
- `rls_fit` / `rls_fit_agg` reported NaN `r_squared`, `adj_r_squared` and
  `residual_std_error`; they are now computed (and match OLS for
  `forgetting_factor = 1` with a diffuse `initial_p_diagonal`).
- `huber_fit_agg` / `huber_fit` with `compute_inference` returned empty
  inference lists; they now carry the Huber asymptotic standard errors
  (as `MASS` `summary.rlm`, `method = "XtX"`).
- `negbinom_fit_agg` returned NULL on ordinary count data and negative
  deviances on zero-heavy data (wrong sign of the `y = 0` unit deviance).
- `residuals_diagnostics_agg` (and `residuals_diagnostics` with `x` but no
  `residual_std_error`) returned NULL standardized and studentized residuals;
  sigma is now estimated as `sqrt(RSS / (n - k - 1))`.
- `t_test_agg` returned `effect_size = NaN`; it is now Cohen's d (pooled SD for
  Student, average-variance standardiser for Welch, `d_z` for paired).
- `pearson_agg` on a constant column raised an internal panic error; it now
  returns NULL.
- Documentation: removed the stale `anofox_stats_` prefixes, the wrong calling
  conventions, the wrong option keys and field names, and the broken links.

## [0.10.0] - 2026-09-02

### Changed (breaking)

- All SQL functions are registered under their unprefixed names only. The
  `anofox_stats_` prefix and the deprecated aliases are gone. See
  [docs/MIGRATION.md](docs/MIGRATION.md).
- `theilsen_*` was renamed to `theil_sen_*`. The `*_predict_agg` aliases were
  removed; use `*_fit_predict_agg`.
- Unknown option keys in an options MAP now raise an error. Before, they were
  silently ignored.

### Added

- A benchmark harness (`scripts/bench.sh`, `bench/`).
- Doc-SQL validation (`scripts/validate_docs_sql.py`) and a CI gate that runs it.
- `docs/API_CONVENTIONS.md`, which describes naming, option keys and return fields.
- WASM support: a load fix, a DuckDB-Wasm test harness and a CI gate.
- GLM results report the curvature (observed information), with a mapping for its rows.

### Fixed

- `ols_fit_agg` used as a window function returns NULL for degenerate frames,
  instead of a saturated fit with NaN statistics.
- Errors name the SQL function that raised them, and numerical failures use a
  distinct error class.

### Performance

- Refactored the FFI marshalling layer, which reduces per-group allocation
  overhead (see `bench/PROFILING.md`).

## [0.9.0] - 2026-08-08

### Added

- MCMC convergence diagnostics: split R-hat and effective sample size.
- AFT models report the full covariance matrix and the observed information.
- Fit errors include a feedback banner and links for reporting issues.

## [0.8.1] - 2026-08-05

### Fixed

- macOS and WASM builds (FFI index type mismatch).

## [0.8.0] - 2026-08-04

### Added

- Hierarchical GLMs (`glmm_fit_agg`, `glmm_fit_by`) with random slopes and
  crossed or nested factors.
- Censored likelihoods: AFT survival models (`aft_fit_agg`, `aft_cdf`, `aft_quantile`).
- Explicit coefficient priors with Laplace intervals.
- Empirical-Bayes shrinkage (`eb_shrink_agg`, `eb_shrink_by`).
- GLM offsets, NULL list elements, and a `converged` field in GLM results.

## [0.7.3] - 2026-07-25

### Changed

- Built against DuckDB v1.5.5, keeping v1.4 LTS support.

## [0.7.2.1] - 2026-07-18

### Changed

- Updated the telemetry dependency.

## [0.7.1] - 2026-07-13

### Changed

- Adopted telemetry schema 2.

### Fixed

- The Rust FFI archive is linked into the WASM build.
- A crash at telemetry teardown.

## [0.7.0] - 2026-06-25

### Added

- LARS / LassoLars regression (`lars_fit_agg`).
- Binary logistic regression (`logistic_fit_agg`).
- GLM aggregates: binomial, negative binomial, gamma and Tweedie.

### Fixed

- An uninitialized-memory read on NULL features in the fit-predict aggregates.

## [0.6.1] - 2026-06-18

### Fixed

- CI: the extension-ci-tools version used for deploys is pinned.

## [0.6.0] - 2026-06-11

### Added

- Robust regression: Huber, RANSAC and Theil-Sen, each with scalar, aggregate,
  window, fit-predict and `*_by` macro forms.
- Publication to the erpl.io distribution bucket.

### Changed

- Upgraded DuckDB to v1.5.3.

## [0.5.3] - 2026-03-10

### Added

- `solver`, `hc_type`, `lambda_scaling` and `glm_lambda` options.

### Changed

- SVD is the default decomposition.
- Upgraded to DuckDB 1.5.0.

## [0.5.2] - 2026-02-13

### Added

- `split` column support in the `*_fit_predict_agg` functions and the `*_fit_predict_by` macros.
- Passthrough columns in the `*_fit_predict_by` macros.

## [0.5.1] - 2026-01-30

### Added

- `aid_by` and `aid_anomaly_by` table macros.

### Changed

- Updated DuckDB from v1.4.3 to v1.4.4.

## [0.5.0] - 2026-01-16

### Added

- PLS, isotonic and quantile regression, as fit-predict aggregates and `*_by` macros.

### Fixed

- A segfault in `aid_anomaly_agg` on very large inputs.

## [0.4.0] - 2026-01-08

### Added

- `*_fit_predict_by` table macros for grouped regression.
- Fit-predict aggregates for BLS, ALM and Poisson.

### Fixed

- Zero-variance feature handling in all regression methods.
- Undefined behaviour in aggregate functions.

## [0.3.0] - 2025-12-19

### Added

- Statistical hypothesis tests: t-test, ANOVA, Mann-Whitney, correlation tests and more.
- Automatic Identification of Demand (AID).

## [0.2.0] - 2025-11-30

### Changed (breaking)

- Function prefix unified from `anofox_statistics_*` to `anofox_stats_*`.

## [0.1.0] - 2025-11-05

### Added

- Initial release: OLS, Ridge, WLS, RLS and Elastic Net regression as scalar
  and aggregate functions, plus diagnostics.

[Unreleased]: https://github.com/DataZooDE/anofox-statistics/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.12.2...v0.13.0
[0.12.2]: https://github.com/DataZooDE/anofox-statistics/compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.2.1...v0.7.3
[0.7.2.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.1...v0.7.2.1
[0.7.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DataZooDE/anofox-statistics/releases/tag/v0.1.0
