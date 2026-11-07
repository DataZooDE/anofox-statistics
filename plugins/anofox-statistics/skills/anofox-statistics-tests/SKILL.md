---
name: anofox-statistics-tests
description: >
  Statistical hypothesis tests, post-hoc comparisons, correlation, effect
  sizes, equivalence testing, classifier evaluation, and time-series and
  survival tests in the anofox_statistics DuckDB extension. Covers the unified
  test-result fields (method, alternative, statistic, df1, df2, p_value,
  estimate, conf_low, conf_high, conf_level, effect_size, effect_name, n) and
  their deprecated aliases, normality tests (Shapiro-Wilk, Jarque-Bera,
  D'Agostino K2), parametric tests (two-sample, one-sample and paired t-tests
  with one-sided intervals, one-way, two-way Type III and repeated-measures
  ANOVA with Mauchly / Greenhouse-Geisser / Huynh-Feldt, Yuen,
  Brown-Forsythe), bootstrap confidence intervals of the mean (IID,
  stationary, circular and moving block), nonparametric tests (Mann-Whitney
  U, Kruskal-Wallis, Wilcoxon signed-rank with rank-biserial, Brunner-Munzel,
  permutation t-test), pairwise post-hoc tests (pairwise_t_test_agg,
  tukey_hsd_agg, dunn_test_agg and their _by macros, ptukey / qtukey),
  multiple-testing correction of any p-values (p_adjust_by, p_adjust_agg,
  p_adjust_list, p_adjust window function on DuckDB 2.0; R p.adjust),
  correlation (Pearson, Spearman, Kendall, partial and semi-partial,
  distance correlation, Spearman rho^2, Hoeffding's D, ICC with all six
  Shrout-Fleiss types), categorical / contingency tests (chi-squared,
  chi-squared GOF, G-test, Fisher's exact, McNemar), effect sizes (Cramér's V,
  phi, contingency coefficient, Cohen's kappa), proportion tests, TOST
  equivalence tests (t, paired, correlation, one-sample, one / two
  proportions, Wilcoxon, Yuen, bootstrap), distribution comparison (energy
  distance, MMD, multivariate energy / MMD permutation tests), evidently-style
  data drift (KS, Anderson-Darling, Cramér-von Mises, Epps-Singleton,
  Wasserstein, PSI, KL, Jensen-Shannon, Hellinger, TVD, z / chi-squared / G
  drift, column_drift_agg, dataset_drift_by, embedding_drift_agg,
  category_proportions_agg), NannyML-style model monitoring (CBPE
  performance estimation without labels: cbpe_fit_agg / cbpe_estimate_agg /
  cbpe_by; PCA reconstruction-error drift; alert_band_agg / alert_band_by
  reference-band thresholds), performance metrics (classification_metrics_agg,
  regression_metrics_agg, ranking_metrics_agg, data_quality_by), forecast-evaluation tests (Diebold-Mariano,
  Clark-West, Hansen SPA, MSPE-adjusted SPA, Model Confidence Set), residual
  autocorrelation and heteroskedasticity tests (Ljung-Box, Box-Pierce,
  Durbin-Watson with p-value, Breusch-Pagan, White), survival tests
  (Kaplan-Meier, log-rank with Peto-Peto rho and strata), causal mediation
  analysis (mediation_by / mediation_agg: ACME, ADE, total effect,
  proportion mediated), and classifier evaluation (ROC / PR / calibration
  curves, AUC, AUC-vs-chance p-values auc_null_p_value / auc_null_quantile,
  Brier score, log loss, roc_curve_by). Also: the nested-model F test
  (ols_anova_by), type I / II ANOVA tables (ols_anova_table_by), and seeded,
  bitwise-reproducible Monte Carlo, bootstrap and permutation tests. Use when
  running a significance test, comparing many groups or repeated conditions,
  testing equivalence, measuring association/effect size, evaluating a
  classifier or forecast models, detecting data or embedding drift, monitoring a deployed model, or
  comparing distributions or survival in SQL.
version: 0.17.0
user-invocable: false
---

# Anofox Statistics — Hypothesis Tests Cheat Sheet

**Extension:** `anofox_statistics` v0.17.0 | **DuckDB:** v1.5.6 / v2.0 | Tests are **aggregates** — use with `GROUP BY` for per-segment tests.

## Gotchas

1. **All tests are `_agg` aggregates.** They consume rows, not literal arrays. Wrap in `(test_agg(...)).field` to read result fields, or select the whole STRUCT.
2. **Two-sample tests take two value columns** `(x, y)` — one value per row in each. Paired tests expect aligned rows.
3. **Every test returns the unified fields** (below), always present and NULL where they do not apply, after the test's own fields (`n1`, `n2`, `ss_between`, …). Read `(r).p_value`, `(r).estimate`, `(r).conf_low` the same way for every test.
4. **Deprecated aliases.** `ci_lower` / `ci_upper` → `conf_low` / `conf_high`; `df` → `df1`; ANOVA `f_statistic` → `statistic`, `df_between` / `df_within` → `df1` / `df2`. They still work but are removed in the next minor release; write the new names.
5. **Correlation has scalar + agg forms for some** (`spearman`, `icc` have both); most others are agg-only.
6. **Group arguments are long format.** Two-sample tests take `t_test_agg(value, group)` with exactly two group labels (BIGINT or VARCHAR); `estimate` is the difference `group1 - group2`. The k-sample tests take `one_way_anova_agg(value DOUBLE, group BIGINT | VARCHAR)` (likewise `kruskal_wallis_agg`, `brown_forsythe_agg`), as do the post-hoc tests. Paired tests (`t_test_paired_agg`, `wilcoxon_signed_rank_agg`, `tost_paired_agg`, `tost_wilcoxon_paired_agg`) take two columns of one row per pair instead; repeated-measures ANOVA takes long `(y, subject, condition)`.
7. **Order-sensitive aggregates need their own `ORDER BY`, after the constant arguments**: `ljung_box_agg(resid, 10 ORDER BY t)`, `durbin_watson_agg(resid, [x1, x2] ORDER BY t)`. In `ljung_box_agg(resid ORDER BY t, 10)` the 10 becomes a second sort key. The options STRUCT too: `spa_test_agg(b, [l1, l2], {'n_bootstrap': 499} ORDER BY t)`; in `spa_test_agg(b, [l1, l2] ORDER BY t, {'n_bootstrap': 499})` the options are silently ignored (the result shows the default `n_bootstrap` 1000). Applies to `acf_agg`, `pacf_agg`, `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg`, and to the block bootstraps (`bootstrap_mean_ci_agg`, `spa_test_agg`, `mspe_adjusted_spa_agg`, `model_confidence_set_agg`).
8. **Behaviour changes in 0.16.0.** One-sided t-tests (`t_test_agg`, `t_test_one_sample_agg`, `t_test_paired_agg` with `'alternative': 'less'` / `'greater'`) now return R's one-sided interval with an infinite bound (`conf_low = -inf` or `conf_high = inf`); statistic and p-value are unchanged. `mmd_test_agg`'s gaussian kernel now defaults to the median-heuristic bandwidth (was 1; pass `{'bandwidth': 1.0}` for the old statistic). `spa_test_agg` / `mspe_adjusted_spa_agg` give all p-values = 1 when no model beats the benchmark. `wilcoxon_signed_rank_agg` has a new `rank_biserial` field (positional STRUCT readers see one more field).
9. **The new TOST variants require bounds.** `tost_one_sample_agg`, `tost_prop_one_agg`, `tost_prop_two_agg`, `tost_wilcoxon_paired_agg`, `tost_wilcoxon_agg`, `tost_yuen_agg`, `tost_bootstrap_agg` raise an error without `delta` or both `bound_lower` / `bound_upper`. The older `tost_t_test_agg` / `tost_paired_agg` / `tost_correlation_agg` still default to [-1, 1]; always set the bounds explicitly anyway, from subject-matter reasoning, before seeing the data.

## Unified result fields

| Field | Meaning |
|---|---|
| `method` | Name of the test |
| `alternative` | `'two_sided'`, `'less'`, `'greater'` (NULL for omnibus tests) |
| `statistic` | Test statistic |
| `df1`, `df2` | (Numerator / denominator) degrees of freedom |
| `p_value` | p-value |
| `estimate`, `conf_low`, `conf_high`, `conf_level` | Point estimate (difference, odds ratio, correlation, …) and its interval |
| `effect_size`, `effect_name` | Effect size and its name (`cohens_d`, `rank_biserial`, `eta_squared`, `r`, `rho`, `tau`, `kappa`, …) |
| `n` | Number of observations |

## Test catalog

### Normality
| Function | Test |
|---|---|
| `shapiro_wilk_agg` | Shapiro-Wilk |
| `jarque_bera_agg` / `jarque_bera` | Jarque-Bera (agg + scalar) |
| `dagostino_k2_agg` | D'Agostino K² |

### Parametric (location / scale)
| Function | Test |
|---|---|
| `t_test_agg(value, group[, opts])` | Two-sample t-test (Welch default, `kind` `'student'`); `alternative`, `confidence_level`, `mu` |
| `t_test_one_sample_agg(x[, opts])` | One-sample t-test, R `t.test(x, mu =)`; `estimate` = mean, `effect_size` Cohen's d |
| `t_test_paired_agg(x, y[, opts])` | Paired t-test on `x - y` (wide format, one row per pair); `effect_size` Cohen's d_z |
| `one_way_anova_agg` | One-way ANOVA |
| `two_way_anova_agg(y, a, b[, {'terms': ['a', 'b']}])` | Factorial ANOVA, Type III SS (`car::Anova(type = 3)`, sum contrasts): `LIST` of 3 rows (A, B, A:B) with `term`, F, `df1`, `df2`, `sum_sq`, partial eta² as `effect_size`. Every A x B cell must be observed (else NULL) |
| `repeated_measures_anova_agg(y, subject, condition[, {'sphericity': true}])` | Within-subjects F (`aov(y ~ cond + Error(subj/cond))`), Mauchly (`mauchly_w`, `mauchly_p_value`), `gg_epsilon` / `gg_p_value`, `hf_epsilon` / `hf_p_value`; long format, subjects missing a condition excluded (`n_excluded_subjects`); duplicates are an error |
| `yuen_agg` | Yuen's trimmed-mean t-test (robust) |
| `brown_forsythe_agg` | Brown-Forsythe equality-of-variance |
| `bootstrap_mean_ci_agg(x[, opts] [ORDER BY t])` | Percentile bootstrap CI of the mean (not a test): `method` `'iid'` (default) / `'stationary'` / `'circular_block'` / `'moving_block'` (`block_length` required for the block methods; `boot::tsboot`), `n_bootstrap` (10000), `confidence_level`, `seed` (42). Returns `estimate, std_error, conf_low, conf_high, …` |

Repeated measures: read `p_value` when Mauchly does not reject, else `gg_p_value` (or `hf_p_value` when `gg_epsilon` > 0.75). For a dependent series use a block bootstrap, the IID interval is too narrow.

### Nonparametric
| Function | Test |
|---|---|
| `mann_whitney_u_agg` | Mann-Whitney U (rank-sum) |
| `kruskal_wallis_agg` | Kruskal-Wallis |
| `wilcoxon_signed_rank_agg(x, y)` | Wilcoxon signed-rank (paired); `rank_biserial` = `effect_size`, positive when `y` tends to be larger (opposite sign to R `effectsize`) |
| `brunner_munzel_agg` | Brunner-Munzel |
| `permutation_t_test_agg` | Permutation t-test; `n_permutations` (10000), `seed` (default 42, so the p-value is reproducible) |

### Correlation
| Function | Measure |
|---|---|
| `pearson_agg` | Pearson r |
| `spearman_agg` / `spearman` | Spearman ρ |
| `kendall_agg` | Kendall τ |
| `partial_cor_agg(x, y, controls DOUBLE[][, {'method': 'spearman'}])` | Partial correlation (controls removed from both), t test on n − 2 − k df; R `ppcor::pcor.test`. No CI |
| `semi_partial_cor_agg(x, y, controls DOUBLE[][, opts])` | Semi-partial (part) correlation: controls removed from `y` only (jointly); `ppcor::spcor.test` |
| `distance_cor_agg` | Distance correlation; permutation test with `n_permutations` (1000), `seed` (default 42) |
| `spearman2_agg(y, x[, {'p': 2}])` | Generalized Spearman ρ² (`Hmisc::spearman2`), quadratic in ranks or categorical (VARCHAR `x`); predictor screening |
| `hoeffding_d_agg(x, y)` | Hoeffding's D (any form of dependence, O(n²)) |
| `icc_agg(value, subject, rater[, {'type': …}])` / `icc` | Intraclass correlation, `psych::ICC` types `icc1` (default), `icc2`, `icc3`, `icc1k`, `icc2k`, `icc3k` (`single` / `average` = `icc1` / `icc1k`). Complete subject x rater data only |

ICC choice: `icc1` = each subject rated by different random raters; `icc2` = same raters, a sample of raters, absolute agreement (rater bias counts); `icc3` = these raters only, consistency; the `k` types are the reliability of the mean of k ratings. `spearman2_agg` / `hoeffding_d_agg` are documented in docs/api/statistics/reduction.md.

### Categorical / contingency
| Function | Test |
|---|---|
| `chisq_test_agg` | Chi-squared independence |
| `chisq_gof_agg` | Chi-squared goodness-of-fit |
| `g_test_agg` | G-test (likelihood ratio) |
| `fisher_exact_agg` | Fisher's exact |
| `mcnemar_agg` | McNemar (paired binary) |

### Effect size / association
| Function | Measure |
|---|---|
| `cramers_v_agg` | Cramér's V |
| `phi_coefficient_agg` | Phi coefficient |
| `contingency_coef_agg` | Contingency coefficient |
| `cohen_kappa_agg` | Cohen's kappa (inter-rater) |

### Proportion
| Function | Test |
|---|---|
| `prop_test_one_agg` | One-proportion z-test |
| `prop_test_two_agg` | Two-proportion test (R `prop.test`); group 1 is `group_id = 0`; `estimate` = p₁ − p₂ (the CI's quantity) |
| `binom_test_agg` | Exact binomial test |

### Equivalence (TOST)
| Function | Test |
|---|---|
| `tost_t_test_agg(value, group[, opts])` | Two one-sided t-tests (Welch / `kind` `'student'`) |
| `tost_paired_agg` | Paired TOST |
| `tost_correlation_agg` | Correlation TOST |
| `tost_one_sample_agg(value, opts)` | One-sample t TOST of `mean - mu` (`TOSTER::TOSTone`); `mu` |
| `tost_prop_one_agg(success, opts)` | One proportion vs `p0` (0.5), z tests |
| `tost_prop_two_agg(success, group, opts)` | `p1 - p2`, unpooled z tests |
| `tost_wilcoxon_paired_agg(x, y, opts)` / `tost_wilcoxon_agg(value, group, opts)` | Rank-based; estimate = Hodges-Lehmann shift |
| `tost_yuen_agg(value, group, opts)` | Trimmed means (`trim` 0.2), robust to outliers |
| `tost_bootstrap_agg(value, group, opts)` | Percentile bootstrap of the mean difference; `n_bootstrap` (10000), `seed` (42), row-order independent |

All return the `tost_t_test_agg` STRUCT: `t_lower`, `t_upper`, `p_lower`, `p_upper`, `p_value` (the larger), `estimate` with its (1 − 2α) interval `ci_lower` / `ci_upper`, `equivalent` (BOOLEAN), bounds, `n`, plus the unified fields. Options: `delta` (symmetric) or `bound_lower` + `bound_upper` (raw units; proportions in [-1, 1]), `alpha` (0.05) or `confidence_level`. Two-group variants: group 1 is the smaller of exactly two BIGINT / VARCHAR labels. A non-significant t-test is not evidence of equivalence; TOST is.

### Distribution comparison
| Function | Measure |
|---|---|
| `energy_distance_agg` | Energy distance; `n_permutations` (1000), `seed` (default 42) |
| `mmd_agg` | Maximum mean discrepancy; `n_permutations` (1000), `seed` (default 42) |
| `energy_distance_test_agg(x DOUBLE[], group[, opts])` | Multivariate energy-distance permutation test (Székely-Rizzo); one vector per row |
| `mmd_test_agg(x DOUBLE[], group[, opts])` | Multivariate unbiased MMD² permutation test; `kernel` `gaussian` (default) / `laplacian` / `linear` / `polynomial`; `bandwidth` default `'median'` heuristic for gaussian (**changed**, was 1), 1 for laplacian; result has the `bandwidth` used |

The `_test_agg` forms: all vectors of a call must have the same length; group 1 (`n1`) is the smaller label; cost O(n_permutations · N² · dim), so lower `n_permutations` on large N.

### Data drift (evidently's stattests)
`(value, group [, options])`; the reference is the smaller group label unless `reference` names it.
| Function | Measure |
|---|---|
| `ks_test_agg` | Two-sample KS (R `ks.test`): exact p when n1·n2 < 10000, else limiting distribution; `alternative`, `exact` |
| `ad_test_agg` | k-sample Anderson-Darling (kSamples, asymptotic p, not capped like scipy); `midrank` (true) |
| `cvm_test_agg` | Two-sample Cramér-von Mises; exact when max(n1, n2) ≤ 20; `method` |
| `epps_singleton_agg` | Epps-Singleton ECF test (works for discrete data); `t` ([0.4, 0.8]) |
| `wasserstein_agg` | `distance` W_p and evidently's `distance_normed` = W_p / population SD of the reference; `p` (1), `n_permutations` (0) |
| `psi_agg` / `kl_divergence_agg` / `jensen_shannon_agg` / `hellinger_agg` | Binned divergences (DOUBLE or VARCHAR values); `binning` auto (evidently: Sturges if > 20 distinct reference values, else per value) / sturges / quantile (`n_bins`) / categorical; `drift` = value ≥ `threshold` (0.1); `epsilon` = constant zero fill (PSI, KL); KL is asymmetric (reference = p); JS `distance` is √divergence |
| `tvd_agg` | Total variation distance of categories + permutation p (1000, seed 42); weaker than `chisq_test_agg` |
| `z_drift_agg` / `chisq_drift_agg` / `g_drift_agg` | evidently's frequency tests (numeric or VARCHAR categories): pooled two-proportion z; GOF of current counts against reference shares (not `chisq_test_agg`'s two-sample test) |
| `category_proportions_agg(category, group)` | `categories`, `p_reference`, `p_current`: which categories drive the drift (not a test) |
| `column_drift_agg(value, group[, {'test': 'auto' \| <evidently name>, 'threshold': …}])` | evidently's per-column drift with its default test choice (`ks` / `chisquare` / `z` for ≤ 1000 reference rows, else `wasserstein` / `jensenshannon`) and drift rule: `STRUCT(test, score, is_p_value, threshold, drifted, n1, n2)` |
| `dataset_drift_by(source, group_col, ['c1', 'c2', …][, options][, drift_share := 0.5])` / `dataset_drift_agg(drifted[, drift_share])` | One row per column plus evidently's dataset verdict (share of drifted columns ≥ `drift_share`) |
| `embedding_drift_agg(embedding DOUBLE[], group \| is_reference[, opts])` | evidently embedding drift: `method` `'distance'` (default, `metric` euclidean / cosine / cityblock / chebyshev, threshold 0.2), `'ratio'` (share of drifted dimensions), `'mmd'` (median bandwidth); `'bootstrap': true` for a bootstrapped threshold. Returns `method, score, threshold, drifted, n1, n2, dim, bandwidth` |

Use `column_drift_agg` / `dataset_drift_by` for monitoring dashboards (evidently-compatible verdicts); use the dedicated tests above when you need the full statistic. Mind multiple testing when many columns are tested.

### Model monitoring and performance metrics (new in 0.17.0)
NannyML / evidently-style monitoring over **chunks** (e.g. weeks). The reference period is a positional BOOLEAN expression (`is_reference`, e.g. `period = 'reference'`), not an option. Fitted models (CBPE calibrator, reconstruction PCA) are STRUCTs you can store in a table and reuse on later chunks.

| Function | What it does |
|---|---|
| `cbpe_fit_agg(y_true, score[, prediction], {'calibration': 'isotonic' \| 'platt' \| 'none', 'threshold': 0.5})` | Calibrates a binary classifier's scores on the labelled reference; returns the model STRUCT (one-class reference → NULL) |
| `cbpe_estimate_agg(model, score[, prediction])` / `cbpe_realized_agg(model, y_true, score[, prediction])` | Estimated (no labels) / realized ROC AUC, F1, precision, recall, specificity, accuracy, confusion counts; each `STRUCT(value, sampling_error, lower_confidence, upper_confidence)` |
| `cbpe_by(source, chunk_col, y_col, score_col, is_reference)` | Long format: one row per chunk × metric with `estimate`, bands and `realized` |
| `pca_reconstruction_fit_agg(x DOUBLE[][, {'variance': 0.65 \| 'n_components': k, 'scale': true}])`, `pca_reconstruction_chunk_agg(model, x)`, `pca_reconstruction_drift_by(source, chunk_col, x_cols, is_reference)` | Multivariate drift by mean PCA reconstruction error per chunk vs the reference band; NULL elements of `x` are imputed with the reference mean (NannyML), unlike `pca_agg` |
| `alert_band_agg(value[, {'method': 'std', 'k': 3, 'lower_limit': 0, 'upper_limit': 1}])` / `alert_band_by(source, chunk_col, metric_col, is_reference)` | Alert thresholds for any chunked metric (drift score, PSI, CBPE estimate): `std` (mean ± k·SD of reference chunks), `constant`, `relative`; value limits; `alert` = strictly outside the band |
| `classification_metrics_agg(y_true, y_pred[, {'average': 'macro' \| 'micro' \| 'weighted' \| 'binary', 'pos_label', 'threshold'}])` | Accuracy, precision, recall, F1, specificity, balanced accuracy, MCC, Cohen's kappa, `labels` and `confusion` (rows = true). Default `average` is **macro** (not scikit-learn's binary). A DOUBLE score needs an explicit `threshold` |
| `regression_metrics_agg(y_true, y_pred)` | MAE, MSE, RMSE, MAPE / sMAPE (fractions), median AE, R², explained variance, max error |
| `ranking_metrics_agg(query_id, relevance, score[, {'k': 10, 'min_relevance', 'include_no_feedback'}])` | NDCG@k (scikit-learn), MAP@k, MRR, precision@k, recall@k, hit rate, averaged over queries; no relevant item anywhere → NULL |
| `data_quality_by(source, ['c1', …], reference := 'train')` | Per column: missing share, distinct count, min / max / mean / sd; vs the reference: out-of-range share, new / dropped categories |

The bands from CBPE and reconstruction drift are sampling-error bands; the alert decision uses the thresholds from the reference chunks. Many chunks × metrics also means many alerts by chance.

### Forecast evaluation
| Function | Test |
|---|---|
| `diebold_mariano_agg` | Diebold-Mariano (equal predictive accuracy) |
| `clark_west_agg` | Clark-West (nested-model forecast comparison) |
| `spa_test_agg` | Hansen SPA: does any of K models (`model_losses DOUBLE[]`) beat the benchmark loss? (`ORDER BY t`) |
| `mspe_adjusted_spa_agg` | MSPE-adjusted SPA on forecast errors, benchmark nested in the alternatives |
| `model_confidence_set_agg` | Model Confidence Set of K loss series: surviving set, elimination order, per-model MCS p-values (`{'method': 'moving_block', 'block_length': k}` = R `MCSprocedure(k)`) |

SPA / MCS take **losses** (MSPE-adjusted SPA takes **errors**), one row per time point, and need `ORDER BY t`. SPA returns `p_value` (= `p_value_consistent`, Hansen's choice), `p_value_lower`, `p_value_upper` (White's Reality Check) and `best_model` (1-based). MCS returns `included_models`, `eliminated_models`, `mcs_p_values`, `elimination_order` (best last); options `alpha` (0.1; R's default is 0.15), `statistic` `range` / `max`, `method` `stationary` (default) / `moving_block` / `circular_block` / `iid`. All share `n_bootstrap` (1000), `block_length` (T^(1/3)), `seed` (42); p-values agree with R within Monte Carlo error only.

### Pairwise post-hoc (after an omnibus test)
| Function | Test |
|---|---|
| `pairwise_t_test_agg(value, group[, opts])` | R `pairwise.t.test`; `p_adjust` (default `'holm'`), `pool_sd` (true), `alternative`, `confidence_level` (0.95). `conf_low`/`conf_high`: CI of each difference, per comparison for `'none'`, Bonferroni otherwise (as `emmeans`) |
| `tukey_hsd_agg(value, group[, opts])` | R `TukeyHSD(aov(...))`; `confidence_level` |
| `dunn_test_agg(value, group[, opts])` | `FSA::dunnTest` (after Kruskal-Wallis); `p_adjust` |
| `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by` `(source, group_col, value_col, factor_col, options := NULL)` | The same per group of `source`, as the long `test` table |
| `ptukey(q, n_means, df[, n_ranges[, lower_tail]])` / `qtukey(p, …)` | Studentized range distribution (R `ptukey` / `qtukey`): Tukey p-values and critical values by hand |

The aggregates return `LIST<STRUCT(group1, group2, estimate, statistic, p_value, p_adj, conf_low, conf_high, method, df1, alternative, conf_level, p_adjust_method)>`, one entry per pair, with `estimate = group2 - group1`; read them with `unnest(…, recursive := true)`. `p_adjust` takes R's `p.adjust` methods: `holm`, `hochberg`, `hommel`, `bonferroni`, `BH` (`fdr`), `BY`, `none`. The `_by` macros return the group column, `model_id`, `test_id`, `method`, `statistic`, `p_value`, `group1`, `group2`, `p_adj`, `df1`, `df2`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name`, `alternative`, `p_adjust_method`.

### Multiple testing (adjust any p-values)
| Function | Use |
|---|---|
| `p_adjust_by(source, group_col, p_col, method := 'holm', n := NULL)` | Every source row (source order) with `p_adjusted`; one family per group. The row-aligned form |
| `p_adjust_agg(p[, method[, n]] ORDER BY …)` | `DOUBLE[]` per group, in input order; `method`/`n` constant |
| `p_adjust_list(p_list[, method[, n]])` | `DOUBLE[]` for a list of p-values |
| `p_adjust(p[, method[, n]]) OVER (PARTITION BY g)` | **DuckDB 2.0+ only.** The row's adjusted p-value inline; the family is the window frame, so leave `ORDER BY` out of `OVER` |

R `p.adjust` semantics: methods `holm` (default), `hochberg`, `hommel`, `bonferroni`, `BH` (`fdr`), `BY`, `none`, case-insensitive (unknown method: error). NULL p-values stay NULL and are not counted; `n` (number of hypotheses) defaults to the non-NULL count and may not be smaller. `p_adjust(...) OVER (...)` exists on DuckDB 2.0 only (check `SELECT version()`); on 1.5 window aggregates can't produce per-row values over a whole partition, so use `p_adjust_by`. For inline use with `n = count(p) OVER (PARTITION BY g)` and `i = row_number() OVER (PARTITION BY g ORDER BY p NULLS LAST)`: Bonferroni `least(1, n*p)`; Holm `max(least(1, (n-i+1)*p)) OVER (PARTITION BY g ORDER BY p NULLS LAST)`; BH `min(least(1, n*p/i)) OVER (PARTITION BY g ORDER BY p DESC NULLS LAST)`. Keep the default RANGE frame and wrap each in `CASE WHEN p IS NOT NULL`, because `least` ignores NULLs (recipe in docs/api/statistics/hypothesis.md).

### Residual autocorrelation and heteroskedasticity
| Function | Test |
|---|---|
| `ljung_box_agg(resid[, lag[, fitdf]] ORDER BY t)` | Ljung-Box, R `Box.test(type = "Ljung-Box")`; `lag` default 1, `fitdf` = p + q for ARMA residuals; adds `lag`, `fitdf` fields |
| `box_pierce_agg(resid[, lag[, fitdf]] ORDER BY t)` | Box-Pierce |
| `durbin_watson_agg(resid ORDER BY t)` | Durbin-Watson statistic only (`p_value` NULL) |
| `durbin_watson_agg(resid, x[, alternative[, fit_intercept]] ORDER BY t)` | With the model's regressors `x`: `lmtest::dwtest` p-value (exact via Imhof for n < 100, else normal approximation); `alternative` `'greater'` (default) / `'less'` / `'two_sided'`; `[]::DOUBLE[]` = intercept-only |
| `breusch_pagan_agg(resid, x[, studentize])` | Breusch-Pagan, `lmtest::bptest`; `studentize` true (Koenker) by default |
| `white_test_agg(resid, x)` | White's test (squares and cross-products of `x`) |

`lag`, `fitdf`, `studentize`, `alternative` and `fit_intercept` must be constants. Statistic near 2 = no first-order autocorrelation; DW well below 2 = positive autocorrelation. The ACF / PACF curves are in the **anofox-statistics-diagnostics** skill.

### Survival
| Function | Returns |
|---|---|
| `log_rank_test_agg(time, event, group)` | k-group log-rank test (R `survdiff`); unified fields, `df1` = groups − 1 |
| `log_rank_test_agg(time, event, group[, columns DOUBLE[]], {'rho': 1, 'strata': k})` | G-rho family (`rho` 1 = Peto-Peto, weights early differences) and stratified test (`strata` = 1-based position in `columns`) |
| `kaplan_meier_agg(time, event[, conf_level[, conf_type]])` | `LIST<STRUCT(time, surv, conf_low, conf_high, n_risk, n_event, n_censor, std_err)>` (R `survfit`); `conf_type` `'log'` (default) / `'log-log'` / `'plain'` |
| `kaplan_meier_by(source, strata_col, time_col, event_col[, conf_level := 0.95, conf_type := 'log'])` | One KM curve per stratum in the `curve` schema (`curve_type = 'km'`) |

`event` is BOOLEAN or numeric (true / non-zero = event observed, else right-censored). Group is VARCHAR or BIGINT. `kaplan_meier_agg` also takes weights / entry times / clusters and the cumulative hazard; Cox and AFT models are in the **anofox-statistics-regression** skill.

### Causal mediation
| Function | Returns |
|---|---|
| `mediation_by(source, group_col, treatment, mediator, outcome[, covariates := [c1, ...]][, options := {...}])` | Per group, one row per effect (`term` = `acme`, `ade`, `total`, `prop_mediated`) in the `terms` schema (`id` = group, `model_id = 'mediation'`) plus `method`, `n_obs`, `n_sims` |
| `mediation_agg(y, m, x[, covariates[, options]])` | The same four rows as `LIST<STRUCT(term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, method, n_obs, n_sims)>` |

Linear models as R `mediation::mediate`: `m ~ x + covariates` (a) and `y ~ x + m + covariates` (b, c'); ACME = a·b·Δ, ADE = c'·Δ, Δ = `treat_value` − `control_value` (default 1 − 0). Options: `method` `'bootstrap'` (default; case resampling, `n_bootstrap` 1000, `seed` 42, `ci_type` `'percentile'` / `'bca'`, p-value as `mediate`), `'sobel'` (delta method, z/t tests) or `'quasi_bayesian'`; `confidence_level`. Rows with a NULL anywhere are dropped from both models. Continuous mediator and outcome only; no treatment × mediator interaction.

```sql
CREATE OR REPLACE TABLE med_demo AS
SELECT i, 's' || (i % 2) AS site, ((i // 2) % 2)::DOUBLE AS treat, sin(i * 1.3) AS age_z,
       0.5 + 0.8 * ((i // 2) % 2) + sin(i * 2.1 + 0.4) * 0.9 AS m
FROM range(100) t(i);
CREATE OR REPLACE TABLE med_demo AS SELECT *, 1 + 0.4 * treat + 0.6 * m + cos(i * 1.7) * 0.8 AS y FROM med_demo;

SELECT site, term, round(estimate, 3) AS estimate, round(conf_low, 3) AS conf_low, round(conf_high, 3) AS conf_high
FROM mediation_by('med_demo', site, treatment := treat, mediator := m, outcome := y, covariates := [age_z]);
```

### Classifier evaluation
| Function | Returns |
|---|---|
| `auc_agg(y_true, score)` | ROC AUC (`DOUBLE`; NULL with one class only) |
| `brier_agg(y_true, prob)`, `log_loss_agg(y_true, prob)` | Brier score, log loss (`prob` in [0, 1]) |
| `roc_agg(y_true, score[, drop_intermediate])` | `LIST<STRUCT(threshold, fpr, tpr)>` (sklearn `roc_curve`) |
| `pr_agg(y_true, score)` | `LIST<STRUCT(threshold, recall, precision)>`; last point has NULL threshold |
| `calibration_agg(y_true, prob, n_bins[, conf_level])` | `LIST<STRUCT(bin, mean_pred, frac_pos, n, conf_low, conf_high)>` with Wilson intervals |
| `roc_curve_by(source, group_col, y_true_col, score_col[, drop_intermediate := true])`, `pr_curve_by(source, group_col, y_true_col, score_col)`, `calibration_curve_by(source, group_col, y_true_col, prob_col[, n_bins := 10, conf_level := 0.95])` | Per-group curves in the `curve` schema (`curve_type` `'roc'` / `'pr'` / `'calibration'`), combinable with `UNION ALL` |
| `auc_null_p_value(auc, n_pos, n_neg)` / `auc_null_quantile(p, n_pos, n_neg)` | Is the AUC better than chance? Exact one-sided p-value / quantile of a random classifier's AUC (R `pwilcox` / `qwilcox` / (n_pos·n_neg); normal approximation beyond n_pos·n_neg = 10000) |

`y_true` is BOOLEAN or numeric 0/1 (anything else is an error); NULL rows are skipped. These are not hypothesis tests: the scalar metrics return a plain `DOUBLE`. Harrell-style accuracy indexes (`concordance_agg`, `somers2_agg`, `rcorr_cens_agg`, `val_prob_agg`, `lrm_stats_agg`) are in docs/api/statistics/accuracy.md and the **anofox-statistics-diagnostics** skill.

### Model-comparison tests (elsewhere)
Likelihood-ratio (`lr_test`), score (`glm_score_test_agg` / `_by`), Wald chunk tests (`wald_anova`, `fastbw`), cluster-robust inference and profile CIs belong to fitted models: see the **anofox-statistics-regression** skill. Per-group runs of any test via `GROUP BY` or the `_by` macros: **anofox-statistics-batch**.

## Worked examples

```sql
-- Paired columns: control and treatment measured on the same units
CREATE TABLE ab AS SELECT * FROM (VALUES
  (10.1, 12.3),(9.8, 11.9),(10.5, 12.8),(10.0, 12.1),(9.9, 12.0)
) t(control, treatment);
-- Two-sample t-test takes long format: t_test_agg(value DOUBLE, group_id INTEGER[, options])
SELECT r.statistic, r.p_value
FROM (
  SELECT t_test_agg(value, group_id) AS r
  FROM (SELECT control::DOUBLE AS value, 0 AS group_id FROM ab
        UNION ALL
        SELECT treatment::DOUBLE, 1 FROM ab)
);
```

```sql
-- Pearson and Spearman correlation (both aggregate forms)
SELECT
  (pearson_agg(control, treatment)).r   AS pearson_r,
  (spearman_agg(control, treatment)).r  AS spearman_rho
FROM ab;
```

```sql
-- Omnibus ANOVA, then Tukey HSD for every pair of arms
CREATE TABLE trial AS SELECT 'site_' || (i % 2) AS site, chr((65 + i % 3)::INTEGER) AS arm,
    (i % 3) + sin(i) AS response FROM range(60) t(i);
SELECT (r).statistic, (r).df1, (r).df2, (r).p_value, (r).effect_size, (r).effect_name
FROM (SELECT one_way_anova_agg(response, arm) AS r FROM trial);
SELECT unnest(tukey_hsd_agg(response, arm), recursive := true) FROM trial;
-- Holm-adjusted pairwise t-tests per site, as a long table
SELECT site, group1, group2, round(p_adj, 4) AS p_adj, round(conf_low, 3) AS lo, round(conf_high, 3) AS hi
FROM pairwise_t_test_by('trial', site, response, arm, options := {'p_adjust': 'holm'})
ORDER BY site, group1, group2;
```

```sql
-- Benjamini-Hochberg within each endpoint, for p-values from any tests
CREATE TABLE results AS SELECT * FROM (VALUES
    ('weight', 'dose_a', 0.012), ('weight', 'dose_b', 0.030), ('weight', 'dose_c', 0.200),
    ('bp', 'dose_a', 0.001), ('bp', 'dose_b', 0.049)) t(endpoint, test, p_value);
SELECT endpoint, test, p_value, p_adjusted
FROM p_adjust_by('results', endpoint, p_value, method := 'BH');
SELECT p_adjust_list([0.01, 0.02, 0.04], 'holm', 10);  -- 10 hypotheses, 3 observed
```

```sql
-- Residual autocorrelation and heteroskedasticity of a fitted model (ORDER BY after the constants)
CREATE TABLE fit_resid AS
SELECT i AS t, i::DOUBLE AS x1, ((i * 3) % 5)::DOUBLE AS x2,
       sin(i * 0.8) * (0.5 + 0.1 * i) + 0.3 * cos(i * 0.3) AS resid
FROM range(60) r(i);
SELECT (ljung_box_agg(resid, 10 ORDER BY t)).p_value               AS ljung_box_p,
       (durbin_watson_agg(resid ORDER BY t)).statistic               AS dw,
       (breusch_pagan_agg(resid, [x1, x2])).p_value                  AS bp_p,
       (white_test_agg(resid, [x1, x2])).p_value                     AS white_p
FROM fit_resid;
```

```sql
-- Durbin-Watson p-value (lmtest::dwtest): pass the regressors of the fitted model
SELECT (durbin_watson_agg(resid, [x1, x2], 'greater' ORDER BY t)).p_value AS dw_p FROM fit_resid;
```

```sql
-- Classifier evaluation per model: AUC, Brier score, and the ROC curve as a long table
CREATE TABLE scored AS SELECT * FROM (VALUES
  ('a', true, 0.9), ('a', false, 0.1), ('a', true, 0.8), ('a', true, 0.65), ('a', false, 0.4),
  ('a', false, 0.3), ('a', true, 0.55), ('a', false, 0.55), ('a', true, 0.7), ('a', false, 0.2)
) t(model, label, score);
SELECT model, auc_agg(label, score) AS auc, brier_agg(label, score) AS brier, log_loss_agg(label, score) AS log_loss
FROM scored GROUP BY model;
SELECT x AS fpr, y AS tpr, threshold FROM roc_curve_by('scored', model, label, score);
```

```sql
-- Survival: log-rank test across plans and the Kaplan-Meier curve per plan
CREATE TABLE churn AS
SELECT CASE WHEN i % 3 = 0 THEN 'annual' ELSE 'monthly' END AS plan,
       least(round(exp(2.0 + 0.6 * (i % 3 = 0)::INTEGER) * (0.3 + ((i * 37) % 100) / 60.0), 1), 24.0) AS months,
       round(exp(2.0 + 0.6 * (i % 3 = 0)::INTEGER) * (0.3 + ((i * 37) % 100) / 60.0), 1) <= 24 AS churned
FROM range(200) r(i);
SELECT (r).statistic, (r).df1, (r).p_value FROM (SELECT log_rank_test_agg(months, churned, plan) AS r FROM churn);
SELECT series, x AS months, y AS surv, y_low, y_high FROM kaplan_meier_by('churn', plan, months, churned) WHERE x <= 3;
```

```sql
-- Paired / one-sample t-tests (wide format) and a bootstrap CI of the mean
SELECT (t_test_paired_agg(treatment, control)).estimate AS mean_diff,
       (t_test_paired_agg(treatment, control)).p_value  AS paired_p,
       (t_test_one_sample_agg(treatment, {'mu': 12, 'alternative': 'greater'})).conf_high AS one_sided_hi, -- inf
       (bootstrap_mean_ci_agg(treatment, {'n_bootstrap': 2000})).conf_low AS boot_lo
FROM ab;
-- Dependent series: block bootstrap in time order
SELECT (bootstrap_mean_ci_agg(x, {'method': 'stationary', 'block_length': 8} ORDER BY t)).std_error AS se
FROM (SELECT i AS t, 10 + sin(i * 0.3) AS x FROM range(120) r(i));
```

```sql
-- Two-way ANOVA (one row per term) and repeated-measures ANOVA with sphericity corrections
SELECT a.term, round(a.statistic, 3) AS f, a.df1, a.df2, round(a.p_value, 4) AS p, round(a.effect_size, 3) AS partial_eta2
FROM (SELECT unnest(two_way_anova_agg(y, fert, irr, {'terms': ['fert', 'irr']})) AS a
      FROM (SELECT ['none', 'low', 'high'][(i % 3) + 1] AS fert, (i // 3) % 2 AS irr,
                   10 + (i % 3) * 1.5 + ((i // 3) % 2) * 0.8 + sin(i * 1.3) AS y FROM range(36) r(i)));
SELECT r.p_value, round(r.mauchly_p_value, 4) AS mauchly_p, round(r.gg_epsilon, 3) AS gg_eps, r.gg_p_value, r.hf_p_value
FROM (SELECT repeated_measures_anova_agg(rt, subj, 'c' || cond) AS r
      FROM (SELECT i // 4 AS subj, i % 4 AS cond,
                   300 + (i % 4) * 6 + (i // 4) * 5 + (hash(i) % 1000) / 100.0 * (1 + i % 4) AS rt
            FROM range(40) t(i)));
```

```sql
-- Equivalence: the new TOST variants need explicit bounds
CREATE TABLE eq AS
SELECT CASE WHEN i % 2 = 0 THEN 'old' ELSE 'new' END AS process, 10 + sin(i * 1.3) AS weight,
       (i % 5 < 2)::BIGINT AS defect
FROM range(60) t(i);
SELECT (tost_one_sample_agg(weight, {'mu': 10, 'delta': 0.5})).equivalent       AS mean_is_10,
       (tost_yuen_agg(weight, process, {'delta': 0.5})).p_value                  AS yuen_p,
       (tost_bootstrap_agg(weight, process, {'bound_lower': -0.4, 'bound_upper': 0.6,
                                             'n_bootstrap': 2000})).equivalent    AS boot_equiv,
       (tost_prop_two_agg(defect, process, {'delta': 0.2})).p_value              AS rate_diff_p
FROM eq;
```

```sql
-- Partial correlation and ICC types
CREATE TABLE growth AS
SELECT (i % 12)::DOUBLE AS age, (100 + 5 * (i % 12) + (i * 7) % 11)::DOUBLE AS height,
       (20 + 2 * (i % 12) + (i * 5) % 7)::DOUBLE AS weight FROM range(60) r(i);
SELECT (pearson_agg(height, weight)).r AS r_raw,
       (partial_cor_agg(height, weight, [age])).r AS r_partial,
       (partial_cor_agg(height, weight, [age])).p_value AS p_partial
FROM growth;
SELECT round((icc_agg(score, p, r, {'type': 'icc2'})).icc, 3) AS icc2_agreement,
       round((icc_agg(score, p, r, {'type': 'icc3'})).icc, 3) AS icc3_consistency
FROM (SELECT p, r, (p * 2 + 3 * (r = 2)::INT + (p * r) % 3)::DOUBLE AS score
      FROM range(10) a(p), range(3) b(r));
```

```sql
-- Drift monitoring: evidently column / dataset verdicts, embeddings, multivariate MMD
CREATE TABLE ev AS
SELECT 0 AS grp, 2 * sin(0.37 * i) AS v, ['a', 'b', 'c'][i % 3 + 1] AS s,
       [sin(0.7 * i), cos(1.3 * i), 0.1 * (i % 5)] AS e FROM range(50) r(i)
UNION ALL
SELECT 1, 2 * sin(0.53 * i) + 0.5, ['a', 'b', 'c'][((i * 7) % 5) % 3 + 1],
       [sin(0.9 * i) + 0.4, cos(1.1 * i) - 0.2, 0.1 * (i % 7) + 0.1] FROM range(40) r(i);
SELECT column_name, test, round(score, 4) AS score, drifted, dataset_drift
FROM dataset_drift_by('ev', grp, ['v', 's']);
SELECT (column_drift_agg(v, grp, {'test': 'wasserstein'})).drifted AS wass_drift,
       (category_proportions_agg(s, grp)).p_current AS shares_now,
       (embedding_drift_agg(e, grp, {'metric': 'cosine'})).score AS emb_cosine,
       (mmd_test_agg(e, grp, {'n_permutations': 199})).p_value AS mmd_p,
       (energy_distance_test_agg(e, grp, {'n_permutations': 199})).p_value AS energy_p
FROM ev;
```

```sql
-- Forecast models: does any beat the benchmark (SPA), which survive (MCS)?
CREATE TABLE fc_losses AS
SELECT t, 2.0 + 0.5 * abs(sin(t * 0.7)) AS naive_loss, 1.0 + 0.3 * abs(sin(t * 0.7 + 1.0)) AS ets_loss,
       3.0 + 0.4 * abs(cos(t * 0.3)) AS arima_loss
FROM range(1, 61) r(t);
SELECT (spa_test_agg(naive_loss, [ets_loss, arima_loss], {'n_bootstrap': 499} ORDER BY t)).p_value AS spa_p,
       (model_confidence_set_agg([naive_loss, ets_loss, arima_loss], {'n_bootstrap': 499} ORDER BY t)).included_models AS mcs
FROM fc_losses;
```

```sql
-- AUC vs chance, and Tukey critical values by hand
SELECT model, auc_agg(label, score) AS auc,
       auc_null_p_value(auc_agg(label, score), count(*) FILTER (WHERE label), count(*) FILTER (WHERE NOT label)) AS p_vs_chance
FROM scored GROUP BY model;
SELECT qtukey(0.95, 3, 12) AS q_crit, ptukey(3.5, 3, 12, 1, false) AS p_upper;
```

```sql skip
-- Per-segment normality test with GROUP BY
SELECT segment, (shapiro_wilk_agg(value)).p_value AS p
FROM measurements GROUP BY segment;
```
