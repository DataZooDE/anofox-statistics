---
name: anofox-statistics-tests
description: >
  Statistical hypothesis tests, post-hoc comparisons, correlation, effect
  sizes, equivalence testing, classifier evaluation, and time-series and
  survival tests in the anofox_statistics DuckDB extension. Covers the
  unified test-result fields (method, alternative, statistic, df1, df2,
  p_value, estimate, conf_low, conf_high, conf_level, effect_size,
  effect_name, n) and their deprecated aliases, normality tests
  (Shapiro-Wilk, Jarque-Bera, D'Agostino K2), parametric tests (t-test,
  one-way ANOVA, Yuen, Brown-Forsythe), nonparametric tests (Mann-Whitney U,
  Kruskal-Wallis, Wilcoxon signed-rank, Brunner-Munzel, permutation t-test),
  pairwise post-hoc tests (pairwise_t_test_agg, tukey_hsd_agg,
  dunn_test_agg and their _by macros, p_adjust holm / BH / bonferroni),
  correlation (Pearson, Spearman, Kendall, distance correlation, ICC),
  categorical / contingency tests (chi-squared, chi-squared GOF, G-test,
  Fisher's exact, McNemar), effect sizes (Cramér's V, phi, contingency
  coefficient, Cohen's kappa), proportion tests, TOST equivalence tests,
  distribution comparison (energy distance, MMD), forecast-evaluation tests
  (Diebold-Mariano, Clark-West), residual autocorrelation and
  heteroskedasticity tests (Ljung-Box, Box-Pierce, Durbin-Watson with
  p-value, Breusch-Pagan, White), survival tests (Kaplan-Meier,
  log-rank), and classifier evaluation (ROC / PR / calibration curves, AUC,
  Brier score, log loss, roc_curve_by). Use when running a significance
  test, comparing many groups, measuring association/effect size,
  evaluating a classifier, or comparing distributions or survival in SQL.
version: 0.12.1
user-invocable: false
---

# Anofox Statistics — Hypothesis Tests Cheat Sheet

**Extension:** `anofox_statistics` v0.12.1 | **DuckDB:** v1.4.5 LTS / v1.5.x | Tests are **aggregates** — use with `GROUP BY` for per-segment tests.

## Gotchas

1. **All tests are `_agg` aggregates.** They consume rows, not literal arrays. Wrap in `(test_agg(...)).field` to read result fields, or select the whole STRUCT.
2. **Two-sample tests take two value columns** `(x, y)` — one value per row in each. Paired tests expect aligned rows.
3. **Every test returns the unified fields** (below), always present and NULL where they do not apply, after the test's own fields (`n1`, `n2`, `ss_between`, …). Read `(r).p_value`, `(r).estimate`, `(r).conf_low` the same way for every test.
4. **Deprecated aliases.** `ci_lower` / `ci_upper` → `conf_low` / `conf_high`; `df` → `df1`; ANOVA `f_statistic` → `statistic`, `df_between` / `df_within` → `df1` / `df2`. They still work but are removed in the next minor release; write the new names.
5. **Correlation has scalar + agg forms for some** (`spearman`, `icc` have both); most others are agg-only.
6. **Group arguments are long format.** Two-sample tests take `t_test_agg(value, group)` with exactly two group labels (BIGINT or VARCHAR); `estimate` is the difference `group1 - group2`. The k-sample tests take `one_way_anova_agg(value DOUBLE, group BIGINT | VARCHAR)` (likewise `kruskal_wallis_agg`, `brown_forsythe_agg`), as do the post-hoc tests.
7. **Order-sensitive aggregates need their own `ORDER BY`, after the constant arguments**: `ljung_box_agg(resid, 10 ORDER BY t)`, `durbin_watson_agg(resid, [x1, x2] ORDER BY t)`. In `ljung_box_agg(resid ORDER BY t, 10)` the 10 becomes a second sort key. Applies to `acf_agg`, `pacf_agg`, `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg`.

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
| `t_test_agg` | Two-sample / paired t-test |
| `one_way_anova_agg` | One-way ANOVA |
| `yuen_agg` | Yuen's trimmed-mean t-test (robust) |
| `brown_forsythe_agg` | Brown-Forsythe equality-of-variance |

### Nonparametric
| Function | Test |
|---|---|
| `mann_whitney_u_agg` | Mann-Whitney U (rank-sum) |
| `kruskal_wallis_agg` | Kruskal-Wallis |
| `wilcoxon_signed_rank_agg` | Wilcoxon signed-rank (paired) |
| `brunner_munzel_agg` | Brunner-Munzel |
| `permutation_t_test_agg` | Permutation t-test |

### Correlation
| Function | Measure |
|---|---|
| `pearson_agg` | Pearson r |
| `spearman_agg` / `spearman` | Spearman ρ |
| `kendall_agg` | Kendall τ |
| `distance_cor_agg` | Distance correlation |
| `icc_agg` / `icc` | Intraclass correlation |

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
| `tost_t_test_agg` | Two one-sided t-tests |
| `tost_paired_agg` | Paired TOST |
| `tost_correlation_agg` | Correlation TOST |

### Distribution comparison
| Function | Measure |
|---|---|
| `energy_distance_agg` | Energy distance |
| `mmd_agg` | Maximum mean discrepancy |

### Forecast evaluation
| Function | Test |
|---|---|
| `diebold_mariano_agg` | Diebold-Mariano (equal predictive accuracy) |
| `clark_west_agg` | Clark-West (nested-model forecast comparison) |

### Pairwise post-hoc (after an omnibus test)
| Function | Test |
|---|---|
| `pairwise_t_test_agg(value, group[, opts])` | R `pairwise.t.test`; `p_adjust` (default `'holm'`), `pool_sd` (true), `alternative`, `confidence_level` (0.95). `conf_low`/`conf_high`: CI of each difference, per comparison for `'none'`, Bonferroni otherwise (as `emmeans`) |
| `tukey_hsd_agg(value, group[, opts])` | R `TukeyHSD(aov(...))`; `confidence_level` |
| `dunn_test_agg(value, group[, opts])` | `FSA::dunnTest` (after Kruskal-Wallis); `p_adjust` |
| `pairwise_t_test_by`, `tukey_hsd_by`, `dunn_test_by` `(source, group_col, value_col, factor_col, options := NULL)` | The same per group of `source`, as the long `test` table |

The aggregates return `LIST<STRUCT(group1, group2, estimate, statistic, p_value, p_adj, conf_low, conf_high, method, df1, alternative, conf_level, p_adjust_method)>`, one entry per pair, with `estimate = group2 - group1`; read them with `unnest(…, recursive := true)`. `p_adjust` takes R's `p.adjust` methods: `holm`, `hochberg`, `hommel`, `bonferroni`, `BH` (`fdr`), `BY`, `none`. The `_by` macros return the group column, `model_id`, `test_id`, `method`, `statistic`, `p_value`, `group1`, `group2`, `p_adj`, `df1`, `df2`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name`, `alternative`, `p_adjust_method`.

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
| `kaplan_meier_agg(time, event[, conf_level[, conf_type]])` | `LIST<STRUCT(time, surv, conf_low, conf_high, n_risk, n_event, n_censor, std_err)>` (R `survfit`); `conf_type` `'log'` (default) / `'log-log'` / `'plain'` |
| `kaplan_meier_by(source, strata_col, time_col, event_col[, conf_level := 0.95, conf_type := 'log'])` | One KM curve per stratum in the `curve` schema (`curve_type = 'km'`) |

`event` is BOOLEAN or numeric (true / non-zero = event observed, else right-censored). Group is VARCHAR or BIGINT.

### Classifier evaluation
| Function | Returns |
|---|---|
| `auc_agg(y_true, score)` | ROC AUC (`DOUBLE`; NULL with one class only) |
| `brier_agg(y_true, prob)`, `log_loss_agg(y_true, prob)` | Brier score, log loss (`prob` in [0, 1]) |
| `roc_agg(y_true, score[, drop_intermediate])` | `LIST<STRUCT(threshold, fpr, tpr)>` (sklearn `roc_curve`) |
| `pr_agg(y_true, score)` | `LIST<STRUCT(threshold, recall, precision)>`; last point has NULL threshold |
| `calibration_agg(y_true, prob, n_bins[, conf_level])` | `LIST<STRUCT(bin, mean_pred, frac_pos, n, conf_low, conf_high)>` with Wilson intervals |
| `roc_curve_by(source, group_col, y_true_col, score_col[, drop_intermediate := true])`, `pr_curve_by(source, group_col, y_true_col, score_col)`, `calibration_curve_by(source, group_col, y_true_col, prob_col[, n_bins := 10, conf_level := 0.95])` | Per-group curves in the `curve` schema (`curve_type` `'roc'` / `'pr'` / `'calibration'`), combinable with `UNION ALL` |

`y_true` is BOOLEAN or numeric 0/1 (anything else is an error); NULL rows are skipped. These are not hypothesis tests: the scalar metrics return a plain `DOUBLE`.

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

```sql skip
-- Per-segment normality test with GROUP BY
SELECT segment, (shapiro_wilk_agg(value)).p_value AS p
FROM measurements GROUP BY segment;
```
