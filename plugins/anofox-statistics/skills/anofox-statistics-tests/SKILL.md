---
name: anofox-statistics-tests
description: >
  Statistical hypothesis tests, correlation, effect sizes, and equivalence
  testing in the anofox_statistics DuckDB extension. Covers normality tests
  (Shapiro-Wilk, Jarque-Bera, D'Agostino K2), parametric tests (t-test,
  one-way ANOVA, Yuen, Brown-Forsythe), nonparametric tests (Mann-Whitney U,
  Kruskal-Wallis, Wilcoxon signed-rank, Brunner-Munzel, permutation t-test),
  correlation (Pearson, Spearman, Kendall, distance correlation, ICC),
  categorical / contingency tests (chi-squared, chi-squared GOF, G-test,
  Fisher's exact, McNemar), effect sizes (Cramér's V, phi, contingency
  coefficient, Cohen's kappa), proportion tests, TOST equivalence tests,
  distribution comparison (energy distance, MMD), and forecast-evaluation
  tests (Diebold-Mariano, Clark-West). Use when running a significance test,
  measuring association/effect size, or comparing distributions in SQL.
version: 0.10.0
user-invocable: false
---

# Anofox Statistics — Hypothesis Tests Cheat Sheet

**Extension:** `anofox_statistics` v0.10.0 | **DuckDB:** v1.4.5 LTS / v1.5.4+ | Tests are **aggregates** — use with `GROUP BY` for per-segment tests.

## Gotchas

1. **All tests are `_agg` aggregates.** They consume rows, not literal arrays. Wrap in `(test_agg(...)).field` to read result fields, or select the whole STRUCT.
2. **Two-sample tests take two value columns** `(x, y)` — one value per row in each. Paired tests expect aligned rows.
3. **Result is a STRUCT** with at least `statistic` and `p_value`; test-specific extras vary (df, effect size, CI). Select the struct first if unsure of field names: `SELECT t_test_agg(a, b) FROM tbl;`.
4. **Correlation has scalar + agg forms for some** (`spearman`, `icc` have both); most others are agg-only.

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
| `prop_test_two_agg` | Two-proportion z-test |
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

## Worked examples

```sql
-- Two-sample t-test between two groups' values
CREATE TABLE ab AS SELECT * FROM (VALUES
  (10.1, 12.3),(9.8, 11.9),(10.5, 12.8),(10.0, 12.1),(9.9, 12.0)
) t(control, treatment);
SELECT (t_test_agg(control, treatment)).* FROM ab;
```

```sql
-- Pearson and Spearman correlation (both aggregate forms)
SELECT
  (pearson_agg(control, treatment)).statistic   AS pearson_r,
  (spearman_agg(control, treatment)).statistic  AS spearman_rho
FROM ab;
```

```sql skip
-- Per-segment normality test with GROUP BY
SELECT segment, (shapiro_wilk_agg(value)).p_value AS p
FROM measurements GROUP BY segment;
```
