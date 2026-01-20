# Statistical Hypothesis Testing Functions

Comprehensive statistical hypothesis testing. All tests are implemented as aggregate functions.

See [Reference: Statistical Tests](../reference/tests/) for detailed documentation of each test.

## Quick Reference

### Normality Tests
- `shapiro_wilk_agg` - Shapiro-Wilk test
- `jarque_bera_agg` - Jarque-Bera test
- `dagostino_k2_agg` - D'Agostino K² test

### Parametric Tests
- `t_test_agg` - Two-sample t-test (Welch/Student)
- `one_way_anova_agg` - One-way ANOVA
- `yuen_agg` - Yuen's trimmed mean test
- `brown_forsythe_agg` - Brown-Forsythe variance test

### Nonparametric Tests
- `mann_whitney_u_agg` - Mann-Whitney U test
- `kruskal_wallis_agg` - Kruskal-Wallis H test
- `wilcoxon_signed_rank_agg` - Wilcoxon signed-rank test
- `brunner_munzel_agg` - Brunner-Munzel test
- `permutation_t_test_agg` - Permutation t-test

### Correlation Tests
- `pearson_agg` - Pearson correlation
- `spearman_agg` - Spearman rank correlation
- `kendall_agg` - Kendall tau correlation
- `distance_cor_agg` - Distance correlation
- `icc_agg` - Intraclass correlation

### Categorical Tests
- `chisq_test_agg` - Chi-square independence test
- `chisq_gof_agg` - Chi-square goodness of fit
- `g_test_agg` - G-test (log-likelihood ratio)
- `fisher_exact_agg` - Fisher's exact test
- `mcnemar_agg` - McNemar's test

### Effect Size Measures
- `cramers_v_agg` - Cramér's V
- `phi_coefficient_agg` - Phi coefficient
- `contingency_coef_agg` - Contingency coefficient
- `cohen_kappa_agg` - Cohen's kappa

### Proportion Tests
- `prop_test_one_agg` - One-sample proportion test
- `prop_test_two_agg` - Two-sample proportion test
- `binom_test_agg` - Exact binomial test

### Equivalence Tests
- `tost_t_test_agg` - TOST two-sample t-test
- `tost_paired_agg` - TOST paired t-test
- `tost_correlation_agg` - TOST correlation equivalence

### Distribution Comparison
- `energy_distance_agg` - Energy distance
- `mmd_agg` - Maximum Mean Discrepancy

### Forecast Comparison
- `diebold_mariano_agg` - Diebold-Mariano test
- `clark_west_agg` - Clark-West test
