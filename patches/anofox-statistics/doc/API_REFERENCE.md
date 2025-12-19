# API Reference

Complete API documentation for anofox-statistics. This document serves as the single source of truth for all function signatures, parameters, and return types.

For runnable code examples demonstrating each test category, see the [examples/](../examples/) directory.

## Table of Contents

- [Parametric Tests](#parametric-tests)
  - [t_test](#t_test)
  - [yuen_test](#yuen_test)
  - [brown_forsythe](#brown_forsythe)
  - [one_way_anova](#one_way_anova)
  - [two_way_anova](#two_way_anova)
  - [repeated_measures_anova](#repeated_measures_anova)
- [Nonparametric Tests](#nonparametric-tests)
  - [rank](#rank)
  - [mann_whitney_u](#mann_whitney_u)
  - [wilcoxon_signed_rank](#wilcoxon_signed_rank)
  - [kruskal_wallis](#kruskal_wallis)
  - [brunner_munzel](#brunner_munzel)
- [Distributional Tests](#distributional-tests)
  - [shapiro_wilk](#shapiro_wilk)
  - [dagostino_k_squared](#dagostino_k_squared)
- [Correlation Tests](#correlation-tests)
  - [pearson](#pearson)
  - [spearman](#spearman)
  - [kendall](#kendall)
  - [partial_cor](#partial_cor)
  - [semi_partial_cor](#semi_partial_cor)
  - [distance_cor](#distance_cor)
  - [distance_cor_test](#distance_cor_test)
  - [icc](#icc)
- [Categorical Tests](#categorical-tests)
  - [chisq_test](#chisq_test)
  - [chisq_goodness_of_fit](#chisq_goodness_of_fit)
  - [g_test](#g_test)
  - [fisher_exact](#fisher_exact)
  - [mcnemar_test](#mcnemar_test)
  - [mcnemar_exact](#mcnemar_exact)
  - [cramers_v](#cramers_v)
  - [phi_coefficient](#phi_coefficient)
  - [contingency_coef](#contingency_coef)
  - [cohen_kappa](#cohen_kappa)
  - [prop_test_one](#prop_test_one)
  - [prop_test_two](#prop_test_two)
  - [binom_test](#binom_test)
- [Resampling Methods](#resampling-methods)
  - [permutation_t_test](#permutation_t_test)
  - [PermutationEngine](#permutationengine)
  - [StationaryBootstrap](#stationarybootstrap)
  - [CircularBlockBootstrap](#circularblockbootstrap)
- [Modern Distribution Tests](#modern-distribution-tests)
  - [energy_distance_test](#energy_distance_test)
  - [mmd_test](#mmd_test)
- [Forecast Evaluation](#forecast-evaluation)
  - [diebold_mariano](#diebold_mariano)
  - [clark_west](#clark_west)
  - [spa_test](#spa_test)
  - [mspe_adjusted_spa](#mspe_adjusted_spa)
  - [model_confidence_set](#model_confidence_set)
- [Equivalence Testing (TOST)](#equivalence-testing-tost)
  - [tost_t_test_one_sample](#tost_t_test_one_sample)
  - [tost_t_test_two_sample](#tost_t_test_two_sample)
  - [tost_t_test_paired](#tost_t_test_paired)
  - [tost_correlation](#tost_correlation)
  - [tost_prop_one](#tost_prop_one)
  - [tost_prop_two](#tost_prop_two)
  - [tost_wilcoxon_paired](#tost_wilcoxon_paired)
  - [tost_wilcoxon_two_sample](#tost_wilcoxon_two_sample)
  - [tost_bootstrap](#tost_bootstrap)
  - [tost_yuen](#tost_yuen)
- [Math Primitives](#math-primitives)
  - [mean](#mean)
  - [stable_mean](#stable_mean)
  - [variance](#variance)
  - [stable_variance](#stable_variance)
  - [std_dev](#std_dev)
  - [median](#median)
  - [trimmed_mean](#trimmed_mean)
  - [skewness](#skewness)
  - [kurtosis](#kurtosis)
- [Enums](#enums)
  - [Alternative](#alternative)
  - [TTestKind](#ttestkind)
  - [AnovaKind](#anovakind)
  - [LossFunction](#lossfunction)
  - [VarEstimator](#varestimator)
  - [MCSStatistic](#mcsstatistic)
  - [Kernel](#kernel)
  - [CorrelationMethod](#correlationmethod)
  - [KendallVariant](#kendallvariant)
  - [ICCType](#icctype)
  - [EquivalenceBounds](#equivalencebounds)
  - [CorrelationTostMethod](#correlationtostmethod)

---

## Parametric Tests

### t_test

Performs t-test comparing two samples.

```rust
pub fn t_test(
    x: &[f64],
    y: &[f64],
    kind: TTestKind,
    alternative: Alternative,
    mu: f64,
    conf_level: Option<f64>,
) -> Result<TTestResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `kind` | `TTestKind` | Type of t-test: `Welch`, `Student`, or `Paired` |
| `alternative` | `Alternative` | Alternative hypothesis: `TwoSided`, `Less`, or `Greater` |
| `mu` | `f64` | Null hypothesis value for the true difference in means |
| `conf_level` | `Option<f64>` | Confidence level for CI (e.g., `Some(0.95)` for 95% CI) |

**Returns:** `TTestResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The t-statistic |
| `df` | `f64` | Degrees of freedom |
| `p_value` | `f64` | The p-value |
| `mean_x` | `f64` | Mean of first sample (or mean difference for paired) |
| `mean_y` | `Option<f64>` | Mean of second sample (`None` for paired) |
| `conf_int` | `Option<TTestConfInt>` | Confidence interval (if `conf_level` specified) |
| `null_value` | `f64` | Null hypothesis value (the `mu` parameter) |

**R equivalent:** `t.test()` (stats)

**References:**
- Student (1908). "The Probable Error of a Mean." *Biometrika*, 6(1), 1–25. [DOI: 10.2307/2331554](https://doi.org/10.2307/2331554)
- Welch, B. L. (1947). "The Generalization of 'Student's' Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1–2), 28–35. [DOI: 10.2307/2332510](https://doi.org/10.2307/2332510)

[Back to top](#table-of-contents)

---

### yuen_test

Performs Yuen's test for comparing trimmed means of two independent samples. Robust alternative to t-test using trimmed means and winsorized variances.

```rust
pub fn yuen_test(
    x: &[f64],
    y: &[f64],
    trim: f64,
    alternative: Alternative,
    conf_level: Option<f64>,
) -> Result<YuenResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `trim` | `f64` | Proportion to trim from each tail, must be in `[0, 0.5)` |
| `alternative` | `Alternative` | Alternative hypothesis: `TwoSided`, `Less`, or `Greater` |
| `conf_level` | `Option<f64>` | Confidence level for CI (e.g., `Some(0.95)` for 95% CI) |

**Returns:** `YuenResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The t-statistic |
| `df` | `f64` | Degrees of freedom (Welch-Satterthwaite) |
| `p_value` | `f64` | The p-value |
| `diff` | `f64` | Difference between trimmed means |
| `trimmed_mean_x` | `f64` | Trimmed mean of first sample |
| `trimmed_mean_y` | `f64` | Trimmed mean of second sample |
| `conf_int` | `Option<YuenConfInt>` | Confidence interval (if `conf_level` specified) |

**R equivalent:** `yuen()` (WRS2)

**Reference:** Yuen, K. K. (1974). "The Two-Sample Trimmed t for Unequal Population Variances." *Biometrika*, 61(1), 165–170. [DOI: 10.2307/2334299](https://doi.org/10.2307/2334299)

[Back to top](#table-of-contents)

---

### brown_forsythe

Performs Brown-Forsythe test for homogeneity of variances. This is Levene's test using the median instead of the mean.

```rust
pub fn brown_forsythe(groups: &[&[f64]]) -> Result<LeveneResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `groups` | `&[&[f64]]` | Slice of slices, each containing one group's data (minimum 2 groups) |

**Returns:** `LeveneResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The F-statistic |
| `df1` | `f64` | Numerator degrees of freedom (k-1) |
| `df2` | `f64` | Denominator degrees of freedom (N-k) |
| `p_value` | `f64` | The p-value |

**R equivalent:** `leveneTest(center=median)` (car)

**Reference:** Brown, M. B., & Forsythe, A. B. (1974). "Robust Tests for the Equality of Variances." *Journal of the American Statistical Association*, 69(346), 364–367. [DOI: 10.2307/2285659](https://doi.org/10.2307/2285659)

[Back to top](#table-of-contents)

---

### one_way_anova

Performs one-way ANOVA for comparing means across multiple groups. Supports both Fisher's (equal variance) and Welch's (unequal variance) variants.

```rust
pub fn one_way_anova(groups: &[&[f64]], kind: AnovaKind) -> Result<OneWayAnovaResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `groups` | `&[&[f64]]` | Slice of slices, each containing one group's data (minimum 2 groups) |
| `kind` | `AnovaKind` | Type of ANOVA: `Fisher` (equal variances) or `Welch` (unequal variances) |

**Returns:** `OneWayAnovaResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The F-statistic |
| `df_between` | `f64` | Degrees of freedom between groups (k-1) |
| `df_within` | `f64` | Degrees of freedom within groups (N-k or Welch-adjusted) |
| `p_value` | `f64` | The p-value |
| `ss_between` | `Option<f64>` | Sum of squares between groups (`None` for Welch) |
| `ss_within` | `Option<f64>` | Sum of squares within groups (`None` for Welch) |
| `ss_total` | `Option<f64>` | Total sum of squares (`None` for Welch) |
| `ms_between` | `Option<f64>` | Mean square between groups (`None` for Welch) |
| `ms_within` | `Option<f64>` | Mean square within groups (`None` for Welch) |
| `n_groups` | `usize` | Number of groups |
| `group_sizes` | `Vec<usize>` | Sample size of each group |
| `group_means` | `Vec<f64>` | Mean of each group |
| `grand_mean` | `Option<f64>` | Grand mean (`None` for Welch) |

**R equivalent:** `oneway.test()` (stats), `aov()` (stats)

**References:**
- Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver and Boyd.
- Welch, B. L. (1951). "On the Comparison of Several Mean Values: An Alternative Approach." *Biometrika*, 38(3–4), 330–336. [DOI: 10.2307/2332579](https://doi.org/10.2307/2332579)

[Back to top](#table-of-contents)

---

### two_way_anova

Performs two-way factorial ANOVA with interaction effects. Uses Type III sum of squares (marginal), supporting both balanced and unbalanced designs.

```rust
pub fn two_way_anova(
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> Result<TwoWayAnovaResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `&[f64]` | Response values |
| `factor_a` | `&[usize]` | Factor A levels (0-indexed) for each observation |
| `factor_b` | `&[usize]` | Factor B levels (0-indexed) for each observation |

**Returns:** `TwoWayAnovaResult`

| Field | Type | Description |
|-------|------|-------------|
| `factor_a` | `AnovaTableRow` | ANOVA results for Factor A |
| `factor_b` | `AnovaTableRow` | ANOVA results for Factor B |
| `interaction` | `AnovaTableRow` | ANOVA results for A×B interaction |
| `residual` | `AnovaTableRow` | Residual (error) row |
| `total` | `AnovaTableRow` | Total row |
| `levels_a` | `usize` | Number of levels in Factor A |
| `levels_b` | `usize` | Number of levels in Factor B |
| `n` | `usize` | Total number of observations |
| `grand_mean` | `f64` | Grand mean of all observations |
| `cell_means` | `Vec<Vec<f64>>` | Cell means where `cell_means[a][b]` is the mean for level a of A and level b of B |
| `marginal_means_a` | `Vec<f64>` | Marginal means for each level of Factor A |
| `marginal_means_b` | `Vec<f64>` | Marginal means for each level of Factor B |

**`AnovaTableRow` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source` | `String` | Source of variation label |
| `ss` | `f64` | Sum of squares |
| `df` | `f64` | Degrees of freedom |
| `ms` | `f64` | Mean square (SS/df) |
| `f_statistic` | `Option<f64>` | F-statistic (`None` for residual/total) |
| `p_value` | `Option<f64>` | p-value (`None` for residual/total) |

**R equivalent:** `Anova(type="III")` (car)

**Reference:** Maxwell, S. E., & Delaney, H. D. (2004). *Designing Experiments and Analyzing Data: A Model Comparison Perspective.* (2nd ed.). Lawrence Erlbaum Associates.

[Back to top](#table-of-contents)

---

### repeated_measures_anova

Performs one-way repeated measures ANOVA for within-subjects designs. Includes Mauchly's sphericity test and Greenhouse-Geisser/Huynh-Feldt corrections.

```rust
pub fn repeated_measures_anova(
    data: &[&[f64]],
    compute_sphericity: bool,
) -> Result<RmAnovaResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[&[f64]]` | Matrix where rows are subjects and columns are conditions |
| `compute_sphericity` | `bool` | Whether to compute sphericity test and corrections (requires k ≥ 3) |

**Returns:** `RmAnovaResult`

| Field | Type | Description |
|-------|------|-------------|
| `within_subjects` | `AnovaTableRow` | Within-subjects (treatment) effect |
| `subjects` | `AnovaTableRow` | Between-subjects (individual differences) |
| `error` | `AnovaTableRow` | Error (subjects × conditions interaction) |
| `total` | `AnovaTableRow` | Total |
| `sphericity` | `Option<SphericityResult>` | Mauchly's sphericity test results |
| `greenhouse_geisser` | `Option<CorrectedResult>` | Greenhouse-Geisser corrected results |
| `huynh_feldt` | `Option<CorrectedResult>` | Huynh-Feldt corrected results |
| `grand_mean` | `f64` | Grand mean of all observations |
| `condition_means` | `Vec<f64>` | Mean of each condition |
| `subject_means` | `Vec<f64>` | Mean of each subject |

**`SphericityResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `w` | `f64` | Mauchly's W statistic |
| `chi_square` | `f64` | Chi-square approximation |
| `df` | `f64` | Degrees of freedom |
| `p_value` | `f64` | p-value for sphericity test |

**`CorrectedResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `epsilon` | `f64` | Epsilon correction factor (GG or HF) |
| `df_num_corrected` | `f64` | Corrected numerator degrees of freedom |
| `df_den_corrected` | `f64` | Corrected denominator degrees of freedom |
| `f_statistic` | `f64` | F-statistic (same as uncorrected) |
| `p_value` | `f64` | Corrected p-value |

**R equivalent:** `ezANOVA()` (ez)

**References:**
- Mauchly, J. W. (1940). "Significance Test for Sphericity of a Normal n-Variate Distribution." *Annals of Mathematical Statistics*, 11(2), 204–209. [DOI: 10.1214/aoms/1177731915](https://doi.org/10.1214/aoms/1177731915)
- Greenhouse, S. W., & Geisser, S. (1959). "On Methods in the Analysis of Profile Data." *Psychometrika*, 24(2), 95–112. [DOI: 10.1007/BF02289823](https://doi.org/10.1007/BF02289823)
- Huynh, H., & Feldt, L. S. (1976). "Estimation of the Box Correction for Degrees of Freedom from Sample Data in Randomized Block and Split-Plot Designs." *Journal of Educational Statistics*, 1(1), 69–82. [DOI: 10.2307/1164736](https://doi.org/10.2307/1164736)

[Back to top](#table-of-contents)

---

## Nonparametric Tests

### rank

Computes ranks with average tie handling.

```rust
pub fn rank(data: &[f64]) -> Result<Vec<f64>>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[f64]` | Input data |

**Returns:** `Vec<f64>` - Ranks (1-indexed, ties receive average rank)

**R equivalent:** `rank(ties.method="average")` (stats)

[Back to top](#table-of-contents)

---

### mann_whitney_u

Performs Mann-Whitney U test (Wilcoxon rank-sum test) for two independent samples.

```rust
pub fn mann_whitney_u(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<MannWhitneyResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `alternative` | `Alternative` | Alternative hypothesis |
| `continuity_correction` | `bool` | Apply continuity correction (normal approximation only) |
| `exact` | `bool` | Compute exact p-value (recommended for small samples without ties) |
| `conf_level` | `Option<f64>` | Confidence level for Hodges-Lehmann CI |
| `mu` | `Option<f64>` | Null hypothesis location shift (default: 0) |

**Returns:** `MannWhitneyResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The U statistic |
| `p_value` | `f64` | The p-value |
| `estimate` | `Option<f64>` | Hodges-Lehmann estimate of location shift |
| `conf_int` | `Option<ConfidenceInterval>` | Confidence interval for location shift |
| `null_value` | `f64` | Null hypothesis value (location shift under H0) |

**R equivalent:** `wilcox.test(paired=FALSE)` (stats)

**References:**
- Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6), 80–83. [DOI: 10.2307/3001968](https://doi.org/10.2307/3001968)
- Mann, H. B., & Whitney, D. R. (1947). "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics*, 18(1), 50–60. [DOI: 10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491)

[Back to top](#table-of-contents)

---

### wilcoxon_signed_rank

Performs Wilcoxon signed-rank test for paired samples.

```rust
pub fn wilcoxon_signed_rank(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<WilcoxonResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample (paired with x) |
| `alternative` | `Alternative` | Alternative hypothesis |
| `continuity_correction` | `bool` | Apply continuity correction |
| `exact` | `bool` | Compute exact p-value |
| `conf_level` | `Option<f64>` | Confidence level for pseudo-median CI |
| `mu` | `Option<f64>` | Null hypothesis median difference (default: 0) |

**Returns:** `WilcoxonResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The V statistic (sum of positive ranks) |
| `p_value` | `f64` | The p-value |
| `estimate` | `Option<f64>` | Hodges-Lehmann pseudo-median of differences |
| `conf_int` | `Option<ConfidenceInterval>` | Confidence interval for pseudo-median |
| `null_value` | `f64` | Null hypothesis value (median difference under H0) |

**R equivalent:** `wilcox.test(paired=TRUE)` (stats)

**Reference:** Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6), 80–83. [DOI: 10.2307/3001968](https://doi.org/10.2307/3001968)

[Back to top](#table-of-contents)

---

### kruskal_wallis

Performs Kruskal-Wallis H test for comparing multiple independent groups. Nonparametric equivalent of one-way ANOVA.

```rust
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `groups` | `&[&[f64]]` | Slice of slices, each containing one group's data |

**Returns:** `KruskalResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The H statistic (chi-squared approximation) |
| `df` | `f64` | Degrees of freedom (k-1) |
| `p_value` | `f64` | The p-value |

**R equivalent:** `kruskal.test()` (stats)

**Reference:** Kruskal, W. H., & Wallis, W. A. (1952). "Use of Ranks in One-Criterion Variance Analysis." *Journal of the American Statistical Association*, 47(260), 583–621. [DOI: 10.2307/2280779](https://doi.org/10.2307/2280779)

[Back to top](#table-of-contents)

---

### brunner_munzel

Performs Brunner-Munzel test for stochastic equality. Robust alternative to Mann-Whitney U that handles unequal variances.

```rust
pub fn brunner_munzel(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    alpha: Option<f64>,
) -> Result<BrunnerMunzelResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample (minimum 2 observations) |
| `y` | `&[f64]` | Second sample (minimum 2 observations) |
| `alternative` | `Alternative` | Alternative hypothesis |
| `alpha` | `Option<f64>` | Significance level for CI (e.g., `Some(0.05)` for 95% CI) |

**Returns:** `BrunnerMunzelResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The test statistic |
| `df` | `f64` | Degrees of freedom (Welch-Satterthwaite) |
| `p_value` | `f64` | The p-value |
| `estimate` | `f64` | Estimated P(X < Y) + 0.5 * P(X = Y) |
| `conf_int` | `Option<BrunnerMunzelConfInt>` | Confidence interval for the estimate |

**R equivalent:** `brunner.munzel.test()` (lawstat)

**References:**
- Brunner, E., & Munzel, U. (2000). "The Nonparametric Behrens-Fisher Problem: Asymptotic Theory and a Small-Sample Approximation." *Biometrical Journal*, 42(1), 17–25. [DOI: 10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U](https://doi.org/10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U)
- Neubert, K., & Brunner, E. (2007). "A Studentized Permutation Test for the Non-parametric Behrens-Fisher Problem." *Computational Statistics & Data Analysis*, 51(10), 5192–5204. [DOI: 10.1016/j.csda.2006.05.024](https://doi.org/10.1016/j.csda.2006.05.024)

[Back to top](#table-of-contents)

---

## Distributional Tests

### shapiro_wilk

Performs Shapiro-Wilk test for normality. Implementation follows Algorithm AS R94 (Royston, 1995).

```rust
pub fn shapiro_wilk(data: &[f64]) -> Result<ShapiroWilkResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[f64]` | Sample data (3 ≤ n ≤ 5000) |

**Returns:** `ShapiroWilkResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The W statistic |
| `p_value` | `f64` | The p-value |

**R equivalent:** `shapiro.test()` (stats)

**References:**
- Shapiro, S. S., & Wilk, M. B. (1965). "An Analysis of Variance Test for Normality (Complete Samples)." *Biometrika*, 52(3–4), 591–611. [DOI: 10.1093/biomet/52.3-4.591](https://doi.org/10.1093/biomet/52.3-4.591)
- Royston, J. P. (1995). "Remark AS R94: A Remark on Algorithm AS 181: The W-test for Normality." *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, 44(4), 547–551. [DOI: 10.2307/2986146](https://doi.org/10.2307/2986146)

[Back to top](#table-of-contents)

---

### dagostino_k_squared

Performs D'Agostino's K-squared test for normality. Omnibus test combining tests for skewness and kurtosis.

```rust
pub fn dagostino_k_squared(data: &[f64]) -> Result<DAgostinoResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[f64]` | Sample data (n ≥ 8, recommended n ≥ 20) |

**Returns:** `DAgostinoResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The K² test statistic |
| `p_value` | `f64` | The p-value |
| `z_skewness` | `f64` | Z-score for skewness |
| `z_kurtosis` | `f64` | Z-score for kurtosis |

**R equivalent:** `agostino.test()`, `anscombe.test()` (moments)

**References:**
- D'Agostino, R. B. (1971). "An Omnibus Test of Normality for Moderate and Large Sample Size." *Biometrika*, 58(2), 341–348. [DOI: 10.2307/2334522](https://doi.org/10.2307/2334522)
- D'Agostino, R. B., & Pearson, E. S. (1973). "Tests for Departure from Normality." *Biometrika*, 60(3), 613–622. [DOI: 10.2307/2335012](https://doi.org/10.2307/2335012)
- D'Agostino, R. B., Belanger, A., & D'Agostino, R. B. Jr. (1990). "A Suggestion for Using Powerful and Informative Tests of Normality." *The American Statistician*, 44(4), 316–321. [DOI: 10.2307/2684359](https://doi.org/10.2307/2684359)

[Back to top](#table-of-contents)

---

## Correlation Tests

### pearson

Computes Pearson's product-moment correlation coefficient with significance test.

```rust
pub fn pearson(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<CorrelationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable (must have at least 3 observations) |
| `y` | `&[f64]` | Second variable (same length as x) |
| `conf_level` | `Option<f64>` | Confidence level for CI (e.g., `Some(0.95)` for 95%) |

**Returns:** `CorrelationResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Correlation coefficient |
| `statistic` | `f64` | t-statistic |
| `df` | `Option<f64>` | Degrees of freedom (n-2) |
| `p_value` | `f64` | Two-sided p-value |
| `conf_int` | `Option<CorrelationConfInt>` | Confidence interval (Fisher's z-transformation) |
| `method` | `CorrelationMethod` | Method used (`Pearson`) |
| `n` | `usize` | Sample size |

**R equivalent:** `cor.test(x, y, method = "pearson")`

**Reference:** Pearson, K. (1895). "Notes on Regression and Inheritance in the Case of Two Parents." *Proceedings of the Royal Society of London*, 58, 240–242.

[Back to top](#table-of-contents)

---

### spearman

Computes Spearman's rank correlation coefficient with significance test.

```rust
pub fn spearman(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<CorrelationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable (must have at least 3 observations) |
| `y` | `&[f64]` | Second variable (same length as x) |
| `conf_level` | `Option<f64>` | Confidence level for CI |

**Returns:** `CorrelationResult` (same structure as pearson)

**R equivalent:** `cor.test(x, y, method = "spearman")`

**Reference:** Spearman, C. (1904). "The Proof and Measurement of Association between Two Things." *American Journal of Psychology*, 15(1), 72–101.

[Back to top](#table-of-contents)

---

### kendall

Computes Kendall's tau correlation coefficient with significance test.

```rust
pub fn kendall(x: &[f64], y: &[f64], variant: KendallVariant) -> Result<CorrelationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable |
| `y` | `&[f64]` | Second variable |
| `variant` | `KendallVariant` | Which tau variant: `TauA`, `TauB` (default), or `TauC` |

**Returns:** `CorrelationResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Kendall's tau |
| `statistic` | `f64` | z-statistic (normal approximation) |
| `df` | `Option<f64>` | `None` (uses normal approximation) |
| `p_value` | `f64` | Two-sided p-value |

**R equivalent:** `cor.test(x, y, method = "kendall")` (uses tau-b)

**Reference:** Kendall, M. G. (1938). "A New Measure of Rank Correlation." *Biometrika*, 30(1/2), 81–93.

[Back to top](#table-of-contents)

---

### partial_cor

Computes partial correlation between x and y, controlling for z variables.

```rust
pub fn partial_cor(x: &[f64], y: &[f64], z: &[&[f64]]) -> Result<PartialCorResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable |
| `y` | `&[f64]` | Second variable |
| `z` | `&[&[f64]]` | Control variables (each inner slice is one variable) |

**Returns:** `PartialCorResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Partial correlation coefficient |
| `statistic` | `f64` | t-statistic |
| `df` | `f64` | Degrees of freedom (n - k - 2) |
| `p_value` | `f64` | Two-sided p-value |
| `n` | `usize` | Sample size |
| `n_controls` | `usize` | Number of control variables |

**R equivalent:** `ppcor::pcor.test(x, y, z)`

[Back to top](#table-of-contents)

---

### semi_partial_cor

Computes semi-partial (part) correlation between x and y, controlling for z on y only.

```rust
pub fn semi_partial_cor(x: &[f64], y: &[f64], z: &[&[f64]]) -> Result<PartialCorResult>
```

**Parameters:** Same as `partial_cor`

**Returns:** `PartialCorResult`

**R equivalent:** `ppcor::spcor.test(x, y, z)`

[Back to top](#table-of-contents)

---

### distance_cor

Computes distance correlation between two vectors.

```rust
pub fn distance_cor(x: &[f64], y: &[f64]) -> Result<DistanceCorResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable |
| `y` | `&[f64]` | Second variable |

**Returns:** `DistanceCorResult`

| Field | Type | Description |
|-------|------|-------------|
| `dcor` | `f64` | Distance correlation (0 to 1) |
| `dcov` | `f64` | Distance covariance |
| `dvar_x` | `f64` | Distance variance of X |
| `dvar_y` | `f64` | Distance variance of Y |
| `statistic` | `f64` | Test statistic (n × dCov²) |
| `p_value` | `Option<f64>` | `None` (use `distance_cor_test` for p-value) |
| `n` | `usize` | Sample size |

**R equivalent:** `energy::dcor(x, y)`

**Reference:** Székely, G. J., & Rizzo, M. L. (2007). "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics*, 35(6), 2769–2794.

[Back to top](#table-of-contents)

---

### distance_cor_test

Computes distance correlation with permutation test for significance.

```rust
pub fn distance_cor_test(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<DistanceCorResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable |
| `y` | `&[f64]` | Second variable |
| `n_permutations` | `usize` | Number of permutations |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `DistanceCorResult` with `p_value` populated

**R equivalent:** `energy::dcor.test(x, y, R = n_permutations)`

[Back to top](#table-of-contents)

---

### icc

Computes Intraclass Correlation Coefficient for reliability analysis.

```rust
pub fn icc(data: &[Vec<f64>], icc_type: ICCType) -> Result<ICCResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[Vec<f64>]` | Matrix where rows = subjects, columns = raters |
| `icc_type` | `ICCType` | Type of ICC to compute |

**Returns:** `ICCResult`

| Field | Type | Description |
|-------|------|-------------|
| `icc` | `f64` | ICC value (-1 to 1) |
| `icc_type` | `ICCType` | Type computed |
| `f_value` | `f64` | F-statistic |
| `df1` | `f64` | Numerator degrees of freedom |
| `df2` | `f64` | Denominator degrees of freedom |
| `p_value` | `f64` | p-value |
| `conf_int_lower` | `f64` | 95% CI lower bound |
| `conf_int_upper` | `f64` | 95% CI upper bound |
| `n_subjects` | `usize` | Number of subjects |
| `n_raters` | `usize` | Number of raters |

**R equivalent:** `psych::ICC(data)` or `irr::icc(data)`

**Reference:** Shrout, P. E., & Fleiss, J. L. (1979). "Intraclass Correlations: Uses in Assessing Rater Reliability." *Psychological Bulletin*, 86(2), 420–428.

[Back to top](#table-of-contents)

---

## Categorical Tests

### chisq_test

Pearson's chi-square test of independence for contingency tables.

```rust
pub fn chisq_test(observed: &[Vec<usize>], correction: bool) -> Result<ChiSquareResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed` | `&[Vec<usize>]` | Contingency table (at least 2×2) |
| `correction` | `bool` | Apply Yates' continuity correction (2×2 only) |

**Returns:** `ChiSquareResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | Chi-square statistic |
| `df` | `f64` | Degrees of freedom ((r-1)(c-1)) |
| `p_value` | `f64` | p-value |
| `expected` | `Vec<Vec<f64>>` | Expected frequencies |
| `residuals` | `Option<Vec<Vec<f64>>>` | Standardized residuals |

**R equivalent:** `chisq.test(matrix, correct = FALSE)`

**Reference:** Pearson, K. (1900). "On the Criterion that a Given System of Deviations from the Probable in the Case of a Correlated System of Variables is Such that it Can be Reasonably Supposed to have Arisen from Random Sampling." *The London, Edinburgh, and Dublin Philosophical Magazine*, 50(302), 157–175.

[Back to top](#table-of-contents)

---

### chisq_goodness_of_fit

Chi-square goodness-of-fit test.

```rust
pub fn chisq_goodness_of_fit(
    observed: &[usize],
    expected_props: Option<&[f64]>,
) -> Result<ChiSquareResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed` | `&[usize]` | Observed counts |
| `expected_props` | `Option<&[f64]>` | Expected proportions (must sum to 1). `None` = uniform |

**Returns:** `ChiSquareResult`

**R equivalent:** `chisq.test(x, p = expected_props)`

[Back to top](#table-of-contents)

---

### g_test

G-test (log-likelihood ratio test) for contingency tables.

```rust
pub fn g_test(observed: &[Vec<usize>]) -> Result<ChiSquareResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed` | `&[Vec<usize>]` | Contingency table |

**Returns:** `ChiSquareResult` with G statistic (asymptotically chi-square)

**R equivalent:** `DescTools::GTest(matrix)`

**Reference:** Wilks, S. S. (1935). "The Likelihood Test of Independence in Contingency Tables." *Annals of Mathematical Statistics*, 6(4), 190–196.

[Back to top](#table-of-contents)

---

### fisher_exact

Fisher's exact test for 2×2 contingency tables.

```rust
pub fn fisher_exact(table: &[[usize; 2]; 2], alternative: Alternative) -> Result<FisherResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `&[[usize; 2]; 2]` | 2×2 table `[[a, b], [c, d]]` |
| `alternative` | `Alternative` | `TwoSided`, `Less`, or `Greater` |

**Returns:** `FisherResult`

| Field | Type | Description |
|-------|------|-------------|
| `p_value` | `f64` | Exact p-value |
| `odds_ratio` | `f64` | Sample odds ratio (ad/bc) |
| `conf_int_lower` | `f64` | 95% CI lower bound for odds ratio |
| `conf_int_upper` | `f64` | 95% CI upper bound for odds ratio |
| `alternative` | `Alternative` | Alternative hypothesis |

**R equivalent:** `fisher.test(matrix)`

**Reference:** Fisher, R. A. (1922). "On the Interpretation of χ² from Contingency Tables, and the Calculation of P." *Journal of the Royal Statistical Society*, 85(1), 87–94.

[Back to top](#table-of-contents)

---

### mcnemar_test

McNemar's test for paired nominal data.

```rust
pub fn mcnemar_test(table: &[[usize; 2]; 2], correction: bool) -> Result<McNemarkResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `&[[usize; 2]; 2]` | 2×2 table of paired observations |
| `correction` | `bool` | Apply Edwards' continuity correction |

**Returns:** `McNemarkResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | Chi-square statistic |
| `df` | `f64` | Degrees of freedom (always 1) |
| `p_value` | `f64` | p-value |
| `corrected` | `bool` | Whether correction was applied |

**R equivalent:** `mcnemar.test(matrix, correct = FALSE)`

**Reference:** McNemar, Q. (1947). "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages." *Psychometrika*, 12(2), 153–157.

[Back to top](#table-of-contents)

---

### mcnemar_exact

McNemar's exact test using binomial distribution.

```rust
pub fn mcnemar_exact(table: &[[usize; 2]; 2]) -> Result<McNemarkExactResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `&[[usize; 2]; 2]` | 2×2 table of paired observations |

**Returns:** `McNemarkExactResult`

| Field | Type | Description |
|-------|------|-------------|
| `p_value` | `f64` | Exact two-sided p-value |
| `b` | `usize` | Discordant pairs (off-diagonal) |
| `c` | `usize` | Discordant pairs (off-diagonal) |

[Back to top](#table-of-contents)

---

### cramers_v

Cramér's V effect size for chi-square test.

```rust
pub fn cramers_v(observed: &[Vec<usize>]) -> Result<AssociationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed` | `&[Vec<usize>]` | Contingency table |

**Returns:** `AssociationResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Cramér's V (0 to 1) |
| `se` | `Option<f64>` | Standard error (if available) |
| `conf_int_lower` | `Option<f64>` | CI lower bound |
| `conf_int_upper` | `Option<f64>` | CI upper bound |

**Formula:** V = √(χ² / (n × min(r-1, c-1)))

**R equivalent:** `DescTools::CramerV(matrix)`

[Back to top](#table-of-contents)

---

### phi_coefficient

Phi coefficient for 2×2 contingency tables.

```rust
pub fn phi_coefficient(table: &[[usize; 2]; 2]) -> Result<AssociationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `&[[usize; 2]; 2]` | 2×2 table |

**Returns:** `AssociationResult` with phi coefficient (-1 to 1)

**Formula:** φ = (ad - bc) / √((a+b)(c+d)(a+c)(b+d))

**R equivalent:** `psych::phi(matrix)`

[Back to top](#table-of-contents)

---

### contingency_coef

Contingency coefficient (Pearson's C).

```rust
pub fn contingency_coef(observed: &[Vec<usize>]) -> Result<AssociationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed` | `&[Vec<usize>]` | Contingency table |

**Returns:** `AssociationResult` with C (0 to √((k-1)/k))

**Formula:** C = √(χ² / (χ² + n))

[Back to top](#table-of-contents)

---

### cohen_kappa

Cohen's kappa for inter-rater agreement.

```rust
pub fn cohen_kappa(table: &[Vec<usize>], weighted: bool) -> Result<KappaResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | `&[Vec<usize>]` | Square confusion matrix (rows = rater 1, cols = rater 2) |
| `weighted` | `bool` | Use weighted kappa with linear weights |

**Returns:** `KappaResult`

| Field | Type | Description |
|-------|------|-------------|
| `kappa` | `f64` | Kappa coefficient (-1 to 1) |
| `se` | `f64` | Standard error |
| `z` | `f64` | z-statistic |
| `p_value` | `f64` | Two-sided p-value |
| `conf_int_lower` | `f64` | 95% CI lower bound |
| `conf_int_upper` | `f64` | 95% CI upper bound |
| `weighted` | `bool` | Whether weighted kappa was used |

**Formula:** κ = (Po - Pe) / (1 - Pe)

**R equivalent:** `psych::cohen.kappa(matrix)`

**Reference:** Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement*, 20(1), 37–46.

[Back to top](#table-of-contents)

---

### prop_test_one

One-sample proportion test (z-test approximation).

```rust
pub fn prop_test_one(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<PropTestResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `successes` | `usize` | Number of successes |
| `n` | `usize` | Total trials |
| `p0` | `f64` | Null hypothesis proportion |
| `alternative` | `Alternative` | Alternative hypothesis |

**Returns:** `PropTestResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `Vec<f64>` | Estimated proportion(s) |
| `statistic` | `f64` | z-statistic |
| `df` | `Option<f64>` | `None` |
| `p_value` | `f64` | p-value |
| `conf_int_lower` | `f64` | Wilson score CI lower bound |
| `conf_int_upper` | `f64` | Wilson score CI upper bound |
| `null_value` | `f64` | Null proportion |
| `alternative` | `Alternative` | Alternative hypothesis |

**R equivalent:** `prop.test(x, n, p = p0, correct = FALSE)`

[Back to top](#table-of-contents)

---

### prop_test_two

Two-sample proportion test.

```rust
pub fn prop_test_two(
    successes: [usize; 2],
    totals: [usize; 2],
    alternative: Alternative,
    correction: bool,
) -> Result<PropTestResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `successes` | `[usize; 2]` | Successes in each group |
| `totals` | `[usize; 2]` | Total trials in each group |
| `alternative` | `Alternative` | Alternative hypothesis |
| `correction` | `bool` | Apply Yates' continuity correction |

**Returns:** `PropTestResult` with chi-square statistic (z²)

**R equivalent:** `prop.test(c(x1, x2), c(n1, n2))`

[Back to top](#table-of-contents)

---

### binom_test

Exact binomial test.

```rust
pub fn binom_test(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<BinomTestResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `successes` | `usize` | Number of successes |
| `n` | `usize` | Total trials |
| `p0` | `f64` | Null hypothesis probability |
| `alternative` | `Alternative` | Alternative hypothesis |

**Returns:** `BinomTestResult`

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Estimated probability |
| `successes` | `usize` | Number of successes |
| `n` | `usize` | Total trials |
| `p_value` | `f64` | Exact p-value |
| `conf_int_lower` | `f64` | Clopper-Pearson CI lower bound |
| `conf_int_upper` | `f64` | Clopper-Pearson CI upper bound |
| `null_value` | `f64` | Null probability |
| `alternative` | `Alternative` | Alternative hypothesis |

**R equivalent:** `binom.test(x, n, p = p0)`

[Back to top](#table-of-contents)

---

## Resampling Methods

### permutation_t_test

Performs permutation-based t-test.

```rust
pub fn permutation_t_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<PermutationResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `alternative` | `Alternative` | Alternative hypothesis |
| `n_permutations` | `usize` | Number of permutations |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `PermutationResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The observed test statistic |
| `p_value` | `f64` | The p-value from permutation distribution |
| `n_permutations` | `usize` | Number of permutations used |

**References:**
- Fisher, R. A. (1935). *The Design of Experiments.* Oliver and Boyd.
- Pitman, E. J. G. (1937). "Significance Tests Which May be Applied to Samples from Any Populations." *Supplement to the Journal of the Royal Statistical Society*, 4(1), 119–130. [DOI: 10.2307/2984124](https://doi.org/10.2307/2984124)

[Back to top](#table-of-contents)

---

### PermutationEngine

Generic permutation engine for custom test statistics.

```rust
impl PermutationEngine {
    pub fn new(n_permutations: usize, seed: Option<u64>) -> Self;

    pub fn test<F>(
        &self,
        x: &[f64],
        y: &[f64],
        statistic_fn: F,
        alternative: Alternative,
    ) -> Result<PermutationResult>
    where
        F: Fn(&[f64], &[f64]) -> f64;
}
```

[Back to top](#table-of-contents)

---

### StationaryBootstrap

Stationary bootstrap for dependent data (Politis & Romano, 1994).

```rust
impl StationaryBootstrap {
    pub fn new(expected_block_length: f64, seed: Option<u64>) -> Self;

    pub fn resample(&mut self, data: &[f64]) -> Vec<f64>;
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_block_length` | `f64` | Expected block length (determines block switching probability) |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Reference:** Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association*, 89(428), 1303–1313. [DOI: 10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)

[Back to top](#table-of-contents)

---

### CircularBlockBootstrap

Circular block bootstrap for dependent data.

```rust
impl CircularBlockBootstrap {
    pub fn new(block_length: usize, seed: Option<u64>) -> Self;

    pub fn resample(&mut self, data: &[f64]) -> Vec<f64>;
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_length` | `usize` | Fixed block length |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**References:**
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *Annals of Statistics*, 17(3), 1217–1241. [DOI: 10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)
- Politis, D. N., & Romano, J. P. (1992). "A Circular Block-Resampling Procedure for Stationary Data." In R. LePage & L. Billard (Eds.), *Exploring the Limits of Bootstrap* (pp. 263–270). Wiley.

[Back to top](#table-of-contents)

---

## Modern Distribution Tests

### energy_distance_test

Performs energy distance two-sample test for multivariate data.

```rust
pub fn energy_distance_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[Vec<f64>]` | First sample (n₁ observations, each d-dimensional) |
| `y` | `&[Vec<f64>]` | Second sample (n₂ observations, each d-dimensional) |
| `n_permutations` | `usize` | Number of permutations |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `EnergyDistanceResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The energy distance statistic |
| `p_value` | `f64` | The p-value from permutation test |
| `n_permutations` | `usize` | Number of permutations used |

**References:**
- Székely, G. J., & Rizzo, M. L. (2004). "Testing for Equal Distributions in High Dimension." *InterStat*, November (5).
- Székely, G. J., & Rizzo, M. L. (2013). "Energy Statistics: A Class of Statistics Based on Distances." *Journal of Statistical Planning and Inference*, 143(8), 1249–1272. [DOI: 10.1016/j.jspi.2013.03.018](https://doi.org/10.1016/j.jspi.2013.03.018)

[Back to top](#table-of-contents)

---

### mmd_test

Performs Maximum Mean Discrepancy two-sample test.

```rust
pub fn mmd_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    kernel: Kernel,
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<MMDResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[Vec<f64>]` | First sample |
| `y` | `&[Vec<f64>]` | Second sample |
| `kernel` | `Kernel` | Kernel function to use |
| `n_permutations` | `usize` | Number of permutations |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `MMDResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The MMD² test statistic (unbiased estimator) |
| `p_value` | `f64` | The p-value from permutation test |
| `n_permutations` | `usize` | Number of permutations used |

**Reference:** Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). "A Kernel Two-Sample Test." *Journal of Machine Learning Research*, 13, 723–773. [PDF](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)

[Back to top](#table-of-contents)

---

## Forecast Evaluation

### diebold_mariano

Performs Diebold-Mariano test for comparing forecast accuracy.

```rust
pub fn diebold_mariano(
    e1: &[f64],
    e2: &[f64],
    loss: LossFunction,
    h: usize,
    alternative: Alternative,
    varestimator: VarEstimator,
) -> Result<DMResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `e1` | `&[f64]` | Forecast errors from model 1 |
| `e2` | `&[f64]` | Forecast errors from model 2 |
| `loss` | `LossFunction` | Loss function: `SquaredError` or `AbsoluteError` |
| `h` | `usize` | Forecast horizon (for variance adjustment) |
| `alternative` | `Alternative` | Alternative hypothesis |
| `varestimator` | `VarEstimator` | Variance estimator: `Acf` or `Bartlett` |

**Returns:** `DMResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The DM test statistic |
| `p_value` | `f64` | The p-value |
| `horizon` | `usize` | Forecast horizon used |
| `loss_function` | `LossFunction` | Loss function used |
| `varestimator` | `VarEstimator` | Variance estimator used |
| `alternative` | `Alternative` | Alternative hypothesis tested |

**R equivalent:** `dm.test()` (forecast)

**Reference:** Diebold, F. X., & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253–263. [DOI: 10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)

[Back to top](#table-of-contents)

---

### clark_west

Performs Clark-West test for comparing forecasts from nested models.

```rust
pub fn clark_west(e1: &[f64], e2: &[f64], h: usize) -> Result<CWResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `e1` | `&[f64]` | Forecast errors from restricted (null) model |
| `e2` | `&[f64]` | Forecast errors from unrestricted (alternative) model |
| `h` | `usize` | Forecast horizon (for HAC variance adjustment) |

**Returns:** `CWResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | The Clark-West adjusted test statistic |
| `p_value` | `f64` | One-sided p-value (H₁: unrestricted is better) |
| `p_value_two_sided` | `f64` | Two-sided p-value |

**References:**
- Clark, T. E., & West, K. D. (2006). "Using Out-of-Sample Mean Squared Prediction Errors to Test the Martingale Difference Hypothesis." *Journal of Econometrics*, 135(1–2), 155–186. [DOI: 10.1016/j.jeconom.2005.07.014](https://doi.org/10.1016/j.jeconom.2005.07.014)
- Clark, T. E., & West, K. D. (2007). "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *Journal of Econometrics*, 138(1), 291–311. [DOI: 10.1016/j.jeconom.2006.05.023](https://doi.org/10.1016/j.jeconom.2006.05.023)

[Back to top](#table-of-contents)

---

### spa_test

Performs Superior Predictive Ability test for multiple model comparison.

```rust
pub fn spa_test(
    benchmark_losses: &[f64],
    model_losses: &[Vec<f64>],
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<SPAResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `benchmark_losses` | `&[f64]` | Loss values from benchmark model (length T) |
| `model_losses` | `&[Vec<f64>]` | Loss values from K competing models (K × T) |
| `n_bootstrap` | `usize` | Number of bootstrap samples |
| `block_length` | `f64` | Expected block length for stationary bootstrap |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `SPAResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | Maximum standardized performance |
| `p_value_consistent` | `f64` | Consistent p-value (Hansen, 2005) |
| `p_value_upper` | `f64` | Upper p-value (more conservative) |
| `n_bootstrap` | `usize` | Number of bootstrap samples used |
| `best_model_idx` | `Option<usize>` | Index of best performing model |

**Reference:** Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics*, 23(4), 365–380. [DOI: 10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063)

[Back to top](#table-of-contents)

---

### mspe_adjusted_spa

Performs MSPE-Adjusted SPA test combining Clark-West adjustment with bootstrap for nested models.

```rust
pub fn mspe_adjusted_spa(
    benchmark_errors: &[f64],
    model_errors: &[Vec<f64>],
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<MSPEAdjustedResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `benchmark_errors` | `&[f64]` | Forecast errors from benchmark (restricted) model |
| `model_errors` | `&[Vec<f64>]` | Forecast errors from K alternative models |
| `n_bootstrap` | `usize` | Number of bootstrap samples |
| `block_length` | `f64` | Expected block length for stationary bootstrap |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `MSPEAdjustedResult`

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `f64` | Maximum standardized Clark-West adjusted performance |
| `p_value_consistent` | `f64` | Consistent p-value |
| `p_value_upper` | `f64` | Upper p-value (conservative) |
| `n_bootstrap` | `usize` | Number of bootstrap samples used |
| `best_model_idx` | `Option<usize>` | Index of best performing model |

[Back to top](#table-of-contents)

---

### model_confidence_set

Performs Model Confidence Set procedure to identify the set of best models.

```rust
pub fn model_confidence_set(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<MCSResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `losses` | `&[Vec<f64>]` | Loss values for K models (K × T) |
| `alpha` | `f64` | Significance level for elimination (e.g., 0.10) |
| `statistic` | `MCSStatistic` | Test statistic type: `Range` or `Max` |
| `n_bootstrap` | `usize` | Number of bootstrap samples |
| `block_length` | `f64` | Expected block length for stationary bootstrap |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**Returns:** `MCSResult`

| Field | Type | Description |
|-------|------|-------------|
| `included_models` | `Vec<usize>` | Model indices in the confidence set |
| `eliminated_models` | `Vec<usize>` | Model indices eliminated from the set |
| `mcs_p_value` | `f64` | P-value when elimination stopped |
| `elimination_sequence` | `Vec<MCSEliminationStep>` | Full elimination history |
| `n_bootstrap` | `usize` | Number of bootstrap samples used |
| `statistic_type` | `MCSStatistic` | Statistic type used |

**Reference:** Hansen, P. R., Lunde, A., & Nason, J. M. (2011). "The Model Confidence Set." *Econometrica*, 79(2), 453–497. [DOI: 10.3982/ECTA5771](https://doi.org/10.3982/ECTA5771)

[Back to top](#table-of-contents)

---

## Equivalence Testing (TOST)

TOST (Two One-Sided Tests) is used to test equivalence hypotheses, where the goal is to demonstrate that an effect is small enough to be considered practically equivalent to zero (or some other value).

### tost_t_test_one_sample

One-sample TOST to test if a mean is equivalent to a specified value.

```rust
pub fn tost_t_test_one_sample(
    x: &[f64],
    mu: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | Sample data |
| `mu` | `f64` | Value to test equivalence against (usually 0) |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds specification |
| `alpha` | `f64` | Significance level (e.g., 0.05) |

**R equivalent:** `TOSTER::TOSTone()`

[Back to top](#table-of-contents)

---

### tost_t_test_two_sample

Two-sample TOST to test if two means are equivalent.

```rust
pub fn tost_t_test_two_sample(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    pooled: bool,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds specification |
| `alpha` | `f64` | Significance level |
| `pooled` | `bool` | If true, use pooled variance (Student's t); if false, use Welch's t |

**R equivalent:** `TOSTER::TOSTtwo()`

[Back to top](#table-of-contents)

---

### tost_t_test_paired

Paired-samples TOST to test if paired differences are equivalent to zero.

```rust
pub fn tost_t_test_paired(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample (same length as x) |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds specification |
| `alpha` | `f64` | Significance level |

**R equivalent:** `TOSTER::TOSTpaired()`

[Back to top](#table-of-contents)

---

### tost_correlation

TOST for a correlation coefficient.

```rust
pub fn tost_correlation(
    x: &[f64],
    y: &[f64],
    rho_null: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
    method: CorrelationTostMethod,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First variable |
| `y` | `&[f64]` | Second variable |
| `rho_null` | `f64` | Null correlation value (usually 0) |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds (Raw or Symmetric only) |
| `alpha` | `f64` | Significance level |
| `method` | `CorrelationTostMethod` | Pearson or Spearman |

**R equivalent:** `TOSTER::TOSTr()`

[Back to top](#table-of-contents)

---

### tost_prop_one

TOST for a single proportion.

```rust
pub fn tost_prop_one(
    x: usize,
    n: usize,
    p0: f64,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `usize` | Number of successes |
| `n` | `usize` | Total trials |
| `p0` | `f64` | Null proportion to test against |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds (Raw or Symmetric only) |
| `alpha` | `f64` | Significance level |

[Back to top](#table-of-contents)

---

### tost_prop_two

TOST for two independent proportions.

```rust
pub fn tost_prop_two(
    x1: usize,
    n1: usize,
    x2: usize,
    n2: usize,
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x1` | `usize` | Number of successes in group 1 |
| `n1` | `usize` | Total trials in group 1 |
| `x2` | `usize` | Number of successes in group 2 |
| `n2` | `usize` | Total trials in group 2 |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds (Raw or Symmetric only) |
| `alpha` | `f64` | Significance level |

**R equivalent:** `TOSTER::TOSTtwo.prop()`

[Back to top](#table-of-contents)

---

### tost_wilcoxon_paired

Non-parametric TOST for paired samples using Wilcoxon signed-rank test.

```rust
pub fn tost_wilcoxon_paired(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample (same length as x) |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds (Raw or Symmetric only) |
| `alpha` | `f64` | Significance level |

**R equivalent:** `TOSTER::wilcox_TOST(paired = TRUE)`

[Back to top](#table-of-contents)

---

### tost_wilcoxon_two_sample

Non-parametric TOST for two independent samples using Wilcoxon rank-sum test.

```rust
pub fn tost_wilcoxon_two_sample(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds (Raw or Symmetric only) |
| `alpha` | `f64` | Significance level |

**R equivalent:** `TOSTER::wilcox_TOST(paired = FALSE)`

[Back to top](#table-of-contents)

---

### tost_bootstrap

Bootstrap TOST for two independent samples.

```rust
pub fn tost_bootstrap(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds specification |
| `alpha` | `f64` | Significance level |
| `n_bootstrap` | `usize` | Number of bootstrap samples (≥ 100) |
| `seed` | `Option<u64>` | Random seed for reproducibility |

**R equivalent:** `TOSTER::boot_t_TOST()`

[Back to top](#table-of-contents)

---

### tost_yuen

Robust TOST using Yuen's trimmed means.

```rust
pub fn tost_yuen(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
    trim: f64,
) -> Result<TostResult>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `&[f64]` | First sample |
| `y` | `&[f64]` | Second sample |
| `bounds` | `&EquivalenceBounds` | Equivalence bounds specification |
| `alpha` | `f64` | Significance level |
| `trim` | `f64` | Proportion to trim from each tail (0 to 0.5) |

**R equivalent:** `WRS2::yuen.TOST()`

[Back to top](#table-of-contents)

---

### TostResult

Result structure returned by all TOST functions.

| Field | Type | Description |
|-------|------|-------------|
| `estimate` | `f64` | Point estimate of the effect |
| `ci` | `(f64, f64)` | Confidence interval at (1 - 2α) level |
| `bounds` | `(f64, f64)` | Equivalence bounds used (lower, upper) |
| `lower_test` | `OneSidedTestResult` | Result of lower bound test |
| `upper_test` | `OneSidedTestResult` | Result of upper bound test |
| `tost_p_value` | `f64` | TOST p-value: max(p_lower, p_upper) |
| `equivalent` | `bool` | Whether equivalence was established |
| `alpha` | `f64` | Significance level used |
| `n` | `usize` | Sample size |
| `df` | `Option<f64>` | Degrees of freedom (if applicable) |
| `method` | `String` | Name of the test method |

### OneSidedTestResult

| Field | Type | Description |
|-------|------|-------------|
| `hypothesis` | `String` | Null hypothesis description |
| `statistic` | `f64` | Test statistic |
| `p_value` | `f64` | p-value for this one-sided test |
| `rejected` | `bool` | Whether null was rejected |

**Reference:** Schuirmann, D. J. (1987). "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.

[Back to top](#table-of-contents)

---

## Math Primitives

### mean

Computes arithmetic mean.

```rust
pub fn mean(data: &[f64]) -> Result<f64>
```

[Back to top](#table-of-contents)

---

### stable_mean

Computes arithmetic mean using Welford's online algorithm. Numerically stable for data with large magnitude or many observations.

```rust
pub fn stable_mean(data: &[f64]) -> Result<f64>
```

**Reference:** Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics*, 4(3), 419–420. [DOI: 10.2307/1266577](https://doi.org/10.2307/1266577)

[Back to top](#table-of-contents)

---

### variance

Computes sample variance (n-1 denominator).

```rust
pub fn variance(data: &[f64]) -> Result<f64>
```

[Back to top](#table-of-contents)

---

### stable_variance

Computes sample variance using Welford's online algorithm. Numerically stable for data with large magnitude or small variance relative to mean.

```rust
pub fn stable_variance(data: &[f64]) -> Result<f64>
```

**Reference:** Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics*, 4(3), 419–420. [DOI: 10.2307/1266577](https://doi.org/10.2307/1266577)

[Back to top](#table-of-contents)

---

### std_dev

Computes sample standard deviation (n-1 denominator).

```rust
pub fn std_dev(data: &[f64]) -> Result<f64>
```

[Back to top](#table-of-contents)

---

### median

Computes median.

```rust
pub fn median(data: &[f64]) -> Result<f64>
```

[Back to top](#table-of-contents)

---

### trimmed_mean

Computes trimmed mean.

```rust
pub fn trimmed_mean(data: &[f64], trim: f64) -> Result<f64>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `&[f64]` | Input data |
| `trim` | `f64` | Proportion to trim from each tail, must be in `[0, 0.5)` |

[Back to top](#table-of-contents)

---

### skewness

Computes sample skewness (Fisher's definition, type 2, matching R's e1071).

```rust
pub fn skewness(data: &[f64]) -> Result<f64>
```

**R equivalent:** `skewness(type=2)` (e1071)

[Back to top](#table-of-contents)

---

### kurtosis

Computes sample excess kurtosis (Fisher's definition, type 2, matching R's e1071).

```rust
pub fn kurtosis(data: &[f64]) -> Result<f64>
```

**R equivalent:** `kurtosis(type=2)` (e1071)

[Back to top](#table-of-contents)

---

## Enums

### Alternative

Alternative hypothesis direction for hypothesis tests.

```rust
pub enum Alternative {
    TwoSided,  // x ≠ y
    Less,      // x < y
    Greater,   // x > y
}
```

[Back to top](#table-of-contents)

---

### TTestKind

Type of t-test to perform.

```rust
pub enum TTestKind {
    Welch,    // Independent samples, unequal variances
    Student,  // Independent samples, equal variances assumed
    Paired,   // Paired samples
}
```

[Back to top](#table-of-contents)

---

### AnovaKind

Type of one-way ANOVA to perform.

```rust
pub enum AnovaKind {
    Fisher,  // Classic ANOVA (assumes equal variances)
    Welch,   // Welch's ANOVA (robust to unequal variances)
}
```

[Back to top](#table-of-contents)

---

### LossFunction

Loss function for forecast comparison tests.

```rust
pub enum LossFunction {
    SquaredError,   // (e)²
    AbsoluteError,  // |e|
}
```

[Back to top](#table-of-contents)

---

### VarEstimator

Variance estimator for Diebold-Mariano test.

```rust
pub enum VarEstimator {
    Acf,      // ACF-based estimator (default) - uses unweighted autocovariances
    Bartlett, // Bartlett kernel estimator - uses Bartlett weights for positive variance
}
```

[Back to top](#table-of-contents)

---

### MCSStatistic

Test statistic type for Model Confidence Set.

```rust
pub enum MCSStatistic {
    Range,  // T_R: max_{i,j} |t_{ij}| - best against one clearly inferior model
    Max,    // T_max: max_i of average t-statistics - balanced for multiple inferior models
}
```

[Back to top](#table-of-contents)

---

### Kernel

Kernel types for MMD test.

```rust
pub enum Kernel {
    Gaussian { bandwidth: f64 },                           // RBF: exp(-||x-y||²/(2σ²))
    Linear,                                                 // x·y
    Polynomial { degree: u32, scale: f64, offset: f64 },   // (scale·x·y + offset)^degree
    Laplacian { bandwidth: f64 },                          // exp(-||x-y||/σ)
}
```

[Back to top](#table-of-contents)

---

### CorrelationMethod

Correlation method used in correlation tests.

```rust
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}
```

[Back to top](#table-of-contents)

---

### KendallVariant

Variant of Kendall's tau to compute.

```rust
pub enum KendallVariant {
    TauA,   // No tie adjustment
    TauB,   // Tie-adjusted (default, matches R)
    TauC,   // Stuart's tau-c for rectangular tables
}
```

[Back to top](#table-of-contents)

---

### ICCType

Type of Intraclass Correlation Coefficient to compute.

```rust
pub enum ICCType {
    ICC1,   // One-way random effects, absolute agreement, single rater
    ICC2,   // Two-way random effects, absolute agreement, single rater (default)
    ICC3,   // Two-way mixed effects, consistency, single rater
    ICC1k,  // One-way random effects, absolute agreement, average of k raters
    ICC2k,  // Two-way random effects, absolute agreement, average of k raters
    ICC3k,  // Two-way mixed effects, consistency, average of k raters
}
```

[Back to top](#table-of-contents)

---

### EquivalenceBounds

Specification of equivalence bounds for TOST tests.

```rust
pub enum EquivalenceBounds {
    Raw { lower: f64, upper: f64 },  // Asymmetric bounds in raw units
    Symmetric { delta: f64 },        // Symmetric ±delta bounds
    CohenD { d: f64 },               // Bounds as ±d standard deviations
}
```

| Variant | Description |
|---------|-------------|
| `Raw` | Asymmetric bounds specified in raw measurement units |
| `Symmetric` | Symmetric bounds ±delta in raw units |
| `CohenD` | Bounds as Cohen's d effect sizes (converted to raw using pooled SD) |

**Note:** For correlation and proportion tests, only `Raw` and `Symmetric` bounds are allowed.

[Back to top](#table-of-contents)

---

### CorrelationTostMethod

Method for computing correlation in TOST.

```rust
pub enum CorrelationTostMethod {
    Pearson,   // Pearson product-moment correlation
    Spearman,  // Spearman rank correlation
}
```

[Back to top](#table-of-contents)
