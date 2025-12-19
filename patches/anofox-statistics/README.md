# anofox-statistics

[![CI](https://github.com/sipemu/anofox-statistics-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-statistics-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-statistics.svg)](https://crates.io/crates/anofox-statistics)
[![Documentation](https://docs.rs/anofox-statistics/badge.svg)](https://docs.rs/anofox-statistics)
[![codecov](https://codecov.io/gh/sipemu/anofox-statistics-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/stattests-rs)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Code Health](https://img.shields.io/badge/code%20health-85%25-brightgreen)](.)

A statistical hypothesis testing library for Rust, validated against R ([VALIDATION](R/VALIDATION.md)).

This library provides a wide range of statistical tests commonly used in data analysis, all validated against R's implementations to ensure numerical accuracy.

## Features

- **Math Primitives**
  - Mean, variance, standard deviation, median
  - Numerically stable mean and variance (Welford's algorithm)
  - Trimmed mean (robust to outliers)
  - Skewness and kurtosis (Fisher's definition, matching R's e1071)

- **Parametric Tests**
  - T-tests (Welch, Student, Paired) with all alternatives
  - Yuen's test (robust t-test using trimmed means)
  - Brown-Forsythe test (homogeneity of variances)
  - One-way ANOVA (Fisher's and Welch's)
  - Two-way ANOVA (factorial design with Type III SS)
  - Repeated measures ANOVA (with Mauchly's sphericity test and GG/HF corrections)

- **Nonparametric Tests**
  - Ranking with average tie handling
  - Mann-Whitney U test (Wilcoxon rank-sum)
  - Wilcoxon signed-rank test (paired)
  - Kruskal-Wallis test (k-sample)
  - Brunner-Munzel test (robust rank-based test for stochastic equality)

- **Distributional Tests**
  - Shapiro-Wilk normality test (Royston AS R94)
  - D'Agostino's K-squared test (omnibus normality test using skewness and kurtosis)

- **Correlation Analysis**
  - Pearson's product-moment correlation with CI
  - Spearman's rank correlation
  - Kendall's tau (tau-a, tau-b, tau-c variants)
  - Partial and semi-partial correlation
  - Distance correlation (detects non-linear dependence)
  - Intraclass correlation coefficient (ICC, 6 variants)

- **Categorical Data Analysis**
  - Chi-square test (independence and goodness-of-fit)
  - Fisher's exact test for 2x2 tables
  - G-test (log-likelihood ratio)
  - McNemar's test (standard and exact)
  - Effect sizes: Cramér's V, phi coefficient, contingency coefficient
  - Cohen's kappa (unweighted and weighted)
  - Proportion tests (one-sample and two-sample)
  - Exact binomial test

- **Resampling Methods**
  - Permutation engine with custom statistics
  - Permutation t-test
  - Stationary bootstrap (for dependent data)
  - Circular block bootstrap

- **Modern Distribution Tests**
  - Energy distance test (univariate and multivariate)
  - Maximum Mean Discrepancy (MMD) with multiple kernels (Gaussian, Linear, Polynomial, Laplacian)

- **Forecast Evaluation**
  - Diebold-Mariano test for comparing predictive accuracy
  - Clark-West test for nested model comparison
  - Superior Predictive Ability (SPA) test for multiple model comparison
  - MSPE-Adjusted SPA test for multiple nested models (Clark-West + bootstrap)
  - Model Confidence Set (Hansen, Lunde, & Nason, 2011)

- **Equivalence Testing (TOST)**
  - TOST for means: one-sample, two-sample, and paired t-tests
  - TOST for correlations (Pearson and Spearman)
  - TOST for proportions (one-sample and two-sample)
  - Wilcoxon TOST (non-parametric equivalence)
  - Bootstrap TOST (resampling-based)
  - Yuen TOST (robust trimmed means)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
anofox-statistics = "0.2"
```

## Examples

The library includes runnable examples demonstrating each major feature:

```bash
cargo run --example quickstart      # Overview: t-test, Mann-Whitney, Shapiro-Wilk, permutation test
cargo run --example parametric      # T-tests, Yuen's robust test, Brown-Forsythe
cargo run --example nonparametric   # Ranking, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Brunner-Munzel
cargo run --example normality       # Shapiro-Wilk, D'Agostino's K-squared
cargo run --example resampling      # Permutation tests, bootstrap methods
cargo run --example modern          # Energy distance, MMD with different kernels
cargo run --example forecast        # Diebold-Mariano, Clark-West, SPA, MCS
cargo run --example correlation     # Pearson, Spearman, Kendall, partial, distance, ICC
cargo run --example categorical     # Chi-square, Fisher, McNemar, Cramér's V, kappa
```

## Quick Start

### T-Tests

```rust
use anofox_statistics::{t_test, TTestKind, Alternative};

let group1 = vec![1.2, 2.3, 3.4, 4.5, 5.6];
let group2 = vec![2.1, 3.2, 4.3, 5.4, 6.5];

// Welch t-test (unequal variances), mu=0.0 tests if mean difference equals zero
let result = t_test(&group1, &group2, TTestKind::Welch, Alternative::TwoSided, 0.0, None)
    .expect("t-test should succeed");

println!("t-statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
println!("degrees of freedom: {:.4}", result.df);

// Student t-test (equal variances assumed)
let result = t_test(&group1, &group2, TTestKind::Student, Alternative::TwoSided, 0.0, None)?;

// Paired t-test
let result = t_test(&group1, &group2, TTestKind::Paired, Alternative::Less, 0.0, None)?;

// Test if mean difference equals 0.5 (non-zero null hypothesis)
let result = t_test(&group1, &group2, TTestKind::Welch, Alternative::TwoSided, 0.5, None)?;

// T-test with 95% confidence interval
let result = t_test(&group1, &group2, TTestKind::Welch, Alternative::TwoSided, 0.0, Some(0.95))?;
if let Some(ci) = result.conf_int {
    println!("95% CI: [{:.3}, {:.3}]", ci.lower, ci.upper);
}
```

### Yuen's Robust T-Test

```rust
use anofox_statistics::{yuen_test, Alternative};

// 20% trimmed means (robust to outliers)
let result = yuen_test(&group1, &group2, 0.2, Alternative::TwoSided)?;

println!("Test statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Brown-Forsythe Test

```rust
use anofox_statistics::brown_forsythe;

let groups = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
    vec![7.0, 8.0, 9.0],
];

let result = brown_forsythe(&groups)?;

println!("F-statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### One-Way ANOVA

```rust
use anofox_statistics::{one_way_anova, AnovaKind};

let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let group2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let group3 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
let groups: Vec<&[f64]> = vec![&group1, &group2, &group3];

// Fisher's ANOVA (assumes equal variances)
let result = one_way_anova(&groups, AnovaKind::Fisher)?;
println!("F-statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
println!("Group means: {:?}", result.group_means);

// Welch's ANOVA (robust to unequal variances)
let result = one_way_anova(&groups, AnovaKind::Welch)?;
```

### Two-Way ANOVA

```rust
use anofox_statistics::two_way_anova;

// Values with factor level arrays (long format)
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let factor_a = vec![0, 0, 0, 0, 1, 1, 1, 1];  // 2 levels
let factor_b = vec![0, 0, 1, 1, 0, 0, 1, 1];  // 2 levels

let result = two_way_anova(&values, &factor_a, &factor_b)?;

println!("Factor A: F={:.4}, p={:.4}", result.factor_a.f_statistic.unwrap(), result.factor_a.p_value.unwrap());
println!("Factor B: F={:.4}, p={:.4}", result.factor_b.f_statistic.unwrap(), result.factor_b.p_value.unwrap());
println!("Interaction: F={:.4}, p={:.4}", result.interaction.f_statistic.unwrap(), result.interaction.p_value.unwrap());
```

### Repeated Measures ANOVA

```rust
use anofox_statistics::repeated_measures_anova;

// Matrix format: rows = subjects, columns = conditions
let subject1 = vec![1.0, 2.0, 3.0];
let subject2 = vec![2.0, 3.0, 4.0];
let subject3 = vec![1.5, 2.5, 3.5];
let data: Vec<&[f64]> = vec![&subject1, &subject2, &subject3];

let result = repeated_measures_anova(&data, true)?;  // compute sphericity

println!("F-statistic: {:.4}", result.within_subjects.f_statistic.unwrap());
println!("p-value: {:.4}", result.within_subjects.p_value.unwrap());
println!("Condition means: {:?}", result.condition_means);

// Sphericity test (Mauchly's W) - only for k >= 3 conditions
if let Some(sphericity) = &result.sphericity {
    println!("Mauchly's W: {:.4}, p={:.4}", sphericity.w, sphericity.p_value);
}

// Greenhouse-Geisser corrected p-value
if let Some(gg) = &result.greenhouse_geisser {
    println!("GG epsilon: {:.4}, corrected p={:.4}", gg.epsilon, gg.p_value);
}
```

### Nonparametric Tests

```rust
use anofox_statistics::{mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis, rank, brunner_munzel, Alternative};

// Ranking
let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
let ranks = rank(&data)?;

// Mann-Whitney U test (two-sided, no continuity correction, normal approximation)
let result = mann_whitney_u(&group1, &group2, Alternative::TwoSided, false, false, None, None)?;

// With exact p-values (for small samples without ties)
let result = mann_whitney_u(&group1, &group2, Alternative::TwoSided, false, true, None, None)?;

// With confidence interval (Hodges-Lehmann estimate)
let result = mann_whitney_u(&group1, &group2, Alternative::TwoSided, false, true, Some(0.95), None)?;
if let Some(ci) = result.conf_int {
    println!("95% CI: [{:.3}, {:.3}]", ci.lower, ci.upper);
}

// Test if location shift equals 0.5 (non-zero null hypothesis)
let result = mann_whitney_u(&group1, &group2, Alternative::TwoSided, false, false, None, Some(0.5))?;

// Wilcoxon signed-rank test (paired)
let result = wilcoxon_signed_rank(&group1, &group2, Alternative::TwoSided, false, false, None, None)?;

// Wilcoxon with non-zero null hypothesis (test if median difference equals 0.5)
let result = wilcoxon_signed_rank(&group1, &group2, Alternative::TwoSided, false, false, None, Some(0.5))?;

// Kruskal-Wallis test
let result = kruskal_wallis(&groups)?;

// Brunner-Munzel test (robust alternative to Mann-Whitney)
let result = brunner_munzel(&group1, &group2, Alternative::TwoSided, None)?;
println!("Estimate P(X < Y): {:.4}", result.estimate);

// Brunner-Munzel with 95% confidence interval
let result = brunner_munzel(&group1, &group2, Alternative::TwoSided, Some(0.05))?;
if let Some(ci) = result.conf_int {
    println!("95% CI for P(X < Y): [{:.3}, {:.3}]", ci.lower, ci.upper);
}
```

### Normality Tests

```rust
use anofox_statistics::{shapiro_wilk, dagostino_k_squared};

let data = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];

// Shapiro-Wilk test
let result = shapiro_wilk(&data)?;
println!("W statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// D'Agostino's K-squared test (omnibus test using skewness and kurtosis)
let result = dagostino_k_squared(&data)?;
println!("K-squared: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Resampling Methods

```rust
use anofox_statistics::resampling::{permutation_t_test, StationaryBootstrap, CircularBlockBootstrap};

// Permutation t-test
let result = permutation_t_test(&group1, &group2, 10000, Some(42))?;
println!("p-value: {:.4}", result.p_value);

// Stationary bootstrap for time series
let bootstrap = StationaryBootstrap::new(&time_series, 10.0, Some(42))?;
let samples: Vec<Vec<f64>> = bootstrap.take(1000).collect();

// Circular block bootstrap
let bootstrap = CircularBlockBootstrap::new(&time_series, 5, Some(42))?;
```

### Modern Distribution Tests

```rust
use anofox_statistics::modern::{energy_distance_test, mmd_test, Kernel};

// Energy distance test
let result = energy_distance_test(&sample1, &sample2, 1000, Some(42))?;
println!("Energy distance: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// Maximum Mean Discrepancy with Gaussian kernel
let result = mmd_test(&sample1, &sample2, Kernel::Gaussian(1.0), 1000, Some(42))?;
println!("MMD: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// MMD with automatic bandwidth selection
let result = mmd_test(&sample1, &sample2, Kernel::GaussianMedian, 1000, Some(42))?;
```

### Correlation Analysis

```rust
use anofox_statistics::{pearson, spearman, kendall, partial_cor, distance_cor, icc,
                        KendallVariant, ICCType};

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9, 18.1, 19.9];

// Pearson correlation with 95% CI
let result = pearson(&x, &y, Some(0.95))?;
println!("Pearson r = {:.4}, p = {:.4}", result.estimate, result.p_value);

// Spearman rank correlation
let result = spearman(&x, &y, None)?;
println!("Spearman rho = {:.4}", result.estimate);

// Kendall's tau-b (default, matches R)
let result = kendall(&x, &y, KendallVariant::TauB)?;
println!("Kendall tau = {:.4}", result.estimate);

// Partial correlation (controlling for z)
let z = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
let result = partial_cor(&x, &y, &[&z])?;
println!("Partial r = {:.4}", result.estimate);

// Distance correlation (detects non-linear dependence)
let result = distance_cor(&x, &y)?;
println!("Distance correlation = {:.4}", result.dcor);

// ICC for inter-rater reliability
let ratings = vec![
    vec![9.0, 2.0, 5.0, 8.0],
    vec![6.0, 1.0, 3.0, 2.0],
    vec![8.0, 4.0, 6.0, 8.0],
];
let result = icc(&ratings, ICCType::ICC2)?;
println!("ICC(2,1) = {:.4}", result.icc);
```

### Categorical Data Analysis

```rust
use anofox_statistics::{chisq_test, chisq_goodness_of_fit, fisher_exact, mcnemar_test,
                        cramers_v, phi_coefficient, cohen_kappa, binom_test, Alternative};

// Chi-square test of independence
let observed = vec![
    vec![10, 20, 30],
    vec![15, 25, 35],
];
let result = chisq_test(&observed, false)?;
println!("Chi-square = {:.4}, p = {:.4}", result.statistic, result.p_value);

// Chi-square goodness-of-fit (test if die is fair)
let rolls = vec![16, 18, 14, 17, 15, 20];
let result = chisq_goodness_of_fit(&rolls, None)?;
println!("Chi-square = {:.4}, p = {:.4}", result.statistic, result.p_value);

// Fisher's exact test for 2x2 tables
let table = [[3, 1], [1, 3]];
let result = fisher_exact(&table, Alternative::TwoSided)?;
println!("p-value = {:.4}, odds ratio = {:.4}", result.p_value, result.odds_ratio);

// McNemar's test for paired data
let before_after = [[10, 20], [5, 65]];
let result = mcnemar_test(&before_after, false)?;
println!("Chi-square = {:.4}, p = {:.4}", result.statistic, result.p_value);

// Effect sizes
let result = cramers_v(&observed)?;
println!("Cramér's V = {:.4}", result.estimate);

let result = phi_coefficient(&table)?;
println!("Phi = {:.4}", result.estimate);

// Cohen's kappa for inter-rater agreement
let confusion = vec![
    vec![20, 5, 0],
    vec![10, 30, 5],
    vec![0, 5, 25],
];
let result = cohen_kappa(&confusion, false)?;
println!("Kappa = {:.4}, p = {:.4}", result.kappa, result.p_value);

// Exact binomial test
let result = binom_test(7, 10, 0.5, Alternative::TwoSided)?;
println!("p-value = {:.4}", result.p_value);
```

### Forecast Evaluation

```rust
use anofox_statistics::{diebold_mariano, clark_west, spa_test, mspe_adjusted_spa,
                        model_confidence_set, LossFunction, MCSStatistic, Alternative};

// Forecast errors from two competing models
let errors_model1 = vec![0.1, -0.2, 0.3, -0.1, 0.2];
let errors_model2 = vec![0.2, -0.3, 0.4, -0.2, 0.3];

// Diebold-Mariano test (two-sided)
let result = diebold_mariano(&errors_model1, &errors_model2, LossFunction::SquaredError, 1, Alternative::TwoSided)?;
println!("DM statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// Clark-West test for nested models (e.g., AR(1) vs AR(2))
let restricted_errors = vec![0.3, -0.2, 0.4, -0.3, 0.2];   // Benchmark/restricted model
let unrestricted_errors = vec![0.2, -0.1, 0.3, -0.2, 0.1]; // Alternative/unrestricted model
let result = clark_west(&restricted_errors, &unrestricted_errors, 1)?;
println!("CW statistic: {:.4}", result.statistic);
println!("p-value (one-sided): {:.4}", result.p_value);

// Superior Predictive Ability test (compare benchmark vs multiple models)
let benchmark_losses = vec![0.5, 0.6, 0.4, 0.7, 0.5];
let model_losses = vec![
    vec![0.4, 0.5, 0.3, 0.6, 0.4],  // Model 1
    vec![0.6, 0.7, 0.5, 0.8, 0.6],  // Model 2
];
let result = spa_test(&benchmark_losses, &model_losses, 1000, 10.0, Some(42))?;
println!("SPA statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value_consistent);

// MSPE-Adjusted SPA for multiple nested models
// Combines Clark-West adjustment with bootstrap for multiple testing
let benchmark_errors = vec![0.5, 0.4, 0.6, 0.3, 0.5];
let nested_model_errors = vec![
    vec![0.4, 0.3, 0.5, 0.2, 0.4],  // Nested model 1
    vec![0.3, 0.2, 0.4, 0.1, 0.3],  // Nested model 2
];
let result = mspe_adjusted_spa(&benchmark_errors, &nested_model_errors, 1000, 5.0, Some(42))?;
println!("Best model: {:?}", result.best_model_idx);
println!("p-value (adjusted): {:.4}", result.p_value_consistent);

// Model Confidence Set - identify the set of best models
let losses = vec![
    vec![0.5, 0.6, 0.4, 0.7, 0.5],  // Model 0
    vec![0.4, 0.5, 0.3, 0.6, 0.4],  // Model 1
    vec![0.8, 0.9, 0.7, 1.0, 0.8],  // Model 2 (worst)
];
let result = model_confidence_set(&losses, 0.10, MCSStatistic::Range, 1000, 5.0, Some(42))?;
println!("Models in MCS: {:?}", result.included_models);
println!("Eliminated: {:?}", result.eliminated_models);
```

### Equivalence Testing (TOST)

TOST (Two One-Sided Tests) tests whether an effect is small enough to be considered practically equivalent to zero, rather than just testing if it differs from zero.

```rust
use anofox_statistics::{tost_t_test_two_sample, tost_correlation, tost_yuen,
                        EquivalenceBounds, CorrelationTostMethod};

let group1 = vec![10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0];
let group2 = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.9, 10.0, 10.1];

// Two-sample TOST: test if mean difference is within ±0.5
let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
let result = tost_t_test_two_sample(&group1, &group2, &bounds, 0.05, false)?;
println!("Equivalent: {}", result.equivalent);
println!("TOST p-value: {:.4}", result.tost_p_value);
println!("90% CI: [{:.4}, {:.4}]", result.ci.0, result.ci.1);

// Using Cohen's d effect size bounds (±0.5 SD)
let bounds = EquivalenceBounds::CohenD { d: 0.5 };
let result = tost_t_test_two_sample(&group1, &group2, &bounds, 0.05, false)?;

// Correlation TOST: test if correlation is equivalent to zero
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 4.9, 5.0, 5.1, 4.9];  // Near-zero correlation
let bounds = EquivalenceBounds::Symmetric { delta: 0.3 };
let result = tost_correlation(&x, &y, 0.0, &bounds, 0.05, CorrelationTostMethod::Pearson)?;

// Robust TOST using trimmed means (resistant to outliers)
let data_with_outlier = vec![10.0, 11.0, 12.0, 13.0, 14.0, 100.0];  // Outlier
let normal_data = vec![10.1, 11.1, 12.1, 13.1, 14.1, 15.1];
let bounds = EquivalenceBounds::Symmetric { delta: 2.0 };
let result = tost_yuen(&data_with_outlier, &normal_data, &bounds, 0.05, 0.2)?;  // 20% trim
```

## Validation

This library is developed using Test-Driven Development (TDD) with R as the oracle (ground truth). All implementations are validated against R's statistical functions:

| Rust Function | R Equivalent | Package |
|---------------|--------------|---------|
| `t_test()` | `t.test()` | stats |
| `yuen_test()` | `yuen()` | WRS2 |
| `brown_forsythe()` | `leveneTest(center=median)` | car |
| `one_way_anova()` | `oneway.test()`, `aov()` | stats |
| `two_way_anova()` | `Anova(type="III")` | car |
| `repeated_measures_anova()` | `ezANOVA()` | ez |
| `mann_whitney_u()`, `wilcoxon_signed_rank()` | `wilcox.test()` | stats |
| `kruskal_wallis()` | `kruskal.test()` | stats |
| `brunner_munzel()` | `brunner.munzel.test()` | lawstat |
| `shapiro_wilk()` | `shapiro.test()` | stats |
| `dagostino_k_squared()` | `agostino.test()`, `anscombe.test()` | moments |
| `skewness()`, `kurtosis()` | `skewness()`, `kurtosis()` | e1071 |
| `diebold_mariano()` | `dm.test()` | forecast |
| `pearson()`, `spearman()` | `cor.test()` | stats |
| `kendall()` | `cor.test(method="kendall")` | stats |
| `partial_cor()`, `semi_partial_cor()` | `pcor.test()`, `spcor.test()` | ppcor |
| `distance_cor()` | `dcor()` | energy |
| `icc()` | `ICC()` | psych |
| `chisq_test()` | `chisq.test()` | stats |
| `chisq_goodness_of_fit()` | `chisq.test(p=...)` | stats |
| `fisher_exact()` | `fisher.test()` | stats |
| `g_test()` | `GTest()` | DescTools |
| `mcnemar_test()` | `mcnemar.test()` | stats |
| `cramers_v()` | `CramerV()` | DescTools |
| `phi_coefficient()` | `phi()` | psych |
| `cohen_kappa()` | `cohen.kappa()` | psych |
| `binom_test()` | `binom.test()` | stats |
| `prop_test_one()`, `prop_test_two()` | `prop.test()` | stats |
| `tost_t_test_*()` | `TOSTone()`, `TOSTtwo()`, `TOSTpaired()` | TOSTER |
| `tost_correlation()` | `TOSTr()` | TOSTER |
| `tost_prop_*()` | `TOSTtwo.prop()` | TOSTER |
| `tost_wilcoxon_*()` | `wilcox_TOST()` | TOSTER |
| `tost_bootstrap()` | `boot_t_TOST()` | TOSTER |
| `tost_yuen()` | `yuen.TOST()` | WRS2 |

All 303 test cases ensure numerical agreement with R within appropriate tolerances (typically 1e-10, with documented exceptions for algorithm-dependent tests like Shapiro-Wilk).

**For complete transparency on the validation process, see [`R/VALIDATION.md`](R/VALIDATION.md)**, which documents:
- All 76 reference data files and their R generation code
- Tolerance rationale for each test category
- Step-by-step reproduction instructions
- R package dependencies

## Dependencies

- [statrs](https://crates.io/crates/statrs) - Statistical distributions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation for resampling

## Attribution

This library incorporates Rust implementations of algorithms from several open-source projects. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for complete attribution and license information.

- **statrs** (MIT) - Statistical distributions
- **rand** (MIT/Apache-2.0) - Random number generation
- **R Statistical Computing** - Algorithm validation and methodology

## License

MIT License
