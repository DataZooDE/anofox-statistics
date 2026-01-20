# Statistical Hypothesis Tests Reference

Comprehensive statistical hypothesis testing powered by the `anofox-statistics` crate.

## Test Categories

### [Normality Tests](normality.md)
- Shapiro-Wilk, Jarque-Bera, D'Agostino K²

### [Parametric Tests](parametric.md)
- t-test, ANOVA, Yuen's trimmed mean, Brown-Forsythe

### [Nonparametric Tests](nonparametric.md)
- Mann-Whitney U, Kruskal-Wallis, Wilcoxon, Brunner-Munzel, Permutation

### [Correlation Tests](correlation.md)
- Pearson, Spearman, Kendall, Distance correlation, ICC

### [Categorical Tests](categorical.md)
- Chi-square, G-test, Fisher exact, McNemar

### [Effect Sizes](effect-sizes.md)
- Cramér's V, Phi coefficient, Contingency coefficient, Cohen's kappa

### [Proportion Tests](proportions.md)
- One-sample, Two-sample, Binomial exact

### [Equivalence Tests](equivalence.md)
- TOST for t-tests, paired, correlation

### [Distribution Comparison](distribution.md)
- Energy distance, Maximum Mean Discrepancy (MMD)

### [Forecast Comparison](forecast.md)
- Diebold-Mariano, Clark-West

## Test Selection Guide

| Question | Test |
|----------|------|
| Is data normal? | Shapiro-Wilk, Jarque-Bera |
| Are two group means different? | t-test (parametric), Mann-Whitney (nonparametric) |
| Are multiple group means different? | ANOVA (parametric), Kruskal-Wallis (nonparametric) |
| Are two variables correlated? | Pearson (linear), Spearman (monotonic) |
| Are categorical variables associated? | Chi-square, Fisher exact |
| Are two groups equivalent? | TOST |
| Which forecast is better? | Diebold-Mariano |
