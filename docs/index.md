# Anofox Statistics Extension Documentation

**Version:** 0.6.0 | **DuckDB:** 1.4.3+ | **Backend:** Rust

Comprehensive regression analysis and statistical hypothesis testing for DuckDB.

## Quick Links

- [Quick Start Guide](guides/01_quick_start.md)
- [API Reference](api/00-api-design.md)
- [Function Aliases](api/21-aliases.md)

## Documentation Structure

### [API Reference](api/)

Complete API documentation organized by topic:

- [API Design & Function Types](api/00-api-design.md)
- **Regression Models:** [OLS](api/01-regression-ols.md) | [Ridge](api/02-regression-ridge.md) | [Elastic Net](api/03-regression-elasticnet.md) | [WLS](api/04-regression-wls.md) | [RLS](api/05-regression-rls.md)
- **Advanced Models:** [PLS](api/06-regression-pls.md) | [Isotonic](api/07-regression-isotonic.md) | [Quantile](api/08-regression-quantile.md) | [GLM](api/09-regression-glm.md) | [ALM](api/10-regression-alm.md) | [BLS/NNLS](api/11-regression-bls-nnls.md)
- **Utilities:** [AID Functions](api/12-aid-functions.md) | [Diagnostics](api/18-diagnostics.md)
- **Statistical Tests:** [Overview](api/13-hypothesis-tests.md)
- **Fit-Predict:** [Window](api/14-fit-predict-window.md) | [Aggregate](api/15-fit-predict-aggregate.md) | [Table Macros](api/16-table-macros.md)
- **Reference:** [Common Options](api/19-common-options.md) | [Return Types](api/20-return-types.md) | [Error Handling](api/22-error-handling.md) | [Performance](api/23-performance.md)

### [User Guides](guides/)

Step-by-step guides for common workflows:

- [Quick Start](guides/01_quick_start.md) - Get up and running
- [Technical Guide](guides/02_technical_guide.md) - Deep dive into features
- [Business Guide](guides/03_business_guide.md) - Real-world applications
- [Advanced Use Cases](guides/04_advanced_use_cases.md) - Complex scenarios

### [Reference](reference/)

Detailed documentation for models and statistical tests:

#### [Regression Models](reference/models/)
- [OLS](reference/models/ols.md) - Ordinary Least Squares
- [Ridge](reference/models/ridge.md) - L2 Regularization
- [Elastic Net](reference/models/elasticnet.md) - L1+L2 Regularization
- [WLS](reference/models/wls.md) - Weighted Least Squares
- [RLS](reference/models/rls.md) - Recursive/Adaptive
- [PLS](reference/models/pls.md) - Partial Least Squares
- [Isotonic](reference/models/isotonic.md) - Monotonic Regression
- [Quantile](reference/models/quantile.md) - Quantile Regression
- [GLM](reference/models/glm.md) - Generalized Linear Models
- [ALM](reference/models/alm.md) - Augmented Linear Models
- [BLS](reference/models/bls.md) / [NNLS](reference/models/nnls.md) - Constrained

#### [Statistical Tests](reference/tests/)
- [Normality](reference/tests/normality.md) - Shapiro-Wilk, Jarque-Bera
- [Parametric](reference/tests/parametric.md) - t-test, ANOVA
- [Nonparametric](reference/tests/nonparametric.md) - Mann-Whitney, Kruskal-Wallis
- [Correlation](reference/tests/correlation.md) - Pearson, Spearman, Kendall
- [Categorical](reference/tests/categorical.md) - Chi-square, Fisher exact
- [Effect Sizes](reference/tests/effect-sizes.md) - Cramér's V, Cohen's kappa
- [Proportions](reference/tests/proportions.md) - z-tests, Binomial
- [Equivalence](reference/tests/equivalence.md) - TOST procedures
- [Distribution](reference/tests/distribution.md) - Energy, MMD
- [Forecast](reference/tests/forecast.md) - Diebold-Mariano, Clark-West

## Installation

```sql
INSTALL anofox_statistics FROM community;
LOAD anofox_statistics;
```

## Quick Example

```sql
-- Load extension
LOAD anofox_statistics;

-- Fit OLS regression
SELECT ols_fit_agg(sales, [advertising, price])
FROM monthly_data;

-- Per-group regression
SELECT
    region,
    (ols_fit_agg(sales, [advertising])).r_squared as r2
FROM data
GROUP BY region;

-- Statistical test
SELECT (t_test_agg(score, treatment_group)).*
FROM experiment;
```

## Legacy API Reference

The original monolithic [API_REFERENCE.md](API_REFERENCE.md) is maintained for backwards compatibility but new documentation uses the structured format above.
