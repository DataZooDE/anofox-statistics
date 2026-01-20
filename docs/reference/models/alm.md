# Augmented Linear Models (ALM)

## Overview

ALM supports 24 error distributions, enabling robust regression for non-Gaussian data.

## Implementation

- **Algorithm:** Maximum Likelihood Estimation
- **Backend:** Rust
- **Optimization:** Gradient descent with line search

## Supported Distributions

### Light-tailed
- `normal` - Gaussian
- `logistic` - Logistic
- `uniform` - Uniform

### Heavy-tailed (Robust)
- `student_t` - Student's t (df=3)
- `cauchy` - Cauchy
- `laplace` - Double exponential

### Skewed
- `gumbel` - Gumbel (extreme value)
- `weibull` - Weibull
- `gamma` - Gamma
- `lognormal` - Log-normal

### And more...

## When to Use

- Outliers present (use heavy-tailed)
- Skewed residuals (use asymmetric)
- Known error distribution
- Robust estimation needs

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `alm_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `alm_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `alm_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| distribution | required | Error distribution name |
| fit_intercept | true | Include intercept |
| max_iterations | 1000 | Max iterations |

## Example

```sql
-- Robust regression with Student-t errors
SELECT alm_fit_agg(y, [x], 'student_t')
FROM data_with_outliers;

-- Laplace (L1 loss equivalent)
SELECT alm_fit_agg(y, [x], 'laplace')
FROM data;

-- Heavy-tailed for financial data
SELECT alm_fit_agg(returns, [market_return], 'cauchy')
FROM stock_data;
```

## Distribution Selection Guide

| Data Characteristic | Recommended |
|--------------------|-------------|
| Moderate outliers | student_t |
| Heavy outliers | cauchy |
| Right-skewed | gumbel, gamma |
| Bounded data | uniform |
| Standard | normal |

## Related

- [Quantile](quantile.md) - Non-parametric robustness
- [OLS](ols.md) - Gaussian assumption
- [API Reference](../../api/10-regression-alm.md)
