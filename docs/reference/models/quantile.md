# Quantile Regression

## Overview

Quantile regression models conditional quantiles rather than the mean, providing robust estimates.

$$\hat{\beta} = \arg\min_\beta \sum_i \rho_\tau(y_i - x_i'\beta)$$

where ρ_τ is the check function for quantile τ.

## Implementation

- **Algorithm:** Interior point / Linear programming
- **Backend:** Rust
- **Complexity:** O(n²p) worst case

## When to Use

- Robust regression (τ = 0.5 for median)
- Modeling distribution tails (τ = 0.1 or 0.9)
- Heterogeneous effects across distribution
- Presence of outliers

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| quantile | 0.5 | Target quantile τ ∈ (0, 1) |
| fit_intercept | true | Include intercept |
| max_iterations | 1000 | Max iterations |
| tolerance | 1e-6 | Convergence tolerance |

## Common Quantiles

- **τ = 0.5:** Median regression (most robust)
- **τ = 0.25, 0.75:** Interquartile range
- **τ = 0.1, 0.9:** Distribution tails
- **τ = 0.05, 0.95:** Extreme values

## Example

```sql
-- Median regression
SELECT quantile_fit_agg(y, [x], {'quantile': 0.5})
FROM data_with_outliers;

-- Model upper tail (90th percentile)
SELECT quantile_fit_agg(income, [education, experience], {'quantile': 0.9})
FROM wage_data;

-- Compare quantiles
SELECT
    0.25 as quantile,
    (quantile_fit_agg(y, [x], {'quantile': 0.25})).coefficients[1] as slope
FROM data
UNION ALL
SELECT 0.5, (quantile_fit_agg(y, [x], {'quantile': 0.5})).coefficients[1] FROM data
UNION ALL
SELECT 0.75, (quantile_fit_agg(y, [x], {'quantile': 0.75})).coefficients[1] FROM data;
```

## Related

- [OLS](ols.md) - Mean regression
- [ALM](alm.md) - Robust via distribution choice
- [API Reference](../../api/08-regression-quantile.md)
