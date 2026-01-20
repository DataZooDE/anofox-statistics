# Partial Least Squares (PLS)

## Overview

PLS finds latent components that maximize covariance between X and y, handling multicollinearity and high-dimensional data.

## Implementation

- **Algorithm:** SIMPLS
- **Backend:** Rust
- **Complexity:** O(npk) where k = components

## When to Use

- More features than observations (p > n)
- Severe multicollinearity
- Chemometrics, spectroscopy data
- When features are noisy measurements of latent factors

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_components | min(p, 10) | Number of PLS components |
| fit_intercept | true | Include intercept |
| scale | true | Standardize features |

## Choosing Components

- Start with 1-2 components
- Use cross-validation to select optimal number
- More components = more flexibility but risk of overfitting
- Typically 2-10 components sufficient

## Example

```sql
-- PLS with default components
SELECT pls_fit_agg(y, [x1, x2, ..., x100])
FROM spectroscopy_data;

-- Specify number of components
SELECT pls_fit_agg(y, features, {'n_components': 5})
FROM high_dim_data;
```

## Comparison with Other Methods

| Method | Multicollinearity | Feature Selection | Interpretability |
|--------|-------------------|-------------------|------------------|
| OLS | Poor | No | High |
| Ridge | Good | No | Medium |
| PLS | Excellent | Implicit | Medium |
| Elastic Net | Good | Yes | High |

## Related

- [Ridge](ridge.md) - Alternative for multicollinearity
- [API Reference](../../api/06-regression-pls.md)
