# Elastic Net Regression

## Overview

Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization.

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 + \alpha \left[ \rho\|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|^2 \right]$$

## Implementation

- **Algorithm:** Coordinate descent
- **Backend:** Rust
- **Complexity:** O(np × iterations)

## When to Use

- Feature selection with regularization
- Correlated features (where pure Lasso struggles)
- High-dimensional data (p > n)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | required | Overall regularization strength |
| l1_ratio | required | Mix ratio: 0=Ridge, 1=Lasso |
| max_iterations | 1000 | Max coordinate descent iterations |
| tolerance | 1e-6 | Convergence tolerance |

## L1 Ratio Guide

- **l1_ratio = 0:** Pure Ridge (no feature selection)
- **l1_ratio = 0.5:** Balanced L1/L2
- **l1_ratio = 1:** Pure Lasso (sparse solutions)
- **l1_ratio = 0.1-0.3:** Mostly Ridge with some sparsity

## Example

```sql
-- Elastic Net with balanced regularization
SELECT elasticnet_fit_agg(y, [x1, x2, x3, x4, x5], 0.1, 0.5)
FROM high_dim_data;

-- Lasso-like (mostly L1)
SELECT elasticnet_fit_agg(y, features, 0.1, 0.9)
FROM data;
```

## Related

- [Ridge](ridge.md) - Pure L2
- [API Reference](../../api/03-regression-elasticnet.md)
