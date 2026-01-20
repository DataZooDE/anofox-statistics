# Generalized Linear Models (GLM)

## Overview

GLM extends linear regression to non-Gaussian response distributions using a link function.

## Poisson Regression

For count data with log link:

$$\log(\mathbb{E}[y]) = X\beta$$

### Implementation

- **Algorithm:** Iteratively Reweighted Least Squares (IRLS)
- **Backend:** Rust
- **Convergence:** Typically 3-10 iterations

### When to Use

- Count data (0, 1, 2, ...)
- Event rates
- Exposure-adjusted counts

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| fit_intercept | true | Include intercept |
| max_iterations | 25 | Max IRLS iterations |
| tolerance | 1e-8 | Convergence tolerance |

### Example

```sql
-- Count regression
SELECT poisson_fit_agg(accidents, [age, mileage])
FROM driver_data;

-- With exposure offset
SELECT poisson_fit_agg(events, [risk_factor], {'offset': log(exposure)})
FROM event_data;
```

### Interpretation

- Coefficients represent log rate ratios
- exp(β) = multiplicative effect on count
- β = 0.1 → ~10% increase per unit change

## Related

- [ALM](alm.md) - More distribution choices
- [OLS](ols.md) - For continuous outcomes
- [API Reference](../../api/09-regression-glm.md)
