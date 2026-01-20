# Non-Negative Least Squares (NNLS)

## Overview

NNLS constrains all coefficients to be non-negative.

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 \quad \text{s.t.} \quad \beta \geq 0$$

## Implementation

- **Algorithm:** Lawson-Hanson active set
- **Backend:** Rust
- **Complexity:** O(np²) typical

## When to Use

- Mixture decomposition (proportions ≥ 0)
- Signal processing (non-negative components)
- Physical constraints (concentrations, counts)
- Portfolio weights (long-only)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| fit_intercept | true | Include intercept (not constrained) |

## Example

```sql
-- Spectral unmixing
SELECT nnls_fit_agg(spectrum, [component1, component2, component3])
FROM spectral_data;

-- Long-only portfolio
SELECT nnls_fit_agg(returns, [asset1, asset2, asset3])
FROM portfolio_data;
```

## Mathematical Properties

- Solution is unique (convex optimization)
- Some coefficients may be exactly zero
- Sparse solutions common

## Related

- [BLS](bls.md) - General bounds
- [OLS](ols.md) - Unconstrained
- [API Reference](../../api/11-regression-bls-nnls.md)
