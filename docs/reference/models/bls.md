# Bounded Least Squares (BLS)

## Overview

BLS constrains coefficients to lie within specified bounds.

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 \quad \text{s.t.} \quad l \leq \beta \leq u$$

## Implementation

- **Algorithm:** Active set / Projected gradient
- **Backend:** Rust
- **Complexity:** O(np² × iterations)

## When to Use

- Physical constraints (e.g., concentrations ≥ 0)
- Prior knowledge about coefficient ranges
- Mixture models (coefficients sum to 1)
- Sign constraints

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `bls_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `bls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `bls_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lower_bounds | -∞ | Lower bounds per coefficient |
| upper_bounds | +∞ | Upper bounds per coefficient |
| fit_intercept | true | Include intercept (not bounded) |

## Example

```sql
-- Coefficients between 0 and 1
SELECT bls_fit_agg(y, [x1, x2], [0, 0], [1, 1])
FROM mixture_data;

-- Only lower bounds (non-negative)
SELECT bls_fit_agg(y, [x1, x2], [0, 0], [NULL, NULL])
FROM data;

-- Mixed constraints
SELECT bls_fit_agg(y, [x1, x2, x3], [-1, 0, 0], [1, NULL, 10])
FROM data;
```

## Related

- [NNLS](nnls.md) - Special case: all ≥ 0
- [OLS](ols.md) - Unconstrained
- [API Reference](../../api/11-regression-bls-nnls.md)
