# Recursive Least Squares (RLS)

## Overview

RLS updates model parameters incrementally with each new observation, supporting exponential forgetting for adaptive learning.

## Implementation

- **Algorithm:** Sherman-Morrison-Woodbury updates
- **Backend:** Rust
- **Complexity:** O(p²) per observation

## When to Use

- Online/streaming data
- Adaptive systems (parameters change over time)
- Real-time forecasting
- Non-stationary processes

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `rls_fit` | Scalar | Fit on array data |
| `rls_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `rls_fit_predict` | Window | Fit and predict in window context |
| `rls_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `rls_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| forgetting_factor | 1.0 | λ ∈ (0, 1], lower = faster adaptation |
| initial_p_diagonal | 100.0 | Initial covariance matrix diagonal |
| fit_intercept | true | Include intercept term |

## Forgetting Factor Guide

- **λ = 1.0:** No forgetting, equivalent to OLS
- **λ = 0.99:** Mild forgetting, ~100 effective observations
- **λ = 0.95:** Moderate forgetting, ~20 effective observations
- **λ = 0.90:** Strong forgetting, ~10 effective observations

Effective window ≈ 1/(1-λ)

## Example

```sql
-- Adaptive regression with forgetting
SELECT rls_fit_agg(y, [x], 0.95)
FROM streaming_data;

-- Rolling window equivalent using OVER
SELECT
    date,
    (rls_fit_agg(y, [x], 0.99) OVER (
        ORDER BY date
    )).coefficients[1] as adaptive_beta
FROM time_series;
```

## Related

- [OLS](ols.md) - Batch equivalent
- [WLS](wls.md) - Fixed weights
- [API Reference](../../api/05-regression-rls.md)
