# Weighted Least Squares (WLS)

## Overview

WLS accounts for heteroscedasticity by weighting observations differently.

$$\hat{\beta} = \arg\min_\beta \sum_i w_i (y_i - x_i'\beta)^2$$

## Implementation

- **Algorithm:** Weighted QR decomposition
- **Backend:** Rust faer library

## When to Use

- Heteroscedastic errors (non-constant variance)
- Known observation reliability differences
- Combining data from different sources
- Survey data with sampling weights

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| weights | required | Observation weights (same length as y) |
| fit_intercept | true | Include intercept term |

## Weight Interpretation

- Higher weight = more influence on fit
- Weights should be proportional to 1/variance
- For heteroscedasticity: w_i = 1/σ²_i

## Example

```sql
-- WLS with inverse variance weights
SELECT wls_fit_agg(y, [x], 1.0 / variance)
FROM data_with_known_variance;

-- Survey data with sampling weights
SELECT wls_fit_agg(response, [predictor], sample_weight)
FROM survey_data;

-- Higher weight for recent observations
SELECT wls_fit_agg(y, [x], exp(-0.1 * days_old))
FROM time_series;
```

## Related

- [OLS](ols.md) - Equal weights
- [RLS](rls.md) - Exponential forgetting
- [API Reference](../../api/04-regression-wls.md)
