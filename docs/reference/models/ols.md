# Ordinary Least Squares (OLS)

## Overview

OLS minimizes the sum of squared residuals to find the best linear fit.

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2$$

## Implementation

- **Algorithm:** QR decomposition (numerically stable)
- **Backend:** Rust faer library
- **Complexity:** O(np²) where n = observations, p = features

## When to Use

- Standard regression with well-conditioned data
- When interpretability is important
- Baseline model for comparison

## When NOT to Use

- Multicollinearity (use Ridge/PLS)
- More features than observations (use regularization)
- Non-Gaussian errors (use ALM/GLM)

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit` | Scalar | Fit on array data |
| `ols_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `ols_fit_predict` | Window | Fit and predict in window context |
| `ols_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `ols_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| fit_intercept | true | Include intercept term |
| compute_inference | false | Compute standard errors, t-stats, p-values |
| confidence_level | 0.95 | CI confidence level |

## Inference Output

When `compute_inference=true`, returns:

- **Standard Errors:** Based on MSE and (X'X)⁻¹
- **t-statistics:** coefficient / standard_error
- **p-values:** Two-sided t-test
- **Confidence Intervals:** coefficient ± t_crit × SE
- **F-statistic:** Overall model significance

## Example

```sql
-- Basic OLS
SELECT ols_fit_agg(sales, [advertising, price])
FROM monthly_data;

-- With inference
SELECT ols_fit_agg(sales, [advertising, price], true, true, 0.95)
FROM monthly_data;

-- Access specific results
SELECT
    (ols_fit_agg(y, [x])).coefficients[1] as slope,
    (ols_fit_agg(y, [x])).r_squared as r2
FROM data;

-- Fit per group and get predictions table
FROM ols_fit_predict_by('sales_data', store_id, revenue, [ads, traffic]);
```

## Related

- [Ridge](ridge.md) - Add L2 regularization
- [WLS](wls.md) - Handle heteroscedasticity
- [API Reference](../../api/01-regression-ols.md)
