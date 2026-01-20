# Ridge Regression

## Overview

Ridge regression adds L2 regularization to OLS, shrinking coefficients toward zero.

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|^2$$

## Implementation

- **Algorithm:** Augmented normal equations
- **Backend:** Rust faer library
- **Complexity:** O(np²)

## When to Use

- Multicollinearity among features
- High-dimensional data (p approaching n)
- Preventing overfitting
- When all features should contribute (no feature selection)

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `ridge_fit` | Scalar | Fit on array data |
| `ridge_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `ridge_fit_predict` | Window | Fit and predict in window context |
| `ridge_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `ridge_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | required | L2 regularization strength (≥ 0) |
| fit_intercept | true | Include intercept (not regularized) |

## Choosing Alpha

- **alpha = 0:** Equivalent to OLS
- **Small alpha (0.01-0.1):** Mild regularization
- **Large alpha (1-10):** Strong regularization, more shrinkage

Use cross-validation to select optimal alpha.

## Example

```sql
-- Ridge with alpha=0.1
SELECT ridge_fit_agg(y, [x1, x2, x3], 0.1)
FROM high_correlation_data;

-- Compare different alpha values
SELECT
    0.01 as alpha,
    (ridge_fit_agg(y, [x], 0.01)).r_squared as r2
FROM data
UNION ALL
SELECT
    0.1,
    (ridge_fit_agg(y, [x], 0.1)).r_squared
FROM data;
```

## Mathematical Properties

- Coefficients shrink toward zero but never exactly zero
- Handles rank-deficient X'X matrices
- Bias-variance tradeoff: increases bias, reduces variance

## Related

- [OLS](ols.md) - No regularization
- [Elastic Net](elasticnet.md) - Combined L1/L2
- [API Reference](../../api/02-regression-ridge.md)
