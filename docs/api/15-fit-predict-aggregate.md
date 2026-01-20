# Fit-Predict Aggregate Functions

Aggregate functions that fit a model and return predictions for all input rows.

## Overview

These functions return an array of predictions matching the input order:

```sql
SELECT ols_fit_predict_agg(y, [x1, x2]) as predictions
FROM data
GROUP BY category;
```

## Available Functions

| Function | Description |
|----------|-------------|
| `ols_fit_predict_agg` | OLS regression |
| `ridge_fit_predict_agg` | Ridge regression |
| `elasticnet_fit_predict_agg` | Elastic Net regression |
| `wls_fit_predict_agg` | Weighted Least Squares |
| `rls_fit_predict_agg` | Recursive Least Squares |
| `bls_fit_predict_agg` | Bounded Least Squares |
| `alm_fit_predict_agg` | Augmented Linear Model |
| `poisson_fit_predict_agg` | Poisson regression |
| `pls_fit_predict_agg` | Partial Least Squares |
| `isotonic_fit_predict_agg` | Isotonic regression |
| `quantile_fit_predict_agg` | Quantile regression |

## Deprecation Notice

The old `*_predict_agg` names (`ols_predict_agg`, etc.) are deprecated but still work for backwards compatibility. Use `*_fit_predict_agg` instead.
