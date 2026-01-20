# Fit-Predict Table Macros

Table macros for convenient grouped fit-predict operations.

## Overview

Table macros provide a simplified interface for fitting models by group and returning predictions:

```sql
FROM ols_fit_predict_by(
    'source_table',
    'group_column',
    'y_column',
    ['x1_column', 'x2_column']
);
```

## Available Macros

| Macro | Description |
|-------|-------------|
| `ols_fit_predict_by` | OLS regression by group |
| `ridge_fit_predict_by` | Ridge regression by group |
| `elasticnet_fit_predict_by` | Elastic Net by group |
| `wls_fit_predict_by` | WLS by group |
| `rls_fit_predict_by` | RLS by group |
| `bls_fit_predict_by` | BLS by group |
| `alm_fit_predict_by` | ALM by group |
| `poisson_fit_predict_by` | Poisson by group |
| `pls_fit_predict_by` | PLS by group |
| `isotonic_fit_predict_by` | Isotonic by group |
| `quantile_fit_predict_by` | Quantile by group |

## Example

```sql
-- Fit OLS per store and get predictions
FROM ols_fit_predict_by(
    'sales_data',
    'store_id',
    'revenue',
    ['advertising', 'foot_traffic']
);
```
