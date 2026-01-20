# Fit-Predict Table Macros

Table macros for convenient grouped fit-predict operations.

## Overview

Table macros provide a simplified interface for fitting models by group and returning predictions in long format (one row per observation):

```sql
FROM ols_fit_predict_by(
    'source_table',
    group_column,
    y_column,
    [x1_column, x2_column],
    {'fit_intercept': true}
);
```

## Common Signature

```sql
*_fit_predict_by(
    source VARCHAR,           -- Table name (quoted string)
    group_col COLUMN,         -- Grouping column
    y_col COLUMN,             -- Response variable column
    x_cols LIST(COLUMN),      -- Feature columns as list
    [options MAP]             -- Optional parameters
) -> TABLE
```

## Common Options

All table macros support the `null_policy` option for handling NULL values:

| Value | Behavior |
|-------|----------|
| `'drop'` (default) | Drop rows with NULL y from training, include in output with predictions |
| `'drop_y_zero_x'` | Drop rows with NULL y OR zero x values from training |

See [Common Options](19-common-options.md#null-handling-options) for details.

## Return Columns

All table macros return:

| Column | Type | Description |
|--------|------|-------------|
| group_id | ANY | Group identifier |
| y | DOUBLE | Actual response value |
| x | LIST(DOUBLE) | Feature values |
| yhat | DOUBLE | Predicted value |
| yhat_lower | DOUBLE | Lower prediction interval |
| yhat_upper | DOUBLE | Upper prediction interval |
| is_training | BOOLEAN | Whether row was used for training |

## Available Macros

### ols_fit_predict_by

OLS regression by group.

```sql
FROM ols_fit_predict_by('sales', store_id, revenue, [ads, traffic]);
```

**Options:** `fit_intercept`, `confidence_level`, `null_policy`

### ridge_fit_predict_by

Ridge regression by group.

```sql
FROM ridge_fit_predict_by('data', category, y, [x1, x2], {'alpha': 0.1});
```

**Options:** `alpha`, `fit_intercept`, `confidence_level`, `null_policy`

### elasticnet_fit_predict_by

Elastic Net regression by group.

```sql
FROM elasticnet_fit_predict_by('data', grp, y, [x1, x2], {'alpha': 0.1, 'l1_ratio': 0.5});
```

**Options:** `alpha`, `l1_ratio`, `max_iterations`, `tolerance`, `fit_intercept`, `confidence_level`, `null_policy`

### wls_fit_predict_by

Weighted Least Squares by group. Note: requires weight column.

```sql
FROM wls_fit_predict_by('data', grp, y, [x1, x2], weight_col, {'fit_intercept': true});
```

**Options:** `fit_intercept`, `confidence_level`, `null_policy`

### rls_fit_predict_by

Recursive Least Squares by group.

```sql
FROM rls_fit_predict_by('streaming', sensor_id, value, [temp, pressure], {'forgetting_factor': 0.95});
```

**Options:** `forgetting_factor`, `initial_p_diagonal`, `fit_intercept`, `confidence_level`, `null_policy`

### bls_fit_predict_by

Bounded Least Squares by group.

```sql
FROM bls_fit_predict_by('mixture', batch, y, [x1, x2], {'lower_bounds': [0, 0], 'upper_bounds': [1, 1]});
```

**Options:** `lower_bounds`, `upper_bounds`, `fit_intercept`, `null_policy`

### alm_fit_predict_by

Augmented Linear Model by group.

```sql
FROM alm_fit_predict_by('robust_data', grp, y, [x], {'distribution': 'student_t'});
```

**Options:** `distribution`, `fit_intercept`, `null_policy`

### poisson_fit_predict_by

Poisson GLM by group.

```sql
FROM poisson_fit_predict_by('counts', region, events, [exposure, risk], {'fit_intercept': true});
```

**Options:** `fit_intercept`, `max_iterations`, `tolerance`, `null_policy`

### pls_fit_predict_by

Partial Least Squares by group.

```sql
FROM pls_fit_predict_by('spectral', sample_id, concentration, [wavelength1, wavelength2], {'n_components': 3});
```

**Options:** `n_components`, `fit_intercept`, `scale`, `null_policy`

### isotonic_fit_predict_by

Isotonic regression by group. Note: single x column, not a list.

```sql
FROM isotonic_fit_predict_by('calibration', model_id, actual, predicted, {'increasing': true});
```

**Options:** `increasing`, `null_policy`

### quantile_fit_predict_by

Quantile regression by group.

```sql
FROM quantile_fit_predict_by('sales', region, revenue, [price, promo], {'quantile': 0.5});
```

**Options:** `quantile`, `fit_intercept`, `max_iterations`, `tolerance`, `null_policy`
