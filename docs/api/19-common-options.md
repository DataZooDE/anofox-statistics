# Common Options

Options MAP parameters used across multiple functions.

## Regression Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| compute_inference | BOOLEAN | false | Compute t-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | Confidence level for intervals |

## NULL Handling Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| null_policy | VARCHAR | 'drop' | How to handle NULL values |

**`null_policy` Values:**

| Value | Behavior |
|-------|----------|
| `'drop'` | Drop rows with NULL y from training, but include them in output with predictions |
| `'drop_y_zero_x'` | Drop rows with NULL y OR zero x values from training |

The `'drop'` policy (default) is useful for forecasting scenarios where you want predictions for future dates that don't have y values yet. Rows with NULL y are excluded from model fitting but still appear in the output with their predicted values.

The `'drop_y_zero_x'` policy additionally excludes rows where all x features are zero, which can be useful for sparse data or when zero features indicate missing data.

## Regularization Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | - | Regularization strength |
| l1_ratio | DOUBLE | - | L1/L2 mix (0=Ridge, 1=Lasso) |
| max_iterations | INTEGER | 1000 | Max iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

## Hypothesis Test Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |
| correction | BOOLEAN | true | Apply continuity correction |

## Examples

**Regression with inference:**
```sql
SELECT ols_fit_agg(y, [x], {
    'fit_intercept': true,
    'compute_inference': true,
    'confidence_level': 0.99
}) FROM data;
```

**Using null_policy for forecasting:**
```sql
-- Get predictions including future dates with NULL y values
FROM ols_fit_predict_by('sales', store_id, revenue, [ads, traffic], {
    'null_policy': 'drop'
});
```

**Excluding zero features:**
```sql
-- Exclude rows where all x features are zero
FROM ridge_fit_predict_by('sparse_data', group_id, y, [x1, x2, x3], {
    'null_policy': 'drop_y_zero_x',
    'alpha': 0.1
});
```
