# Common Options

Options MAP parameters used across multiple functions.

## Regression Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| compute_inference | BOOLEAN | false | Compute t-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | Confidence level for intervals |

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

## Example

```sql
SELECT ols_fit_agg(y, [x], {
    'fit_intercept': true,
    'compute_inference': true,
    'confidence_level': 0.99
}) FROM data;
```
