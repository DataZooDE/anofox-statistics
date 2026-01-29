# Diagnostic Functions

Model diagnostics and evaluation functions for regression analysis.

## Variance Inflation Factor (VIF)

### vif / anofox_stats_vif

Compute Variance Inflation Factor for multicollinearity detection.

**Signature:**
```sql
vif(x LIST(LIST(DOUBLE))) -> LIST(DOUBLE)
```

**Interpretation:**
| VIF | Interpretation |
|-----|----------------|
| VIF = 1 | No correlation |
| VIF 1-5 | Moderate correlation |
| VIF > 5 | High correlation (warning) |
| VIF > 10 | Very high correlation (problematic) |

**Example:**
```sql
SELECT vif([[x1_vals], [x2_vals], [x3_vals]]) as vif_values;
```

### vif_agg / anofox_stats_vif_agg

Streaming VIF aggregate function.

```sql
SELECT vif_agg([x1, x2, x3]) FROM data;
```

## Model Selection Criteria

### aic / anofox_stats_aic

Compute Akaike Information Criterion. Lower is better.

**Signature:**
```sql
aic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| rss | DOUBLE | Residual Sum of Squares |
| n | BIGINT | Number of observations |
| k | BIGINT | Number of parameters (including intercept) |

**Example:**
```sql
SELECT aic(100.0, 50, 3) as aic_value;
```

**Formula:** AIC = n × ln(RSS/n) + 2k

### bic / anofox_stats_bic

Compute Bayesian Information Criterion. Lower is better. Penalizes complexity more than AIC.

**Signature:**
```sql
bic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

**Example:**
```sql
SELECT bic(100.0, 50, 3) as bic_value;
```

**Formula:** BIC = n × ln(RSS/n) + k × ln(n)

### Choosing Between AIC and BIC

| Criterion | Best for |
|-----------|----------|
| AIC | Prediction, when true model may not be in candidate set |
| BIC | Model identification, converges to true model as n→∞ |

## Normality Tests

### jarque_bera / anofox_stats_jarque_bera

Jarque-Bera test for normality of residuals.

**Signature:**
```sql
jarque_bera(data LIST(DOUBLE)) -> STRUCT
```

**Returns:**
```
STRUCT(
    statistic DOUBLE,
    p_value DOUBLE,
    skewness DOUBLE,
    kurtosis DOUBLE,
    n BIGINT
)
```

**Example:**
```sql
SELECT jarque_bera(residuals).p_value as normality_pvalue;
```

### jarque_bera_agg / anofox_stats_jarque_bera_agg

Streaming Jarque-Bera aggregate function.

```sql
SELECT jarque_bera_agg(residual) FROM fitted_data;
```

## Residual Analysis

### residuals_diagnostics / anofox_stats_residuals_diagnostics

Compute comprehensive residual diagnostics.

**Signature:**
```sql
residuals_diagnostics(
    y LIST(DOUBLE),
    y_hat LIST(DOUBLE),
    [x LIST(LIST(DOUBLE))],
    [residual_std_error DOUBLE],
    [include_studentized BOOLEAN]
) -> STRUCT
```

**Returns:**
```
STRUCT(
    raw LIST(DOUBLE),           -- Raw residuals (y - ŷ)
    standardized LIST(DOUBLE),  -- Standardized residuals
    studentized LIST(DOUBLE),   -- Studentized residuals
    leverage LIST(DOUBLE)       -- Leverage values (hat matrix diagonal)
)
```

**Example:**
```sql
SELECT residuals_diagnostics(
    actual_values,
    predicted_values
) as diagnostics;
```

### residuals_diagnostics_agg / anofox_stats_residuals_diagnostics_agg

Streaming residuals diagnostics aggregate function.

```sql
SELECT residuals_diagnostics_agg(y, y_hat, [x]) FROM data;
```

## Residual Types

| Type | Formula | Use |
|------|---------|-----|
| Raw | e = y - ŷ | Basic residuals |
| Standardized | e / σ | Scale-free comparison |
| Studentized | e / (σ × √(1-h)) | Account for leverage |

## Detecting Problems

### High Leverage Points
- Leverage > 2(k+1)/n suggests influential point
- Check studentized residuals for these points

### Outliers
- |Studentized residual| > 3 suggests outlier
- Use Jarque-Bera to test overall normality

### Multicollinearity
- VIF > 5 indicates moderate collinearity
- VIF > 10 indicates severe collinearity
- Consider Ridge regression or variable selection

## See Also

- [OLS](../regression/ols.md) - Standard regression
- [Ridge](../regression/ridge.md) - Regularization for multicollinearity
- [Hypothesis Tests](../statistics/hypothesis.md) - Statistical tests
