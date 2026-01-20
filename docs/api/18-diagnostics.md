# Diagnostic Functions

Model diagnostics and validation utilities.

## VIF (Variance Inflation Factor)

### vif

Calculate VIF for detecting multicollinearity.

**Signature:**
```sql
vif(x LIST(LIST(DOUBLE))) -> LIST(DOUBLE)
```

### vif_agg

Streaming VIF aggregate function.

```sql
SELECT vif_agg([x1, x2, x3]) FROM data;
```

## Model Selection Criteria

### aic

Akaike Information Criterion.

```sql
SELECT aic(log_likelihood, n_parameters) FROM model_stats;
```

### bic

Bayesian Information Criterion.

```sql
SELECT bic(log_likelihood, n_parameters, n_observations) FROM model_stats;
```

## Residual Diagnostics

### residuals_diagnostics_agg

Comprehensive residual analysis.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| mean | DOUBLE | Mean of residuals |
| std | DOUBLE | Standard deviation |
| skewness | DOUBLE | Skewness |
| kurtosis | DOUBLE | Kurtosis |
| durbin_watson | DOUBLE | Durbin-Watson statistic |

## Jarque-Bera Test

### jarque_bera_agg

Test for normality of residuals.

```sql
SELECT (jarque_bera_agg(residual)).p_value FROM diagnostics;
```
