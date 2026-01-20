# Forecast Comparison Tests

Tests for comparing predictive accuracy of forecasting models.

## diebold_mariano_agg

Diebold-Mariano test for equal predictive accuracy.

**Signature:**
```sql
diebold_mariano_agg(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| loss | 'squared' | 'squared', 'absolute' |
| h | 1 | Forecast horizon |
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |

**Returns:**
- `statistic` - DM statistic
- `p_value` - p-value
- `mean_diff` - Mean loss differential

**Example:**
```sql
SELECT (diebold_mariano_agg(actual, model1_pred, model2_pred)).*
FROM forecast_comparison;

-- One-sided: is model1 better?
SELECT diebold_mariano_agg(actual, model1, model2, {'alternative': 'less'})
FROM forecasts;
```

**Interpretation:**
- Positive statistic: forecast2 is better
- Negative statistic: forecast1 is better
- p < 0.05: Significant difference in accuracy

**Notes:**
- HAC standard errors for serial correlation
- Valid for h-step ahead forecasts
- Asymptotically normal

---

## clark_west_agg

Clark-West test for nested model comparison.

**Signature:**
```sql
clark_west_agg(actual DOUBLE, forecast_restricted DOUBLE, forecast_unrestricted DOUBLE, [options MAP]) -> STRUCT
```

**Returns:**
- `statistic` - CW statistic
- `p_value` - One-sided p-value

**Example:**
```sql
-- Compare restricted (AR) vs unrestricted (ARX) model
SELECT (clark_west_agg(actual, ar_forecast, arx_forecast)).*
FROM nested_model_comparison;
```

**Notes:**
- Adjusts for noise in unrestricted model
- One-sided test: unrestricted ≥ restricted
- For nested models only

## Choosing a Test

| Models | Test |
|--------|------|
| Non-nested | Diebold-Mariano |
| Nested | Clark-West |
| Multiple models | Model Confidence Set |

## Best Practices

1. Use out-of-sample forecasts
2. Account for forecast horizon
3. Consider multiple loss functions
4. Report both statistical and economic significance
