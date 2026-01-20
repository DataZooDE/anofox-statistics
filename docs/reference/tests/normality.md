# Normality Tests

Tests for whether data follows a normal distribution.

## shapiro_wilk_agg

The gold standard for normality testing with small to moderate samples.

**Signature:**
```sql
shapiro_wilk_agg(value DOUBLE) -> STRUCT
```

**Returns:**
- `statistic` - W statistic (closer to 1 = more normal)
- `p_value` - p-value (low = reject normality)
- `n` - Sample size
- `method` - "Shapiro-Wilk"

**Example:**
```sql
SELECT (shapiro_wilk_agg(residual)).*
FROM model_diagnostics;
```

**Notes:**
- Best for n < 5000
- Sensitive to ties
- W close to 1 indicates normality

---

## jarque_bera_agg

Tests normality based on skewness and kurtosis.

**Signature:**
```sql
jarque_bera_agg(value DOUBLE) -> STRUCT
```

**Returns:**
- `statistic` - JB statistic
- `p_value` - p-value
- `skewness` - Sample skewness
- `kurtosis` - Sample excess kurtosis

**Example:**
```sql
SELECT (jarque_bera_agg(returns)).*
FROM financial_data;
```

**Notes:**
- Good for large samples
- Chi-square distribution with df=2
- Focuses on shape (skew + kurtosis)

---

## dagostino_k2_agg

D'Agostino and Pearson's K² test combining skewness and kurtosis tests.

**Signature:**
```sql
dagostino_k2_agg(value DOUBLE) -> STRUCT
```

**Returns:**
- `statistic` - K² statistic
- `p_value` - p-value
- `z_skewness` - z-score for skewness
- `z_kurtosis` - z-score for kurtosis

**Example:**
```sql
SELECT (dagostino_k2_agg(measurement)).*
FROM lab_data;
```

**Notes:**
- Requires n ≥ 20
- Omnibus test (skewness + kurtosis)

## Choosing a Test

| Sample Size | Recommended |
|-------------|-------------|
| n < 50 | Shapiro-Wilk |
| 50 ≤ n < 5000 | Shapiro-Wilk or D'Agostino |
| n ≥ 5000 | Jarque-Bera |
