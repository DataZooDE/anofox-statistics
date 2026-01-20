# Correlation Tests

Tests for association between two continuous variables.

## pearson_agg

Pearson product-moment correlation for linear relationships.

**Signature:**
```sql
pearson_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Returns:**
- `coefficient` - Pearson r (-1 to 1)
- `p_value` - Two-sided p-value
- `ci_lower`, `ci_upper` - Fisher-transformed CI
- `n` - Sample size

**Example:**
```sql
SELECT (pearson_agg(height, weight)).*
FROM anthropometrics;
```

**Notes:**
- Assumes linear relationship
- Sensitive to outliers
- r² = proportion of variance explained

---

## spearman_agg

Spearman rank correlation for monotonic relationships.

**Signature:**
```sql
spearman_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Returns:**
- `coefficient` - Spearman ρ (-1 to 1)
- `p_value` - p-value
- `n` - Sample size

**Example:**
```sql
SELECT (spearman_agg(rank_a, rank_b)).*
FROM ordinal_data;
```

**Notes:**
- Robust to outliers
- Works with ordinal data
- Detects monotonic (not just linear) relationships

---

## kendall_agg

Kendall tau correlation based on concordant/discordant pairs.

**Signature:**
```sql
kendall_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Returns:**
- `coefficient` - Kendall τ (-1 to 1)
- `p_value` - p-value
- `n` - Sample size

**Example:**
```sql
SELECT (kendall_agg(judge1_rank, judge2_rank)).*
FROM ratings;
```

**Notes:**
- Most robust to ties
- More interpretable than Spearman for small samples
- τ ≈ 0.67 × ρ (approximately)

---

## distance_cor_agg

Distance correlation detects any dependence, not just linear.

**Signature:**
```sql
distance_cor_agg(x DOUBLE, y DOUBLE) -> STRUCT
```

**Returns:**
- `coefficient` - Distance correlation (0 to 1)
- `p_value` - p-value (permutation-based)

**Example:**
```sql
SELECT (distance_cor_agg(x, y)).*
FROM nonlinear_data;
```

**Notes:**
- Detects non-linear relationships
- dCor = 0 iff independent
- Computationally intensive

---

## icc_agg

Intraclass correlation coefficient for reliability/agreement.

**Signature:**
```sql
icc_agg(value DOUBLE, rater_id INTEGER, subject_id INTEGER, [options MAP]) -> STRUCT
```

**Returns:**
- `coefficient` - ICC value
- `ci_lower`, `ci_upper` - Confidence interval
- `f_value`, `p_value` - F-test results

**Example:**
```sql
SELECT (icc_agg(score, rater, subject)).*
FROM inter_rater_study;
```

## Choosing a Correlation Measure

| Relationship | Data Type | Test |
|--------------|-----------|------|
| Linear | Continuous | Pearson |
| Monotonic | Ordinal/Continuous | Spearman |
| Monotonic (small n) | Ordinal | Kendall |
| Any dependence | Continuous | Distance |
| Reliability | Repeated measures | ICC |
