# Categorical Tests

Tests for categorical data and contingency tables.

## chisq_test_agg

Chi-square test of independence for contingency tables.

**Signature:**
```sql
chisq_test_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
- `statistic` - χ² statistic
- `p_value` - p-value
- `df` - Degrees of freedom
- `expected` - Expected frequencies

**Example:**
```sql
SELECT (chisq_test_agg(gender, preference)).*
FROM survey;
```

**Notes:**
- Expected cell counts should be ≥ 5
- For 2×2 with small counts, use Fisher exact

---

## chisq_gof_agg

Chi-square goodness-of-fit test.

**Signature:**
```sql
chisq_gof_agg(observed INTEGER, expected DOUBLE) -> STRUCT
```

**Example:**
```sql
SELECT (chisq_gof_agg(count, expected_count)).*
FROM frequency_data;
```

---

## g_test_agg

G-test (log-likelihood ratio test) for contingency tables.

**Signature:**
```sql
g_test_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Example:**
```sql
SELECT (g_test_agg(category, outcome)).*
FROM data;
```

**Notes:**
- Alternative to chi-square
- Better for small samples
- Additive across subtables

---

## fisher_exact_agg

Fisher's exact test for 2×2 tables.

**Signature:**
```sql
fisher_exact_agg(row_var INTEGER, col_var INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |

**Returns:**
- `p_value` - Exact p-value
- `odds_ratio` - Odds ratio

**Example:**
```sql
SELECT (fisher_exact_agg(treatment, outcome)).*
FROM small_clinical_trial;
```

**Notes:**
- Exact test (no approximation)
- Best for small samples
- Only for 2×2 tables

---

## mcnemar_agg

McNemar's test for paired nominal data.

**Signature:**
```sql
mcnemar_agg(before INTEGER, after INTEGER) -> STRUCT
```

**Example:**
```sql
-- Test if treatment changed responses
SELECT (mcnemar_agg(before_treatment, after_treatment)).*
FROM paired_binary_data;
```

**Notes:**
- For matched pairs
- Tests marginal homogeneity
