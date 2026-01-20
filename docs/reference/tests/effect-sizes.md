# Effect Size Measures

Measures of association strength for categorical data.

## cramers_v_agg

Cramér's V for association in contingency tables.

**Signature:**
```sql
cramers_v_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
- `coefficient` - V statistic (0 to 1)
- `p_value` - p-value

**Interpretation:**
| V | Association |
|---|-------------|
| < 0.1 | Negligible |
| 0.1-0.3 | Weak |
| 0.3-0.5 | Moderate |
| > 0.5 | Strong |

**Example:**
```sql
SELECT (cramers_v_agg(education, income_bracket)).*
FROM demographics;
```

---

## phi_coefficient_agg

Phi coefficient for 2×2 tables.

**Signature:**
```sql
phi_coefficient_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
- `coefficient` - φ statistic (-1 to 1)
- `p_value` - p-value

**Example:**
```sql
SELECT (phi_coefficient_agg(exposed, diseased)).*
FROM epidemiology_data;
```

**Notes:**
- Equivalent to Pearson r for binary variables
- Sign indicates direction of association

---

## contingency_coef_agg

Pearson's contingency coefficient.

**Signature:**
```sql
contingency_coef_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
- `coefficient` - C statistic (0 to √((k-1)/k))

**Example:**
```sql
SELECT (contingency_coef_agg(category, outcome)).*
FROM data;
```

---

## cohen_kappa_agg

Cohen's kappa for inter-rater agreement.

**Signature:**
```sql
cohen_kappa_agg(rater1 INTEGER, rater2 INTEGER) -> STRUCT
```

**Returns:**
- `coefficient` - κ statistic (-1 to 1)
- `p_value` - p-value
- `se` - Standard error
- `ci_lower`, `ci_upper` - Confidence interval

**Interpretation:**
| κ | Agreement |
|---|-----------|
| < 0 | Less than chance |
| 0.01-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

**Example:**
```sql
SELECT (cohen_kappa_agg(diagnosis_a, diagnosis_b)).*
FROM inter_rater_study;
```

## Choosing an Effect Size

| Table Size | Measure |
|------------|---------|
| 2×2 | Phi coefficient |
| r×c | Cramér's V |
| Agreement | Cohen's kappa |
