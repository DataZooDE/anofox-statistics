# Parametric Tests

Tests that assume normally distributed data.

## t_test_agg

Two-sample t-test comparing means.

**Signature:**
```sql
t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |
| kind | 'welch' | 'welch' or 'student' |
| confidence_level | 0.95 | CI confidence level |
| mu | 0.0 | Hypothesized mean difference |

**Returns:**
- `statistic` - t-statistic
- `p_value` - p-value
- `df` - Degrees of freedom
- `effect_size` - Cohen's d
- `ci_lower`, `ci_upper` - Confidence interval

**Example:**
```sql
-- Welch's t-test (default, unequal variances)
SELECT (t_test_agg(score, group)).*
FROM experiment;

-- Student's t-test (equal variances assumed)
SELECT t_test_agg(score, group, {'kind': 'student'})
FROM experiment;

-- One-sided test
SELECT t_test_agg(score, group, {'alternative': 'greater'})
FROM experiment;
```

---

## one_way_anova_agg

One-way Analysis of Variance for comparing multiple group means.

**Signature:**
```sql
one_way_anova_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Returns:**
- `f_statistic` - F-statistic
- `p_value` - p-value
- `df_between` - Between-groups df
- `df_within` - Within-groups df
- `ss_between`, `ss_within` - Sum of squares

**Example:**
```sql
SELECT (one_way_anova_agg(response, treatment)).*
FROM clinical_trial;
```

---

## yuen_agg

Yuen's test for trimmed means (robust to outliers).

**Signature:**
```sql
yuen_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| trim | 0.2 | Proportion to trim from each tail |

**Example:**
```sql
SELECT (yuen_agg(score, group, {'trim': 0.2})).*
FROM data_with_outliers;
```

---

## brown_forsythe_agg

Brown-Forsythe test for equality of variances.

**Signature:**
```sql
brown_forsythe_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Example:**
```sql
SELECT (brown_forsythe_agg(measurement, group)).*
FROM heteroscedastic_data;
```

**Notes:**
- More robust than Levene's test
- Uses median instead of mean
