# Nonparametric Tests

Tests that don't assume normal distribution.

## mann_whitney_u_agg

Mann-Whitney U test (Wilcoxon rank-sum). Non-parametric alternative to t-test.

**Signature:**
```sql
mann_whitney_u_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |
| correction | true | Continuity correction |

**Returns:**
- `statistic` - U statistic
- `p_value` - p-value
- `effect_size` - Rank-biserial correlation

**Example:**
```sql
SELECT (mann_whitney_u_agg(rating, group)).*
FROM ordinal_data;
```

---

## kruskal_wallis_agg

Kruskal-Wallis H test. Non-parametric alternative to one-way ANOVA.

**Signature:**
```sql
kruskal_wallis_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Returns:**
- `statistic` - H statistic
- `p_value` - p-value
- `df` - Degrees of freedom

**Example:**
```sql
SELECT (kruskal_wallis_agg(satisfaction, department)).*
FROM survey;
```

---

## wilcoxon_signed_rank_agg

Wilcoxon signed-rank test for paired samples.

**Signature:**
```sql
wilcoxon_signed_rank_agg(diff DOUBLE, [options MAP]) -> STRUCT
```

**Example:**
```sql
-- Test if before-after difference is significant
SELECT (wilcoxon_signed_rank_agg(after - before)).*
FROM paired_data;
```

---

## brunner_munzel_agg

Brunner-Munzel test. Robust to heteroscedasticity and non-normality.

**Signature:**
```sql
brunner_munzel_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Example:**
```sql
SELECT (brunner_munzel_agg(score, treatment)).*
FROM heterogeneous_data;
```

**Notes:**
- More robust than Mann-Whitney
- Valid under heteroscedasticity

---

## permutation_t_test_agg

Permutation test for comparing two groups.

**Signature:**
```sql
permutation_t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| n_permutations | 10000 | Number of permutations |

**Example:**
```sql
SELECT (permutation_t_test_agg(outcome, treatment)).*
FROM small_sample_data;
```

## When to Use Nonparametric Tests

- Non-normal data
- Ordinal scales
- Small samples with unknown distribution
- Presence of outliers
