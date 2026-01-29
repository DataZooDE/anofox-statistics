# Hypothesis Tests

Comprehensive statistical hypothesis testing functions for comparing groups and distributions.

## Parametric Tests

### t_test_agg

Two-sample t-test comparing means of two groups. Supports both Welch's (default) and Student's t-test.

**Signature:**
```sql
t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |
| kind | VARCHAR | 'welch' | 'welch' (default) or 'student' |
| mu | DOUBLE | 0.0 | Hypothesized mean difference |

**Returns:**
```
STRUCT(
    statistic DOUBLE,     -- t-statistic
    p_value DOUBLE,       -- p-value
    df DOUBLE,            -- Degrees of freedom
    effect_size DOUBLE,   -- Cohen's d
    ci_lower DOUBLE,      -- CI lower bound
    ci_upper DOUBLE,      -- CI upper bound
    n1 BIGINT,            -- Group 1 sample size
    n2 BIGINT,            -- Group 2 sample size
    method VARCHAR        -- "Welch's t-test" or "Student's t-test"
)
```

**Example:**
```sql
-- Compare treatment vs control (group_id: 0 = control, 1 = treatment)
SELECT (t_test_agg(outcome, treatment_group)).*
FROM experiment;

-- One-sided test (treatment > control)
SELECT t_test_agg(score, group, {'alternative': 'greater'})
FROM test_results;
```

### one_way_anova_agg

One-way Analysis of Variance for comparing means across multiple groups.

**Signature:**
```sql
one_way_anova_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Returns:**
```
STRUCT(
    f_statistic DOUBLE,   -- F-statistic
    p_value DOUBLE,       -- p-value
    df_between BIGINT,    -- Between-groups degrees of freedom
    df_within BIGINT,     -- Within-groups degrees of freedom
    ss_between DOUBLE,    -- Between-groups sum of squares
    ss_within DOUBLE,     -- Within-groups sum of squares
    n_groups BIGINT,      -- Number of groups
    n BIGINT,             -- Total sample size
    method VARCHAR        -- "One-Way ANOVA"
)
```

**Example:**
```sql
-- Compare means across treatment groups
SELECT (one_way_anova_agg(response, treatment_group)).*
FROM clinical_trial;
```

### yuen_agg

Yuen's trimmed mean test - robust alternative to t-test.

**Signature:**
```sql
yuen_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| trim | DOUBLE | 0.2 | Proportion to trim from each tail |

### brown_forsythe_agg

Brown-Forsythe test for equality of variances.

**Signature:**
```sql
brown_forsythe_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

## Nonparametric Tests

### mann_whitney_u_agg

Mann-Whitney U test (Wilcoxon rank-sum). Non-parametric alternative to t-test.

**Signature:**
```sql
mann_whitney_u_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |
| correction | BOOLEAN | true | Apply continuity correction |

**Returns:**
```
STRUCT(
    statistic DOUBLE,     -- U statistic
    p_value DOUBLE,       -- p-value
    effect_size DOUBLE,   -- Rank-biserial correlation
    ci_lower DOUBLE,      -- CI lower bound
    ci_upper DOUBLE,      -- CI upper bound
    n1 BIGINT,            -- Group 1 sample size
    n2 BIGINT,            -- Group 2 sample size
    method VARCHAR        -- "Mann-Whitney U"
)
```

### kruskal_wallis_agg

Kruskal-Wallis H test. Non-parametric alternative to ANOVA.

**Signature:**
```sql
kruskal_wallis_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

### wilcoxon_signed_rank_agg

Wilcoxon signed-rank test for paired samples.

**Signature:**
```sql
wilcoxon_signed_rank_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

### brunner_munzel_agg

Brunner-Munzel test - robust to unequal variances and non-normality.

**Signature:**
```sql
brunner_munzel_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

### permutation_t_test_agg

Permutation t-test - exact test without distributional assumptions.

**Signature:**
```sql
permutation_t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_permutations | INTEGER | 10000 | Number of permutations |

## Normality Tests

### shapiro_wilk_agg

Shapiro-Wilk test for normality.

**Signature:**
```sql
shapiro_wilk_agg(value DOUBLE) -> STRUCT
```

**Returns:**
```
STRUCT(
    statistic DOUBLE,    -- W statistic (closer to 1 = more normal)
    p_value DOUBLE,      -- p-value (low = reject normality)
    n BIGINT,            -- Sample size
    method VARCHAR       -- "Shapiro-Wilk"
)
```

### jarque_bera_agg

Jarque-Bera test for normality based on skewness and kurtosis.

**Signature:**
```sql
jarque_bera_agg(value DOUBLE) -> STRUCT
```

### dagostino_k2_agg

D'Agostino K² test for normality.

**Signature:**
```sql
dagostino_k2_agg(value DOUBLE) -> STRUCT
```

## Equivalence Tests (TOST)

### tost_t_test_agg

Two One-Sided Tests (TOST) for equivalence.

**Signature:**
```sql
tost_t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| equivalence_margin | DOUBLE | 0.5 | Equivalence margin |
| confidence_level | DOUBLE | 0.95 | Confidence level |

### tost_paired_agg

TOST for paired samples.

### tost_correlation_agg

TOST for correlation equivalence.

## Distribution Comparison

### energy_distance_agg

Energy distance between two distributions.

**Signature:**
```sql
energy_distance_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

### mmd_agg

Maximum Mean Discrepancy test.

**Signature:**
```sql
mmd_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

## Forecast Evaluation

### diebold_mariano_agg

Diebold-Mariano test for comparing forecast accuracy.

**Signature:**
```sql
diebold_mariano_agg(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE, [options MAP]) -> STRUCT
```

### clark_west_agg

Clark-West test for nested forecast models.

**Signature:**
```sql
clark_west_agg(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE) -> STRUCT
```

## See Also

- [Correlation](correlation.md) - Correlation tests
- [Categorical](categorical.md) - Tests for categorical data
