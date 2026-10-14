# Hypothesis Tests

Comprehensive statistical hypothesis testing functions for comparing groups and distributions.

## Unified Result Fields

Every test aggregate returns the integration contract's `test` fields (#159), always
present and NULL when not applicable:

| Field | Type | Meaning |
|-------|------|---------|
| `method` | VARCHAR | Name of the test |
| `alternative` | VARCHAR | `two_sided`, `less` or `greater` (NULL for omnibus tests) |
| `statistic` | DOUBLE | Test statistic |
| `df1`, `df2` | DOUBLE | (Numerator / denominator) degrees of freedom |
| `p_value` | DOUBLE | p-value |
| `estimate` | DOUBLE | Point estimate (difference `group1 - group2`, odds ratio, correlation, ...) |
| `conf_low`, `conf_high`, `conf_level` | DOUBLE | Confidence interval of `estimate` and its level |
| `effect_size`, `effect_name` | DOUBLE, VARCHAR | Effect size and its name (`cohens_d`, `rank_biserial`, `eta_squared`, `r`, `rho`, `tau`, `kappa`, ...) |
| `n` | BIGINT | Number of observations |

A test keeps its existing fields in their order and appends the unified fields it
did not have. `ci_lower` / `ci_upper`, `df` and the ANOVA `f_statistic`,
`df_between`, `df_within` are deprecated aliases of `conf_low` / `conf_high`, `df1`,
`statistic`, `df1`, `df2`; they are removed in the next minor release.

```sql
SELECT r.df1, r.estimate, r.conf_low, r.conf_high, r.effect_size, r.effect_name
FROM (SELECT t_test_agg(value, group_id) AS r
      FROM (VALUES (4.2, 0), (5.1, 0), (4.8, 0), (5.5, 0), (6.3, 1), (6.9, 1), (7.1, 1), (6.5, 1)) t(value, group_id));
```

## Parametric Tests

### t_test_agg

Two-sample t-test comparing means of two groups. Supports both Welch's (default) and Student's t-test.

**Signature:**
```text
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
CREATE OR REPLACE TABLE experiment AS
SELECT (i % 2)::INTEGER AS treatment_group,
       (10 + 1.5 * (i % 2) + ((i * 37) % 11) / 2.0)::DOUBLE AS outcome
FROM range(80) r(i);

-- Compare treatment vs control (group_id: 0 = control, 1 = treatment)
SELECT unnest(t_test_agg(outcome, treatment_group)) FROM experiment;

-- One-sided test
SELECT t_test_agg(outcome, treatment_group, {'alternative': 'less'}) AS result
FROM experiment;
```

### one_way_anova_agg

One-way Analysis of Variance for comparing means across multiple groups.

**Signature:**
```text
one_way_anova_agg(value DOUBLE, group ANY) -> STRUCT
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
-- Compare means across three treatment groups
SELECT unnest(one_way_anova_agg(response, treatment_group))
FROM (SELECT (i % 3)::INTEGER AS treatment_group,
             (5 + (i % 3) + ((i * 37) % 7) / 3.0)::DOUBLE AS response
      FROM range(90) r(i)) clinical_trial;
```

### yuen_agg

Yuen's trimmed mean test - robust alternative to t-test.

**Signature:**
```text
yuen_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| trim | DOUBLE | 0.2 | Proportion to trim from each tail |

### brown_forsythe_agg

Brown-Forsythe test for equality of variances.

**Signature:**
```text
brown_forsythe_agg(value DOUBLE, group ANY) -> STRUCT
```

## Nonparametric Tests

### mann_whitney_u_agg

Mann-Whitney U test (Wilcoxon rank-sum). Non-parametric alternative to t-test.

**Signature:**
```text
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
```text
kruskal_wallis_agg(value DOUBLE, group ANY) -> STRUCT
```

### wilcoxon_signed_rank_agg

Wilcoxon signed-rank test for paired samples.

**Signature:**
```text
wilcoxon_signed_rank_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

### brunner_munzel_agg

Brunner-Munzel test - robust to unequal variances and non-normality.

**Signature:**
```text
brunner_munzel_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

### permutation_t_test_agg

Permutation t-test - exact test without distributional assumptions.

**Signature:**
```text
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
```text
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
```text
jarque_bera_agg(value DOUBLE) -> STRUCT
```

### dagostino_k2_agg

D'Agostino K² test for normality.

**Signature:**
```text
dagostino_k2_agg(value DOUBLE) -> STRUCT
```

## Equivalence Tests (TOST)

### tost_t_test_agg

Two One-Sided Tests (TOST) for equivalence.

**Signature:**
```text
tost_t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| delta (alias `equivalence_bound`) | DOUBLE | — | Symmetric equivalence bounds `[-delta, delta]`; overrides the explicit bounds |
| bound_lower (aliases `lower`, `low`) | DOUBLE | -1.0 | Lower equivalence bound (used when `delta` is not set) |
| bound_upper (aliases `upper`, `high`) | DOUBLE | 1.0 | Upper equivalence bound (used when `delta` is not set) |
| confidence_level | DOUBLE | 0.95 | Confidence level |
| kind | VARCHAR | 'welch' | `'welch'` or `'student'` |
| mu | DOUBLE | 0.0 | Hypothesised difference |

### tost_paired_agg

TOST for paired samples.

### tost_correlation_agg

TOST for correlation equivalence.

## Distribution Comparison

### energy_distance_agg

Energy distance between two distributions.

**Signature:**
```text
energy_distance_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

### mmd_agg

Maximum Mean Discrepancy test.

**Signature:**
```text
mmd_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

## Forecast Evaluation

### diebold_mariano_agg

Diebold-Mariano test for comparing forecast accuracy.

**Signature:**
```text
diebold_mariano_agg(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE, [options MAP]) -> STRUCT
```

### clark_west_agg

Clark-West test for nested forecast models.

**Signature:**
```text
clark_west_agg(actual DOUBLE, forecast1 DOUBLE, forecast2 DOUBLE) -> STRUCT
```

## Pairwise Post-hoc Tests

`pairwise_t_test_agg(value, group [, options])` (R `pairwise.t.test`; options
`p_adjust` default `'holm'`, `pool_sd` default true, `alternative`),
`tukey_hsd_agg(value, group [, options])` (R `TukeyHSD(aov(...))`; option
`confidence_level`) and `dunn_test_agg(value, group [, options])` (`FSA::dunnTest`;
option `p_adjust`) compare every pair of groups. Each returns
`LIST(STRUCT(group1, group2, estimate, statistic, p_value, p_adj, conf_low, conf_high,
method, df1, alternative, conf_level, p_adjust_method))` with `estimate = group2 -
group1`. `p_adjust` accepts R's `p.adjust` methods: holm, hochberg, hommel,
bonferroni, BH (fdr), BY, none.

```sql
SELECT unnest(tukey_hsd_agg(value, grp), recursive := true)
FROM (VALUES ('A', 4.2), ('A', 5.1), ('A', 4.8), ('B', 6.3), ('B', 6.9), ('B', 7.1),
             ('C', 5.5), ('C', 5.0), ('C', 5.9)) t(grp, value);
```

`pairwise_t_test_by(source, group_col, value_col, factor_col, options := NULL)`,
`tukey_hsd_by` and `dunn_test_by` return the long `test` table (group column,
`model_id`, `test_id`, `method`, `statistic`, `p_value`, `group1`, `group2`, `p_adj`,
`df1`, `df2`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`,
`effect_name`, `alternative`, `p_adjust_method`), ready for significance brackets.

```sql
CREATE TABLE trial AS SELECT 'site_' || (i % 2) AS site, chr((65 + i % 3)::INTEGER) AS arm,
    (i % 3) + sin(i) AS response FROM range(60) t(i);
SELECT site, group1, group2, p_adj FROM pairwise_t_test_by('trial', site, response, arm);
```

## See Also

- [Correlation](correlation.md) - Correlation tests
- [Categorical](categorical.md) - Tests for categorical data
