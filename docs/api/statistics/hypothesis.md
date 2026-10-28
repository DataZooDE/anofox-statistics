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

## Reproducibility

The tests return bitwise-identical results for the same rows, whatever the thread
count, the row order or the order in which the group labels first appear: each
finalizer sorts the samples (the paired tests sort the pairs, and the k-sample tests
and `log_rank_test_agg` order the groups by label) before computing. This covers the
one-sample, paired, two-sample, k-sample and post-hoc tests; the time-series tests
(`durbin_watson_agg`, `ljung_box_agg`, `diebold_mariano_agg`, ...) depend on the row
order and need an `ORDER BY` inside the aggregate. See
[Methodology](../../METHODOLOGY.md#reproducibility-across-threads). The Monte Carlo tests (`permutation_t_test_agg`,
`energy_distance_agg`, `mmd_agg`, and `distance_cor_agg` in
[Correlation](correlation.md)) seed their permutation draw with 42 unless you pass
`'seed'`, so repeated calls return the same p-value. Pass a different seed to draw a
different set of permutations:

```sql
SELECT (permutation_t_test_agg(v, g)).p_value AS p_default,              -- seed 42
       (permutation_t_test_agg(v, g, {'seed': 7})).p_value AS p_seed_7
FROM (SELECT (i % 2)::INTEGER AS g, sin(i * 0.37) + (i % 2) * 0.2 AS v
      FROM range(60) r(i)) samples;
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
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the permutation draw; without it the default 42 makes the p-value reproducible |

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
energy_distance_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_permutations (alias `permutations`) | INTEGER | 1000 | Number of permutations |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the permutation draw; without it the default 42 makes the p-value reproducible |

### mmd_agg

Maximum Mean Discrepancy test.

**Signature:**
```text
mmd_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_permutations (alias `permutations`) | INTEGER | 1000 | Number of permutations |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the permutation draw; without it the default 42 makes the p-value reproducible |

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
`p_adjust` default `'holm'`, `pool_sd` default true, `alternative`,
`confidence_level` default 0.95),
`tukey_hsd_agg(value, group [, options])` (R `TukeyHSD(aov(...))`; option
`confidence_level`) and `dunn_test_agg(value, group [, options])` (`FSA::dunnTest`;
option `p_adjust`) compare every pair of groups. Each returns
`LIST(STRUCT(group1, group2, estimate, statistic, p_value, p_adj, conf_low, conf_high,
method, df1, alternative, conf_level, p_adjust_method))` with `estimate = group2 -
group1`. `p_adjust` accepts R's `p.adjust` methods: holm, hochberg, hommel,
bonferroni, BH (fdr), BY, none.

`pairwise_t_test_agg` fills `conf_low` / `conf_high` with the confidence interval of
each difference, from the same standard error and degrees of freedom as its t test
(pooled SD with N − k df when `pool_sd`, Welch otherwise); one-sided intervals
(`±inf`) for a one-sided `alternative`. With `p_adjust` `'none'` it is the
per-comparison interval at `confidence_level`. With any other adjustment it is the
Bonferroni simultaneous interval at `1 − (1 − confidence_level) / m` for the m pairs,
as `emmeans::confint(pairs(...), adjust = )`, which also uses Bonferroni for
step-wise and FDR methods. `conf_level` is the family-wise level `confidence_level`.
R's `pairwise.t.test` has no intervals; the pooled-SD values equal
`confint(pairs(emmeans(lm(value ~ group), ~ group), reverse = TRUE), adjust = )`.
`dunn_test_agg` has no interval.

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
SELECT site, group1, group2, p_adj, estimate, conf_low, conf_high
FROM pairwise_t_test_by('trial', site, response, arm);
```

## Multiple Testing

Adjust p-values from any set of tests for multiple comparisons, with R's
`p.adjust(p, method, n)` semantics (#219). Methods: `'holm'` (default), `'hochberg'`,
`'hommel'`, `'bonferroni'`, `'BH'` (alias `'fdr'`), `'BY'`, `'none'`; names are
case-insensitive and an unknown method is an error. NULL p-values stay NULL and are
not counted. `n` is the number of hypotheses in the family, for when not every test
appears as a row; it defaults to the number of non-NULL p-values and may not be
smaller (an error, as `stopifnot(n >= lp)` in R).

| Function | Signature | Returns |
|----------|-----------|---------|
| `p_adjust_by` | `(source, group_col, p_col, method := 'holm', n := NULL)` | Every source row, in source order, with `p_adjusted`; each group of `group_col` is one family |
| `p_adjust_agg` | `(p DOUBLE [, method VARCHAR [, n BIGINT]])` | `DOUBLE[]`, the adjusted p-values in input order (`ORDER BY` inside the call fixes it) |
| `p_adjust_list` | `(p DOUBLE[] [, method VARCHAR [, n BIGINT]])` | `DOUBLE[]`, the adjusted list in the same order |

`method` and `n` of `p_adjust_agg` must be constants; `p_adjust_list` also takes
per-row values.

```sql
CREATE TABLE results AS SELECT * FROM (VALUES
    ('weight', 'dose_a', 0.012), ('weight', 'dose_b', 0.030), ('weight', 'dose_c', 0.200),
    ('bp', 'dose_a', 0.001), ('bp', 'dose_b', 0.049)) t(endpoint, test, p_value);

-- One family per endpoint, Benjamini-Hochberg; the rows keep their order.
SELECT endpoint, test, p_value, p_adjusted
FROM p_adjust_by('results', endpoint, p_value, method := 'BH');

-- The same as lists, per endpoint.
SELECT endpoint, list(test ORDER BY test) AS tests,
       p_adjust_agg(p_value, 'BH' ORDER BY test) AS p_adj
FROM results GROUP BY endpoint;

-- A list of p-values, with 10 hypotheses of which only three were observed.
SELECT p_adjust_list([0.01, 0.02, 0.04], 'holm', 10);
```

There is no `p_adjust(p) OVER (PARTITION BY ...)` window function: in DuckDB 1.x an
extension's window aggregate is not told which row of a whole-partition frame it is
computing, so it cannot return a different value per row (a window function for
DuckDB 2.0 is tracked in #224). `p_adjust_by` is the supported way to get row-aligned
results.

### Row-aligned with plain window functions

For inline use in a larger query, Bonferroni, Holm and Benjamini-Hochberg can be written
with built-in window functions. With `n` = the number of non-NULL p-values in the
family and `i` = the rank of `p` in ascending order:

- Bonferroni: `least(1, n·p)`
- Holm: the running maximum, over `p` ascending, of `least(1, (n − i + 1)·p)`
- BH: the running minimum, over `p` descending, of `least(1, n·p / i)`

This gives the same result as `p_adjust_by` and R `p.adjust`, including for ties, NULLs
and p = 0 or 1 (`test/sql/reference/p_adjust_reference.test`). Keep the default `RANGE`
frame (ties share it) and `NULLS LAST`, and guard NULLs with `CASE`. `least` ignores
NULL arguments, and a NULL row's frame contains the non-NULL p-values.

```sql
WITH ranked AS (
    SELECT *,
           count(p_value) OVER (PARTITION BY endpoint) AS n_tests,
           row_number() OVER (PARTITION BY endpoint ORDER BY p_value NULLS LAST) AS i
    FROM results
)
SELECT endpoint, test, p_value,
       CASE WHEN p_value IS NOT NULL THEN least(1, n_tests * p_value) END AS p_bonferroni,
       CASE WHEN p_value IS NOT NULL THEN
           max(least(1, (n_tests - i + 1) * p_value))
               OVER (PARTITION BY endpoint ORDER BY p_value NULLS LAST) END AS p_holm,
       CASE WHEN p_value IS NOT NULL THEN
           min(least(1, n_tests * p_value / i))
               OVER (PARTITION BY endpoint ORDER BY p_value DESC NULLS LAST) END AS p_bh
FROM ranked
ORDER BY endpoint, test;
```

For the other methods (hochberg, hommel, BY) or an explicit `n`, use `p_adjust_by`.

## See Also

- [Correlation](correlation.md) - Correlation tests
- [Categorical](categorical.md) - Tests for categorical data
