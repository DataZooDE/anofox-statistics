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
    ci_lower DOUBLE,      -- CI lower bound (-inf for alternative 'less')
    ci_upper DOUBLE,      -- CI upper bound (inf for alternative 'greater')
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

The interval is for the difference of means at `confidence_level`. For a
one-sided `alternative` it is R's one-sided interval, `(-inf, d + t s]` for
`'less'` and `[d - t s, inf)` for `'greater'` (`t` the `confidence_level` quantile
of the t distribution, `s` the standard error), as `t.test(alternative = )`. Up to
0.15.0 the two-sided interval was returned for every alternative.

### t_test_one_sample_agg

One-sample t-test of H0: mean(x) = `mu` (R `t.test(x, mu = mu)`), with Cohen's d
`(mean(x) - mu) / sd(x)` as `effect_size`. NULL and NaN values are skipped; fewer
than two values give NULL.

**Signature:**
```text
t_test_one_sample_agg(x DOUBLE, [options MAP]) -> STRUCT
```

**Options:** `alternative` ('two_sided', 'less', 'greater'), `confidence_level`
(default 0.95), `mu` (default 0.0). `kind` and `paired` do not apply and raise an error.

**Returns:** `statistic`, `p_value`, `df`, `effect_size` (Cohen's d), `ci_lower`,
`ci_upper` (CI of the mean at `confidence_level`; one-sided with an infinite bound for a one-sided alternative, as `t.test`), `n`, `method`
(`One Sample t-test`), `alternative`, then the unified fields (`estimate` is the
sample mean, `effect_name` is `cohens_d`).

### t_test_paired_agg

Paired t-test of H0: mean(x - y) = `mu` (R `t.test(x, y, paired = TRUE, mu = mu)`),
with Cohen's d_z `(mean(d) - mu) / sd(d)` of the differences `d = x - y` as
`effect_size`. A row whose `x` or `y` is NULL or NaN is dropped as a pair; fewer
than two pairs give NULL. Options and result fields as `t_test_one_sample_agg`
(`method` is `Paired t-test`, `estimate` the mean difference).

**Signature:**
```text
t_test_paired_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Example:**
```sql
CREATE OR REPLACE TABLE crossover AS
SELECT i AS subject,
       (70 + (i * 13) % 7)::DOUBLE AS before,
       (70 + (i * 13) % 7 + 1.2 + ((i * 7) % 5 - 2) / 4.0)::DOUBLE AS after
FROM range(20) r(i);

-- Did the score change? (H0: mean(before - after) = 0)
SELECT r.statistic, r.df, r.p_value, r.estimate, r.ci_lower, r.ci_upper, r.effect_size
FROM (SELECT t_test_paired_agg(before, after) AS r FROM crossover);

-- Is the mean score after treatment above 72?
SELECT r.statistic, r.p_value, r.effect_size
FROM (SELECT t_test_one_sample_agg(after, {'mu': 72, 'alternative': 'greater'}) AS r FROM crossover);
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

### two_way_anova_agg

Two-way (factorial) ANOVA with interaction: tests the main effect of each factor and
their interaction. Sums of squares are Type III (each effect adjusted for all others),
matching R's `car::Anova(lm(y ~ a * b, contrasts = list(a = "contr.sum", b = "contr.sum")), type = 3)`;
for a balanced design they equal `summary(aov(y ~ a * b))`. Unbalanced designs are
supported, but every combination of the two factors must be observed (otherwise the
result is NULL).

**Signature:**
```text
two_way_anova_agg(y DOUBLE, factor_a BIGINT | VARCHAR, factor_b BIGINT | VARCHAR, [options]) -> LIST(STRUCT)
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| terms | LIST(VARCHAR) | ['factor_a', 'factor_b'] | Labels of the two factors; the interaction is labelled `a:b` |

**Returns:** one list element per F test (factor A, factor B, interaction):
```
LIST(STRUCT(
    term VARCHAR,              -- 'factor_a', 'factor_b', 'factor_a:factor_b' (or the terms labels)
    method VARCHAR,            -- 'Two-way ANOVA: factor A', ...
    statistic DOUBLE,          -- F
    df1 DOUBLE,                -- effect df
    df2 DOUBLE,                -- residual df (n - levels_a * levels_b)
    p_value DOUBLE,
    sum_sq DOUBLE,             -- Type III sum of squares of the effect
    mean_sq DOUBLE,
    residual_sum_sq DOUBLE,
    residual_mean_sq DOUBLE,
    n_observations BIGINT,
    ... unified fields: effect_size is partial eta squared (effect_name 'partial_eta_squared')
))
```

A row is skipped when `y` is NULL or NaN or a factor is NULL. Levels are ordered by
label, so the result does not depend on the row order.

**Example:**
```sql
-- Yield by fertilizer (3 levels) and irrigation (2 levels), one row per term
SELECT t.term, round(t.statistic, 4) AS f, t.df1, t.df2, round(t.p_value, 6) AS p_value,
       round(t.effect_size, 4) AS partial_eta_sq
FROM (SELECT unnest(two_way_anova_agg(yield, fertilizer, irrigation,
                                      {'terms': ['fertilizer', 'irrigation']})) AS t
      FROM (SELECT ['none', 'low', 'high'][(i % 3) + 1] AS fertilizer,
                   (i // 3) % 2 AS irrigation,
                   10 + (i % 3) * 1.5 + ((i // 3) % 2) * 0.8 + sin(i * 1.3) AS yield
            FROM range(36) r(i)) field_trial);
```

### repeated_measures_anova_agg

One-way repeated-measures ANOVA: compares the means of conditions measured on the same
subjects (long format, one row per subject and condition). The F test is R's
`summary(aov(y ~ condition + Error(subject / condition)))`. With three or more
conditions it adds Mauchly's test of sphericity and the Greenhouse-Geisser (`gg_*`) and
Huynh-Feldt (`hf_*`) corrected tests, as `afex::aov_ez` / `car::Anova` on the wide
matrix report them.

**Signature:**
```text
repeated_measures_anova_agg(y DOUBLE, subject BIGINT | VARCHAR, condition BIGINT | VARCHAR, [options]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| sphericity | BOOLEAN | true | Compute Mauchly's test and the Greenhouse-Geisser and Huynh-Feldt corrections |

**Returns:**
```
STRUCT(
    method VARCHAR,              -- 'Repeated measures ANOVA'
    statistic DOUBLE,            -- F of the within-subjects effect
    df1 DOUBLE,                  -- k - 1
    df2 DOUBLE,                  -- (n - 1)(k - 1)
    p_value DOUBLE,              -- uncorrected p-value
    ss_condition DOUBLE,
    ss_subjects DOUBLE,
    ss_error DOUBLE,             -- subject x condition residual
    ms_condition DOUBLE,
    ms_error DOUBLE,
    n_subjects BIGINT,           -- complete subjects used
    n_conditions BIGINT,
    n_excluded_subjects BIGINT,  -- subjects missing at least one condition
    mauchly_w DOUBLE,            -- Mauchly's W (NULL for k = 2 or sphericity := false)
    mauchly_statistic DOUBLE,    -- chi-square approximation
    mauchly_df DOUBLE,
    mauchly_p_value DOUBLE,
    gg_epsilon DOUBLE,           -- Greenhouse-Geisser epsilon
    gg_df1 DOUBLE,
    gg_df2 DOUBLE,
    gg_p_value DOUBLE,           -- Greenhouse-Geisser corrected p-value
    hf_epsilon DOUBLE,           -- Huynh-Feldt epsilon (Lecoutre-corrected, capped at 1)
    hf_df1 DOUBLE,
    hf_df2 DOUBLE,
    hf_p_value DOUBLE,           -- Huynh-Feldt corrected p-value
    ... unified fields: effect_size is partial eta squared, n is n_subjects
)
```

Subjects that lack a value for some condition are excluded (listwise, like afex and
ez) and counted in `n_excluded_subjects`; a second value for the same subject and
condition is an error (aggregate replicates first). The result is NULL with fewer than
two complete subjects or two conditions. `mauchly_statistic` is the chi-square
approximation with Bartlett's correction factor and `mauchly_p_value` includes its
second-order term, as R's `mauchly.test`. `hf_epsilon` is the Huynh-Feldt epsilon with
the Lecoutre (1991) correction, `(n p ε_GG - 2) / (p (n - 1 - p ε_GG))` with
`p = k - 1`, capped at 1: car and afex print the uncapped value but compute the
corrected p-value with the capped one, which `hf_p_value` matches.

**Example:**
```sql
-- Reaction time of 10 participants under 4 conditions
-- (the spread grows with the condition, so sphericity is doubtful)
SELECT round(r.statistic, 4) AS f, r.df1, r.df2, r.p_value,
       round(r.mauchly_w, 4) AS mauchly_w, round(r.mauchly_p_value, 4) AS mauchly_p,
       round(r.gg_epsilon, 4) AS gg_epsilon, r.gg_p_value,
       round(r.hf_epsilon, 4) AS hf_epsilon, r.hf_p_value
FROM (SELECT repeated_measures_anova_agg(rt, participant, 'c' || condition) AS r
      FROM (SELECT i // 4 AS participant, i % 4 AS condition,
                   300 + (i % 4) * 6 + (i // 4) * 5 + (hash(i) % 1000) / 100.0 * (1 + i % 4) AS rt
            FROM range(40) t(i)) trials);
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
    ci_lower DOUBLE,      -- CI lower bound (-inf for alternative 'less')
    ci_upper DOUBLE,      -- CI upper bound (inf for alternative 'greater')
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

**Returns:** `statistic` (V), `p_value`, `ci_lower`, `ci_upper`, `n`, `method`,
`alternative`, `rank_biserial`, then the unified fields (`effect_size` = `rank_biserial`,
`effect_name` = `rank_biserial`).

`rank_biserial` is the matched-pairs rank-biserial correlation `(T- - T+) / (T+ + T-)`,
where `T+` / `T-` are the rank sums of `|x - y|` over positive / negative differences
(zero differences dropped, average ranks for ties). It lies in [-1, 1] and is
positive when `y` tends to be larger than `x` -- the sign convention of
`mann_whitney_u_agg`'s rank-biserial; R `effectsize::rank_biserial(x, y, paired = TRUE)`
reports the opposite sign. NULL when every difference is zero.

```sql
SELECT r.statistic, r.p_value, r.rank_biserial
FROM (SELECT wilcoxon_signed_rank_agg(before, after) AS r
      FROM (VALUES (68.5, 70.2), (72.3, 74.5), (69.1, 68.3), (71.8, 73.9),
                   (67.2, 69.4), (73.5, 75.8), (70.2, 72.1), (68.9, 71.2)) t(before, after));
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

### bootstrap_mean_ci_agg

Percentile bootstrap confidence interval of the mean (anofox-statistics
`bootstrap_mean_ci_with_options`): the IID bootstrap, or for a dependent series the
stationary, circular block or moving block bootstrap. The block methods resample
blocks of consecutive values in the aggregate's `ORDER BY` order (as `acf_agg`), so
pass the time column there; without `ORDER BY` the row order, and so the result, is not
defined. The IID bootstrap does not depend on the row order. NULL and NaN values are
skipped; fewer than two values give NULL. The draw is seeded (42 by default), so
repeated calls return the same interval.

**Signature:**
```text
bootstrap_mean_ci_agg(x DOUBLE [, options MAP] [ORDER BY t]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| method | VARCHAR | 'iid' | `'iid'`, `'stationary'`, `'circular_block'` or `'moving_block'` |
| n_bootstrap (alias `n_boot`) | INTEGER | 10000 | Number of bootstrap replicates (>= 2) |
| block_length (alias `l`, `block_size`, `expected_block_length`) | DOUBLE | - | Required for the block methods, not allowed for `'iid'`: the expected block length of `'stationary'` (>= 1, may be fractional), the block length of `'circular_block'` / `'moving_block'` (a whole number >= 2) |
| confidence_level (alias `confidence`, `conf_level`, `level`) | DOUBLE | 0.95 | Level of the percentile interval |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the bootstrap draw |

**Returns:** `estimate` (the mean), `std_error` (standard deviation of the bootstrap
means), `conf_low`, `conf_high` (percentile interval: the `floor(alpha/2 B)`-th and
`ceil((1 - alpha/2) B)`-th sorted replicate, 0-based), `conf_level`, `n`,
`n_bootstrap`, `block_length` (DOUBLE; NULL for `'iid'`), `method`. The schemes:

| method | Resampling | R |
|---|---|---|
| `'stationary'` | Politis-Romano stationary bootstrap: blocks of geometric length with mean `block_length`, wrapping around | `boot::tsboot(sim = "geom")` |
| `'circular_block'` | blocks of `block_length` starting anywhere, wrapping around | `tsboot(sim = "fixed", endcorr = TRUE)` |
| `'moving_block'` | blocks of `block_length` that do not wrap around | `tsboot(sim = "fixed", endcorr = FALSE)` |

A block length that divides `n` gives the textbook block bootstraps; otherwise the
last block of each resample is cut to length `n`.

```sql
SELECT r.estimate, r.std_error, r.conf_low, r.conf_high, r.method
FROM (SELECT bootstrap_mean_ci_agg(x) AS r
      FROM (SELECT 10 + sin(i * 0.7) AS x FROM range(60) t(i)));
```

For a dependent series the block bootstrap gives a wider (more honest) interval:

```sql
SELECT (bootstrap_mean_ci_agg(x ORDER BY t)).std_error AS se_iid,
       (bootstrap_mean_ci_agg(x, {'method': 'circular_block', 'block_length': 8} ORDER BY t)).std_error AS se_block,
       (bootstrap_mean_ci_agg(x, {'method': 'stationary', 'block_length': 8} ORDER BY t)).std_error AS se_stationary
FROM (SELECT i AS t, 10 + sin(i * 0.3) + 0.1 * cos(i * 2.1) AS x FROM range(120) r(i));
```

Per group, with a 90% interval:

```sql
SELECT sensor, (r).conf_low, (r).conf_high
FROM (SELECT sensor, bootstrap_mean_ci_agg(reading, {'method': 'circular_block', 'block_length': 5,
                                                     'confidence_level': 0.9, 'n_bootstrap': 2000} ORDER BY t) AS r
      FROM (SELECT i % 2 AS sensor, i // 2 AS t, 20 + (i % 2) + sin(i * 0.4) AS reading FROM range(100) s(i))
      GROUP BY sensor)
ORDER BY sensor;
```


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

### TOST variants

One-sample, proportion, rank-based, robust and bootstrap TOST. All seven return
the `tost_t_test_agg` STRUCT: the two one-sided tests (`t_lower`, `t_upper`,
`p_lower`, `p_upper`), `p_value` (the larger of the two), `estimate` with its
`(1 - 2 alpha)` interval `ci_lower` / `ci_upper`, the bounds, `equivalent`
(the interval lies inside the bounds), `n`, `method` and the unified fields.
`t_lower` / `t_upper` hold the statistic of the respective test: t (one-sample,
Yuen), z (proportions), Wilcoxon V (paired) or U (two-sample), and the point
estimate for the bootstrap. `df` is NULL where the test has none.

**Signatures:**
```text
tost_one_sample_agg(value DOUBLE, [options]) -> STRUCT
tost_prop_one_agg(success BIGINT, [options]) -> STRUCT
tost_prop_two_agg(success BIGINT, group_id BIGINT | VARCHAR, [options]) -> STRUCT
tost_wilcoxon_paired_agg(x DOUBLE, y DOUBLE, [options]) -> STRUCT
tost_wilcoxon_agg(value DOUBLE, group_id BIGINT | VARCHAR, [options]) -> STRUCT
tost_yuen_agg(value DOUBLE, group_id BIGINT | VARCHAR, [options]) -> STRUCT
tost_bootstrap_agg(value DOUBLE, group_id BIGINT | VARCHAR, [options]) -> STRUCT
```

| Function | Estimate | Method (upstream `equivalence`) |
|----------|----------|----------------------------------|
| `tost_one_sample_agg` | `mean(value) - mu` | one-sample t-tests (`TOSTER::TOSTone`) |
| `tost_prop_one_agg` | `p_hat - p0` | z-tests, SE from `p_hat` (`TOSTER::TOSTone.prop`) |
| `tost_prop_two_agg` | `p1 - p2` | z-tests, unpooled SE (`TOSTER::TOSTtwo.prop`) |
| `tost_wilcoxon_paired_agg` | Hodges-Lehmann pseudo-median of `x - y` | shifted signed-rank tests, normal approximation with continuity correction |
| `tost_wilcoxon_agg` | Hodges-Lehmann shift `group1 - group2` | shifted rank-sum tests, normal approximation with continuity correction |
| `tost_yuen_agg` | difference of trimmed means | Yuen's t-tests, Welch df |
| `tost_bootstrap_agg` | `mean(group1) - mean(group2)` | percentile bootstrap |

Two-group variants take any two group labels (BIGINT or VARCHAR); group 1 is
the smaller label, and a third label is an error. A success is a non-zero
`success`, as in `prop_test_one_agg`. Rows with a NULL or NaN input are
skipped; too little data (one-sample < 2 rows, Wilcoxon < 3 per group or pairs,
Yuen < 4 per group, bootstrap < 3 per group, no trials) gives NULL.

**Options:**
| Key | Applies to | Default | Description |
|-----|-----------|---------|-------------|
| delta (alias `equivalence_bound`) | all | required* | Symmetric bounds `[-delta, delta]`; overrides the explicit bounds |
| bound_lower / bound_upper (aliases `lower`/`low`, `upper`/`high`) | all | required* | Asymmetric bounds (raw units; proportions within [-1, 1]); give both |
| alpha | all | 0.05 | Level of each one-sided test, in (0, 0.5) |
| confidence_level | all | 0.95 | Alternative to alpha: `alpha = 1 - confidence_level` |
| mu | `tost_one_sample_agg` | 0.0 | Hypothesised mean |
| p0 (alias `p`) | `tost_prop_one_agg` | 0.5 | Hypothesised proportion |
| trim (aliases `trim_proportion`, `tr`) | `tost_yuen_agg` | 0.2 | Proportion trimmed from each tail, in [0, 0.5) |
| n_bootstrap (aliases `bootstrap_samples`, `n_boot`) | `tost_bootstrap_agg` | 10000 | Resamples (at least 100) |
| seed (alias `random_state`) | `tost_bootstrap_agg` | 42 | RNG seed; the result is reproducible and independent of row order |

\* The equivalence bounds have no default: these seven functions raise an error
unless `delta` or both `bound_lower` and `bound_upper` are given. (The older
`tost_t_test_agg`, `tost_paired_agg` and `tost_correlation_agg` keep their
default bounds.)

**Example:**
```sql
CREATE TABLE eq AS
SELECT i, CASE WHEN i % 2 = 0 THEN 'old' ELSE 'new' END AS process,
       10 + sin(i * 1.3) + 0.05 * (i % 2) AS weight,
       10 + sin(i * 0.7) AS before, 10 + sin(i * 0.7) + 0.1 * cos(i * 1.9) AS after,
       (i % 5 < 2)::BIGINT AS defect
FROM range(60) t(i);

SELECT (tost_one_sample_agg(weight, {'mu': 10, 'delta': 0.5})).equivalent AS mean_is_10,
       (tost_wilcoxon_paired_agg(before, after, {'delta': 0.2})).p_value AS paired_p,
       (tost_wilcoxon_agg(weight, process, {'delta': 0.5})).estimate AS hl_shift,
       (tost_yuen_agg(weight, process, {'delta': 0.5, 'trim': 0.1})).equivalent AS trimmed_equivalent,
       (tost_bootstrap_agg(weight, process, {'delta': 0.5, 'n_bootstrap': 2000})).ci_upper AS boot_ci_upper,
       (tost_prop_one_agg(defect, {'p0': 0.4, 'delta': 0.15})).equivalent AS rate_is_40pct,
       (tost_prop_two_agg(defect, process, {'delta': 0.2})).p_value AS rate_diff_p
FROM eq;
```

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

## Data Drift

Two-sample statistics that compare a **current** sample with a **reference** sample:
the drift tests of [evidently](https://github.com/evidentlyai/evidently)
(`legacy/calculations/stattests`), validated against R (see
`validation/generators/make_reference_drift.R`). They take
`(value, group [, options])`. The group may be BIGINT or VARCHAR, with exactly two
labels (except `ad_test_agg`, which takes two or more). The **reference is the smaller
label** unless the `reference` option names it. This matters for the asymmetric
measures (KL, normed Wasserstein, quantile-binned PSI). NULL values and groups are
skipped. One group, or an empty group, gives NULL. Every result also carries the
[unified test fields](#unified-result-fields).

| Function | evidently | Values | Measure |
|---|---|---|---|
| `ks_test_agg` | `ks` | numeric | Two-sample Kolmogorov-Smirnov test |
| `ad_test_agg` | `anderson` | numeric | k-sample Anderson-Darling test |
| `cvm_test_agg` | `cramer_von_mises` | numeric | Two-sample Cramér-von Mises test |
| `epps_singleton_agg` | `es` | numeric | Epps-Singleton test (empirical characteristic function) |
| `wasserstein_agg` | `wasserstein` | numeric | 1-D Wasserstein distance, raw and SD-normed |
| `psi_agg` | `psi` | numeric or VARCHAR | Population Stability Index |
| `kl_divergence_agg` | `kl_div` | numeric or VARCHAR | Kullback-Leibler divergence KL(reference ‖ current) |
| `jensen_shannon_agg` | `jensenshannon` | numeric or VARCHAR | Jensen-Shannon distance |
| `hellinger_agg` | `hellinger` | numeric or VARCHAR | Hellinger distance |
| `tvd_agg` | `TVD` | numeric or VARCHAR | Total variation distance + permutation p-value |
| `category_proportions_agg` | (shares) | numeric or VARCHAR | Share of each category in both groups |
| `z_drift_agg` | `z` | numeric or VARCHAR | Two-proportion z test |
| `chisq_drift_agg` | `chisquare` | numeric or VARCHAR | Chi-squared goodness-of-fit against the reference shares |
| `g_drift_agg` | `g_test` | numeric or VARCHAR | G goodness-of-fit against the reference shares |
| `column_drift_agg` | all tests | numeric or VARCHAR | evidently's column drift (default test, threshold, rule) |
| `dataset_drift_agg`, `dataset_drift_by` | dataset drift | BOOLEAN flags / table | Share of drifted columns |

```sql
CREATE OR REPLACE TABLE drift AS
SELECT 0 AS grp, 2*sin(0.37*i) AS v, (i % 4)::INTEGER AS c FROM range(50) r(i)
UNION ALL
SELECT 1 AS grp, 2*sin(0.53*i) + 0.5 AS v, ((i*3) % 5)::INTEGER AS c FROM range(40) r(i);

SELECT (ks_test_agg(v, grp)).p_value         AS ks_p,        -- 0.0575 (exact)
       (wasserstein_agg(v, grp)).distance_normed AS wasserstein, -- 0.439
       (psi_agg(v, grp)).psi                 AS psi_numeric, -- 0.571
       (jensen_shannon_agg(c, grp)).distance AS js_categorical, -- 0.274
       (tvd_agg(c::VARCHAR, grp)).p_value    AS tvd_p
FROM drift;
```

### ks_test_agg

`ks_test_agg(value DOUBLE, group, [options]) -> STRUCT(statistic, p_value, n1, n2, method, exact, ...)`

Two-sample Kolmogorov-Smirnov test, as R `ks.test`. `statistic` is D = sup |F₁ − F₂| over
the distinct pooled values (D⁺ / D⁻ for one-sided alternatives). The p-value is exact
(R's `psmirnov`, also with ties) when n1·n2 < 10000. Otherwise it comes from the
**limiting Kolmogorov distribution**, which is R's `exact = FALSE` value. It is not
scipy's finite-n approximation. `method` and `exact` say which was used. evidently
flags drift at p ≤ 0.05.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group (the first sample of `ks.test`) |
| alternative | `'two_sided'` | `'two_sided'`, `'less'`, `'greater'` (R's convention) |
| exact | NULL (auto) | Force the exact (`true`) or asymptotic (`false`) p-value |

### ad_test_agg

`ad_test_agg(value DOUBLE, group, [options]) -> STRUCT(statistic, ad, p_value, k, n, method, ...)`

k-sample Anderson-Darling test (Scholz and Stephens, 1987), as
`kSamples::ad.test(method = "asymptotic")`, for k ≥ 2 groups. `statistic` is the
standardized T.AD, `ad` the raw AD criterion. The p-value is kSamples' asymptotic one.
It is **not capped to [0.001, 0.25]** like scipy's `anderson_ksamp` (evidently's
`anderson`), so the p-values differ from evidently by design. The statistics agree.

| Key | Default | Description |
|---|---|---|
| midrank | `true` | Version 2 (midranks, correct with ties); `false` for version 1 |

### cvm_test_agg

`cvm_test_agg(value DOUBLE, group, [options]) -> STRUCT(statistic, p_value, n1, n2, method, exact, ...)`

Two-sample Cramér-von Mises test: Anderson's T with midranks. With `method = 'auto'`,
the p-value is exact when max(n1, n2) ≤ 20, otherwise asymptotic (the limiting
CvM distribution, `goftest::pCvM`, as scipy). The exact p-value enumerates the lattice
paths (Xiao, Gordon and Yakovlev, 2006). With ties, it is exact for the observed tie
pattern; scipy uses the no-ties distribution there.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |
| method | `'auto'` | `'auto'`, `'exact'`, `'asymptotic'` |

### epps_singleton_agg

`epps_singleton_agg(value DOUBLE, group, [options]) -> STRUCT(statistic, p_value, df, n1, n2, method, ...)`

Epps-Singleton test on the empirical characteristic function, as
`scipy.stats.epps_singleton_2samp`. It also works for discrete data. The evaluation
points `t` are scaled by the pooled semi-IQR. The statistic gets the small-sample
correction when max(n1, n2) < 25. `df` is the rank of the covariance matrix, which can
be below 2·len(t). It returns NULL when a group has fewer than 5 values, and fails when
the pooled IQR is 0.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |
| t | `[0.4, 0.8]` | Evaluation points (LIST of positive numbers) |

### wasserstein_agg

`wasserstein_agg(value DOUBLE, group, [options]) -> STRUCT(distance, distance_normed, p_value, n1, n2, method, ...)`

1-D Wasserstein-p distance (earth mover's distance for p = 1), as
`transport::wasserstein1d`. `distance_normed` is evidently's score
W_p / max(SD(reference), 0.001), with the **population** SD (ddof = 0). evidently uses
p = 1 and flags drift at `distance_normed ≥ 0.1`. With `n_permutations > 0`, `p_value` is
a seeded permutation p-value on W_p, (1 + #{W* ≥ W}) / (1 + B); otherwise it is NULL.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Group whose SD norms `distance_normed` |
| p | 1 | Order of the distance (≥ 1) |
| n_permutations (alias `permutations`) | 0 | Permutations of the p-value |
| seed (alias `random_state`) | 42 | Seed of the permutation draw |

### psi_agg, kl_divergence_agg, jensen_shannon_agg, hellinger_agg

```text
psi_agg(value DOUBLE | VARCHAR, group, [options])            -> STRUCT(psi, n_bins, n1, n2, drift, method, ...)
kl_divergence_agg(value DOUBLE | VARCHAR, group, [options])  -> STRUCT(kl, n_bins, n1, n2, drift, method, ...)
jensen_shannon_agg(value DOUBLE | VARCHAR, group, [options]) -> STRUCT(distance, divergence, n_bins, n1, n2, drift, method, ...)
hellinger_agg(value DOUBLE | VARCHAR, group, [options])      -> STRUCT(distance, n_bins, n1, n2, drift, method, ...)
```

These are divergences between the bin proportions p (reference) and q (current), on
bins shared by both groups:

- **PSI** = Σ (p − q) ln(p / q), on zero-filled proportions. This is the Jeffreys
  divergence, `philentropy` `"jeffreys"`. The usual reading: < 0.1 stable, 0.1–0.25
  moderate, > 0.25 significant.
- **KL** = Σ p ln(p / q), **asymmetric**: the reference is p. The proportions are
  zero-filled and renormalized, as `scipy.stats.entropy`.
- **Jensen-Shannon**: `distance` is the square root of the JS divergence, as
  `scipy.spatial.distance.jensenshannon` and evidently report it. `divergence` is the
  JS divergence itself (`philentropy::JSD`). It is symmetric and bounded by √ln 2 (1 in base 2). There is no zero fill.
- **Hellinger** = √(1 − Σ √(p q)), in [0, 1]. `philentropy`'s `"hellinger"` is twice
  this value. There is no zero fill.

The zero fill (PSI and KL) is evidently's by default: an empty bin gets 1e-4, or
min_nonzero / 1e6 when the smallest non-zero proportion is ≤ 1e-4. The `epsilon` option
replaces every empty bin by that constant instead (KL renormalizes afterwards).

Binning:

- `'auto'` (the default) is evidently's rule. When the reference has more than 20
  distinct values, it uses equal-width **Sturges** bins over the pooled sample (numpy
  `histogram_bin_edges(..., "sturges")`; R's `hist(breaks = "Sturges")` uses `pretty()`
  breaks instead). Otherwise it uses one bin per distinct value.
- For `hellinger_agg`, `'auto'` also bins per value up to 20 distinct reference values.
  Above that it uses evidently's numeric Hellinger rule: Sturges bins whose count comes
  from the number of **distinct** pooled values.
- `'quantile'` uses `n_bins` bins at the reference quantiles (R type 7), the classic
  credit-scoring PSI. The outer edges are open, so current values outside the
  reference range fall into the first or last bin. Tied breaks are merged.
- VARCHAR values always get one bin per category: the categories are the labels seen
  in either group, sorted by byte order (DuckDB's default `ORDER BY`). They go to
  upstream's `*_categorical` variants, so a label missing from one group is an empty bin
  (zero-filled for PSI and KL). Only `'auto'` and `'categorical'` binning apply.
  `category_proportions_agg` returns the shares p and q these divergences use.

`drift` is `statistic ≥ threshold`, with evidently's default threshold 0.1.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group (p) |
| binning | `'auto'` | `'auto'`, `'sturges'`, `'quantile'`, `'categorical'` |
| n_bins | 10 | Bins of `binning = 'quantile'` |
| threshold | 0.1 | Drift threshold of the `drift` flag |
| base | e | Logarithm base (`kl_divergence_agg`, `jensen_shannon_agg`) |
| epsilon | evidently's fill | Constant zero fill, > 0 (`psi_agg`, `kl_divergence_agg`) |

```sql
SELECT (psi_agg(v, grp, {'binning': 'quantile', 'n_bins': 10})).psi AS psi_deciles,
       (kl_divergence_agg(c, grp, {'base': 2})).kl              AS kl_bits,
       (hellinger_agg(c::VARCHAR, grp)).distance                AS hellinger
FROM drift;
```

On string labels (here `'amber'` only occurs in the reference and `'violet'` only in the
current sample):

```sql
CREATE OR REPLACE TABLE colors AS
SELECT 'ref' AS grp, ['red','green','blue','amber'][((i*7) % 11) % 4 + 1] AS s FROM range(60) r(i)
UNION ALL
SELECT 'cur' AS grp, ['red','green','blue','violet'][((i*5) % 13) % 4 + 1] AS s FROM range(45) r(i);

SELECT (psi_agg(s, grp, {'reference': 'ref'})).psi                AS psi,       -- 3.247
       (kl_divergence_agg(s, grp, {'reference': 'ref'})).kl       AS kl,        -- 1.549
       (jensen_shannon_agg(s, grp, {'reference': 'ref'})).distance AS js,       -- 0.385
       (hellinger_agg(s, grp, {'reference': 'ref'})).distance     AS hellinger, -- 0.462
       (tvd_agg(s, grp, {'reference': 'ref'})).statistic          AS tvd        -- 0.267
FROM colors;
```

### tvd_agg

`tvd_agg(category DOUBLE | VARCHAR, group, [options]) -> STRUCT(statistic, p_value, n_categories, n1, n2, method, ...)`

Total variation distance ½ Σ |p_k − q_k| between the category distributions of the two
groups, with a seeded permutation p-value (1 + #{T* ≥ T}) / (1 + B), so p is never 0.
evidently flags drift at p < 0.1. TVD is much less powerful than a chi-squared test on
the same table. On the example data, TVD gives p ≈ 0.29, while `chisq_test_agg` rejects.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |
| n_permutations (alias `permutations`) | 1000 | Number of permutations |
| seed (alias `random_state`) | 42 | Seed of the permutation draw |

### category_proportions_agg

`category_proportions_agg(category DOUBLE | VARCHAR, group, [options]) -> STRUCT(categories, p_reference, p_current, n_categories, n1, n2)`

The share of each category in the reference (`p_reference`) and the current group
(`p_current`), over the categories seen in either group, in sorted order (upstream
`category_proportions`). These are the p and q of `psi_agg`, `kl_divergence_agg`,
`jensen_shannon_agg`, `hellinger_agg` and `tvd_agg` on categories, before any zero fill,
so they show which categories drive the drift. `categories` is a `VARCHAR[]` for VARCHAR
values and a `DOUBLE[]` (one category per distinct value) for numeric values. The
result has no unified test fields: it is not a test.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |

```sql
SELECT r.categories, r.p_reference, r.p_current
FROM (SELECT category_proportions_agg(s, grp, {'reference': 'ref'}) AS r FROM colors);
-- [amber, blue, green, red, violet], [0.2, 0.283, 0.25, 0.267, 0.0], [0.0, 0.244, 0.222, 0.311, 0.222]
```

### z_drift_agg, chisq_drift_agg, g_drift_agg

```
z_drift_agg(category DOUBLE | VARCHAR, group, [options])     -> STRUCT(statistic, p_value, p_reference, p_current, n_categories, n1, n2, method, ...)
chisq_drift_agg(category DOUBLE | VARCHAR, group, [options]) -> STRUCT(statistic, df, p_value, n_categories, n1, n2, method, ...)
g_drift_agg(category DOUBLE | VARCHAR, group, [options])     -> STRUCT(statistic, df, p_value, n_categories, n1, n2, method, ...)
```

evidently's frequency tests for categorical (or low-cardinality numeric) columns. Every
distinct value is a category.

- `z_drift_agg` (evidently `z`): pooled two-proportion z test, without continuity
  correction, of the share of labels other than the smallest category.
  `p_reference` / `p_current` are these shares, and `statistic` is
  z = (p_reference − p_current) / √(P(1 − P)(1/n1 + 1/n2)). It matches
  R `prop.test(correct = FALSE)` with z² = X². One category gives z = 0, p = 1.
  evidently flags drift at p < 0.05.
- `chisq_drift_agg` (evidently `chisquare`): Pearson goodness-of-fit test of the current
  counts against the reference shares, with expected counts e = r · n2 / n1 on k − 1 df,
  as R `chisq.test(o, p = r / sum(r))`. The reference is the known distribution, not a
  second sample (that is `chisq_test_agg`). A category that appears only in the current
  sample gives `statistic = inf` and p = 0. evidently flags drift at p < 0.05.
- `g_drift_agg` (evidently `g_test`): the same comparison with G = 2 Σ o ln(o / e).
  evidently flags drift at p < 0.1. evidently passes the unscaled reference counts as
  expected counts, so its value agrees only for equal sample sizes. Here the expected
  counts are always scaled to the current size.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |
| alternative | two_sided | `z_drift_agg` only: `two_sided`, `less` (p_reference < p_current) or `greater` |

```sql
CREATE OR REPLACE TABLE ev AS
SELECT 0 AS grp, (i % 3 = 0)::INTEGER AS b, ['a','b','c'][i % 3 + 1] AS s FROM range(50) r(i)
UNION ALL
SELECT 1, (i % 2 = 0)::INTEGER, ['a','b','c'][((i*7) % 5) % 3 + 1] FROM range(40) r(i);

SELECT (z_drift_agg(b, grp)).p_value     AS z_p,     -- 0.125
       (chisq_drift_agg(s, grp)).p_value AS chisq_p, -- 0.266
       (g_drift_agg(s, grp)).statistic   AS g        -- 2.881
FROM ev;
```

### column_drift_agg

`column_drift_agg(value DOUBLE | VARCHAR, group, [options]) -> STRUCT(test, score, is_p_value, threshold, drifted, n1, n2)`

evidently's per-column drift detection. It runs one of evidently's tests (by default
evidently's choice for the column) and compares the score with the test's threshold
using evidently's drift rule. `score` is a p-value when `is_p_value` is true, and a
distance otherwise. Numeric values are tested as numbers. VARCHAR values are categories.
The default test (`'test': 'auto'`) is:

| Reference rows | Column | Test |
|---|---|---|
| ≤ 1000 | numeric, > 5 distinct values | `ks` |
| ≤ 1000 | numeric with 3–5 distinct values, or VARCHAR with > 2 categories | `chisquare` |
| ≤ 1000 | ≤ 2 distinct values | `z` |
| > 1000 | numeric, > 5 distinct values | `wasserstein` (normed) |
| > 1000 | otherwise | `jensenshannon` |

Tests (evidently's names): `ks`, `anderson`, `cramer_von_mises`, `es`, `mannw`, `t_test`,
`ed`, `empirical_mmd`, `wasserstein`, `psi`, `kl_div`, `jensenshannon`, `hellinger`,
`chisquare`, `g_test`, `z`, `fisher_exact`, `TVD`. On VARCHAR columns only the frequency
tests and the divergences apply (`psi`, `kl_div`, `jensenshannon`, `hellinger`,
`chisquare`, `g_test`, `z`, `fisher_exact`, `TVD`). The default threshold is 0.05 for
`ks`, `es`, `mannw`, `t_test`, `chisquare` and `z`, and 0.1 for the others. The drift
rules are:

- `ks` and `cramer_von_mises`: score ≤ threshold;
- the other p-values: score < threshold;
- `ed`: score > threshold;
- the other distances: score ≥ threshold.

The scores follow the routines of the dedicated aggregates. Where scipy and R differ,
R is followed: `anderson` is the uncapped kSamples p-value, and `mannw` is exact for
n < 50 without ties. The permutation p-values of `TVD` and `empirical_mmd` are seeded.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group |
| test | auto | evidently's test name (case-sensitive), or `auto` |
| threshold | the test's default | Drift threshold |
| binning, n_bins, epsilon | as `psi_agg` | Binning and zero fill of `psi`, `kl_div`, `jensenshannon`, `hellinger` (numeric columns) |
| n_permutations | 1000 | Permutations of `TVD` / `empirical_mmd` |
| seed (alias `random_state`) | 42 | Seed of the permutations |

```sql
CREATE OR REPLACE TABLE ev AS
SELECT 0 AS grp, 2*sin(0.37*i) AS v, ['a','b','c'][i % 3 + 1] AS s FROM range(50) r(i)
UNION ALL
SELECT 1, 2*sin(0.53*i) + 0.5, ['a','b','c'][((i*7) % 5) % 3 + 1] FROM range(40) r(i);

SELECT column_drift_agg(v, grp)                   AS numeric_auto, -- ks, p = 0.0575, not drifted
       column_drift_agg(s, grp)                   AS category_auto, -- chisquare, p = 0.266
       column_drift_agg(v, grp, {'test': 'psi', 'threshold': 0.2}) AS psi
FROM ev;
```

### dataset_drift_agg, dataset_drift_by

```
dataset_drift_agg(drifted BOOLEAN, [drift_share DOUBLE]) -> STRUCT(n_columns, n_drifted, share_drifted, drift_share, dataset_drift)
dataset_drift_by(source, group_col, columns, options := NULL, drift_share := 0.5)
```

evidently's dataset drift: the dataset has drifted when the share of drifted columns is at
least `drift_share` (default 0.5, in [0, 1]). `dataset_drift_agg` aggregates per-column
flags, skipping NULL flags. No flags gives NULL.

`dataset_drift_by` runs `column_drift_agg(column, group_col, options)` on every column
named in `columns`. Numeric columns are tested as numbers, and all other types (VARCHAR,
BOOLEAN, ...) as categories. It returns one row per column, in the order of `columns`:
`column_name, test, score, is_p_value, threshold, drifted, n1, n2`, plus the dataset
summary `n_columns, n_drifted, share_drifted, drift_share, dataset_drift`. `group_col`
splits the reference rows from the current rows, as in the aggregates.

```sql
CREATE OR REPLACE TABLE ev AS
SELECT 0 AS grp, 2*sin(0.37*i) AS v, (i % 4)::INTEGER AS c,
       (i % 3 = 0)::INTEGER AS b, ['a','b','c'][i % 3 + 1] AS s
FROM range(50) r(i)
UNION ALL
SELECT 1, 2*sin(0.53*i) + 0.5, ((i*3) % 5)::INTEGER,
       (i % 2 = 0)::INTEGER, ['a','b','c'][((i*7) % 5) % 3 + 1]
FROM range(40) r(i);

-- v: ks, c: chisquare (drifted), b: z, s: chisquare -> share 0.25, no dataset drift
SELECT column_name, test, score, drifted, share_drifted, dataset_drift
FROM dataset_drift_by('ev', grp, ['v', 'c', 'b', 's']);

-- Per segment, from per-column flags
SELECT (dataset_drift_agg(d, 0.5)).dataset_drift
FROM (VALUES (true), (false), (true)) t(d);
```

### embedding_drift_agg, energy_distance_test_agg, mmd_test_agg

```
embedding_drift_agg(embedding DOUBLE[], group | is_reference BOOLEAN, [options]) -> STRUCT(method, score, threshold, drifted, n1, n2, dim)
energy_distance_test_agg(x DOUBLE[], group, [options]) -> STRUCT(statistic, p_value, n1, n2, dim, n_permutations, ...)
mmd_test_agg(x DOUBLE[], group, [options]) -> STRUCT(statistic, p_value, n1, n2, dim, n_permutations, ...)
```

Drift and two-sample tests on vectors, such as text or image embeddings: one `DOUBLE[]`
per row. Every vector in a call must have the same length (an error otherwise). Rows
with a NULL vector, a NULL group, or a NULL or NaN element are skipped. One group, or an
empty group, gives NULL. `group` is a BIGINT or VARCHAR label with two values. For
`embedding_drift_agg` the reference is the smaller label unless the `reference` option
names it, and `group` may instead be a BOOLEAN `is_reference` (TRUE rows are the
reference). For the two tests, group 1 (`n1`) is the smaller label. The seeded results
(bootstrap, permutations) do not depend on the row order.

`embedding_drift_agg` is evidently's embedding drift
(`legacy/metrics/data_drift/embedding_drift_methods.py`). `n1` is the reference size,
`n2` the current size and `dim` the vector length. The drift is `score > threshold`:

| method | score | default threshold |
|---|---|---|
| `distance` (default) | `metric` between the mean vectors: `euclidean` (default), `cosine` (1 − cosine similarity), `cityblock`, `chebyshev` | 0.2 |
| `ratio` | share of the dimensions that drift, each tested as a numeric column with `component_test` (as `column_drift_agg`) | 0.2 |
| `mmd` | unbiased MMD² with the Gaussian kernel `exp(-‖a − b‖² / s²)`, `s` the median distance between reference and current rows (floored at 0) | 0.015 |

With `'bootstrap': true` (`distance`, `mmd`) the threshold is the `quantile` of the scores
between two with-replacement resamples of the reference, of sizes `m²/(m + n)` and
`mn/(m + n)`, over `n_bootstrap` resamples. evidently's PCA reduction (`pca_components`)
and its classifier method (`model`) are not part of this function.

| Key | Default | Description |
|---|---|---|
| reference | smaller label | Reference group (not with a BOOLEAN `is_reference`) |
| method | `distance` | `distance`, `ratio` or `mmd` |
| metric (alias `dist`) | `euclidean` | `distance` only |
| threshold | per method, see above | Fixed threshold (not with `bootstrap`) |
| bootstrap | false | Bootstrapped threshold (`distance`, `mmd`) |
| n_bootstrap (alias `n_boot`) | 100 | Bootstrap resamples |
| quantile (alias `quantile_probability`) | 0.95 (`distance`), 0.05 (`mmd`) | Quantile of the bootstrap scores, in [0, 1] |
| component_test (alias `drift_method`) | `wasserstein` | `ratio` only: evidently's test name, as in `column_drift_agg` |
| component_threshold (alias `drift_threshold`) | the test's default | `ratio` only |
| seed (alias `random_state`) | 42 | Seed of the bootstrap and of permutation component tests |

`energy_distance_test_agg` is the multivariate energy distance test (Székely and Rizzo).
The statistic is `2 E‖X − Y‖ − E‖X − X'‖ − E‖Y − Y'‖`, with the within-sample means taken
over distinct pairs. `energy::eqdist.etest` reports `n1 n2 / (n1 + n2)` times the
version with the means over all pairs, so its value differs by that scale and by the
zero diagonal. `mmd_test_agg` is the unbiased MMD² (Gretton et al.). Both p-values are
seeded permutation p-values `(1 + #{permuted ≥ observed}) / (1 + n_permutations)`. Both
also carry the unified test fields. The cost is O(n_permutations · N² · dim) for N rows.

| Key | Default | Description |
|---|---|---|
| n_permutations (alias `permutations`) | 1000 | Permutations |
| seed (alias `random_state`) | 42 | Seed of the permutations |
| kernel | `gaussian` | `mmd_test_agg` only: `gaussian` `exp(-‖a − b‖² / (2 h²))`, `laplacian` `exp(-‖a − b‖₁ / h)`, `linear` `a·b`, `polynomial` `(scale a·b + offset)^degree` |
| bandwidth (alias `sigma`) | 1 | `h` of the gaussian / laplacian kernel. It is not chosen from the data, so set it to the scale of the distances |
| degree, scale, offset | 2, 1, 1 | Polynomial kernel |

```sql
CREATE OR REPLACE TABLE emb AS
SELECT 0 AS grp, [sin(0.7*i), cos(1.3*i), 0.1*(i % 5)] AS e FROM range(30) r(i)
UNION ALL
SELECT 1, [sin(0.9*i) + 0.4, cos(1.1*i) - 0.2, 0.1*(i % 7) + 0.1] FROM range(25) r(i);

SELECT embedding_drift_agg(e, grp)                                  AS distance, -- 0.497 > 0.2: drifted
       embedding_drift_agg(e, grp = 0, {'metric': 'cosine'})        AS cosine,   -- is_reference
       embedding_drift_agg(e, grp, {'method': 'ratio'})             AS ratio,    -- 3 of 3 dimensions
       embedding_drift_agg(e, grp, {'method': 'mmd', 'bootstrap': true}) AS mmd
FROM emb;

SELECT (energy_distance_test_agg(e, grp, {'n_permutations': 999})).p_value AS energy_p,
       (mmd_test_agg(e, grp, {'kernel': 'gaussian', 'bandwidth': 1.0})).p_value AS mmd_p
FROM emb;
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

### spa_test_agg

Hansen's (2005) test for Superior Predictive Ability: does any of K competing
models have a lower expected loss than the benchmark? H0: no competitor beats the
benchmark. The arguments are **losses** (for example squared errors), one row per
time point; `model_losses` holds the K competitors' losses of that time point.
The stationary bootstrap resamples the series, so the aggregate needs
`ORDER BY t`.

**Signature:**
```text
spa_test_agg(benchmark_loss DOUBLE, model_losses DOUBLE[] ORDER BY t, [options]) -> STRUCT
```

| Key | Default | Description |
|---|---|---|
| n_bootstrap (aliases `n_boot`, `bootstrap`) | 1000 | Bootstrap samples |
| block_length (alias `block_size`) | T^(1/3) | Expected block length of the stationary bootstrap; 0 or less selects the default |
| seed (alias `random_state`) | 42 | Seed of the bootstrap |

The result has `statistic`, `p_value` (= `p_value_consistent`), `p_value_lower`,
`p_value_consistent`, `p_value_upper`, `best_model` (1-based position in
`model_losses` of the model with the largest standardized differential), `n_models`,
`n`, `n_bootstrap`, `block_length`, `method`, `alternative` (`greater`) and the
unified test fields. Rows with a NULL or NaN benchmark, a NULL list or a NULL/NaN
element are skipped; every list must have the same length (otherwise an error).
Fewer than 3 complete rows give NULL.

With `d_k = L_benchmark − L_k` the statistic is Hansen's
`T_SPA = max(0, max_k sqrt(T) mean(d_k) / ω_k)`, `ω_k²` the stationary-bootstrap
long-run variance of `d_k`. Each bootstrap replicate draws one set of time indices
and applies it to all models jointly. The three p-values differ in how the models
are re-centred under H0: `p_value_lower` (only models that beat the benchmark),
`p_value_consistent` (Hansen's recommended choice: models that are not significantly
worse) and `p_value_upper` (every model, White's Reality Check), so
`p_value_lower <= p_value_consistent <= p_value_upper`. They match a base-R
implementation of Hansen (2005) within Monte Carlo error.

The p-values are `P(T* >= T_SPA)` over the bootstrap replicates. When no
competitor has a lower average loss than the benchmark, `statistic` is 0 and all
three p-values are 1: the test does not reject.

```sql
CREATE TABLE fc_losses AS
SELECT t,
       2.0 + 0.5 * abs(sin(t * 0.7))       AS naive_loss,
       1.0 + 0.3 * abs(sin(t * 0.7 + 1.0)) AS ets_loss,
       3.0 + 0.4 * abs(cos(t * 0.3))       AS arima_loss
FROM range(1, 61) r(t);

SELECT r.statistic, r.p_value, r.p_value_upper, r.best_model
FROM (SELECT spa_test_agg(naive_loss, [ets_loss, arima_loss],
                          {'n_bootstrap': 999, 'block_length': 4} ORDER BY t) AS r
      FROM fc_losses);
```

### mspe_adjusted_spa_agg

The MSPE-adjusted SPA test (Clark and McCracken, 2012) for a benchmark that is
nested in each alternative: the SPA bootstrap on the Clark-West adjusted
differentials `(e_b² − e_k²) + (e_b − e_k)²`. The arguments are **forecast errors**
(actual − forecast), not losses. Options and result are those of `spa_test_agg`.

**Signature:**
```text
mspe_adjusted_spa_agg(benchmark_error DOUBLE, model_errors DOUBLE[] ORDER BY t, [options]) -> STRUCT
```

```sql
SELECT (mspe_adjusted_spa_agg(sin(t / 4.0) + 0.2 * cos(t * 1.3),
                              [0.1 * sin(t / 4.0) + 0.2 * cos(t * 1.3)],
                              {'n_bootstrap': 499} ORDER BY t)).p_value
FROM range(1, 101) r(t);
```

### model_confidence_set_agg

Hansen, Lunde and Nason's (2011) Model Confidence Set: the set of models that
contains the best model with confidence 1 − alpha. Starting from all K models, the
worst model is eliminated while the bootstrap test of equal predictive ability
rejects at level alpha.

**Signature:**
```text
model_confidence_set_agg(model_losses DOUBLE[] ORDER BY t, [options]) -> STRUCT
```

| Key | Default | Description |
|---|---|---|
| alpha | 0.1 | Elimination level, in (0, 1) |
| statistic | `range` | `range` (T_R = max \|t_ij\|) or `max` (T_max) |
| method (alias `bootstrap_method`) | `stationary` | Bootstrap scheme: `stationary`, `moving_block`, `circular_block` or `iid` |
| n_bootstrap (aliases `n_boot`, `bootstrap`) | 1000 | Bootstrap samples |
| block_length (alias `block_size`) | T^(1/3) | Expected block length of `stationary`; the block length of `moving_block` / `circular_block` (a whole number; default `round(T^(1/3))`); ignored by `iid` |
| seed (alias `random_state`) | 42 | Seed of the bootstrap |

The default is the stationary bootstrap with expected block length `T^(1/3)`, the
scheme of the SPA aggregates. `{'method': 'moving_block', 'block_length': k}`
reproduces R's `MCS::MCSprocedure(k = k)` (`k = 1` is its IID bootstrap); note that
R's default `alpha` is 0.15.

The result has `included_models` and `eliminated_models` (1-based positions in
`model_losses`; eliminated in elimination order), `mcs_p_value` (the MCS p-value at
which elimination stopped; 1 for a single model), `mcs_p_values` (the MCS p-value of
every model, indexed by position: R's `MCS p-Value` column, 1 for the best model; a
model is in the set iff its value is >= `alpha`), `elimination_order` (all models,
worst first and the best last: R's `elimination.order`), `elimination_steps`
(`STRUCT(model, p_value, mcs_p_value, eliminated)[]`, the steps up to the first
retained model; `p_value` is the step's test, `mcs_p_value` the running maximum),
`statistic_type`, `alpha`, `n_models`, `n`, `n_bootstrap`, `block_length` (the one
used), `bootstrap_method` and `method`. NULL handling is that of `spa_test_agg`.

Each replicate draws one set of time indices for all models, and the contrasts are
standardized with bootstrap variances, as `MCSprocedure`. The result is
reproducible for a fixed seed, with any number of models.

```sql
SELECT r.included_models, r.eliminated_models, r.mcs_p_values, r.elimination_order
FROM (SELECT model_confidence_set_agg([naive_loss, ets_loss, arima_loss],
                                      {'alpha': 0.1, 'n_bootstrap': 999} ORDER BY t) AS r
      FROM fc_losses);

-- As R MCS::MCSprocedure(L, alpha = 0.1, statistic = "Tmax", k = 3)
SELECT (model_confidence_set_agg([naive_loss, ets_loss, arima_loss],
                                 {'statistic': 'max', 'method': 'moving_block', 'block_length': 3,
                                  'n_bootstrap': 2000} ORDER BY t)).mcs_p_values
FROM fc_losses;
```

Per series, group as usual:

```sql
SELECT series, (model_confidence_set_agg([naive_loss, ets_loss] ORDER BY t)).included_models
FROM (SELECT t % 2 AS series, * FROM fc_losses)
GROUP BY series ORDER BY series;
```

The SPA and MCS p-values come from the bootstrap: their digits depend on the
random stream, so they agree with other implementations (R's `MCS` package, a
base-R Hansen SPA) within Monte Carlo error (about 0.005 at 20000 replicates), not
to the last digit. The SPA statistic and the MCS elimination order are compared
exactly.

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

### Studentized range distribution: ptukey, qtukey

```text
ptukey(q DOUBLE, n_means DOUBLE, df DOUBLE [, n_ranges DOUBLE [, lower_tail BOOLEAN]]) -> DOUBLE
qtukey(p DOUBLE, n_means DOUBLE, df DOUBLE [, n_ranges DOUBLE [, lower_tail BOOLEAN]]) -> DOUBLE
```

The distribution and quantile functions of the studentized range statistic, ports of
R's `ptukey` / `qtukey` (`n_ranges` default 1, `lower_tail` default true) (#304). They
give Tukey HSD p-values and critical values for given group counts and residual
degrees of freedom. As R, `qtukey` stops its secant iteration once successive iterates
differ by less than 1e-4, so it matches R's `qtukey` rather than the exact quantile.

A NULL argument gives NULL and a NaN argument NaN. `n_means < 2`, `df < 2`,
`n_ranges < 1` or `p` outside [0, 1] is an error (R returns NaN with a warning).
`ptukey(q <= 0)` is 0, `ptukey(inf)` is 1, `qtukey(0)` is 0 and `qtukey(1)` is `inf`
(lower tail).

```sql
SELECT ptukey(3.5, 3, 12) AS p_lower,
       ptukey(3.5, 3, 12, 1, false) AS p_upper,
       qtukey(0.95, 3, 12) AS q_crit;
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
| `p_adjust` | `(p DOUBLE [, method VARCHAR [, n BIGINT]]) OVER (...)` | `DOUBLE`, the row's adjusted p-value; window-only, **DuckDB 2.0+** (see below) |

`method` and `n` of `p_adjust_agg` and `p_adjust` must be constants; `p_adjust_list`
also takes per-row values.

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

### Window function `p_adjust` (DuckDB 2.0+)

On DuckDB 2.0, `p_adjust(p [, method [, n]]) OVER (...)` returns each row's adjusted
p-value inline (#224), the same values as `p_adjust_by`:

```sql skip
-- DuckDB 2.0+ only: one family per endpoint, Benjamini-Hochberg.
SELECT endpoint, test, p_value,
       p_adjust(p_value, 'BH') OVER (PARTITION BY endpoint) AS p_adjusted
FROM results
ORDER BY endpoint, test;
```

The family of a row is its window frame. Leave `ORDER BY` out of `OVER` (or use
`ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`) so that the family is the
whole partition; with `ORDER BY` the default frame ends at the current row and the
family grows row by row. A row outside its own frame (e.g. `EXCLUDE CURRENT ROW`) gets
NULL, and rows removed by `FILTER (WHERE ...)` are NULL and not counted. `p_adjust` is
window-only: without `OVER` it is a binder error. Each distinct frame is adjusted once.

On DuckDB 1.5, `p_adjust` does not exist: an extension's window aggregate there is not
told which row of a whole-partition frame it is computing, so it cannot return a
different value per row. Use `p_adjust_by`, or the recipe below for Bonferroni, Holm
and BH.

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
