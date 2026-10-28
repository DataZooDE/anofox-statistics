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
- VARCHAR values always get one bin per category.

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
