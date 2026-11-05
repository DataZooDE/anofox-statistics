# Correlation Functions

Correlation coefficients and tests for measuring relationships between variables.

The correlation aggregates sort their (x, y) pairs before computing, so the same rows
give bitwise-identical results whatever the thread count or row order (see
[Methodology](../../METHODOLOGY.md#reproducibility-across-threads)).

## Pearson Correlation

### pearson_agg

Pearson product-moment correlation with significance test.

**Signature:**
```text
pearson_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |

**Returns:**
```
STRUCT(
    r DOUBLE,             -- Correlation coefficient (-1 to 1)
    statistic DOUBLE,     -- t-statistic
    p_value DOUBLE,       -- p-value (test r ≠ 0)
    ci_lower DOUBLE,      -- CI lower bound (Fisher z-transformed)
    ci_upper DOUBLE,      -- CI upper bound
    n BIGINT,             -- Sample size
    method VARCHAR        -- "Pearson"
)
```

**Example:**
```sql
CREATE OR REPLACE TABLE measurements AS
SELECT 'R' || (i % 3) AS region,
       (150 + i % 40)::DOUBLE AS height,
       (50 + 0.6 * (i % 40) + (i * 7) % 9)::DOUBLE AS weight
FROM range(120) r(i);

-- Test correlation between two variables
SELECT unnest(pearson_agg(height, weight)) FROM measurements;

-- Per-group correlation with 99% CI
SELECT region, unnest(pearson_agg(height, weight, {'confidence_level': 0.99}))
FROM measurements
GROUP BY region
ORDER BY region;
```

**Interpretation:**
- r = 1: Perfect positive correlation
- r = 0: No linear relationship
- r = -1: Perfect negative correlation

## Spearman Correlation

### spearman_agg

Spearman rank correlation. Robust to outliers and non-linear relationships.

**Signature:**
```text
spearman_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |

**Returns:** Same structure as pearson_agg with method "Spearman"

**Example:**
```sql
-- Rank correlation for ordinal data
SELECT unnest(spearman_agg(height, weight)) FROM measurements;
```

**Use Cases:**
- Ordinal data
- Non-linear monotonic relationships
- Data with outliers

## Kendall Correlation

### kendall_agg

Kendall tau correlation. Based on concordant/discordant pairs.

**Signature:**
```text
kendall_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |

**Example:**
```sql
SELECT unnest(kendall_agg(height, weight)) FROM measurements;

-- Tau-a instead of the default tau-b
SELECT (kendall_agg(height, weight, {'variant': 'tau_a'})).tau FROM measurements;
```

**Comparison with Spearman:**
- More robust to ties
- More interpretable (probability of concordance)
- Generally smaller magnitude than Spearman

## Distance Correlation

### distance_cor_agg

Distance correlation. Measures both linear and non-linear dependence.

**Signature:**
```text
distance_cor_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_permutations | INTEGER | 1000 | Number of permutations of the test |
| seed (alias `random_state`) | INTEGER | 42 | Seed of the permutation draw; without it the default 42 makes the p-value reproducible |

**Returns:**
```
STRUCT(
    dcor DOUBLE,          -- Distance correlation (0 to 1)
    statistic DOUBLE,     -- Test statistic
    p_value DOUBLE,       -- p-value (permutation test)
    n BIGINT,             -- Sample size
    method VARCHAR        -- e.g. "Distance correlation test (1000 permutations)"
)
```

**Example:**
```sql
-- Detect non-linear relationships: y = x^2 has zero Pearson correlation
SELECT unnest(distance_cor_agg(x, x * x))
FROM (SELECT (i - 20)::DOUBLE AS x FROM range(41) r(i)) nonlinear_data;
```

**Key Properties:**
- dcor = 0 if and only if independent (for continuous distributions)
- Detects non-linear relationships unlike Pearson
- Always positive (0 to 1)

## Intraclass Correlation

### icc_agg

Intraclass Correlation Coefficient for reliability/agreement.

**Signature:**
```text
icc_agg(value DOUBLE, subject_id BIGINT, rater_id BIGINT [, options MAP]) -> STRUCT
```

**Options** (pass as a `MAP {...}` literal):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| type | VARCHAR | 'icc1' | Shrout & Fleiss (1979) type, named as in `psych::ICC` (case-insensitive): `'icc1'`, `'icc2'`, `'icc3'`, `'icc1k'`, `'icc2k'`, `'icc3k'`. `'single'` and `'average'` are aliases of `'icc1'` and `'icc1k'` |

| Type | Model | Definition | Unit |
|------|-------|------------|------|
| `icc1` | one-way random | absolute agreement | single rater |
| `icc2` | two-way random | absolute agreement | single rater |
| `icc3` | two-way mixed | consistency | single rater |
| `icc1k` | one-way random | absolute agreement | mean of k raters |
| `icc2k` | two-way random | absolute agreement | mean of k raters |
| `icc3k` | two-way mixed | consistency | mean of k raters |

Use `icc1` when each subject is rated by a different random set of raters;
`icc2` when the same raters rate every subject and they are a sample of a larger
population of raters (systematic rater differences count against agreement);
`icc3` when those raters are the only raters of interest (only consistency
matters). The `k` variants give the reliability of the mean of the k ratings.

The `f_statistic` / `p_value` test H0: ICC = 0, with F = MS_R / MS_W for the
one-way types and F = MS_R / MS_E for the two-way types (as `psych::ICC`).
The 95% interval (`ci_lower`, `ci_upper`) matches `psych::ICC` for `icc1`,
`icc3`, `icc1k` and `icc3k`. For `icc2` and `icc2k` it is `NULL`: the upstream
library does not yet compute the Shrout & Fleiss approximate-F interval for the
absolute-agreement two-way types.

Every subject must be rated by every rater; incomplete data returns `NULL`.

**Returns:**
```
STRUCT(
    icc DOUBLE,           -- ICC value
    f_statistic DOUBLE,   -- F-statistic
    ci_lower DOUBLE,      -- CI lower bound
    ci_upper DOUBLE,      -- CI upper bound
    n_subjects BIGINT,    -- Number of subjects
    n_raters BIGINT,      -- Number of raters
    method VARCHAR,       -- "ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k" or "ICC3k"
    ...                   -- statistic (= F), p_value, n and the unified test fields
)
```

**Example:**
```sql
-- Inter-rater reliability: 3 raters scoring 10 patients
SELECT unnest(icc_agg(score, patient_id, rater_id))
FROM (SELECT p AS patient_id, r AS rater_id,
             (p * 2 + r + (p * r) % 3)::DOUBLE AS score
      FROM range(10) a(p), range(3) b(r)) ratings;

-- All six types side by side: the same 3 raters score every patient, and
-- rater 2 scores systematically higher, so ICC2 (agreement) < ICC3 (consistency)
SELECT round((icc_agg(score, patient_id, rater_id, {'type': 'icc1'})).icc, 3) AS icc1,
       round((icc_agg(score, patient_id, rater_id, {'type': 'icc2'})).icc, 3) AS icc2,
       round((icc_agg(score, patient_id, rater_id, {'type': 'icc3'})).icc, 3) AS icc3,
       round((icc_agg(score, patient_id, rater_id, {'type': 'icc1k'})).icc, 3) AS icc1k,
       round((icc_agg(score, patient_id, rater_id, {'type': 'icc2k'})).icc, 3) AS icc2k,
       round((icc_agg(score, patient_id, rater_id, {'type': 'icc3k'})).icc, 3) AS icc3k
FROM (SELECT p AS patient_id, r AS rater_id,
             (p * 2 + 3 * (r = 2)::INT + (p * r) % 3)::DOUBLE AS score
      FROM range(10) a(p), range(3) b(r)) ratings;
```

**Interpretation:**
- ICC < 0.5: Poor reliability
- ICC 0.5-0.75: Moderate reliability
- ICC 0.75-0.9: Good reliability
- ICC > 0.9: Excellent reliability

## Choosing a Correlation Method

| Scenario | Recommended |
|----------|-------------|
| Linear relationship, normal data | Pearson |
| Ordinal data or outliers | Spearman |
| Many ties, small samples | Kendall |
| Non-linear relationships | Distance correlation |
| Inter-rater reliability | ICC |

## See Also

- [Hypothesis Tests](hypothesis.md) - Statistical tests
- [Categorical](categorical.md) - Association measures for categorical data
