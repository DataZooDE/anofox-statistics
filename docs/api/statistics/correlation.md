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

## Partial and Semi-Partial Correlation

### partial_cor_agg

Partial correlation of `x` and `y` controlling for one or more variables (the
controls are removed from both `x` and `y`), with its t test. Matches R
`ppcor::pcor.test`.

**Signature:**
```text
partial_cor_agg(x DOUBLE, y DOUBLE, controls DOUBLE[] [, options MAP]) -> STRUCT
```

### semi_partial_cor_agg

Semi-partial (part) correlation: the correlation of `x` with `y` after the
control is removed from `y` only. Matches R `ppcor::spcor.test(x, y, z)`.
Takes at most one control variable.

**Signature:**
```text
semi_partial_cor_agg(x DOUBLE, y DOUBLE, controls DOUBLE[] [, options MAP]) -> STRUCT
```

**Options** (both functions):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| method | VARCHAR | 'pearson' | `'pearson'`, or `'spearman'` (every column is ranked first, average ties, as ppcor does) |

**Returns:**
```
STRUCT(
    r DOUBLE,             -- Partial (semi-partial) correlation coefficient
    statistic DOUBLE,     -- t = r * sqrt(df / (1 - r^2))
    p_value DOUBLE,       -- Two-sided p-value
    df DOUBLE,            -- n - 2 - n_controls
    n BIGINT,             -- Complete rows used
    n_controls BIGINT,    -- Number of control variables
    method VARCHAR,       -- e.g. "Partial correlation (controlling for 2 variable(s))"
    alternative VARCHAR   -- "two_sided"
    -- plus the unified test fields (estimate, df1, effect_size, ...)
)
```

A row with a NULL or NaN in `x`, `y`, the control list or one of its elements is
skipped. Every row must have the same number of controls. Fewer than
`n_controls + 3` complete rows, or a constant column, returns `NULL`. An empty
control list (`[]::DOUBLE[]`) gives the plain Pearson (or Spearman) correlation.
There is no confidence interval (ppcor reports none either).

**Example:**
```sql
CREATE OR REPLACE TABLE growth AS
SELECT i,
       (i % 12)::DOUBLE AS age,
       (100 + 5 * (i % 12) + (i * 7) % 11)::DOUBLE AS height,
       (20 + 2 * (i % 12) + (i * 5) % 7 + 0.1 * ((i * 7) % 11))::DOUBLE AS weight,
       ((i * 3) % 5)::DOUBLE AS activity
FROM range(60) r(i);

-- Height and weight both grow with age: the partial correlation removes it
SELECT (pearson_agg(height, weight)).r AS r_raw,
       (partial_cor_agg(height, weight, [age])).r AS r_partial
FROM growth;

-- Two controls, rank-based
SELECT unnest(partial_cor_agg(height, weight, [age, activity], {'method': 'spearman'}))
FROM growth;

-- Semi-partial: height vs the part of weight not explained by age
SELECT unnest(semi_partial_cor_agg(height, weight, [age])) FROM growth;
```

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
| type | VARCHAR | 'single' | `'single'` (single-rater ICC) or `'average'` (average of k raters) |

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
    method VARCHAR        -- e.g. "ICC1"
)
```

**Example:**
```sql
-- Inter-rater reliability: 3 raters scoring 10 patients
SELECT unnest(icc_agg(score, patient_id, rater_id))
FROM (SELECT p AS patient_id, r AS rater_id,
             (p * 2 + r + (p * r) % 3)::DOUBLE AS score
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
| Association net of other variables | Partial / semi-partial correlation |
| Non-linear relationships | Distance correlation |
| Inter-rater reliability | ICC |

## See Also

- [Hypothesis Tests](hypothesis.md) - Statistical tests
- [Categorical](categorical.md) - Association measures for categorical data
