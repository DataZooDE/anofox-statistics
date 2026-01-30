# Correlation Functions

Correlation coefficients and tests for measuring relationships between variables.

## Pearson Correlation

### pearson_agg

Pearson product-moment correlation with significance test.

**Signature:**
```sql
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
-- Test correlation between two variables
SELECT (pearson_agg(height, weight)).*
FROM measurements;

-- Per-group correlation with 99% CI
SELECT
    region,
    (pearson_agg(income, spending, {'confidence_level': 0.99})).*
FROM economic_data
GROUP BY region;
```

**Interpretation:**
- r = 1: Perfect positive correlation
- r = 0: No linear relationship
- r = -1: Perfect negative correlation

## Spearman Correlation

### spearman_agg

Spearman rank correlation. Robust to outliers and non-linear relationships.

**Signature:**
```sql
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
SELECT (spearman_agg(rank_x, rank_y)).*
FROM ranked_data;
```

**Use Cases:**
- Ordinal data
- Non-linear monotonic relationships
- Data with outliers

## Kendall Correlation

### kendall_agg

Kendall tau correlation. Based on concordant/discordant pairs.

**Signature:**
```sql
kendall_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| confidence_level | DOUBLE | 0.95 | Confidence level for CI |

**Example:**
```sql
SELECT (kendall_agg(x, y)).*
FROM data;
```

**Comparison with Spearman:**
- More robust to ties
- More interpretable (probability of concordance)
- Generally smaller magnitude than Spearman

## Distance Correlation

### distance_cor_agg

Distance correlation. Measures both linear and non-linear dependence.

**Signature:**
```sql
distance_cor_agg(x DOUBLE, y DOUBLE) -> STRUCT
```

**Returns:**
```
STRUCT(
    dcor DOUBLE,          -- Distance correlation (0 to 1)
    dcov DOUBLE,          -- Distance covariance
    p_value DOUBLE,       -- p-value (permutation test)
    n BIGINT,             -- Sample size
    method VARCHAR        -- "Distance Correlation"
)
```

**Example:**
```sql
-- Detect non-linear relationships
SELECT (distance_cor_agg(x, y)).*
FROM nonlinear_data;
```

**Key Properties:**
- dcor = 0 if and only if independent (for continuous distributions)
- Detects non-linear relationships unlike Pearson
- Always positive (0 to 1)

## Intraclass Correlation

### icc_agg

Intraclass Correlation Coefficient for reliability/agreement.

**Signature:**
```sql
icc_agg(value DOUBLE, subject_id INTEGER, rater_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| type | VARCHAR | 'icc2' | ICC type: 'icc1', 'icc2', 'icc3' |
| definition | VARCHAR | 'single' | 'single' or 'average' |

**Returns:**
```
STRUCT(
    icc DOUBLE,           -- ICC value
    ci_lower DOUBLE,      -- CI lower bound
    ci_upper DOUBLE,      -- CI upper bound
    f_value DOUBLE,       -- F-statistic
    p_value DOUBLE,       -- p-value
    n BIGINT,             -- Number of subjects
    k BIGINT,             -- Number of raters
    method VARCHAR        -- "ICC"
)
```

**Example:**
```sql
-- Inter-rater reliability
SELECT (icc_agg(score, patient_id, rater_id)).*
FROM ratings;
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
