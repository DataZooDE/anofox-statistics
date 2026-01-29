# Categorical Tests

Statistical tests and association measures for categorical data.

## Independence Tests

### chisq_test_agg

Chi-square test of independence for categorical variables.

**Signature:**
```sql
chisq_test_agg(row_var INTEGER, col_var INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| correction | BOOLEAN | false | Apply Yates' continuity correction |

**Returns:**
```
STRUCT(
    statistic DOUBLE,    -- Chi-square statistic
    p_value DOUBLE,      -- p-value
    df BIGINT,           -- Degrees of freedom
    method VARCHAR       -- "Chi-Square"
)
```

**Example:**
```sql
-- Test independence of two categorical variables
SELECT (chisq_test_agg(gender, preference)).*
FROM survey;

-- With Yates correction for 2x2 tables
SELECT chisq_test_agg(group, outcome, {'correction': true})
FROM clinical_data;
```

### g_test_agg

G-test (log-likelihood ratio test) for contingency tables.

**Signature:**
```sql
g_test_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
```
STRUCT(
    statistic DOUBLE,    -- G statistic
    p_value DOUBLE,      -- p-value
    df BIGINT,           -- Degrees of freedom
    method VARCHAR       -- "G-test"
)
```

### fisher_exact_agg

Fisher's exact test for 2x2 contingency tables. Exact test for small samples.

**Signature:**
```sql
fisher_exact_agg(row_var INTEGER, col_var INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |

**Returns:**
```
STRUCT(
    odds_ratio DOUBLE,   -- Odds ratio
    p_value DOUBLE,      -- p-value
    ci_lower DOUBLE,     -- CI lower bound
    ci_upper DOUBLE,     -- CI upper bound
    method VARCHAR       -- "Fisher's Exact Test"
)
```

**Example:**
```sql
-- Fisher's exact test for small samples
SELECT (fisher_exact_agg(treatment, outcome)).*
FROM small_study;
```

## Goodness of Fit

### chisq_gof_agg

Chi-square goodness of fit test. Tests whether observed frequencies match expected.

**Signature:**
```sql
chisq_gof_agg(observed INTEGER, expected DOUBLE) -> STRUCT
```

**Returns:**
```
STRUCT(
    statistic DOUBLE,    -- Chi-square statistic
    p_value DOUBLE,      -- p-value
    df BIGINT,           -- Degrees of freedom
    method VARCHAR       -- "Chi-Square Goodness of Fit"
)
```

**Example:**
```sql
-- Test if observed frequencies match expected
SELECT (chisq_gof_agg(observed_count, expected_count)).*
FROM frequency_data;
```

## Paired Data

### mcnemar_agg

McNemar's test for paired nominal data. Tests marginal homogeneity in 2x2 tables.

**Signature:**
```sql
mcnemar_agg(var1 INTEGER, var2 INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| correction | BOOLEAN | true | Apply continuity correction |

**Example:**
```sql
-- Before/after comparison
SELECT (mcnemar_agg(before_treatment, after_treatment)).*
FROM paired_study;
```

## Effect Size Measures

### cramers_v_agg

Cramér's V - effect size for chi-square tests (0 to 1).

**Signature:**
```sql
cramers_v_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

**Returns:**
```
STRUCT(
    v DOUBLE,            -- Cramér's V (0 to 1)
    chi_sq DOUBLE,       -- Chi-square statistic
    df BIGINT,           -- Degrees of freedom
    n BIGINT,            -- Sample size
    method VARCHAR       -- "Cramér's V"
)
```

**Interpretation:**
- V = 0.1: Small effect
- V = 0.3: Medium effect
- V = 0.5: Large effect

### phi_coefficient_agg

Phi coefficient for 2x2 tables (-1 to 1).

**Signature:**
```sql
phi_coefficient_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

### contingency_coef_agg

Contingency coefficient (Pearson's C).

**Signature:**
```sql
contingency_coef_agg(row_var INTEGER, col_var INTEGER) -> STRUCT
```

### cohen_kappa_agg

Cohen's kappa for inter-rater agreement.

**Signature:**
```sql
cohen_kappa_agg(rater1 INTEGER, rater2 INTEGER) -> STRUCT
```

**Returns:**
```
STRUCT(
    kappa DOUBLE,        -- Kappa coefficient
    se DOUBLE,           -- Standard error
    ci_lower DOUBLE,     -- CI lower bound
    ci_upper DOUBLE,     -- CI upper bound
    z DOUBLE,            -- Z-statistic
    p_value DOUBLE,      -- p-value
    method VARCHAR       -- "Cohen's Kappa"
)
```

**Interpretation:**
- κ < 0: Less than chance agreement
- κ = 0: Agreement equals chance
- κ = 0.01-0.20: Slight agreement
- κ = 0.21-0.40: Fair agreement
- κ = 0.41-0.60: Moderate agreement
- κ = 0.61-0.80: Substantial agreement
- κ = 0.81-1.00: Almost perfect agreement

## Proportion Tests

### prop_test_one_agg

One-sample proportion test.

**Signature:**
```sql
prop_test_one_agg(success INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| p0 | DOUBLE | 0.5 | Null hypothesis proportion |
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |

### prop_test_two_agg

Two-sample proportion test.

**Signature:**
```sql
prop_test_two_agg(success INTEGER, group_id INTEGER, [options MAP]) -> STRUCT
```

### binom_test_agg

Exact binomial test.

**Signature:**
```sql
binom_test_agg(success INTEGER, [options MAP]) -> STRUCT
```

## Choosing a Test

| Scenario | Recommended |
|----------|-------------|
| 2x2 table, large sample | Chi-square |
| 2x2 table, small sample | Fisher's exact |
| Larger tables | Chi-square or G-test |
| Paired nominal data | McNemar's |
| Inter-rater agreement | Cohen's kappa |
| Effect size needed | Cramér's V |

## See Also

- [Hypothesis Tests](hypothesis.md) - Tests for continuous data
- [Correlation](correlation.md) - Correlation measures
