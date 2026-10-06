# Categorical Tests

Statistical tests and association measures for categorical data.

## Independence Tests

### chisq_test_agg

Chi-square test of independence for categorical variables.

**Signature:**
```text
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
-- Survey: gender (0/1) vs preferred product (0/1/2)
CREATE OR REPLACE TABLE survey AS
SELECT (i % 2)::INTEGER AS gender,
       (CASE WHEN i % 2 = 0 THEN (i * 7) % 3 ELSE least(2, (i * 7) % 4) END)::INTEGER AS preference
FROM range(120) r(i);

-- Test independence of two categorical variables
SELECT unnest(chisq_test_agg(gender, preference)) FROM survey;

-- With Yates correction for a 2x2 table
SELECT chisq_test_agg(arm, outcome, {'correction': true}) AS result
FROM (VALUES (0, 1), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (0, 1),
             (1, 0), (1, 0), (1, 1), (1, 0), (1, 0), (1, 0), (1, 1), (1, 0)) t(arm, outcome);
```

### g_test_agg

G-test (log-likelihood ratio test) for contingency tables.

**Signature:**
```text
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
```text
fisher_exact_agg(row_var INTEGER, col_var INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |

**Returns:**
```
STRUCT(
    statistic DOUBLE,    -- Test statistic (the sample odds ratio)
    p_value DOUBLE,      -- p-value
    odds_ratio DOUBLE,   -- Odds ratio
    ci_lower DOUBLE,     -- CI lower bound for the odds ratio
    ci_upper DOUBLE,     -- CI upper bound for the odds ratio
    n BIGINT,            -- Sample size
    method VARCHAR       -- "Fisher's exact test"
)
```

**Example:**
```sql
-- Fisher's exact test for small samples
SELECT unnest(fisher_exact_agg(treatment, outcome))
FROM (VALUES (1, 1), (1, 1), (1, 1), (1, 0), (1, 1),
             (0, 0), (0, 0), (0, 1), (0, 0), (0, 0)) small_study(treatment, outcome);
```

## Goodness of Fit

### chisq_gof_agg

Chi-square goodness of fit test. Tests whether observed frequencies match
expected proportions. One row per category: the observed count and the
expected **probability** of that category (the probabilities must sum to 1;
otherwise the result is `NULL`).

**Signature:**
```text
chisq_gof_agg(observed BIGINT, expected_prob DOUBLE) -> STRUCT
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
-- Is a four-sided die fair?
SELECT unnest(chisq_gof_agg(observed_count, expected_prob))
FROM (VALUES (18, 0.25), (22, 0.25), (29, 0.25), (31, 0.25))
     frequency_data(observed_count, expected_prob);
```

## Paired Data

### mcnemar_agg

McNemar's test for paired nominal data. Tests marginal homogeneity in 2x2 tables.

**Signature:**
```text
mcnemar_agg(var1 BIGINT, var2 BIGINT [, options MAP]) -> STRUCT
```

Returns `STRUCT(statistic DOUBLE, p_value DOUBLE, df BIGINT, method VARCHAR)`.

**Options** (pass as a `MAP {...}` literal):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| correction | BOOLEAN | true | Apply continuity correction |

**Example:**
```sql
-- Before/after comparison (1 = symptom present)
SELECT unnest(mcnemar_agg(before_treatment, after_treatment))
FROM (SELECT CASE WHEN i < 30 THEN 1 ELSE 0 END AS before_treatment,
             CASE WHEN i < 10 OR i BETWEEN 30 AND 34 THEN 1 ELSE 0 END AS after_treatment
      FROM range(60) r(i)) paired_study;
```

## Effect Size Measures

### cramers_v_agg

Cramér's V - effect size for chi-square tests (0 to 1).

**Signature:**
```text
cramers_v_agg(row_var BIGINT, col_var BIGINT) -> DOUBLE
```

**Returns:** Cramér's V as a `DOUBLE` in `[0, 1]`.

```sql
SELECT cramers_v_agg(gender, preference) AS v FROM survey;
```

**Interpretation:**
- V = 0.1: Small effect
- V = 0.3: Medium effect
- V = 0.5: Large effect

### phi_coefficient_agg

Phi coefficient for 2x2 tables (-1 to 1).

**Signature:**
```text
phi_coefficient_agg(row_var BIGINT, col_var BIGINT) -> DOUBLE
```

**Returns:** the phi coefficient as a `DOUBLE` in `[-1, 1]`.

```sql
SELECT phi_coefficient_agg(a, b) AS phi
FROM (VALUES (1, 1), (1, 1), (1, 0), (0, 0), (0, 0), (0, 1), (1, 1), (0, 0)) t(a, b);
```

### contingency_coef_agg

Contingency coefficient (Pearson's C).

**Signature:**
```text
contingency_coef_agg(row_var BIGINT, col_var BIGINT) -> DOUBLE
```

**Returns:** Pearson's contingency coefficient C as a `DOUBLE` in `[0, 1)`.

```sql
SELECT contingency_coef_agg(gender, preference) AS c FROM survey;
```

### cohen_kappa_agg

Cohen's kappa for inter-rater agreement.

**Signature:**
```text
cohen_kappa_agg(rater1 BIGINT, rater2 BIGINT [, options MAP]) -> STRUCT
```

**Options** (pass as a `MAP {...}` literal):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| weighted | BOOLEAN | false | Weighted kappa for ordinal categories |

**Returns:**
```
STRUCT(
    kappa DOUBLE,        -- Kappa coefficient
    se DOUBLE,           -- Standard error
    ci_lower DOUBLE,     -- CI lower bound
    ci_upper DOUBLE,     -- CI upper bound
    z DOUBLE,            -- Z-statistic
    p_value DOUBLE       -- p-value
)
```

```sql
SELECT unnest(cohen_kappa_agg(r1, r2))
FROM (VALUES (1, 1), (2, 2), (3, 3), (1, 2), (2, 2), (3, 3), (1, 1), (3, 2), (2, 2), (1, 1)) t(r1, r2);
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
```text
prop_test_one_agg(value BIGINT [, options MAP]) -> STRUCT
```

`value` is 1 for a success and 0 for a failure, one row per trial.

**Options** (pass as a `MAP {...}` literal):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| p0 (alias `p`) | DOUBLE | 0.5 | Null hypothesis proportion |
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |

### prop_test_two_agg

Two-sample proportion test.

**Signature:**
```text
prop_test_two_agg(value BIGINT, group_id BIGINT [, options MAP]) -> STRUCT
```

**Options** (pass as a `MAP {...}` literal):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alternative | VARCHAR | 'two_sided' | 'two_sided', 'less', 'greater' |
| correction | BOOLEAN | true | Continuity correction |

Both proportion tests return `STRUCT(statistic DOUBLE, p_value DOUBLE,
estimate DOUBLE, ci_lower DOUBLE, ci_upper DOUBLE, n BIGINT, method VARCHAR)`.

```sql
SELECT unnest(prop_test_one_agg(success, MAP {'p0': 0.3}))
FROM (SELECT CASE WHEN i % 5 < 2 THEN 1 ELSE 0 END AS success FROM range(100) r(i));
```

### binom_test_agg

Exact binomial test.

**Signature:**
```text
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
