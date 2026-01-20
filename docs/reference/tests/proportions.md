# Proportion Tests

Tests for binary proportions.

## prop_test_one_agg

One-sample proportion test (z-test).

**Signature:**
```sql
prop_test_one_agg(success INTEGER, n INTEGER, p0 DOUBLE, [options MAP]) -> STRUCT
```

**Parameters:**
- `success` - Number of successes
- `n` - Total trials
- `p0` - Hypothesized proportion

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |
| confidence_level | 0.95 | CI confidence level |

**Returns:**
- `statistic` - z-statistic
- `p_value` - p-value
- `proportion` - Sample proportion
- `ci_lower`, `ci_upper` - Confidence interval

**Example:**
```sql
-- Test if proportion differs from 0.5
SELECT (prop_test_one_agg(successes, total, 0.5)).*
FROM summary_data;
```

---

## prop_test_two_agg

Two-sample proportion test.

**Signature:**
```sql
prop_test_two_agg(success1 INTEGER, n1 INTEGER, success2 INTEGER, n2 INTEGER, [options MAP]) -> STRUCT
```

**Returns:**
- `statistic` - z-statistic
- `p_value` - p-value
- `diff` - Difference in proportions
- `ci_lower`, `ci_upper` - CI for difference

**Example:**
```sql
-- Compare conversion rates
SELECT (prop_test_two_agg(
    conversions_a, visitors_a,
    conversions_b, visitors_b
)).*
FROM ab_test;
```

---

## binom_test_agg

Exact binomial test.

**Signature:**
```sql
binom_test_agg(success INTEGER, n INTEGER, p0 DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| alternative | 'two_sided' | 'two_sided', 'less', 'greater' |

**Returns:**
- `p_value` - Exact p-value
- `proportion` - Sample proportion

**Example:**
```sql
-- Exact test for small samples
SELECT (binom_test_agg(successes, trials, 0.5)).*
FROM small_sample;
```

**Notes:**
- Exact test (no normal approximation)
- Better for small samples (n < 30)
- No continuity correction needed

## When to Use Each Test

| Sample Size | Test |
|-------------|------|
| n·p > 10 and n·(1-p) > 10 | prop_test (z-test) |
| Small samples | binom_test (exact) |
