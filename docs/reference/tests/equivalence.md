# Equivalence Tests

TOST (Two One-Sided Tests) procedures for demonstrating equivalence.

## Overview

Equivalence tests flip the null hypothesis: H0 is that groups are NOT equivalent.

- Traditional test: H0: μ1 = μ2 (no difference)
- Equivalence test: H0: |μ1 - μ2| ≥ Δ (not equivalent)

Rejection of H0 demonstrates equivalence within margin Δ.

## tost_t_test_agg

TOST for two independent samples.

**Signature:**
```sql
tost_t_test_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| equivalence_margin | required | Δ, the equivalence margin |
| confidence_level | 0.90 | 90% CI for equivalence tests |

**Returns:**
- `p_value` - Max of two one-sided p-values
- `ci_lower`, `ci_upper` - (1-2α) CI for difference
- `lower_p`, `upper_p` - Individual one-sided p-values
- `equivalent` - Boolean: is p_value < α?

**Example:**
```sql
-- Test if groups are equivalent within ±0.5 units
SELECT (tost_t_test_agg(score, group, {'equivalence_margin': 0.5})).*
FROM bioequivalence_study;
```

---

## tost_paired_agg

TOST for paired samples.

**Signature:**
```sql
tost_paired_agg(diff DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| equivalence_margin | required | Δ |

**Example:**
```sql
SELECT (tost_paired_agg(after - before, {'equivalence_margin': 0.3})).*
FROM paired_data;
```

---

## tost_correlation_agg

TOST for correlation equivalence.

**Signature:**
```sql
tost_correlation_agg(x DOUBLE, y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| r_margin | required | Margin around hypothesized r |
| r_null | 0 | Hypothesized correlation |

**Example:**
```sql
-- Test if correlation is equivalent to 0.8 ± 0.1
SELECT (tost_correlation_agg(x, y, {
    'r_null': 0.8,
    'r_margin': 0.1
})).*
FROM reliability_data;
```

## Choosing Equivalence Margin

The margin Δ should be:
- Clinically/practically meaningful
- Pre-specified before data collection
- Based on domain expertise

Common approaches:
- 10-20% of control mean
- Based on minimal important difference
- Regulatory guidelines (e.g., FDA bioequivalence: 80-125%)
