# Distribution Comparison Tests

Tests for comparing two distributions.

## energy_distance_agg

Energy distance measures distributional difference.

**Signature:**
```sql
energy_distance_agg(value DOUBLE, group_id INTEGER) -> STRUCT
```

**Returns:**
- `statistic` - Energy distance (≥ 0)
- `p_value` - Permutation p-value

**Example:**
```sql
SELECT (energy_distance_agg(measurement, sample_id)).*
FROM two_sample_data;
```

**Interpretation:**
- Distance = 0: Identical distributions
- Larger distance = More different distributions

**Notes:**
- Detects any distributional difference
- Based on distances between observations
- Permutation test for inference

---

## mmd_agg

Maximum Mean Discrepancy with kernel embedding.

**Signature:**
```sql
mmd_agg(value DOUBLE, group_id INTEGER, [options MAP]) -> STRUCT
```

**Options:**
| Key | Default | Description |
|-----|---------|-------------|
| kernel | 'gaussian' | Kernel type |
| bandwidth | 'median' | Kernel bandwidth or 'median' heuristic |

**Returns:**
- `statistic` - MMD statistic
- `p_value` - Permutation p-value

**Example:**
```sql
SELECT (mmd_agg(feature, group)).*
FROM distribution_comparison;
```

**Notes:**
- Embeds distributions in reproducing kernel Hilbert space
- Can detect subtle differences
- Commonly used in machine learning

## Comparison

| Test | Detects | Computation |
|------|---------|-------------|
| Energy distance | Location, scale, shape | O(n²) |
| MMD | Any difference (kernel-dependent) | O(n²) |
| KS test | Maximum CDF difference | O(n log n) |

## When to Use

- **Energy distance:** General-purpose, interpretable
- **MMD:** When specific kernel characteristics matter
- Use over KS test when interested in any distributional difference, not just location
