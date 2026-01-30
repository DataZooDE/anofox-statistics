# Poisson GLM

Poisson regression for count data using maximum likelihood estimation via Iteratively Reweighted Least Squares (IRLS).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `poisson_fit_agg` | Aggregate | Fit Poisson GLM to count data |
| `poisson_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `poisson_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_poisson_fit_agg / poisson_fit_agg

Poisson regression for count data using maximum likelihood estimation.

**Signature:**
```sql
anofox_stats_poisson_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| link | VARCHAR | 'log' | Link function: 'log', 'identity', 'sqrt' |
| max_iterations | INTEGER | 100 | Maximum IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Compute z-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | CI confidence level |

**Returns:** [GlmFitResult](../reference/return_types.md#glmfitresult-structure) STRUCT

**Example:**
```sql
-- Basic Poisson regression for count data
SELECT poisson_fit_agg(count, [x1, x2])
FROM event_counts;

-- With inference and custom link
SELECT poisson_fit_agg(
    accidents,
    [traffic_volume, weather_score],
    {'compute_inference': true, 'link': 'log'}
)
FROM daily_accidents;

-- Per-group Poisson regression
SELECT
    region,
    (poisson_fit_agg(sales_count, [price, ads])).coefficients
FROM sales_data
GROUP BY region;
```

## Link Functions

| Link | Formula | Use Case |
|------|---------|----------|
| `log` (default) | μ = exp(Xβ) | Ensures positive predictions, multiplicative effects |
| `identity` | μ = Xβ | Additive effects, can produce negative predictions |
| `sqrt` | μ = (Xβ)² | Compromise between log and identity |

## Interpreting Coefficients

With the log link (default):
- Coefficients represent log rate ratios
- exp(β) gives the multiplicative effect on the count
- A coefficient of 0.1 means a 1-unit increase in x multiplies the expected count by exp(0.1) ≈ 1.105

## Use Cases

- **Count data**: Events, occurrences, frequencies
- **Rate modeling**: With exposure offsets
- **Insurance claims**: Number of claims per policy
- **Website analytics**: Page views, clicks
- **Quality control**: Defect counts
- **Epidemiology**: Disease incidence rates

## See Also

- [ALM](alm.md) - Flexible distributions including negative binomial
- [Table Macros](../macros/table_macros.md#poisson_fit_predict_by) - Per-group predictions
