# ALM Functions

Augmented Linear Models supporting 24 error distributions.

## anofox_stats_alm_fit_agg

Augmented Linear Model with flexible error distributions.

**Signature:**
```sql
anofox_stats_alm_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    distribution VARCHAR,
    [options MAP]
) -> STRUCT
```

**Supported Distributions:**

- `normal` - Gaussian
- `laplace` - Double exponential
- `student_t` - Student's t
- `cauchy` - Cauchy
- `logistic` - Logistic
- And 19 more...

**Example:**
```sql
-- Robust regression with Student's t errors
SELECT anofox_stats_alm_fit_agg(y, [x], 'student_t')
FROM data_with_outliers;
```

## Short Aliases

- `alm_fit_agg` -> `anofox_stats_alm_fit_agg`
