# ALM Functions

Augmented Linear Models supporting 24 error distributions.

## Function Overview

| Function | Type | Description |
|----------|------|-------------|
| `alm_fit_agg` | Aggregate | ALM with GROUP BY support |
| `alm_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `alm_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## alm_fit_agg

Augmented Linear Model with flexible error distributions.

**Signature:**
```sql
alm_fit_agg(
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
SELECT alm_fit_agg(y, [x], 'student_t')
FROM data_with_outliers;
```

## alm_fit_predict_agg

Aggregate function returning predictions array.

## alm_fit_predict_by

**Recommended for predictions.** Table macro for grouped fit-predict.

```sql
FROM alm_fit_predict_by('robust_data', grp, y, [x], {'distribution': 'student_t'});
```

**Options:** `distribution`, `fit_intercept`, `null_policy`

## Short Aliases

- `alm_fit_agg` -> `anofox_stats_alm_fit_agg`
- `alm_fit_predict_agg` -> `anofox_stats_alm_fit_predict_agg`
