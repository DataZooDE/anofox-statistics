# ALM (Augmented Linear Models)

Augmented Linear Models with 24 error distribution families for flexible regression.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `alm_fit_agg` | Aggregate | Fit ALM with choice of distribution |
| `alm_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `alm_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_alm_fit_agg / alm_fit_agg

Fit an Augmented Linear Model with choice of distribution and loss function.

**Signature:**
```sql
anofox_stats_alm_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| distribution | VARCHAR | 'normal' | Error distribution (see below) |
| loss | VARCHAR | 'likelihood' | Loss: 'likelihood', 'mse', 'mae', 'ham', 'role' |
| max_iterations | INTEGER | 100 | Maximum iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| quantile | DOUBLE | 0.5 | Quantile for asymmetric_laplace |
| role_trim | DOUBLE | 0.05 | Trim parameter for ROLE loss |
| compute_inference | BOOLEAN | false | Compute t-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | CI confidence level |

**Returns:** [AlmFitResult](../reference/return_types.md#almfitresult-structure) STRUCT

## Supported Distributions

### Continuous (Unbounded)
| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `normal` | Gaussian | Standard regression |
| `laplace` | Double exponential | Robust/median regression |
| `student_t` | Heavy tails | Outlier-robust regression |
| `logistic` | Logistic | Bounded tails |
| `asymmetric_laplace` | Quantile regression | Specific quantiles |
| `generalised_normal` | Flexible shape | Variable tail behavior |
| `s` | S distribution | Heavy tails |

### Continuous (Positive)
| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `log_normal` | Log-normal | Multiplicative processes |
| `log_laplace` | Log-Laplace | Robust positive outcomes |
| `log_s` | Log-S | Heavy-tailed positive |
| `log_generalised_normal` | Log-GN | Flexible positive |
| `gamma` | Gamma | Positive with variance ~ μ² |
| `inverse_gaussian` | Inverse Gaussian | Positive with variance ~ μ³ |
| `exponential` | Exponential | Memoryless positive |

### Continuous (Bounded)
| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `folded_normal` | Folded normal | Absolute values |
| `rectified_normal` | Rectified normal | Zero-inflated positive |
| `box_cox_normal` | Box-Cox normal | Power transforms |
| `beta` | Beta (0-1) | Proportions, rates |
| `logit_normal` | Logit-normal | Proportions |

### Count
| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `poisson` | Poisson | Equidispersed counts |
| `negative_binomial` | Negative binomial | Overdispersed counts |
| `binomial` | Binomial | Bounded counts |
| `geometric` | Geometric | Count until success |

### Ordinal
| Distribution | Description | Use Case |
|--------------|-------------|----------|
| `cumulative_logistic` | Cumulative logit | Ordinal outcomes |
| `cumulative_normal` | Cumulative probit | Ordinal outcomes |

## Examples

```sql
-- Robust regression with Laplace distribution (median regression)
SELECT alm_fit_agg(y, [x1, x2], {'distribution': 'laplace'})
FROM data_with_outliers;

-- Quantile regression (75th percentile)
SELECT alm_fit_agg(
    price,
    [sqft, bedrooms],
    {'distribution': 'asymmetric_laplace', 'quantile': 0.75}
)
FROM housing;

-- Gamma regression for positive data
SELECT alm_fit_agg(
    claim_amount,
    [age, risk_score],
    {'distribution': 'gamma', 'compute_inference': true}
)
FROM insurance_claims;

-- Beta regression for proportions (0-1)
SELECT alm_fit_agg(
    conversion_rate,
    [ad_spend, page_views],
    {'distribution': 'beta'}
)
FROM marketing_data;
```

## Use Cases

- **Robust regression**: Laplace, Student-t for outliers
- **Quantile regression**: asymmetric_laplace for specific quantiles
- **Positive outcomes**: gamma, log_normal for claims, prices
- **Proportions/rates**: beta, logit_normal for (0,1) data
- **Overdispersed counts**: negative_binomial when Poisson fails

## See Also

- [Poisson](poisson.md) - Standard GLM for counts
- [Quantile Regression](../regression/quantile.md) - Alternative quantile approach
- [Table Macros](../macros/table_macros.md#alm_fit_predict_by) - Per-group predictions
