# ALM (Augmented Linear Models)

Augmented Linear Models with 25 error distribution families for flexible regression.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `alm_fit_agg` | Aggregate | Fit ALM with choice of distribution |
| `alm_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `alm_fit_predict_by` | Table Macro | Per-group regression with long-format output |

The examples on this page use this table:

```sql
CREATE OR REPLACE TABLE housing AS
SELECT i AS id,
       'R' || (i % 3) AS region,
       (50 + (i % 40) * 3)::DOUBLE AS sqft,
       (1 + i % 4)::DOUBLE AS bedrooms,
       (20 + 1.5 * (50 + (i % 40) * 3) + 8 * (1 + i % 4) + ((i * 37) % 23) - 11
           + CASE WHEN i % 50 = 0 THEN 200 ELSE 0 END)::DOUBLE AS price,  -- a few outliers
       0.05 + 0.002 * (i % 40) + ((i * 37) % 10) / 200.0 AS conversion_rate
FROM range(200) r(i);
```

## alm_fit_agg

Fit an Augmented Linear Model with a choice of error distribution and loss.

**Signature:**

```text
alm_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include intercept term |
| `distribution` (alias `dist`) | VARCHAR | `'normal'` | Error distribution (see below) |
| `loss` | VARCHAR | `'likelihood'` | Loss: `'likelihood'`, `'mse'`, `'mae'`, `'ham'`, `'role'` |
| `max_iterations` | INTEGER | `100` | Maximum optimiser iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `quantile` | DOUBLE | `0.5` | Quantile for `asymmetric_laplace` |
| `role_trim` | DOUBLE | `0.05` | Trim proportion for the ROLE loss |
| `compute_inference` | BOOLEAN | `false` | Compute t-tests, p-values, CIs |
| `confidence_level` | DOUBLE | `0.95` | CI confidence level |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Feature coefficients (excluding intercept) |
| `intercept` | DOUBLE | Intercept |
| `log_likelihood` | DOUBLE | Log-likelihood at the optimum |
| `aic` | DOUBLE | Akaike information criterion |
| `bic` | DOUBLE | Bayesian information criterion |
| `scale` | DOUBLE | Estimated scale parameter of the distribution |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | Optimiser iterations (0 for closed-form fits) |
| `std_errors` | DOUBLE[] | Only with `compute_inference` |
| `t_values` | DOUBLE[] | Only with `compute_inference` |
| `p_values` | DOUBLE[] | Only with `compute_inference` |
| `ci_lower` | DOUBLE[] | Only with `compute_inference` |
| `ci_upper` | DOUBLE[] | Only with `compute_inference` |

## alm_fit_predict_agg

Fits on the training rows and returns one prediction per input row, in input
order. Works as an aggregate or a window function.

```text
alm_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP])
alm_fit_predict_agg(y DOUBLE, x DOUBLE[], split VARCHAR [, options MAP])
    -> LIST(STRUCT(y DOUBLE, yhat DOUBLE, yhat_lower DOUBLE,
                   yhat_upper DOUBLE, is_training BOOLEAN))
```

Rows with a NULL `y` are predicted but not trained on; with the `split` form
only rows whose split value is `'train'` are trained on. Options: the
`alm_fit_agg` keys except `compute_inference`, plus `null_policy`
(`'drop'` default, or `'drop_y_zero_x'`).

## alm_fit_predict_by

```text
alm_fit_predict_by(source VARCHAR, group_col, y_col, x_cols
                   [, options MAP] [, split VARCHAR column]) -> TABLE
```

Returns every source row plus `yhat`, `yhat_lower`, `yhat_upper`,
`is_training`. See [Table Macros](../macros/table_macros.md#alm_fit_predict_by).

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
-- Robust regression with a Laplace distribution (median regression)
SELECT alm_fit_agg(price, [sqft, bedrooms], {'distribution': 'laplace'}) AS fit
FROM housing;

-- 75th-percentile regression
SELECT (alm_fit_agg(price, [sqft, bedrooms],
                    {'distribution': 'asymmetric_laplace', 'quantile': 0.75})).coefficients
FROM housing;

-- Gamma distribution for a positive response, with inference
SELECT (alm_fit_agg(price, [sqft, bedrooms],
                    {'distribution': 'gamma', 'compute_inference': true})).std_errors
FROM housing;

-- Beta regression for proportions in (0, 1)
SELECT (alm_fit_agg(conversion_rate, [sqft], {'distribution': 'beta'})).coefficients
FROM housing;

-- Per-region fit + predictions
SELECT region, id, price, round(yhat, 1) AS yhat, is_training
FROM alm_fit_predict_by('housing', region, price, [sqft, bedrooms],
                        {'distribution': 'laplace'})
LIMIT 5;
```

## NULL Handling

- `alm_fit_agg` skips rows where `y` is NULL, the `x` list is NULL, or the list
  contains a NULL element.
- All `x` lists must have the same length, otherwise an error is raised.
- Fewer than two usable rows, or a failed fit, returns `NULL`.

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
