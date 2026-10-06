# Binomial GLM

Regression for binary outcomes or proportions in `[0, 1]`, with a choice of
link function (logit, probit, complementary log-log).

For a plain logistic regression with training accuracy, see
[Logistic regression](logistic.md).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `binomial_fit_agg` | Aggregate | Fit a Binomial GLM |

## binomial_fit_agg

**Signature:**

```text
binomial_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Response in `[0, 1]`: 0/1 outcomes or observed proportions |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include an intercept |
| `binomial_link` | VARCHAR | `'logit'` | Link function: `'logit'`, `'probit'`, `'cloglog'` |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Standard errors, z-tests, p-values, intervals |
| `confidence_level` | DOUBLE | `0.95` | Interval level |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the design) |
| `feature_names`, `prior`, `vcov` | | | See [Explicit priors](priors.md) |

Note that the link key is `binomial_link`; the `link` key belongs to the
Poisson GLM.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Feature coefficients on the link scale; `NaN` for an aliased column |
| `intercept` | DOUBLE | Intercept |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Intercept-only deviance |
| `pseudo_r_squared` | DOUBLE | `1 - deviance / null_deviance` |
| `aic` | DOUBLE | Akaike information criterion |
| `dispersion` | DOUBLE | Fixed at 1.0 |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | IRLS iterations performed |
| `converged` | BOOLEAN | Whether IRLS reached the tolerance |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] | Only with `compute_inference` |

**Example:**

```sql
CREATE OR REPLACE TABLE trials AS
SELECT i,
       (i % 10)::DOUBLE AS dose,
       CASE WHEN ((i * 37) % 100) / 100.0 < 1 / (1 + exp(-(0.6 * (i % 10) - 3)))
            THEN 1.0 ELSE 0.0 END AS response
FROM range(200) r(i);

-- Logit link (default) with inference
SELECT binomial_fit_agg(response, [dose], {'compute_inference': true}) AS fit
FROM trials;

-- Probit link
SELECT (binomial_fit_agg(response, [dose], {'binomial_link': 'probit'})).coefficients
FROM trials;
```

## Link Functions

| Link | Inverse link | Notes |
|------|--------------|-------|
| `logit` (default) | `1 / (1 + exp(-eta))` | Coefficients are log odds ratios |
| `probit` | `Phi(eta)` | Latent normal threshold model |
| `cloglog` | `1 - exp(-exp(eta))` | Asymmetric; rare events, grouped survival data |

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped; a NULL element inside `x`
  or any non-finite value drops the row. `n_observations` counts the rows used.
- Fewer than two usable rows, a `y` outside `[0, 1]`, or a failed fit (e.g.
  perfect separation that prevents convergence) returns `NULL`.

## See Also

- [Logistic regression](logistic.md) — binary 0/1 outcome with accuracy
- [Mixed-effects GLMs](glmm.md) — `'family': 'binomial'` with a random intercept
- [Poisson GLM](poisson.md) — field descriptions shared by all GLM aggregates
