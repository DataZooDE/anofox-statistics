# Logistic Regression

Binary logistic regression (Binomial GLM with logit link), returning the
training-set classification accuracy at a configurable threshold alongside the
usual GLM fit statistics.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `logistic_fit_agg` | Aggregate | Fit a binary logistic regression |

## logistic_fit_agg

**Signature:**

```text
logistic_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Binary response: exactly `0.0` or `1.0` |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include an intercept |
| `threshold` | DOUBLE | `0.5` | Classification threshold in `[0, 1]` used for `accuracy` |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Standard errors, z-tests, p-values, intervals |
| `confidence_level` | DOUBLE | `0.95` | Interval level |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the design) |
| `feature_names`, `prior`, `vcov` | | | See [Explicit priors](priors.md) |

The link is always logit. Use [`binomial_fit_agg`](binomial.md) for probit or
cloglog links, or for proportions.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Log odds ratios; `NaN` for an aliased column |
| `intercept` | DOUBLE | Intercept (log odds at x = 0) |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Intercept-only deviance |
| `pseudo_r_squared` | DOUBLE | `1 - deviance / null_deviance` |
| `aic` | DOUBLE | Akaike information criterion |
| `accuracy` | DOUBLE | Share of training rows classified correctly at `threshold` |
| `threshold` | DOUBLE | Threshold used (echoed from options) |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | IRLS iterations performed |
| `converged` | BOOLEAN | Whether IRLS reached the tolerance |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] | Only with `compute_inference` |

**Example:**

```sql
CREATE OR REPLACE TABLE churn AS
SELECT i AS customer_id,
       (i % 10)::DOUBLE AS support_tickets,
       CASE WHEN ((i * 37) % 100) / 100.0 < 1 / (1 + exp(-(0.6 * (i % 10) - 3)))
            THEN 1.0 ELSE 0.0 END AS churned
FROM range(200) r(i);

SELECT (fit).coefficients[1] AS log_odds_per_ticket,
       exp((fit).coefficients[1]) AS odds_ratio,
       (fit).p_values[1] AS p_value,
       (fit).accuracy
FROM (
    SELECT logistic_fit_agg(churned, [support_tickets],
                            {'compute_inference': true}) AS fit
    FROM churn
);

-- Accuracy at a stricter threshold
SELECT (logistic_fit_agg(churned, [support_tickets], {'threshold': 0.7})).accuracy
FROM churn;
```

## Interpreting Coefficients

Coefficients are log odds ratios: `exp(beta)` is the multiplicative change in
the odds of `y = 1` per unit increase in the feature. Predicted probabilities
are `1 / (1 + exp(-(intercept + x'beta)))`.

`accuracy` is measured on the training data and is optimistic; evaluate on a
held-out set for a fair estimate.

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped; a NULL element inside `x`
  or any non-finite value drops the row. `n_observations` counts the rows used.
- Fewer than two usable rows, a `y` other than 0 or 1, a `threshold` outside
  `[0, 1]`, or a failed fit (e.g. perfect separation) returns `NULL`.

## See Also

- [Binomial GLM](binomial.md) — other links, proportions
- [Poisson GLM](poisson.md) — field descriptions shared by all GLM aggregates
- [Explicit priors](priors.md) — regularise individual coefficients
