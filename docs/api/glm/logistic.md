# Logistic Regression

Binary logistic regression (Binomial GLM with logit link), returning the
training-set classification accuracy at a configurable threshold alongside the
usual GLM fit statistics.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `logistic_fit_agg` | Aggregate | Fit a binary logistic regression |
| `logistic_fit_predict_by` | Table macro | Per-group fit, predicted probabilities for every row, see [Table macros](../macros/table_macros.md#logistic_fit_predict_by) |

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
| `family` | VARCHAR | `'binomial'`; used by [`predict`](../regression/model_tools.md#predict) |
| `link` | VARCHAR | `'logit'` |

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

## Prediction

Apply a fitted model to new rows with
[`predict(model, x)`](../regression/model_tools.md#predict). The model's `link`
field maps the linear predictor back to the response scale; pass
`{'type': 'link'}` for the log odds instead. [`tidy`](../regression/model_tools.md#tidy)
and [`glance`](../regression/model_tools.md#glance) give the per-term table and
the fit summary.

```sql
WITH fit AS (SELECT logistic_fit_agg(churned, [support_tickets]) AS m FROM churn)
SELECT m.family, m.link,
       round(predict(m, [8.0]), 4) AS response,   -- churn probability with 8 tickets
       round(predict(m, [8.0], {'type': 'link'}), 4) AS linear_predictor
FROM fit;
```

### logistic_fit_predict_by

```text
logistic_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fits `logistic_fit_agg` on each group's training rows (`y` not NULL and `split`
NULL or `'train'`) and applies `predict(model, x)` to every row. Returns all
source columns plus `yhat` (response scale), `yhat_lower` and `yhat_upper`
(NULL: no interval for this GLM) and `is_training`, ordered by the group
column. A group whose fit fails gets NULL `yhat`. See
[Table macros](../macros/table_macros.md#logistic_fit_predict_by).

```sql
CREATE OR REPLACE VIEW churn_seg AS
SELECT *,
       CASE WHEN customer_id % 2 = 0 THEN 'even' ELSE 'odd' END AS segment,
       CASE WHEN customer_id % 10 < 8 THEN 'train' ELSE 'test' END AS split
FROM churn;

SELECT segment, customer_id, churned, round(yhat, 3) AS yhat, is_training
FROM logistic_fit_predict_by('churn_seg', segment, churned, [support_tickets], split := split)
WHERE NOT is_training
ORDER BY segment, customer_id
LIMIT 4;
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
