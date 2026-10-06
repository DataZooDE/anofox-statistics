# Binomial GLM

Regression for binary outcomes or proportions in `[0, 1]`, with a choice of
link function (logit, probit, complementary log-log).

For a plain logistic regression with training accuracy, see
[Logistic regression](logistic.md).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `binomial_fit_agg` | Aggregate | Fit a Binomial GLM |
| `binomial_fit_predict_by` | Table macro | Per-group fit, predictions (rates) for every row, see [Table macros](../macros/table_macros.md#binomial_fit_predict_by) |

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
| `family` | VARCHAR | `'binomial'`; used by [`predict`](../regression/model_tools.md#predict) |
| `link` | VARCHAR | The link used: `'logit'`, `'probit'` or `'cloglog'` |

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

## Prediction

Apply a fitted model to new rows with
[`predict(model, x)`](../regression/model_tools.md#predict). The model's `link`
field maps the linear predictor back to the response scale; pass
`{'type': 'link'}` for the log odds instead. [`tidy`](../regression/model_tools.md#tidy)
and [`glance`](../regression/model_tools.md#glance) give the per-term table and
the fit summary.

```sql
WITH fit AS (SELECT binomial_fit_agg(response, [dose]) AS m FROM trials)
SELECT m.family, m.link,
       round(predict(m, [7.0]), 4) AS response,   -- probability of a response at dose 7
       round(predict(m, [7.0], {'type': 'link'}), 4) AS linear_predictor
FROM fit;
```

### binomial_fit_predict_by

```text
binomial_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fits `binomial_fit_agg` on each group's training rows (`y` not NULL and `split`
NULL or `'train'`) and applies `predict(model, x)` to every row. Returns all
source columns plus `yhat` (response scale), `yhat_lower` and `yhat_upper`
(NULL: no interval for this GLM) and `is_training`, ordered by the group
column. A group whose fit fails gets NULL `yhat`. See
[Table macros](../macros/table_macros.md#binomial_fit_predict_by).

```sql
CREATE OR REPLACE VIEW trials_seg AS
SELECT *,
       CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END AS segment,
       CASE WHEN i % 10 < 8 THEN 'train' ELSE 'test' END AS split
FROM trials;

SELECT segment, i, response, round(yhat, 3) AS yhat, is_training
FROM binomial_fit_predict_by('trials_seg', segment, response, [dose], split := split)
WHERE NOT is_training
ORDER BY segment, i
LIMIT 4;
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
