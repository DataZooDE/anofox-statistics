# Poisson GLM

Poisson regression for count data, fitted by maximum likelihood via
Iteratively Reweighted Least Squares (IRLS).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `poisson_fit_agg` | Aggregate | Fit a Poisson GLM |
| `poisson_fit_predict_agg` | Aggregate | Fit on training rows, predict every row |
| `poisson_fit_predict_by` | Table Macro | Per-group fit + predict with long-format output |

The examples on this page use this table:

```sql
-- Weekly unit sales for 20 SKUs, half the weeks on promotion
CREATE OR REPLACE TABLE demand AS
SELECT i AS week,
       'SKU-' || (i % 4) AS sku,
       (i % 2)::DOUBLE AS promo,
       ((i % 3) + 1)::DOUBLE AS shelf_facings,
       (((i * 7) % 11) + 3 * (i % 2) + (i % 3))::DOUBLE AS qty
FROM range(200) r(i);
```

## poisson_fit_agg

**Signature:**

```text
poisson_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Response; non-negative counts |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include an intercept term |
| `link` | VARCHAR | `'log'` | Link function: `'log'`, `'identity'`, `'sqrt'` |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Add standard errors, z-tests, p-values and confidence intervals |
| `confidence_level` | DOUBLE | `0.95` | Confidence level for the intervals |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength; 0 disables it |
| `offset` | INTEGER | none | 1-based index into `x` of a column used as an offset: added to the linear predictor with coefficient fixed at 1 and removed from the design. Values are used as-is, so take logs upstream for the log link (e.g. `log(exposure)`). |
| `feature_names`, `prior`, `vcov` | | | Explicit coefficient priors; see [Explicit priors](priors.md) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Feature coefficients (excluding intercept); `NaN` for an aliased column |
| `intercept` | DOUBLE | Intercept (0 when `fit_intercept` is false) |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Deviance of the intercept-only model (fitted with the offset, if any; without an intercept the deviance at `μ = exp(offset)`), as R's `glm` |
| `pseudo_r_squared` | DOUBLE | McFadden-style deviance ratio `1 - deviance / null_deviance` |
| `aic` | DOUBLE | Akaike information criterion |
| `dispersion` | DOUBLE | Always 1.0: standard errors use the Poisson variance, exactly R's `summary(glm(..., family = poisson))`. |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | IRLS iterations performed |
| `converged` | BOOLEAN | Whether IRLS reached the tolerance |
| `std_errors` | DOUBLE[] | Only with `compute_inference` |
| `z_values` | DOUBLE[] | Only with `compute_inference` |
| `p_values` | DOUBLE[] | Only with `compute_inference` |
| `ci_lower` | DOUBLE[] | Only with `compute_inference` |
| `ci_upper` | DOUBLE[] | Only with `compute_inference` |
| `family` | VARCHAR | `'poisson'`; used by [`predict`](../regression/model_tools.md#predict) |
| `link` | VARCHAR | The link used: `'log'`, `'identity'` or `'sqrt'` |

The same struct (with `dispersion`, `iterations`, `converged`, `family` and
`link`) is returned by
all GLM aggregates: `poisson`, [`binomial`](binomial.md),
[`negbinom`](negbinom.md), [`gamma`](gamma.md), [`tweedie`](tweedie.md); the
[`logistic`](logistic.md) aggregate replaces `dispersion` with `accuracy` and
`threshold`.

**Examples:**

```sql
-- Basic Poisson regression
SELECT poisson_fit_agg(qty, [promo, shelf_facings]) AS fit FROM demand;

-- With inference
SELECT (fit).coefficients, (fit).p_values, (fit).dispersion
FROM (
    SELECT poisson_fit_agg(qty, [promo, shelf_facings],
                           {'compute_inference': true}) AS fit
    FROM demand
);

-- One model per SKU
SELECT sku, (poisson_fit_agg(qty, [promo])).coefficients AS coefficients
FROM demand
GROUP BY sku
ORDER BY sku;

-- Expected count for a new row (response scale) and its log rate (link scale)
WITH fit AS (SELECT poisson_fit_agg(qty, [promo, shelf_facings]) AS m FROM demand)
SELECT round(predict(m, [1.0, 3.0]), 3) AS expected_qty,
       round(predict(m, [1.0, 3.0], {'type': 'link'}), 3) AS log_rate
FROM fit;

-- Ridge-penalised fit with an exposure offset (x-column 2 holds log(exposure))
SELECT poisson_fit_agg(qty, [promo, ln(shelf_facings)],
                       {'glm_lambda': 0.01, 'offset': 2}) AS fit
FROM demand;
```

## poisson_fit_predict_agg

Fits on the training rows and returns one prediction per input row, in input
order. Use it as an aggregate (`GROUP BY`) or as a window function
(`OVER (PARTITION BY ...)`).

**Signature:**

```text
poisson_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP])
poisson_fit_predict_agg(y DOUBLE, x DOUBLE[], split VARCHAR [, options MAP])
    -> LIST(STRUCT(y DOUBLE, yhat DOUBLE, yhat_lower DOUBLE,
                   yhat_upper DOUBLE, is_training BOOLEAN))
```

A row is a training row when `y` is not NULL. With the `split` form, only rows
whose split value is `'train'` or `'training'` (case-insensitive) and whose `y`
is not NULL are used for training; every row still gets a prediction.

**Options:** `fit_intercept`, `link`, `max_iterations`, `tolerance`,
`confidence_level`, `glm_lambda` (as above), plus `null_policy`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `null_policy` | VARCHAR | `'drop'` | `'drop'`: rows with NULL `y` are predicted but not trained on. `'drop_y_zero_x'`: additionally excludes training rows with any NULL feature. |

`yhat` is the predicted mean on the response scale. `yhat_lower`/`yhat_upper`
form an approximate interval derived from the dispersion on the link scale
(not the full coefficient uncertainty); `yhat_lower` is floored at 0. A row
with a NULL feature gets NULL predictions.

```sql
-- Hold out the last 20 weeks: their y is passed as NULL
SELECT p.y, round(p.yhat, 2) AS yhat, p.is_training
FROM (
    SELECT poisson_fit_predict_agg(
               CASE WHEN week < 180 THEN qty END, [promo, shelf_facings]) AS preds
    FROM demand
), UNNEST(preds) AS u(p)
LIMIT 5;
```

## poisson_fit_predict_by

Table macro that runs `poisson_fit_predict_agg` per group and returns every
source row with its prediction.

```text
poisson_fit_predict_by(source VARCHAR, group_col, y_col, x_cols
                       [, options MAP] [, split := column] [, order_by := column]) -> TABLE
```

Output: all source columns plus `yhat`, `yhat_lower`, `yhat_upper`,
`is_training`, ordered by the group column. When a split column is given, rows
whose split value is neither `'train'` nor NULL are predicted but not trained
on. `order_by` orders the rows inside each group so the alignment of
predictions to rows is deterministic.

```sql
SELECT sku, week, qty, round(yhat, 2) AS yhat, is_training
FROM poisson_fit_predict_by('demand', sku, qty, [promo, shelf_facings], order_by := week)
ORDER BY sku, week
LIMIT 5;
```

## Link Functions

| Link | Formula | Use Case |
|------|---------|----------|
| `log` (default) | μ = exp(Xβ) | Positive predictions, multiplicative effects |
| `identity` | μ = Xβ | Additive effects; can produce negative predictions |
| `sqrt` | μ = (Xβ)² | Compromise between log and identity |

## Interpreting Coefficients

With the log link, coefficients are log rate ratios: `exp(β)` is the
multiplicative effect on the expected count. A coefficient of 0.1 means a
one-unit increase multiplies the expected count by exp(0.1) ≈ 1.105.

Standard errors assume Poisson variance (dispersion 1, as R's `glm`). If the
Pearson χ² divided by the residual degrees of freedom is well above 1, the data
are overdispersed and these standard errors are too small; use
[Negative Binomial](negbinom.md), which models the extra variation explicitly.

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped.
- A NULL element inside `x`, or any non-finite value, drops that row from the fit;
  `n_observations` reports the rows actually used.
- All `x` lists must have the same length, otherwise an error is raised.
- Fewer than two usable rows, a negative `y`, or a failed fit returns `NULL`.

## Use Cases

- **Count data**: events, occurrences, frequencies
- **Rate modelling**: with an exposure `offset`
- **Insurance**: number of claims per policy
- **Quality control**: defect counts

## See Also

- [Negative Binomial](negbinom.md) — overdispersed counts
- [ALM](alm.md) — flexible distributions including negative binomial
- [Mixed-effects GLMs](glmm.md) — Poisson with a random intercept
- [Table Macros](../macros/table_macros.md#poisson_fit_predict_by)
