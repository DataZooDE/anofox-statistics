# Gamma GLM

Regression for strictly positive, right-skewed continuous responses whose
variance grows with the square of the mean (claim amounts, durations, prices).
Fitted with a log link by IRLS.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `gamma_fit_agg` | Aggregate | Fit a Gamma GLM (log link) |
| `gamma_fit_predict_by` | Table macro | Per-group fit, expected values for every row, see [Table macros](../macros/table_macros.md#gamma_fit_predict_by) |

## gamma_fit_agg

**Signature:**

```text
gamma_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Response; strictly positive |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include an intercept |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Standard errors, z-tests, p-values, intervals |
| `confidence_level` | DOUBLE | `0.95` | Interval level |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the design) |
| `feature_names`, `prior`, `vcov` | | | See [Explicit priors](priors.md) |

The link is always log.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Coefficients on the log scale; `NaN` for an aliased column |
| `intercept` | DOUBLE | Intercept |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Intercept-only deviance (fitted with the offset, if any), as R's `glm` |
| `pseudo_r_squared` | DOUBLE | `1 - deviance / null_deviance` |
| `aic` | DOUBLE | Akaike information criterion; the log-likelihood uses the dispersion `deviance / n`, exactly R's `AIC(glm(..., family = Gamma))` |
| `dispersion` | DOUBLE | Pearson χ² / residual df; scales the standard errors |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | IRLS iterations performed |
| `converged` | BOOLEAN | Whether IRLS reached the tolerance |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] | Only with `compute_inference` |
| `family` | VARCHAR | `'gamma'`; used by [`predict`](../regression/model_tools.md#predict) |
| `link` | VARCHAR | `'log'` |

**Example:**

```sql
CREATE OR REPLACE TABLE claims AS
SELECT i AS claim_id,
       (i % 5)::DOUBLE AS risk_score,
       exp(0.5 + 0.3 * (i % 5)) * (0.5 + ((i * 37) % 100) / 100.0) AS amount
FROM range(200) r(i);

SELECT (fit).coefficients[1] AS log_effect,
       exp((fit).coefficients[1]) AS multiplicative_effect,
       (fit).dispersion,
       (fit).p_values[1] AS p_value
FROM (
    SELECT gamma_fit_agg(amount, [risk_score], {'compute_inference': true}) AS fit
    FROM claims
);
```

## Prediction

Apply a fitted model to new rows with
[`predict(model, x)`](../regression/model_tools.md#predict). The model's `link`
field maps the linear predictor back to the response scale; pass
`{'type': 'link'}` for the log mean instead. [`tidy`](../regression/model_tools.md#tidy)
and [`glance`](../regression/model_tools.md#glance) give the per-term table and
the fit summary.

```sql
WITH fit AS (SELECT gamma_fit_agg(amount, [risk_score]) AS m FROM claims)
SELECT m.family, m.link,
       round(predict(m, [4.0]), 4) AS response,   -- expected amount at risk score 4
       round(predict(m, [4.0], {'type': 'link'}), 4) AS linear_predictor
FROM fit;
```

### gamma_fit_predict_by

```text
gamma_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fits `gamma_fit_agg` on each group's training rows (`y` not NULL and `split`
NULL or `'train'`) and applies `predict(model, x)` to every row. Returns all
source columns plus `yhat` (response scale), `yhat_lower` and `yhat_upper`
(NULL: no interval for this GLM) and `is_training`, ordered by the group
column. A group whose fit fails gets NULL `yhat`. See
[Table macros](../macros/table_macros.md#gamma_fit_predict_by).

```sql
CREATE OR REPLACE VIEW claims_seg AS
SELECT *,
       CASE WHEN claim_id % 2 = 0 THEN 'even' ELSE 'odd' END AS segment,
       CASE WHEN claim_id % 10 < 8 THEN 'train' ELSE 'test' END AS split
FROM claims;

SELECT segment, claim_id, amount, round(yhat, 3) AS yhat, is_training
FROM gamma_fit_predict_by('claims_seg', segment, amount, [risk_score], split := split)
WHERE NOT is_training
ORDER BY segment, claim_id
LIMIT 4;
```

## Interpreting Coefficients

With the log link, `exp(beta)` is the multiplicative effect on the expected
response per unit increase in the feature. The model is equivalent to a Tweedie
GLM with `power = 2`.

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped; a NULL element inside `x`
  or any non-finite value drops the row. `n_observations` counts the rows used.
- Fewer than two usable rows, a `y` that is zero or negative, or a failed fit
  returns `NULL`. For responses that include exact zeros use
  [Tweedie](tweedie.md).

## See Also

- [Tweedie GLM](tweedie.md) — non-negative responses with exact zeros
- [AFT survival regression](../survival/aft.md) — positive durations with censoring
- [ALM](alm.md) — `'distribution': 'gamma'` and other positive distributions
- [Poisson GLM](poisson.md) — field descriptions shared by all GLM aggregates
